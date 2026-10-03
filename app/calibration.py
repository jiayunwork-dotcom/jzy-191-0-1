"""Kinetic parameter calibration.

Strategy
--------
Directly fitting (A, Ea) pairs is badly conditioned: A spans many orders of
magnitude and is almost perfectly correlated with Ea when experiments cover a
narrow temperature range.  We therefore fit, for each rate constant i:

    x = [ln k_ref_i, gamma_i]  with  gamma_i = Ea_i / R

and

    ln k_i(T) = ln k_ref_i + gamma_i * (1/T_ref - 1/T).

T_ref is fixed from the *data* -- the mean of the distinct batch
temperatures -- so ln k_ref describes the rate in the middle of the
experimental window (nearly orthogonal to gamma there).  Reported results are
always converted back to (A, Ea).

Identifiability
---------------
With data at a single temperature T0, the model depends on parameters only
through ln k(T0) = ln k_ref + gamma*(1/T_ref - 1/T0).  With T_ref = T0 the
gamma column of the sensitivity matrix is exactly zero, so gamma (= Ea/R) is
structurally non-identifiable; A is then unidentifiable too (any A/Ea pair on
the Arrhenius line fits), but k(T0) itself is identifiable.  This is detected
both structurally (distinct temperatures < 2 => all Ea non-identifiable) and
numerically via the rank of the (column-scaled) Jacobian.  Non-identifiable
parameters are reported with ``identifiable: false`` and a reason instead of
an arbitrary value.

Optimiser: Levenberg-Marquardt with a Gauss-Newton polish, written from
scratch (NumPy only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .chemistry import CompiledNetwork, R_GAS
from .integrator import IntegrationError, integrate
from .sensitivity import simulate_with_sensitivity
from .simulation import kvec_from_reparam, reparam_to_original, validate_batch

MAX_ITER = 200
FTOL = 1e-10
# internal ODE tolerances during fitting; much tighter than the 1e-4 target,
# but loosening below the user-facing 1e-10 default keeps the optimiser fast
CAL_RTOL = 1e-9
CAL_ATOL = 1e-11


@dataclass
class PreparedDataset:
    net: CompiledNetwork
    batches: list[dict]
    temperatures: np.ndarray
    T_ref: float
    labels: list[str]
    n_obs: int
    single_temperature: bool


def prepare_dataset(net: CompiledNetwork, batches: list[dict]) -> PreparedDataset:
    if not isinstance(batches, list) or not batches:
        from .chemistry import ValidationError
        raise ValidationError([("batches", "must be a non-empty list")])
    norm = [validate_batch(net, b, prefix=f"batches[{i}].") for i, b in enumerate(batches)]
    temps = np.array([b["temperature"] for b in norm], dtype=float)
    distinct = np.unique(np.round(temps, 8))
    T_ref = float(np.mean(distinct))
    labels = net.rate_constant_labels()
    n_obs = sum(b["Y"].size - int(np.isnan(b["Y"]).sum()) for b in norm)
    return PreparedDataset(
        net, norm, temps, T_ref, labels, n_obs,
        single_temperature=len(distinct) < 2,
    )


# ---------------------------------------------------------------------------
# forward model / residuals (with analytic sensitivity)
# ---------------------------------------------------------------------------

def _batch_residual_and_jac(beta, ds: PreparedDataset, bi: int):
    """Return (r, J) for one batch via one augmented integration.

    Raises :class:`IntegrationError` if the trial parameters make the model
    blow up; callers (line search) treat that as a rejected trial.
    """
    batch = ds.batches[bi]
    T = float(ds.temperatures[bi])
    C, S = simulate_with_sensitivity(
        ds.net, beta, T, ds.T_ref, batch["c0"], batch["times"],
        land=batch["times"], rtol=CAL_RTOL, atol=CAL_ATOL,
    )
    sim = C[:, batch["cols"]]
    # missing observations (NaN) are masked out
    mask = ~np.isnan(batch["Y"])
    r = (batch["Y"] - sim)[mask]
    # S shape (n_times, n_spec, n_par); select measured species then mask
    Jt = S[:, batch["cols"], :]  # (n_times, n_measured, n_par)
    J = (-Jt)[mask]  # d r / d beta = - d sim / d beta
    return r, J


def residuals_and_jacobian(beta: np.ndarray, ds: PreparedDataset,
                           free_mask: np.ndarray | None = None):
    rs, Js = [], []
    for bi in range(len(ds.batches)):
        r, J = _batch_residual_and_jac(beta, ds, bi)
        rs.append(r)
        Js.append(J)
    r = np.concatenate(rs)
    J = np.concatenate(Js, axis=0)
    if free_mask is not None:
        J = J[:, free_mask]
    return r, J


def _safe_residuals_and_jac(beta, ds, free_mask):
    try:
        return residuals_and_jacobian(beta, ds, free_mask)
    except IntegrationError:
        return None, None


def _state_only(beta, ds: PreparedDataset):
    """Cheap residuals (state integration only, no sensitivities).

    Returns ``None`` if integration blows up for the trial parameters.
    """
    kgrid = kvec_from_reparam(np.asarray(beta), ds.temperatures, ds.T_ref)
    rs = []
    try:
        for bi, batch in enumerate(ds.batches):
            k = kgrid[bi]

            def rhs(y, _k=k):
                return ds.net.rhs(y, _k)

            res = integrate(rhs, batch["c0"], batch["times"][-1],
                            rtol=CAL_RTOL, atol=CAL_ATOL, land=batch["times"])
            sim = res.sample(batch["times"])[:, batch["cols"]]
            r = batch["Y"] - sim
            rs.append(r[~np.isnan(r)])
    except IntegrationError:
        return None
    return np.concatenate(rs)


def residuals(beta: np.ndarray, ds: PreparedDataset,
              free_mask: np.ndarray | None = None) -> np.ndarray:
    """Residual vector r = observed - simulated over all batches."""
    return _state_only(beta, ds)


# ---------------------------------------------------------------------------
# Levenberg-Marquardt
# ---------------------------------------------------------------------------

def _lm(beta0: np.ndarray, ds: PreparedDataset, free_mask: np.ndarray,
        max_iter: int = MAX_ITER):
    beta = beta0.copy()
    r, J = residuals_and_jacobian(beta, ds, free_mask)
    cost = 0.5 * float(r @ r)
    lam = 1e-3
    iterations = 0
    stop_reason = "max_iterations"
    best_beta, best_cost = beta.copy(), cost

    for it in range(1, max_iter + 1):
        iterations = it
        JtJ = J.T @ J
        Jtr = J.T @ r

        # scale the damping by diagonal magnitudes (Marquardt normalisation)
        diag = np.maximum(np.diag(JtJ), 1e-12)
        accepted = False
        trial_lam = lam
        for _ in range(30):
            A = JtJ + trial_lam * np.diag(diag)
            try:
                delta = np.linalg.solve(A, -Jtr)
            except np.linalg.LinAlgError:
                trial_lam *= 10
                continue
            bp = beta.copy()
            bp[free_mask] = beta[free_mask] + delta
            if not np.all(np.isfinite(bp)):
                trial_lam *= 10
                continue
            rp = residuals(bp, ds)
            if rp is None:
                trial_lam *= 10
                continue
            cp = 0.5 * float(rp @ rp)
            if np.all(np.isfinite(rp)) and cp < cost:
                # relative change convergence test
                rel = (cost - cp) / max(cost, 1e-300)
                beta, r, cost = bp, rp, cp
                if cost < best_cost:
                    best_beta, best_cost = beta.copy(), cost
                lam = trial_lam * 0.3
                accepted = True
                if rel < FTOL:
                    stop_reason = "relative_change_below_1e-10"
                break
            trial_lam *= 10

        if accepted:
            r, J = residuals_and_jacobian(beta, ds, free_mask)
        if stop_reason != "max_iterations":
            break
        if not accepted:
            # could not find a downhill step: converged numerically
            stop_reason = "no_downhill_step"
            break

    return best_beta, best_cost, iterations, stop_reason


def _gn_polish(beta: np.ndarray, ds: PreparedDataset, free_mask: np.ndarray):
    """Pure Gauss-Newton polish so the result is the least-squares stationary
    point regardless of the LM damping path (cold/hot start agreement)."""
    r, J = residuals_and_jacobian(beta, ds, free_mask)
    cost = 0.5 * float(r @ r)
    for _ in range(50):
        try:
            delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        except np.linalg.LinAlgError:
            break
        bp = beta.copy()
        bp[free_mask] = beta[free_mask] + delta
        rp, Jp = _safe_residuals_and_jac(bp, ds, free_mask)
        if rp is None:
            break
        cp = 0.5 * float(rp @ rp)
        if cp >= cost:
            break
        rel = (cost - cp) / max(cost, 1e-300)
        beta, r, J, cost = bp, rp, Jp, cp
        if rel < FTOL:
            break
    return beta, cost


# ---------------------------------------------------------------------------
# identifiability
# ---------------------------------------------------------------------------

def _numeric_rank(J: np.ndarray) -> tuple[int, np.ndarray]:
    """Rank of the column-scaled Jacobian, with singular values."""
    scale = np.linalg.norm(J, axis=0)
    scale[scale == 0] = 1.0
    Js = J / scale
    sv = np.linalg.svd(Js, compute_uv=False)
    tol = max(J.shape) * sv[0] * 1e-10 if sv.size else 0.0
    rank = int(np.sum(sv > tol))
    return rank, sv


def _parameter_covariance(J: np.ndarray, r: np.ndarray, n_free: int) -> np.ndarray:
    """Gauss-Newton covariance estimate: sigma^2 (J^T J)^-1."""
    dof = max(r.size - n_free, 1)
    sigma2 = float(r @ r) / dof
    JtJ = J.T @ J
    try:
        cov = sigma2 * np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        cov = np.full((n_free, n_free), np.nan)
    return cov


# ---------------------------------------------------------------------------
# public calibration entry point
# ---------------------------------------------------------------------------

def default_initial(ds: PreparedDataset) -> np.ndarray:
    """Default initial guess: k = 1 min^-1 for every constant at T_ref."""
    n = ds.net.n_constants
    beta = np.zeros(2 * n)
    beta[0::2] = 0.0  # ln k_ref = 0 -> k = 1
    beta[1::2] = 0.0  # gamma = 0
    return beta


def beta_from_previous(prev_params: list[dict], prev_T_ref: float,
                       ds: PreparedDataset) -> np.ndarray:
    """Map a previous version's (A, Ea) estimates into this T_ref frame."""
    n = ds.net.n_constants
    beta = np.zeros(2 * n)
    for i, p in enumerate(prev_params):
        if p is None or p.get("A") is None or p.get("Ea") is None:
            beta[2 * i] = 0.0
            beta[2 * i + 1] = 0.0
        else:
            gamma = float(p["Ea"]) / R_GAS
            k_ref = float(p["A"]) * math.exp(-gamma / ds.T_ref)
            beta[2 * i] = math.log(max(k_ref, 1e-300))
            beta[2 * i + 1] = gamma
    return beta


def calibrate(net: CompiledNetwork, batches: list[dict], *,
              initial_beta: np.ndarray | None = None,
              max_iter: int = MAX_ITER) -> dict:
    ds = prepare_dataset(net, batches)
    n = net.n_constants
    m = 2 * n

    beta0 = default_initial(ds) if initial_beta is None else np.asarray(initial_beta, dtype=float).copy()
    if beta0.shape != (m,):
        raise ValueError(f"initial has shape {beta0.shape}, expected ({m},)")

    # ---- structural identifiability: single temperature -------------
    if ds.single_temperature:
        # fit only ln k(T0) for each constant; gamma columns are fixed at 0
        free_mask = np.zeros(m, dtype=bool)
        free_mask[0::2] = True
        beta = beta0.copy()
        beta[~free_mask] = 0.0
        beta, cost, iters, reason = _lm(beta, ds, free_mask, max_iter=max_iter)
        beta, cost = _gn_polish(beta, ds, free_mask)
        r = residuals(beta, ds)
        _, J_full = residuals_and_jacobian(beta, ds)
        rank, sv = _numeric_rank(J_full)
        single_reason = (
            "all experiments at one temperature: only k(T) is identifiable; "
            "any (A, Ea) pair on the Arrhenius line through k(T) fits"
        )
        non_id = []
        for i in range(n):
            non_id.append({"constant": ds.labels[i], "parameter": "Ea",
                           "reason": single_reason})
            non_id.append({"constant": ds.labels[i], "parameter": "A",
                           "reason": single_reason})
        return _build_result(
            ds, beta, r, cost, iters, reason,
            free_mask=free_mask,
            non_identifiable=non_id,
            singular_values=sv, rank=rank,
        )

    # ---- general case: fit all, then check numerical rank -----------
    free_mask = np.ones(m, dtype=bool)
    beta, cost, iters, reason = _lm(beta0, ds, free_mask, max_iter=max_iter)
    beta, cost = _gn_polish(beta, ds, free_mask)
    r, J = residuals_and_jacobian(beta, ds)
    rank, sv = _numeric_rank(J)

    non_id: list[dict] = []
    if rank < m:
        # Refit on the well-determined subspace and report rank-deficient
        # parameter directions as non-identifiable.
        free_mask, dropped = _identifiable_subspace(beta, ds, J, sv, rank)
        beta[~free_mask] = 0.0
        beta, cost, iters2, reason2 = _lm(beta, ds, free_mask, max_iter=max_iter)
        beta, cost = _gn_polish(beta, ds, free_mask)
        r = residuals(beta, ds)
        iters += iters2
        reason = reason2
        non_id = dropped

    return _build_result(
        ds, beta, r, cost, iters, reason,
        free_mask=free_mask, non_identifiable=non_id,
        singular_values=sv, rank=rank,
    )


def _identifiable_subspace(beta, ds, J, sv, rank):
    """Find columns spanning the rank(J)-dimensional identifiable subspace."""
    scale = np.linalg.norm(J, axis=0)
    scale[scale == 0] = 1.0
    # column-pivoted QR gives a well-conditioned set of independent columns
    _, _, pivots = np.linalg.qr(J / scale, mode="reduced")
    keep_sorted = np.sort(pivots[:rank])
    mask = np.zeros(beta.size, dtype=bool)
    mask[keep_sorted] = True
    dropped = []
    labels = ds.labels
    for pidx in range(beta.size):
        if not mask[pidx]:
            ci, kind = divmod(pidx, 2)
            dropped.append({
                "constant": labels[ci],
                "parameter": "ln_k_ref" if kind == 0 else "Ea",
                "reason": "sensitivity matrix is rank-deficient "
                          f"(rank {rank} < {beta.size}); not enough data to "
                          "separate this parameter",
            })
    return mask, dropped


def _build_result(ds, beta, r, cost, iterations, lm_reason, *,
                  free_mask, non_identifiable, singular_values, rank):
    n = ds.net.n_constants
    rss = float(r @ r)
    dof = max(r.size - int(free_mask.sum()), 1)
    rmse = math.sqrt(rss / max(r.size, 1))

    # covariance in the reparameterised space
    _, Jall = residuals_and_jacobian(beta, ds)
    J = Jall
    free_idx = np.where(free_mask)[0]
    Jf = J[:, free_idx]
    cov_rep = _parameter_covariance(Jf, r, free_idx.size)
    se_rep = np.sqrt(np.maximum(np.diag(cov_rep), 0.0))

    orig = reparam_to_original(beta, ds.T_ref)

    # standard errors in (A, Ea): gradient of transform
    # Ea = R*gamma ; A = exp(ln_k_ref + gamma/T_ref)
    se_all = np.full(2 * n, np.nan)
    free_list = list(free_idx)
    se_map = {pidx: se_rep[j] for j, pidx in enumerate(free_list)}
    for i in range(n):
        i_ln, i_g = 2 * i, 2 * i + 1
        se_g = se_map.get(i_g, np.nan)
        se_all[i_g] = R_GAS * se_g if np.isfinite(se_g) else np.nan
        jl = free_list.index(i_ln) if i_ln in free_list else None
        jg = free_list.index(i_g) if i_g in free_list else None
        if jl is not None and jg is not None:
            # var(A)/A^2 = var(ln_k_ref) + var(gamma)/T_ref^2
            #            + 2 cov(ln_k_ref, gamma)/T_ref
            var = cov_rep[jl, jl] + cov_rep[jg, jg] / ds.T_ref**2 \
                + 2 * cov_rep[jl, jg] / ds.T_ref
            se_all[i_ln] = orig[i]["A"] * math.sqrt(max(var, 0.0))
        else:
            se_all[i_ln] = np.nan

    params = []
    non_id_keys = {(d["constant"], d["parameter"]) for d in non_identifiable}
    for i, label in enumerate(ds.labels):
        ln_id = (label, "ln_k_ref") not in non_id_keys
        g_id = (label, "Ea") not in non_id_keys
        a_id = (label, "A") not in non_id_keys
        params.append({
            "label": label,
            "A": orig[i]["A"],
            "Ea": orig[i]["Ea"],
            "se_A": None if not a_id or not np.isfinite(se_all[2 * i])
            else float(se_all[2 * i]),
            "se_Ea": None if not g_id or not np.isfinite(se_all[2 * i + 1])
            else float(se_all[2 * i + 1]),
            "ln_k_ref": float(beta[2 * i]),
            "gamma": float(beta[2 * i + 1]),
            "k_at_Tref": float(math.exp(beta[2 * i])),
            "identifiable": {"A": a_id, "Ea": g_id, "ln_k_ref": ln_id},
        })

    converged = lm_reason in ("relative_change_below_1e-10", "no_downhill_step")
    return {
        "T_ref": ds.T_ref,
        "parameters": params,
        "non_identifiable": non_identifiable,
        "iterations": iterations,
        "max_iterations": MAX_ITER,
        "stop_reason": lm_reason,
        "termination": "converged" if converged else "max_iterations",
        "n_observations": int(r.size),
        "degrees_of_freedom": dof,
        "rss": rss,
        "rmse": rmse,
        "residuals": [float(v) for v in r],
        "jacobian_rank": int(rank),
        "jacobian_n_parameters": int(2 * n),
        "singular_values": [float(v) for v in singular_values],
        "_beta": beta.tolist(),
    }
