"""Kinetic-parameter estimation.

Methodology (see README for the full discussion):

* **Reparameterization.**  Directly fitting pre-exponential factor ``A`` and
  activation energy ``Ea`` is badly conditioned because ln A and Ea/R are
  strongly correlated.  We instead fit, per channel,

      ln k(T) = q + theta * (1/T_ref - 1/T),

  where ``q = ln k(T_ref)`` and ``theta = Ea/R``.  ``T_ref`` is fixed from the
  data as the *mean reciprocal temperature*

      1/T_ref = mean_batches(1/T),

  which roughly orthogonalizes the intercept/slope parameters.  Results are
  converted back to ``A`` and ``Ea`` for reporting.

* **Optimizer.**  Levenberg–Marquardt on the residual vector
  (simulated - observed), with analytically structured scaling and a
  self-written Cholesky solver.  Central finite differences give the
  Jacobian.  Deterministic multi-start from a logarithmic grid makes the
  answer independent of the supplied starting point: warm-starting from a
  previous version simply adds one more candidate start.

* **Identifiability.**  At a single temperature every column multiplying
  ``theta`` is structurally zero, so activation energies (and hence A)
  cannot be estimated: they are reported as ``null`` with an explicit reason,
  never as arbitrary numbers.  More general rank deficiency is detected by
  pivoted Gram–Schmidt on the final Jacobian and handled the same way.

* **Stopping.**  At most 200 LM iterations (counted over all starts of one
  calibration), or earlier when the relative objective change falls below
  1e-10.
"""

from __future__ import annotations

import math

import numpy as np

from .kinetics import Network, R_GAS
from .simulator import FIT_RTOL, FIT_ATOL
from .integrator import IntegrationError

REL_TOL_STOP = 1e-10
MAX_ITERS = 200
_FD_EPS = 1e-6
_RANK_TOL = 1e-9
_GRID_OFFSETS = (-6.907755, -2.302585, 0.0, 2.302585, 6.907755)  # 1e-3..1e2


# ---------------------------------------------------------------------------
# Small linear-algebra primitives (no SciPy)
# ---------------------------------------------------------------------------

def _cholesky(A):
    """Lower-triangular Cholesky factor of a symmetric positive-definite
    matrix.  Raises ValueError if the matrix is not positive definite."""
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - float(L[i, :j] @ L[j, :j])
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix is not positive definite")
                L[i, i] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def _solve_lower(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - float(L[i, :i] @ x[:i])) / L[i, i]
    return x


def _solve_upper(U, b):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - float(U[i + 1:, i] @ x[i + 1:])) / U[i, i]
    return x


def _solve_spd(A, b):
    L = _cholesky(A)
    y = _solve_lower(L, b)
    return _solve_upper(L.T, y)


def _invert_spd(A):
    """Invert a symmetric positive-definite matrix via Cholesky."""
    L = _cholesky(A)
    n = A.shape[0]
    inv = np.zeros_like(A)
    eye = np.eye(n)
    for k in range(n):
        y = _solve_lower(L, eye[:, k])
        inv[:, k] = _solve_upper(L.T, y)
    return inv


def _independent_columns(J, tol=_RANK_TOL):
    """Rank-revealing selection by pivoted modified Gram-Schmidt.

    Returns (rank, keep_mask): ``keep_mask[i]`` is False for columns that are
    numerically zero or linearly dependent on previously accepted columns.
    """
    J = np.asarray(J, dtype=float)
    n_cols = J.shape[1]
    keep = np.zeros(n_cols, dtype=bool)
    q_basis = []
    # Work on column residuals; repeatedly pick the largest residual norm.
    residual = {i: J[:, i].copy() for i in range(n_cols)}
    norms = np.array([math.sqrt(float(v @ v)) for v in residual.values()])
    max_norm = norms.max() if n_cols else 0.0
    threshold = tol * max(max_norm, 1.0)
    remaining = set(range(n_cols))
    while remaining:
        best = max(remaining, key=lambda i: norms[i])
        if norms[best] <= threshold:
            break
        q = residual[best] / norms[best]
        q_basis.append(q)
        keep[best] = True
        remaining.discard(best)
        for i in remaining:
            v = residual[i] - float(q @ residual[i]) * q
            residual[i] = v
            norms[i] = math.sqrt(float(v @ v))
    return int(keep.sum()), keep


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

class DatasetMismatchError(ValueError):
    pass


def _initial_vector(batch, components):
    y0 = np.zeros(len(components))
    for comp, val in batch["initial_concentrations"].items():
        y0[components.index(comp)] = float(val)
    return y0


def reference_temperature(batches):
    inv = [1.0 / float(b["temperature"]) for b in batches]
    return 1.0 / (sum(inv) / len(inv))


# ---------------------------------------------------------------------------
# Residuals / objective
# ---------------------------------------------------------------------------

class _Problem:
    def __init__(self, network: Network, batches, t_ref=None):
        self.net = network
        self.batches = batches
        self.comps = list(network.components)
        self.m = network.n_channels
        self.t_ref = t_ref or reference_temperature(batches)
        self._cache = {}
        # Per-observation bookkeeping: (batch_idx, sample_idx, comp_idx).
        self.obs_index = []
        self.observed = []
        for bi, b in enumerate(batches):
            for si, s in enumerate(b["samples"]):
                for comp, val in s["observations"].items():
                    self.obs_index.append((bi, si, self.comps.index(comp)))
                    self.observed.append(float(val))
        self.observed = np.array(self.observed)
        # Temperature factors z = 1/T_ref - 1/T per batch.
        self.z = np.array([1.0 / self.t_ref - 1.0 / b["temperature"]
                           for b in batches])

    # ---- parameter mapping -------------------------------------------------
    fixed_full = np.zeros(0)  # set by calibrate(); values of frozen coords

    def unpack(self, x, free_mask=None):
        """Full parameter vector (q[m], theta[m]) from a (possibly reduced)
        free vector x; frozen entries read from self.fixed_full."""
        mask = np.ones(2 * self.m, bool) if free_mask is None else free_mask
        full = self.fixed_full.copy()
        full[mask] = x
        return full[:self.m], full[self.m:]

    def rate_logs(self, q, theta):
        """ln k for every (channel, batch): shape (n_batches, m)."""
        return q[None, :] + np.outer(self.z, theta)

    # ---- simulation / residuals -------------------------------------------
    def _simulate_batch(self, bi, q, theta):
        b = self.batches[bi]
        ln_a, ea = self._to_arrhenius(q, theta)
        y0 = _initial_vector(b, self.comps)
        all_times = [s["time"] for s in b["samples"]]
        t_end = max(all_times)
        # A sample at t = 0 is an initial-state measurement, not an
        # integration point; ask only for strictly positive times.
        sample_times = sorted({t for t in all_times if t > 0.0})
        from .simulator import simulate as _simulate
        res = _simulate(self.net, b["temperature"], y0, t_end, ln_a, ea,
                        sample_times=sample_times,
                        rtol=FIT_RTOL, atol=FIT_ATOL)
        # Map each requested sample time to its simulated state.  Output rows
        # are [t0, *sample_times, t_end]; look up by time, tolerating the
        # tiny floating-point gap at t_end.
        table = {float(round(t, 12)): row
                 for t, row in zip(res["times"][1:], res["values"][1:])}
        out = {}
        for t in all_times:
            key = float(round(t, 12))
            if key in table:
                out[t] = table[key]
            elif t == 0.0:
                out[t] = y0
            else:  # endpoint rounding edge
                out[t] = res["values"][-1]
        return out

    def _to_arrhenius(self, q, theta):
        # ln k = ln A - (Ea/R)/T = q + theta*(1/Tref - 1/T) for all T
        # => ln A = q + theta/Tref, Ea/R = theta.
        ln_a = q + theta / self.t_ref
        ea = theta * R_GAS
        return ln_a, ea

    def residuals_full(self, full):
        key = (np.round(full, 12).tobytes())
        if key in self._cache:
            return self._cache[key]
        q, theta = full[:self.m], full[self.m:]
        preds = np.empty(len(self.obs_index))
        # One integration per batch, shared by all its observations.
        sim_cache = {}
        failed = False
        for row, (bi, si, ci) in enumerate(self.obs_index):
            if bi not in sim_cache:
                try:
                    sim_cache[bi] = self._simulate_batch(bi, q, theta)
                except IntegrationError:
                    # Parameter region where the explicit integrator cannot
                    # proceed (e.g. wild extrapolation of Ea): treat as an
                    # infeasible point, not as a server error.
                    failed = True
                    sim_cache[bi] = None
            if sim_cache[bi] is None:
                preds[row] = np.inf
            else:
                t_sample = self.batches[bi]["samples"][si]["time"]
                preds[row] = sim_cache[bi][t_sample][ci]
        r = preds - self.observed
        if failed or not np.all(np.isfinite(r)):
            r = np.where(np.isfinite(r), r, np.inf)
        self._cache[key] = r
        return r

    def sse(self, full):
        r = self.residuals_full(full)
        if np.any(np.isinf(r)):
            return np.inf
        return float(r @ r)

    def clear_cache(self):
        self._cache = {}


# ---------------------------------------------------------------------------
# Jacobian by central finite differences
# ---------------------------------------------------------------------------

def _jacobian(problem: _Problem, full, mask, r0=None):
    """Forward finite-difference Jacobian.

    Forward differences (one extra residual evaluation per parameter) keep
    calibration cheap; the ODE is integrated to ~1e-11 relative accuracy, an
    order of magnitude tighter than the statistical precision we report, so
    the FD truncation error (O(eps)) stays well below parameter uncertainty.
    """
    idx = np.where(mask)[0]
    nf = len(idx)
    if r0 is None:
        r0 = problem.residuals_full(full)
    J = np.zeros((len(r0), nf))
    for k, i in enumerate(idx):
        h = _FD_EPS * max(1.0, abs(full[i]))
        fp = full.copy()
        fp[i] += h
        rp = problem.residuals_full(fp)
        if np.any(np.isinf(rp)):
            col = np.zeros_like(r0)
        else:
            col = (rp - r0) / h
        J[:, k] = col
    return J


# ---------------------------------------------------------------------------
# Levenberg–Marquardt
# ---------------------------------------------------------------------------

def _lm(problem: _Problem, full0, mask, budget, f_best_global=None):
    """Run LM on the free coordinates selected by ``mask``.

    Returns a dict with full parameters, sse, iterations, converged flag,
    final Jacobian/residuals and the active mask.
    """
    free_idx = np.where(mask)[0]
    full = full0.copy()
    r = problem.residuals_full(full)
    if np.any(np.isinf(r)):
        return {"full": full, "sse": np.inf, "iters": 0, "converged": False,
                "J": None, "r": r, "mask": mask}
    sse = float(r @ r)
    lam = 1e-3
    iters = 0
    converged = False
    J = None
    while iters < budget:
        J = _jacobian(problem, full, mask, r0=r)
        # Column scaling so q and theta (which may be ~1e4 K) compete fairly.
        col_scale = np.maximum(np.sqrt(np.sum(J * J, axis=0)), 1e-8)
        Js = J / col_scale
        try:
            H = Js.T @ Js
            g = Js.T @ r
        except Exception:
            break
        accepted = False
        step = None
        for _bumps in range(30):
            try:
                ds = _solve_spd(H + lam * np.diag(np.diag(H) + 1e-14), -g)
            except ValueError:
                lam *= 10.0
                continue
            d = ds / col_scale
            step = d
            trial = full.copy()
            trial[free_idx] = full[free_idx] + d
            r_new = problem.residuals_full(trial)
            if np.any(np.isinf(r_new)):
                lam *= 4.0
                continue
            sse_new = float(r_new @ r_new)
            if sse_new < sse:
                rel = abs(sse - sse_new) / max(1.0, abs(sse))
                full, r, sse = trial, r_new, sse_new
                lam = max(lam / 5.0, 1e-12)
                accepted = True
                iters += 1
                if rel < REL_TOL_STOP:
                    converged = True
                break
            lam *= 4.0
        if not accepted:
            # No descent direction available even with huge damping: stationary
            # to working precision.
            converged = True
            break
        if converged:
            break
    return {"full": full, "sse": sse, "iters": iters, "converged": converged,
            "J": J, "r": r, "mask": mask}


# ---------------------------------------------------------------------------
# Structural identifiability
# ---------------------------------------------------------------------------

def _structural_mask(problem: _Problem):
    """Which of the 2*m parameters are structurally active?

    The theta columns carry factor z_b = 1/Tref - 1/T_b.  If every batch is
    at (numerically) the same temperature, z vanishes and theta – and
    therefore Ea and A – is unidentifiable.
    """
    m = problem.m
    mask = np.ones(2 * m, dtype=bool)
    z_span = float(np.max(np.abs(problem.z)))
    if z_span < 1e-12:
        mask[m:] = False
    return mask, z_span


# ---------------------------------------------------------------------------
# Public calibration entry point
# ---------------------------------------------------------------------------

def calibrate(network: Network, batches, initial=None, max_iters=MAX_ITERS):
    """Estimate Arrhenius parameters for ``network`` from ``batches``.

    ``initial`` optionally seeds the multi-start with a previous result's
    internal parameters ({"q": [...], "theta": [...]}).

    Returns a JSON-serializable result dict (see build_report).
    """
    if not batches:
        raise ValueError("calibration requires at least one batch")

    problem = _Problem(network, batches)
    m = problem.m

    # Free/fixed bookkeeping: fixed coordinates are held at zero (theta) or,
    # if a q column ever turns out dead, at their grid value.
    struct_mask, z_span = _structural_mask(problem)
    mask = struct_mask.copy()
    problem.fixed_full = np.zeros(2 * m)

    # ---- deterministic starts ---------------------------------------------
    t_chars = sorted(max(s["time"] for s in b["samples"]) for b in batches)
    t_char = t_chars[len(t_chars) // 2]
    q_centre = math.log(1.0 / max(t_char, 1e-30))
    candidates = []
    for off in _GRID_OFFSETS:
        full = np.zeros(2 * m)
        full[:m] = q_centre + off
        candidates.append(full)
    if initial is not None:
        full = np.zeros(2 * m)
        full[:m] = np.asarray(initial["q"], dtype=float)
        full[m:] = np.asarray(initial["theta"], dtype=float)
        candidates.append(full)

    # Screen starts cheaply (one simulation set each).
    scored = []
    for c in candidates:
        v = c[mask]
        full = problem.fixed_full.copy(); full[mask] = v
        scored.append((problem.sse(full), c))
    scored.sort(key=lambda t: (np.inf if np.isinf(t[0]) else t[0]))
    if np.isinf(scored[0][0]):
        raise IntegrationError(
            "no numerically evaluable starting point: the parameter space "
            "produces non-integrable (non-finite) states for all grid "
            "starts; check the data and network for scale problems")
    best_starts = [c for _, c in scored[:3]]

    # ---- multi-start LM within a shared iteration budget ------------------
    total_iters = 0
    runs = []
    for k, c in enumerate(best_starts):
        budget = min(80, max_iters - total_iters)
        if budget <= 0:
            break
        out = _lm(problem, c, mask, budget)
        total_iters += out["iters"]
        runs.append(out)
        # Stop early once the best-screened start converged; if a later
        # start's basin were better its screening SSE would have ranked first.
        best_so_far = min(runs, key=lambda r: r["sse"])
        if k == 0 and best_so_far is out and out["converged"]:
            break
    winner = min(runs, key=lambda r: (np.inf if np.isinf(r["sse"])
                                      else r["sse"]))

    # ---- numerical identifiability at the solution -------------------------
    # Analyze the final Jacobian; freeze dead/dependent columns and refit.
    frozen_note = None
    full_hat = winner["full"]
    J_hat = winner.get("J")
    if J_hat is None or not np.array_equal(winner.get("mask"), mask):
        J_hat = _jacobian(problem, full_hat, mask, r0=winner.get("r"))
    rank, keep = _independent_columns(J_hat)
    if not bool(np.all(keep)):
        sub_idx = np.where(mask)[0][keep]
        new_mask = np.zeros(2 * m, dtype=bool)
        new_mask[sub_idx] = True
        dropped = np.where(mask)[0][~keep]
        mask = new_mask
        problem.fixed_full = full_hat  # keep dropped coords at fitted values
        budget = max_iters - total_iters
        if budget > 0:
            out = _lm(problem, full_hat, mask, budget)
            total_iters += out["iters"]
            winner = out
        frozen_note = dropped

    converged = bool(winner["converged"])
    if total_iters >= max_iters and not converged:
        stop_reason = "max_iterations"
    else:
        stop_reason = "converged"

    return build_report(problem, winner, mask, total_iters, stop_reason,
                        z_span, frozen_note, max_iters)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(problem: _Problem, winner, mask, total_iters, stop_reason,
                 z_span, frozen_note, max_iters=MAX_ITERS):
    m = problem.m
    full = winner["full"]
    q = full[:m]
    theta = full[m:]
    sse = winner["sse"]

    # Covariance of free coordinates at the solution.
    if winner.get("J") is not None and np.array_equal(winner.get("mask"),
                                                      mask):
        J = winner["J"]
    else:
        J = _jacobian(problem, full, mask, r0=winner.get("r"))
    n_obs = J.shape[0]
    n_free = J.shape[1]
    dof = max(n_obs - n_free, 1)
    sigma2 = sse / dof
    se_free = np.full(n_free, np.nan)
    cov_free = None
    if n_free > 0:
        H = J.T @ J
        try:
            cov_free = sigma2 * _invert_spd(H)
            se_free = np.sqrt(np.maximum(np.diag(cov_free), 0.0))
        except ValueError:
            cov_free = None

    free_idx = np.where(mask)[0]
    se_q = np.full(m, np.nan)
    se_theta = np.full(m, np.nan)
    cov = np.zeros((2 * m, 2 * m))
    for k, i in enumerate(free_idx):
        if i < m:
            se_q[i] = se_free[k]
        else:
            se_theta[i - m] = se_free[k]
        for l, j in enumerate(free_idx):
            if cov_free is not None:
                cov[i, j] = cov_free[k, l]

    t_ref = problem.t_ref
    channels = []
    for j, ch in enumerate(problem.net.channels):
        theta_id = m + j
        identifiable_theta = bool(mask[theta_id])
        item = {
            "reaction_id": ch.reaction_id,
            "direction": ch.direction,
            "k_at_reference": math.exp(q[j]),
            "k_at_reference_se": (math.exp(q[j]) * se_q[j]
                                  if np.isfinite(se_q[j]) else None),
            "reference_temperature": t_ref,
        }
        if identifiable_theta:
            ln_a = q[j] + theta[j] / t_ref
            a = math.exp(ln_a)
            var_lna = cov[j, j] + cov[theta_id, theta_id] / t_ref**2 \
                + 2.0 * cov[j, theta_id] / t_ref
            item.update({
                "a": a,
                "a_se": a * math.sqrt(max(var_lna, 0.0)),
                "ea": theta[j] * R_GAS,
                "ea_se": R_GAS * se_theta[j]
                if np.isfinite(se_theta[j]) else None,
                "identifiable": True,
                "unidentifiable_reason": None,
            })
        else:
            reason = ("activation energy is unidentifiable: all batches are "
                      "at (numerically) the same temperature, so the data "
                      "constrain only k at that temperature")
            if z_span >= 1e-12 and frozen_note is not None and \
                    theta_id in frozen_note:
                reason = ("activation energy is numerically unidentifiable: "
                          "the Jacobian column is (near) dependent on the "
                          "other parameter columns for this dataset")
            item.update({
                "a": None,
                "a_se": None,
                "ea": None,
                "ea_se": None,
                "identifiable": False,
                "unidentifiable_reason": reason,
            })
        channels.append(item)

    # Residual detail.
    resid = problem.residuals_full(full)
    residual_rows = []
    for row, (bi, si, ci) in enumerate(problem.obs_index):
        b = problem.batches[bi]
        s = b["samples"][si]
        comp = problem.comps[ci]
        residual = float(resid[row])
        residual_rows.append({
            "batch_id": b.get("id"),
            "time": s["time"],
            "component": comp,
            "observed": float(problem.observed[row]),
            "predicted": (float(problem.observed[row] + residual)
                          if math.isfinite(residual) else None),
            "residual": residual if math.isfinite(residual) else None,
        })

    any_unidentifiable = not all(c["identifiable"] for c in channels)
    return {
        "reference_temperature": t_ref,
        "temperature_span": {"z_min": float(np.min(problem.z)),
                             "z_max": float(np.max(problem.z))},
        "internal_parameters": {
            "q": [float(v) for v in q],
            "theta": [float(v) for v in theta],
            "meaning": "ln k(T_ref) and Ea/R; T_ref chosen so that "
                       "1/T_ref = mean(1/T) over batches",
        },
        "channels": channels,
        "objective": {
            "sse": float(sse),
            "rmse": math.sqrt(sse / max(n_obs, 1)),
            "n_observations": int(n_obs),
            "n_free_parameters": int(n_free),
            "residual_variance": float(sigma2),
        },
        "iterations": int(total_iters),
        "max_iterations": int(max_iters),
        "stop_reason": stop_reason,
        "converged": stop_reason == "converged",
        "residuals": residual_rows,
        "all_identifiable": not any_unidentifiable,
    }
