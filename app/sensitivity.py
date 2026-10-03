"""Sensitivity propagation for calibration.

For a batch at temperature T the state obeys

    dc/dt = f(c, k(T)),   k_j(T) = exp(ln_k_ref_j + gamma_j * (1/T_ref - 1/T)).

Let beta = (ln_k_ref_0, gamma_0, ...).  The sensitivities S = dc/d beta obey
the linear variational equation

    dS/dt = (df/dc) S + df/d beta.

We integrate c together with S in one augmented system.  The RHS is built
from small dense NumPy matrices (numbers of species/reactions are modest),
which keeps every RHS evaluation free of Python-level per-cell loops.
"""

from __future__ import annotations

import numpy as np

from .integrator import integrate

_STATIC_CACHE: dict[int, dict] = {}


def _static(net):
    """Precompute dense structures for a compiled network (cached by id)."""
    cache = _STATIC_CACHE
    key = id(net)
    if key in cache:
        return cache[key]
    n_sp = net.n_species
    n_rx = len(net.reactions)
    F = np.zeros((n_rx, n_sp))
    Fmask = np.zeros((n_rx, n_sp), dtype=bool)
    Rv = np.zeros((n_rx, n_sp))
    Rmask = np.zeros((n_rx, n_sp), dtype=bool)
    for i, law in enumerate(net.forward_laws):
        for idx, o in zip(law.order_idx, law.order_pow):
            F[i, idx] = o
            Fmask[i, idx] = True
    for i, law in enumerate(net.reverse_laws):
        if law is None:
            continue
        for idx, o in zip(law.order_idx, law.order_pow):
            Rv[i, idx] = o
            Rmask[i, idx] = True
    kf_idx = np.zeros(n_rx, dtype=int)
    kr_idx = np.full(n_rx, -1, dtype=int)
    const_rxn = []   # reaction owning each constant
    const_sign = []  # +1 forward, -1 reverse
    j = 0
    for i, rev in enumerate(net.reversible):
        kf_idx[i] = j
        const_rxn.append(i)
        const_sign.append(1.0)
        j += 1
        if rev:
            kr_idx[i] = j
            const_rxn.append(i)
            const_sign.append(-1.0)
            j += 1
    out = {
        "F": F, "Fmask": Fmask, "R": Rv, "Rmask": Rmask,
        "kf_idx": kf_idx, "kr_idx": kr_idx,
        "const_rxn": np.asarray(const_rxn, dtype=int),
        "const_sign": np.asarray(const_sign, dtype=float),
        "stoich": net.stoich,
    }
    cache[key] = out
    return out


def _monomials_and_grads(c, orders, mask):
    """Mass-action monomials and their concentration gradients.

    mon[i]  = prod_s c_s ** orders[i,s]
    grad[i,s] = d mon[i] / d c_s
    """
    n_rx, n_sp = orders.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        # factor 1 where the species does not appear
        factor = np.where(mask, np.broadcast_to(c, orders.shape) ** orders, 1.0)
    mon = factor.prod(axis=1)

    # "rest product": mon / factor[:,s], computed without 0/0 division.
    # cumulative products from left and right.
    left = np.empty_like(factor)
    right = np.empty_like(factor)
    left[:, 0] = 1.0
    right[:, -1] = 1.0
    if n_sp > 1:
        left[:, 1:] = np.cumprod(factor[:, :-1], axis=1)
        right[:, :-1] = np.cumprod(factor[:, :-0:-1], axis=1)[:, ::-1]
    rest = left * right

    # d factor_s / d c_s = order * c^(order-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        deriv = orders * np.where(
            mask,
            np.where(c[None, :] > 0.0, c[None, :] ** (orders - 1.0),
                     np.where(orders == 1.0, 1.0, 0.0)),
            0.0,
        )
    grad = deriv * rest
    return mon, grad


def augmented_rhs(net, kvec, dlnk_factor, y, n_spec, n_par, st):
    """RHS of the augmented state.

    ``dlnk_factor`` shape (n_const, n_par): d ln k_j / d beta.
    """
    c = y[:n_spec]
    Z = y[n_spec:].reshape(n_spec, n_par)

    mon_f, grad_f = _monomials_and_grads(c, st["F"], st["Fmask"])
    if st["Rmask"].any():
        mon_r, grad_r = _monomials_and_grads(c, st["R"], st["Rmask"])
    else:
        mon_r = np.zeros_like(mon_f)
        grad_r = np.zeros_like(st["R"])

    kf = kvec[st["kf_idx"]]
    kr = np.zeros_like(kf)
    rev_sel = st["kr_idx"] >= 0
    kr[rev_sel] = kvec[st["kr_idx"][rev_sel]]
    rates = kf * mon_f - kr * mon_r
    dc = rates @ st["stoich"]

    # df/dc: dr_i/dc_s = kf_i grad_f - kr_i grad_r
    Jr = kf[:, None] * grad_f - kr[:, None] * grad_r
    dfdc = st["stoich"].T @ Jr

    # df/dk_j (j over constants): sign_j * stoich[rxn_j] * mon(direction)
    mon_sel = np.where(st["const_sign"] > 0, mon_f[st["const_rxn"]],
                       mon_r[st["const_rxn"]])
    k_sel = kvec
    dfdk = (st["const_sign"][:, None] * mon_sel[:, None]
            * st["stoich"][st["const_rxn"]]) * k_sel[:, None]
    # df/d beta = (d k / d beta) relation: dk_j/dbeta = k_j * dlnk_j/dbeta
    dfdbeta = dfdk.T @ dlnk_factor  # (n_spec, n_par)

    dS = dfdc @ Z + dfdbeta
    return np.concatenate([dc, dS.reshape(-1)])


def dlnk_matrix(n_constants: int, inv_T: float, inv_Tref: float) -> np.ndarray:
    """d ln k_j / d beta_p: each k depends on ln_k_ref_j and gamma_j."""
    M = np.zeros((n_constants, 2 * n_constants))
    delta = inv_Tref - inv_T
    for j in range(n_constants):
        M[j, 2 * j] = 1.0
        M[j, 2 * j + 1] = delta
    return M


def simulate_with_sensitivity(net, beta, T, T_ref, c0, times, land=None,
                              rtol=1e-10, atol=1e-13):
    """Integrate state and sensitivities.

    Returns C (n_t, n_spec) and S (n_t, n_spec, n_par).
    """
    from .simulation import kvec_from_reparam

    st = _static(net)
    n_spec = net.n_species
    n_par = 2 * net.n_constants
    kvec = kvec_from_reparam(np.asarray(beta), float(T), float(T_ref))
    dlnk = dlnk_matrix(net.n_constants, 1.0 / float(T), 1.0 / float(T_ref))

    y0 = np.concatenate([np.asarray(c0, dtype=float), np.zeros(n_spec * n_par)])

    def rhs(y):
        return augmented_rhs(net, kvec, dlnk, y, n_spec, n_par, st)

    res = integrate(rhs, y0, float(times[-1]), rtol=rtol, atol=atol,
                    land=list(times))
    Y = res.sample(times)
    C = Y[:, :n_spec]
    S = Y[:, n_spec:].reshape(len(times), n_spec, n_par)
    return C, S
