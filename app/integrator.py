"""Adaptive ODE integration for batch reactor concentration profiles.

Implements the Dormand-Prince RK5(4) embedded pair (the same coefficients used
by classic ``ode45``), with first-step-same-as-last (FSAL) reuse, local error
control, and output exactly at requested sample times.

No SciPy is used: the pair, step controller and dense output are written here.
Dense output is exact polynomial interpolation between the 5th-order stages,
which is both cheaper and more accurate than restarting the method at every
sample time.
"""

from __future__ import annotations

import math

import numpy as np


# --- Dormand-Prince coefficients -------------------------------------------
_C = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
# 5th order solution weights (stage 7 equals this: FSAL).
_B = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
# Error weights = b5 - b4 (Dormand-Prince embedded 4th order estimate).
_E = [
    71 / 57600,
    0.0,
    -71 / 16695,
    71 / 1920,
    -17253 / 339200,
    22 / 525,
    -1 / 40,
]
# Dense-output coefficient matrix (Shampine's optimum c6 construction).
# y(theta) = y0 + h * K^T P * [theta, theta**2, theta**3, theta**4]^T
# Coefficient values are the standard published Dormand-Prince continuous
# extension (transcribed from the tableau in SciPy's rk.py; the integration
# and step control here are our own implementation).
_P = [
    [1.0, -8048581381 / 2820520608, 8663915743 / 2820520608,
     -12715105075 / 11282082432],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 131558114200 / 32700410799, -68118460800 / 10900136933,
     87487479700 / 32700410799],
    [0.0, -1754552775 / 470086768, 14199869525 / 1410260304,
     -10690763975 / 1880347072],
    [0.0, 127303824393 / 49829197408, -318862633887 / 49829197408,
     701980252875 / 199316789632],
    [0.0, -282668133 / 205662961, 2019193451 / 616988883,
     -1453857185 / 822651844],
    [0.0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
]


class IntegrationError(RuntimeError):
    """Raised when the integration cannot meet the requested accuracy."""


def _initial_step(rhs, y0, t0, rtol, atol, order=5):
    """Hairer/Norsett/Wanner initial step-size heuristic."""
    f0 = rhs(t0, y0)
    scale = atol + np.abs(y0) * rtol
    d0 = math.sqrt(np.mean((y0 / scale) ** 2))
    d1 = math.sqrt(np.mean((f0 / scale) ** 2))
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    y1 = y0 + h0 * f0
    f1 = rhs(t0 + h0, y1)
    d2 = math.sqrt(np.mean(((f1 - f0) / scale) ** 2)) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))
    return min(100 * h0, h1)


def _dense_value(theta, y, h, k):
    """Continuous extension at theta=(t-t0)/h, theta in [0, 1]."""
    powers = np.array([theta, theta**2, theta**3, theta**4])
    q = np.zeros_like(y)
    for i, ki in enumerate(k):
        row = _P[i]
        coeff = row[0] * powers[0] + row[1] * powers[1] \
            + row[2] * powers[2] + row[3] * powers[3]
        if coeff != 0.0:
            q += coeff * ki
    return y + h * q


def integrate(rhs, y0, t0, t_end, t_eval=None, rtol=1e-9, atol=1e-12,
              max_steps=200_000):
    """Integrate dy/dt = rhs(t, y) from t0 to t_end with RK5(4).

    Returns a dict with:
        times          – strictly increasing sample times (t0 first)
        values         – state matrix (len(times), len(y0))
        n_steps        – number of accepted steps
        n_rejected     – number of rejected steps
        max_error      – largest accepted scaled local-error norm (estimate
                         of the local truncation error actually incurred)
    """
    y = np.asarray(y0, dtype=float)
    if not np.all(np.isfinite(y)):
        raise IntegrationError("initial state contains non-finite values")

    direction = 1.0 if t_end >= t0 else -1.0
    span = abs(t_end - t0)
    if span == 0.0:
        times = [t0]
        values = [y.copy()]
        return {"times": times, "values": np.array(values), "n_steps": 0,
                "n_rejected": 0, "max_error": 0.0}

    # Merge requested samples into a sorted queue inside the interval.
    if t_eval is None:
        samples = []
    else:
        samples = [float(t) for t in t_eval
                   if (t - t0) * direction > 0.0 and (t_end - t) * direction >= 0.0]
        samples = sorted(samples)

    h = direction * _initial_step(rhs, y, t0, rtol, atol)
    h = min(abs(h), span) * direction

    k = [None] * 7
    k[0] = rhs(t0, y)
    t = t0

    out_t = [t0]
    out_y = [y.copy()]
    sample_idx = 0
    n_steps = 0
    n_rejected = 0
    max_error = 0.0
    h_min = span * 1e-14

    while (t_end - t) * direction > 0.0:
        if n_steps > max_steps:
            raise IntegrationError(f"exceeded {max_steps} steps")
        if abs(h) < h_min:
            raise IntegrationError(
                f"step size collapsed below {h_min:.3e} near t={t:g}; "
                "the system may be too stiff for the explicit integrator")

        # Step to t_end, but not past it; interior sample times are served by
        # the continuous extension, so natural (large) steps are preserved.
        h_try = h
        if (t_end - (t + h_try)) * direction < 0.0:
            h_try = t_end - t

        accepted = False
        for _attempt in range(50):
            k[1] = rhs(t + _C[1] * h_try, y + h_try * _A[1][0] * k[0])
            k[2] = rhs(t + _C[2] * h_try,
                       y + h_try * (_A[2][0] * k[0] + _A[2][1] * k[1]))
            k[3] = rhs(t + _C[3] * h_try,
                       y + h_try * (_A[3][0] * k[0] + _A[3][1] * k[1]
                                    + _A[3][2] * k[2]))
            k[4] = rhs(t + _C[4] * h_try,
                       y + h_try * (_A[4][0] * k[0] + _A[4][1] * k[1]
                                    + _A[4][2] * k[2] + _A[4][3] * k[3]))
            k[5] = rhs(t + _C[5] * h_try,
                       y + h_try * (_A[5][0] * k[0] + _A[5][1] * k[1]
                                    + _A[5][2] * k[2] + _A[5][3] * k[3]
                                    + _A[5][4] * k[4]))
            y_new = y + h_try * (_B[0] * k[0] + _B[2] * k[2] + _B[3] * k[3]
                                 + _B[4] * k[4] + _B[5] * k[5])
            k[6] = rhs(t + h_try, y_new)
            err_vec = h_try * (_E[0] * k[0] + _E[2] * k[2] + _E[3] * k[3]
                               + _E[4] * k[4] + _E[5] * k[5] + _E[6] * k[6])
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            err_norm = math.sqrt(np.mean((err_vec / scale) ** 2))

            if not np.all(np.isfinite(y_new)) or not np.isfinite(err_norm):
                factor = 0.2
                err_norm = np.inf
            elif err_norm <= 1.0:
                accepted = True
                factor = min(5.0, 0.9 * err_norm ** (-1.0 / 5)
                             if err_norm > 0.0 else 5.0)
                factor = max(0.2, factor)
                break
            else:
                factor = max(0.1, 0.9 * err_norm ** (-1.0 / 5))

            h_try *= factor
            if abs(h_try) < h_min:
                raise IntegrationError(
                    f"step size collapsed below {h_min:.3e} near t={t:g}")
            n_rejected += 1
            # k[0] stays valid because (t, y) is unchanged (FSAL).
        if not accepted:
            raise IntegrationError(f"could not complete a step near t={t:g}")

        t_new = t + h_try
        max_error = max(max_error, float(err_norm))
        n_steps += 1

        # Emit all requested samples that fall in (t, t_new] via the
        # continuous extension; a sample at t_new uses the exact 5th-order
        # state instead of the interpolant.
        while sample_idx < len(samples) and \
                (t_new - samples[sample_idx]) * direction >= -1e-14 * span:
            ts = samples[sample_idx]
            if abs(ts - t_new) <= 1e-14 * span:
                ys = y_new
            else:
                theta = (ts - t) / h_try
                ys = _dense_value(theta, y, h_try, k)
            out_t.append(ts)
            out_y.append(ys)
            sample_idx += 1

        t, y = t_new, y_new
        # Grow the step for next interval (cap at remaining span).
        h = h_try * factor
        if abs(h) > abs(t_end - t) and (t_end - t) * direction > 0:
            h = t_end - t
        # FSAL: stage 7 of this step is stage 0 of the next.
        k[0] = k[6]

    # Record the final state (unless a requested sample equal to t_end was
    # already emitted).
    if out_t[-1] != t_end:
        out_t.append(t_end)
        out_y.append(y.copy())

    return {
        "times": out_t,
        "values": np.array(out_y),
        "n_steps": n_steps,
        "n_rejected": n_rejected,
        "max_error": max_error,
    }
