"""Adaptive Dormand-Prince RK5(4) integrator with dense (Hermite) output.

Only NumPy array arithmetic is used -- no SciPy or other solver libraries.

The method is the classic 5(4) embedded pair with FSAL (the 7th stage equals
the first stage of the next step).  Between accepted nodes, states are
recovered with a cubic Hermite interpolant built from endpoint states and
derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
_E = [71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40]

class IntegrationError(Exception):
    """Integration could not satisfy the requested tolerances."""


@dataclass
class IntegrationResult:
    times: np.ndarray           # accepted node times
    states: np.ndarray          # (n_nodes, n_species)
    derivatives: np.ndarray     # rhs at each node
    accepted_steps: int
    rejected_steps: int
    estimated_error: float      # max scaled local error norm over accepted steps

    def dense(self, t: float) -> np.ndarray:
        """Cubic Hermite interpolation at time ``t`` within the range.

        Times exactly equal to an accepted node (forced landing points are
        reached by exact step subtraction) are returned as the node value
        without interpolation.
        """
        j = int(np.searchsorted(self.times, t, side="left"))
        if j > 0 and j < len(self.times):
            if self.times[j] == t or self.times[j - 1] == t:
                kk = j if self.times[j] == t else j - 1
                return self.states[kk].copy()
        if j <= 0:
            return self.states[0].copy()
        if j >= len(self.times):
            return self.states[-1].copy()
        h = self.times[j] - self.times[j - 1]
        theta = (t - self.times[j - 1]) / h
        y0, y1 = self.states[j - 1], self.states[j]
        f0, f1 = self.derivatives[j - 1], self.derivatives[j]
        h00 = (1 + 2 * theta) * (1 - theta) ** 2
        h10 = theta * (1 - theta) ** 2
        h01 = theta**2 * (3 - 2 * theta)
        h11 = theta**2 * (theta - 1)
        return h00 * y0 + h10 * h * f0 + h01 * y1 + h11 * h * f1

    def sample(self, ts) -> np.ndarray:
        return np.vstack([self.dense(float(t)) for t in np.asarray(ts, dtype=float)])


def integrate(rhs, y0, t_end, rtol=1e-10, atol=1e-13,
              t_start=0.0, max_steps=200_000, min_step=None, land=None):
    """Integrate autonomous dy/dt = rhs(y) from ``t_start`` to ``t_end``.

    ``t_end`` must be >= ``t_start``.  The trajectory ends exactly at t_end.
    ``land`` is an optional iterable of times at which the integrator takes an
    exact accepted node (within the interval), so values at those times are
    5th-order step values rather than interpolants.
    """
    if t_end < t_start:
        raise ValueError("backward integration is not supported")
    y = np.asarray(y0, dtype=float).copy()
    span = t_end - t_start
    if min_step is None:
        # relative to the interval: protects against endless shrinking when
        # atol is extremely tight for species born from zero, while keeping
        # steps tiny compared to any physically relevant scale
        min_step = max(span * 1e-13, 1e-14) if span > 0 else 1e-14

    land_times = sorted(float(x) for x in (land or []) if t_start < float(x) < t_end)
    land_pos = 0

    f = rhs(y)
    ts = [t_start]
    ys = [y.copy()]
    fs = [f.copy()]
    accepted = rejected = 0
    max_err = 0.0
    t = t_start

    scale0 = atol + np.abs(y) * rtol
    d0 = math.sqrt(float(np.mean((y / scale0) ** 2)))
    d1 = math.sqrt(float(np.mean((f / scale0) ** 2)))
    if d0 < 1e-5 or d1 < 1e-5 or span == 0.0:
        h = 1e-6 * span or 1e-6
    else:
        h = min(0.01 * d0 / d1, span)
    h = max(h, min_step)

    k1 = f
    for steps in range(1, max_steps + 2):
        if steps > max_steps:
            raise IntegrationError(f"exceeded {max_steps} steps at t={t}")
        # land exactly on the final time and any requested intermediate nodes
        target = t_end
        if land_pos < len(land_times) and land_times[land_pos] - t > min_step:
            target = min(target, land_times[land_pos])
        # never step past the next target; after a rejected landing step the
        # shrunk h stays below target-t, so we reach it via small steps and a
        # final exact step
        remaining = target - t
        if h > remaining:
            h = remaining

        stages = [k1]
        for s in range(1, 7):
            yi = y.copy()
            for j in range(s):
                aij = _A[s][j]
                if aij != 0.0:
                    yi += h * aij * stages[j]
            stages.append(rhs(yi))
        # stages now holds k1..k7 (k7 is the FSAL stage at the trial point)

        y5 = y.copy()
        err_vec = np.zeros_like(y)
        for s in range(7):
            if _B[s] != 0.0:
                y5 += h * _B[s] * stages[s]
            if _E[s] != 0.0:
                err_vec += h * _E[s] * stages[s]

        if not np.all(np.isfinite(y5)):
            rejected += 1
            h *= 0.2
            if h < min_step:
                raise IntegrationError(f"non-finite state at t={t}; step collapsed")
            continue

        sc = atol + np.maximum(np.abs(y), np.abs(y5)) * rtol
        err_norm = math.sqrt(float(np.mean((err_vec / sc) ** 2)))

        # accept if accurate enough, or if the step has shrunk to the
        # minimum allowed size (atol cannot be resolved more tightly)
        if err_norm <= 1.0 or h <= min_step:
            t += h
            y = y5
            f = stages[6]  # FSAL: k7(t+h) == k1 of the next step
            k1 = f
            ts.append(t)
            ys.append(y.copy())
            fs.append(f.copy())
            accepted += 1
            max_err = max(max_err, err_norm)
            if land_pos < len(land_times) and t >= land_times[land_pos] - 1e-12 * max(1.0, abs(t)):
                land_pos += 1
            if t >= t_end:
                break
            factor = 5.0 if err_norm == 0.0 else min(5.0, max(0.2, 0.9 * err_norm ** -0.2))
            h = max(h * factor, min_step)
        else:
            rejected += 1
            h = max(h * max(0.1, 0.9 * err_norm ** -0.2), min_step)

    return IntegrationResult(
        np.asarray(ts), np.vstack(ys), np.vstack(fs),
        accepted, rejected, max_err,
    )
