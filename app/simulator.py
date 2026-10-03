"""Batch-reactor simulation service (ODE integration wrapper)."""

from __future__ import annotations

import numpy as np

from .integrator import integrate, IntegrationError
from .kinetics import Network


DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12
# Tighter tolerances used while calibrating, so that finite-difference
# Jacobians are not contaminated by integration error.
FIT_RTOL = 1e-11
FIT_ATOL = 1e-14


def simulate(network: Network, temperature, initial_concentrations, t_end,
             ln_a, ea, sample_times=None, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Integrate one batch experiment.

    Raises IntegrationError on numerical failure.
    """
    y0 = np.asarray(initial_concentrations, dtype=float)
    ln_a = np.asarray(ln_a, dtype=float)
    ea = np.asarray(ea, dtype=float)

    def rhs(t, y):
        return network.rhs(y, temperature, ln_a, ea)

    # Requested samples inside (0, t_end); t = 0 is served by the initial
    # row and t = t_end by the final row.
    times = sorted(set([float(t) for t in (sample_times or [])
                        if 0.0 < float(t) < float(t_end)]))
    res = integrate(rhs, y0, 0.0, float(t_end), t_eval=times,
                    rtol=rtol, atol=atol)
    values = np.asarray(res["values"])
    if not np.all(np.isfinite(values)):  # defensive; integrator normally raises
        raise IntegrationError("simulation produced non-finite concentrations")
    return {
        "times": [float(t) for t in res["times"]],
        "values": values,
        "n_steps": res["n_steps"],
        "n_rejected": res["n_rejected"],
        "max_scaled_local_error": float(res["max_error"]),
    }
