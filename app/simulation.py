"""Kinetic simulation and validation of experiment/batch records.

Original parameters per rate constant are (A, Ea) with
``k(T) = A * exp(-Ea / (R*T))``.  For fitting we use the reparameterised
pair (see :mod:`app.calibration`):

    ln_k_ref = ln(k(T_ref))
    gamma    = Ea / R          (temperature, in kelvin)

so that ``ln k(T) = ln_k_ref + gamma * (1/T_ref - 1/T)``.
"""

from __future__ import annotations

import math

import numpy as np

from .chemistry import CompiledNetwork, R_GAS, ValidationError, _is_finite_number
from .integrator import IntegrationError, integrate


# ---------------------------------------------------------------------------
# parameter helpers
# ---------------------------------------------------------------------------

def arrhenius_k(A: float, Ea: float, T: float) -> float:
    return A * math.exp(-Ea / (R_GAS * T))


def kvec_from_original(net: CompiledNetwork, params: list[dict], T: float) -> np.ndarray:
    """Build the flat rate-constant vector at temperature T."""
    labels = net.rate_constant_labels()
    if not isinstance(params, list) or len(params) != len(labels):
        raise ValidationError([
            ("parameters", f"expected {len(labels)} rate constants, got "
                           f"{len(params) if isinstance(params, list) else 'non-list'}")
        ])
    out = []
    for i, p in enumerate(params):
        path = f"parameters[{i}]"
        if not isinstance(p, dict):
            raise ValidationError([(path, "must be an object with A and Ea")])
        A, Ea = p.get("A"), p.get("Ea")
        if not _is_finite_number(A) or float(A) <= 0.0:
            raise ValidationError([(f"{path}.A", "must be a positive finite number")])
        if not _is_finite_number(Ea):
            raise ValidationError([(f"{path}.Ea", "must be a finite number")])
        out.append(arrhenius_k(float(A), float(Ea), T))
    return np.asarray(out, dtype=float)


def kvec_from_reparam(beta: np.ndarray, temperatures, T_ref: float) -> np.ndarray:
    """beta layout: [ln_k_ref_0, gamma_0, ln_k_ref_1, gamma_1, ...].

    ``temperatures`` scalar -> 1-D array; array -> 2-D (n_temp, n_constants).
    """
    beta = np.asarray(beta, dtype=float)
    q = beta.reshape(-1, 2)
    ln_kref = q[:, 0]
    gamma = q[:, 1]
    scalar = np.isscalar(temperatures) or np.ndim(temperatures) == 0
    T = np.atleast_1d(np.asarray(temperatures, dtype=float))
    ln_k = ln_kref[None, :] + gamma[None, :] * (1.0 / T_ref - 1.0 / T[:, None])
    k = np.exp(ln_k)
    return k[0] if scalar else k


def reparam_to_original(beta: np.ndarray, T_ref: float) -> list[dict]:
    """Convert fitted reparameterised vector to [{A, Ea}, ...]."""
    q = np.asarray(beta, dtype=float).reshape(-1, 2)
    out = []
    for ln_kref, gamma in q:
        Ea = gamma * R_GAS
        A = math.exp(ln_kref + gamma / T_ref)
        out.append({"A": A, "Ea": Ea})
    return out


# ---------------------------------------------------------------------------
# batch validation
# ---------------------------------------------------------------------------

def validate_initial_concentrations(net: CompiledNetwork, ic) -> np.ndarray:
    if not isinstance(ic, dict) or not ic:
        raise ValidationError([("initial_concentrations", "must be a non-empty object")])
    c0 = np.zeros(net.n_species)
    for sp, v in ic.items():
        if sp not in net.species_idx:
            raise ValidationError([(f"initial_concentrations.{sp}", f"undefined species '{sp}'")])
        if not _is_finite_number(v) or float(v) < 0.0:
            raise ValidationError([(f"initial_concentrations.{sp}", "must be a non-negative finite number")])
        c0[net.species_idx[sp]] = float(v)
    return c0


def validate_batch(net: CompiledNetwork, batch: dict, prefix: str = "") -> dict:
    """Validate one batch record; return normalised data.

    Normalised form::

        {"temperature": float,
         "c0": np.ndarray,
         "times": [float, ...],
         "species": [str, ...],
         "Y": ndarray (n_times, n_measured) with NaN for missing}
    """
    errors: list[tuple[str, str]] = []

    T = batch.get("temperature")
    if not _is_finite_number(T) or float(T) <= 0.0:
        errors.append((f"{prefix}temperature", "must be a positive finite number (kelvin)"))

    ic = batch.get("initial_concentrations", batch.get("c0"))
    c0 = np.zeros(net.n_species)
    if not isinstance(ic, dict) or not ic:
        errors.append((f"{prefix}initial_concentrations", "must be a non-empty object"))
    else:
        for sp, v in ic.items():
            if sp not in net.species_idx:
                errors.append((f"{prefix}initial_concentrations.{sp}", f"undefined species '{sp}'"))
            elif not _is_finite_number(v) or float(v) < 0.0:
                errors.append((f"{prefix}initial_concentrations.{sp}", "must be a non-negative finite number"))
            else:
                c0[net.species_idx[sp]] = float(v)

    samples = batch.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append((f"{prefix}samples", "must be a non-empty list"))
        samples = []

    times: list[float] = []
    measured_species: list[str] = []
    for i, s in enumerate(samples):
        sp = f"{prefix}samples[{i}]"
        if not isinstance(s, dict):
            errors.append((sp, "must be an object"))
            continue
        t = s.get("time")
        if not _is_finite_number(t) or float(t) < 0.0:
            errors.append((f"{sp}.time", "must be a non-negative finite number"))
        elif times and float(t) <= times[-1]:
            errors.append((f"{sp}.time", "sample times must be strictly increasing"))
        else:
            times.append(float(t))
        conc = s.get("concentrations", {})
        if not isinstance(conc, dict) or not conc:
            errors.append((f"{sp}.concentrations", "must be a non-empty object"))
            conc = {}
        for spname, v in conc.items():
            if spname not in net.species_idx:
                errors.append((f"{sp}.concentrations.{spname}", f"undefined species '{spname}'"))
            elif not _is_finite_number(v) or float(v) < 0.0:
                errors.append((f"{sp}.concentrations.{spname}", "must be a non-negative finite number"))
            if spname not in measured_species:
                measured_species.append(spname)

    if errors:
        raise ValidationError(errors)

    cols = [net.species_idx[s] for s in measured_species]
    Y = np.full((len(times), len(measured_species)), np.nan)
    for i, s in enumerate(samples):
        for j, spname in enumerate(measured_species):
            v = s["concentrations"].get(spname)
            if v is not None:
                Y[i, j] = float(v)

    return {
        "temperature": float(T),
        "c0": c0,
        "times": times,
        "species": measured_species,
        "cols": np.asarray(cols, dtype=int),
        "Y": Y,
    }


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def simulate(net: CompiledNetwork, params: list[dict], temperature: float,
             initial_concentrations: dict, times, *, rtol=1e-9, atol=1e-12):
    """Integrate the network and return concentrations at ``times``.

    Returns a dict with times, concentrations (rows aligned to times),
    step counters and the integrator's estimated (scaled) local error.
    """
    if not _is_finite_number(temperature) or float(temperature) <= 0.0:
        raise ValidationError([("temperature", "must be a positive finite number (kelvin)")])
    c0 = validate_initial_concentrations(net, initial_concentrations)

    if not isinstance(times, list) or not times:
        raise ValidationError([("times", "must be a non-empty list")])
    tq = []
    for i, t in enumerate(times):
        if not _is_finite_number(t) or float(t) < 0.0:
            raise ValidationError([(f"times[{i}]", "must be a non-negative finite number")])
        if tq and float(t) <= tq[-1]:
            raise ValidationError([(f"times[{i}]", "times must be strictly increasing")])
        tq.append(float(t))

    kvec = kvec_from_original(net, params, float(temperature))

    def rhs(y):
        return net.rhs(y, kvec)

    # one integration through to the final requested time, landing exactly
    # on every requested time so reported values are 5th-order node values
    try:
        res = integrate(rhs, c0, tq[-1], rtol=rtol, atol=atol, land=tq)
    except IntegrationError as exc:
        raise ValidationError([
            ("parameters", f"simulation produced non-finite concentrations "
                           f"({exc}); check rate constants and time span")
        ])
    C = res.sample(tq)
    if not np.all(np.isfinite(C)):
        raise ValidationError([("parameters", "simulation produced non-finite concentrations")])
    return {
        "times": tq,
        "concentrations": {sp: C[:, j].tolist() for j, sp in enumerate(net.species)},
        "accepted_steps": res.accepted_steps,
        "rejected_steps": res.rejected_steps,
        "steps": res.accepted_steps,
        "estimated_error": res.estimated_error,
        "_result": res,
    }
