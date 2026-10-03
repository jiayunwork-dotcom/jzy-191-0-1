"""Request validation with explicit, field-specific error messages."""

from __future__ import annotations

import math


class ValidationError(Exception):
    def __init__(self, errors):
        # errors: list of {"field": path, "message": str}
        self.errors = errors
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))


def _is_number(x):
    # bool is a subclass of int in Python; reject it explicitly.
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite_number(x):
    return _is_number(x) and math.isfinite(float(x))


def _err(errors, field, message):
    errors.append({"field": field, "message": message})


def validate_network(spec):
    errors: list[dict] = []
    if not isinstance(spec, dict):
        raise ValidationError([{"field": ".", "message": "must be an object"}])

    comps = spec.get("components")
    if not isinstance(comps, list) or not comps:
        _err(errors, "components", "must be a non-empty list of component names")
        comps = []
    names = []
    for i, c in enumerate(comps):
        if not isinstance(c, str) or not c.strip():
            _err(errors, f"components[{i}]", "must be a non-empty string")
        else:
            names.append(c)
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        _err(errors, "components", f"duplicate component names: {dup}")
    name_set = set(names)

    rxns = spec.get("reactions")
    if not isinstance(rxns, list) or not rxns:
        _err(errors, "reactions", "must be a non-empty list")
        rxns = []
    rx_ids = []
    for i, r in enumerate(rxns):
        base = f"reactions[{i}]"
        if not isinstance(r, dict):
            _err(errors, base, "must be an object")
            continue
        rid = r.get("id")
        if not isinstance(rid, str) or not rid.strip():
            _err(errors, f"{base}.id", "must be a non-empty string")
        else:
            rx_ids.append(rid)
        sto = r.get("stoichiometry")
        if not isinstance(sto, dict) or not sto:
            _err(errors, f"{base}.stoichiometry",
                 "must be a non-empty object mapping component -> coefficient")
        else:
            for comp, coef in sto.items():
                if comp not in name_set:
                    _err(errors, f"{base}.stoichiometry.{comp}",
                          f"references undefined component {comp!r}")
                if not _finite_number(coef) or float(coef) == 0.0:
                    _err(errors, f"{base}.stoichiometry.{comp}",
                          "must be a finite non-zero number")
        orders = r.get("orders", {})
        if orders is None:
            orders = {}
        if not isinstance(orders, dict):
            _err(errors, f"{base}.orders", "must be an object")
        else:
            for comp, p in orders.items():
                if comp not in name_set:
                    _err(errors, f"{base}.orders.{comp}",
                          f"references undefined component {comp!r}")
                elif not _finite_number(p) or float(p) < 0:
                    _err(errors, f"{base}.orders.{comp}",
                          "must be a finite non-negative number")
        rev_orders = r.get("reverse_orders")
        if rev_orders is not None:
            if not isinstance(rev_orders, dict):
                _err(errors, f"{base}.reverse_orders", "must be an object")
            else:
                for comp, p in rev_orders.items():
                    if comp not in name_set:
                        _err(errors, f"{base}.reverse_orders.{comp}",
                              f"references undefined component {comp!r}")
                    elif not _finite_number(p) or float(p) < 0:
                        _err(errors, f"{base}.reverse_orders.{comp}",
                              "must be a finite non-negative number")
        rev = r.get("reversible", False)
        if not isinstance(rev, bool):
            _err(errors, f"{base}.reversible", "must be true or false")

    if len(set(rx_ids)) != len(rx_ids):
        dup = sorted({n for n in rx_ids if rx_ids.count(n) > 1})
        _err(errors, "reactions", f"duplicate reaction ids: {dup}")

    if errors:
        raise ValidationError(errors)
    return {"components": names,
            "reactions": [r for r in rxns if isinstance(r, dict)]}


def validate_batch(batch, component_names):
    errors: list[dict] = []
    if not isinstance(batch, dict):
        raise ValidationError([{"field": ".", "message": "must be an object"}])
    name_set = set(component_names)

    T = batch.get("temperature")
    if not _finite_number(T):
        _err(errors, "temperature", "must be a finite number")
    elif float(T) <= 0.0:
        _err(errors, "temperature", f"must be positive, got {T}")

    y0 = batch.get("initial_concentrations")
    if not isinstance(y0, dict):
        _err(errors, "initial_concentrations",
             "must be an object mapping component -> concentration")
    else:
        for comp, val in y0.items():
            if comp not in name_set:
                _err(errors, f"initial_concentrations.{comp}",
                      f"undefined component {comp!r}")
            elif not _finite_number(val):
                _err(errors, f"initial_concentrations.{comp}",
                     "must be a finite number")
            elif float(val) < 0.0:
                _err(errors, f"initial_concentrations.{comp}",
                     f"concentration must not be negative, got {val}")

    samples = batch.get("samples")
    if not isinstance(samples, list) or not samples:
        _err(errors, "samples", "must be a non-empty list")
    else:
        prev_t = 0.0
        first = True
        for i, s in enumerate(samples):
            base = f"samples[{i}]"
            if not isinstance(s, dict):
                _err(errors, base, "must be an object")
                continue
            t = s.get("time")
            if not _finite_number(t):
                _err(errors, f"{base}.time", "must be a finite number")
            else:
                t = float(t)
                if t < 0.0:
                    _err(errors, f"{base}.time",
                         f"sample time must not be negative, got {t}")
                elif not first and t <= prev_t:
                    _err(errors, f"{base}.time",
                         f"sample times must be strictly increasing: "
                         f"{t} follows {prev_t}")
                else:
                    prev_t = t
                first = False
            obs = s.get("observations")
            if not isinstance(obs, dict) or not obs:
                _err(errors, f"{base}.observations",
                     "must be a non-empty object mapping component -> "
                     "measured concentration")
            else:
                for comp, val in obs.items():
                    if comp not in name_set:
                        _err(errors, f"{base}.observations.{comp}",
                              f"undefined component {comp!r}")
                    elif not _finite_number(val):
                        _err(errors, f"{base}.observations.{comp}",
                             "must be a finite number")
                    elif float(val) < 0.0:
                        _err(errors, f"{base}.observations.{comp}",
                             f"measured concentration must not be negative, "
                             f"got {val}")

    if errors:
        raise ValidationError(errors)
    return True


def validate_simulation_request(req, component_names, n_channels):
    """Validate a simulation request; also accepts Arrhenius parameters in
    either raw (A, Ea) or fitted (ln_A, Ea) form."""
    errors: list[dict] = []
    if not isinstance(req, dict):
        raise ValidationError([{"field": ".", "message": "must be an object"}])

    T = req.get("temperature")
    if not _finite_number(T):
        _err(errors, "temperature", "must be a finite number")
    elif float(T) <= 0.0:
        _err(errors, "temperature", f"must be positive, got {T}")

    y0 = req.get("initial_concentrations")
    name_set = set(component_names)
    parsed_y0 = [0.0] * len(component_names)
    if not isinstance(y0, dict) or not y0:
        _err(errors, "initial_concentrations",
             "must be a non-empty object")
    else:
        for comp, val in y0.items():
            if comp not in name_set:
                _err(errors, f"initial_concentrations.{comp}",
                      f"undefined component {comp!r}")
            elif not _finite_number(val) or float(val) < 0:
                _err(errors, f"initial_concentrations.{comp}",
                     "must be a finite non-negative number")
            else:
                parsed_y0[component_names.index(comp)] = float(val)

    t_end = req.get("t_end")
    if not _finite_number(t_end) or float(t_end) <= 0:
        _err(errors, "t_end", "must be a finite positive number")

    t_eval = req.get("sample_times", [])
    t_end_num = float(t_end) if _finite_number(t_end) else None
    if not isinstance(t_eval, list):
        _err(errors, "sample_times", "must be a list")
    else:
        prev = 0.0
        for i, t in enumerate(t_eval):
            if not _finite_number(t) or float(t) < 0:
                _err(errors, f"sample_times[{i}]",
                     "must be a finite non-negative number")
            elif t_end_num is not None and float(t) > t_end_num:
                _err(errors, f"sample_times[{i}]",
                     f"sample time {t} exceeds t_end {t_end_num}")
            elif i > 0 and float(t) <= prev:
                _err(errors, f"sample_times[{i}]",
                     "must be strictly increasing")
            prev = float(t)

    params = req.get("parameters")
    ln_a, ea = _validate_params(params, n_channels, errors)
    if errors:
        raise ValidationError(errors)
    return {
        "temperature": float(T),
        "initial_concentrations": parsed_y0,
        "t_end": float(t_end),
        "sample_times": [float(t) for t in t_eval],
        "ln_a": ln_a,
        "ea": ea,
    }


def _validate_params(params, n_channels, errors):
    """Accept {"per_channel": [{"ln_a"|"a", "ea"}, ...]} or a legacy two
    parallel arrays form. Returns (ln_a, ea) arrays-as-lists."""
    if not isinstance(params, dict):
        _err(errors, "parameters", "must be an object")
        return None, None
    chans = params.get("per_channel")
    if not isinstance(chans, list) or len(chans) != n_channels:
        _err(errors, "parameters.per_channel",
             f"must be a list of {n_channels} channel parameter objects "
             f"(one per forward/reverse elementary step)")
        return None, None
    ln_a, ea = [], []
    for i, ch in enumerate(chans):
        base = f"parameters.per_channel[{i}]"
        if not isinstance(ch, dict):
            _err(errors, base, "must be an object")
            continue
        has_a = "a" in ch
        has_lna = "ln_a" in ch
        if not (has_a or has_lna):
            _err(errors, f"{base}.a", "must provide pre-exponential factor 'a' "
                                      "(or 'ln_a')")
        else:
            key = "a" if has_a else "ln_a"
            v = ch.get(key)
            if not _finite_number(v):
                _err(errors, f"{base}.{key}", "must be a finite number")
            elif key == "a" and float(v) <= 0:
                _err(errors, f"{base}.a",
                     f"pre-exponential factor must be positive, got {v}")
            else:
                ln_a.append(math.log(float(v)) if key == "a" else float(v))
        v = ch.get("ea", 0.0)
        if not _finite_number(v):
            _err(errors, f"{base}.ea", "activation energy must be a finite "
                                       "number [J/mol]")
        else:
            ea.append(float(v))
    if len(ln_a) != n_channels or len(ea) != n_channels:
        return None, None
    return ln_a, ea
