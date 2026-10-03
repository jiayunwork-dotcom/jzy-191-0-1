"""Shared test data builders."""

import math

import numpy as np

from app.kinetics import R_GAS
from app.simulator import simulate

ABC_NETWORK = {
    "components": ["A", "B", "C"],
    "reactions": [
        {"id": "r1", "stoichiometry": {"A": -1, "B": 1},
         "orders": {"A": 1}},
        {"id": "r2", "stoichiometry": {"B": -1, "C": 1},
         "orders": {"B": 1}},
    ],
}

REVERSIBLE_NETWORK = {
    "components": ["A", "B"],
    "reactions": [
        {"id": "r1", "stoichiometry": {"A": -1, "B": 1},
         "orders": {"A": 1}, "reversible": True},
    ],
}


def arrhenius_channels(k_refs, eas, t_ref):
    """Build per-channel Arrhenius params from k at t_ref and Ea [J/mol]."""
    ln_a = [math.log(k) + ea / (R_GAS * t_ref)
            for k, ea in zip(k_refs, eas)]
    return ln_a, list(eas)


def make_batch(network, temperature, ln_a, ea, y0=None, times=None,
               components=None, observe=None, t_end=None):
    components = components or network.components
    y0 = y0 or [1.0] + [0.0] * (len(components) - 1)
    times = times or [2.0, 5.0, 8.0, 12.0, 18.0, 25.0]
    t_end = t_end or max(times)
    res = simulate(network, temperature, y0, t_end,
                   np.asarray(ln_a), np.asarray(ea),
                   sample_times=times, rtol=1e-11, atol=1e-14)
    samples = []
    for t, v in zip(res["times"][1:], res["values"][1:]):
        obs = {}
        for j, comp in enumerate(components):
            if observe is None or comp in observe:
                obs[comp] = float(v[j])
        samples.append({"time": float(t), "observations": obs})
    y0_map = {components[j]: float(y0[j]) for j in range(len(components))
              if y0[j] != 0.0}
    return {"temperature": float(temperature),
            "initial_concentrations": y0_map,
            "samples": samples}


def post_network(client, spec=ABC_NETWORK):
    rv = client.post("/api/networks", json=spec)
    assert rv.status_code == 201, rv.get_json()
    return rv.get_json()["id"]


def post_batch(client, network_id, batch):
    rv = client.post(f"/api/networks/{network_id}/batches", json=batch)
    assert rv.status_code == 201, rv.get_json()
    return rv.get_json()["id"]


def make_dataset(client, network_id, batches, name="ds"):
    ids = [post_batch(client, network_id, b) for b in batches]
    rv = client.post(f"/api/networks/{network_id}/datasets",
                     json={"name": name, "batch_ids": ids})
    assert rv.status_code == 201, rv.get_json()
    return rv.get_json(), ids
