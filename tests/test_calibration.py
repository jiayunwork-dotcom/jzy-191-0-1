"""Tests of parameter calibration: recovery, identifiability, starts."""

import math

import numpy as np
import pytest

from app.calibration import calibrate, prepare_dataset
from app.chemistry import compile_network
from app.simulation import simulate

ABC = {
    "species": ["A", "B", "C"],
    "reactions": [
        {"name": "r1", "stoichiometry": {"A": -1, "B": 1}},
        {"name": "r2", "stoichiometry": {"B": -1, "C": 1}},
    ],
}
TRUE = [{"A": 1.0e5, "Ea": 3.0e4}, {"A": 2.0e4, "Ea": 2.0e4}]
R = 8.314462618
TIMES = [0.5, 1.0, 2.0, 4.0, 7.0, 12.0]


def make_batches(temperatures, true=TRUE, ic=None):
    net = compile_network(ABC)
    batches = []
    for T in temperatures:
        out = simulate(net, true, T, ic or {"A": 1.0}, TIMES)
        samples = [
            {"time": t,
             "concentrations": {s: out["concentrations"][s][i] for s in ("A", "B", "C")}}
            for i, t in enumerate(TIMES)
        ]
        batches.append({
            "temperature": T,
            "initial_concentrations": ic or {"A": 1.0},
            "samples": samples,
        })
    return batches


def test_recover_known_parameters_1e_4():
    net = compile_network(ABC)
    batches = make_batches([290.0, 300.0, 310.0, 320.0])
    res = calibrate(net, batches)
    assert res["termination"] == "converged"
    assert res["rmse"] < 1e-6
    for p, t in zip(res["parameters"], TRUE):
        assert abs(p["A"] / t["A"] - 1.0) < 1e-4
        assert abs(p["Ea"] / t["Ea"] - 1.0) < 1e-4
        assert p["identifiable"]["Ea"] is True


def test_standard_errors_present_finite():
    net = compile_network(ABC)
    batches = make_batches([290.0, 300.0, 310.0, 320.0])
    res = calibrate(net, batches)
    for p in res["parameters"]:
        assert p["se_A"] is not None and math.isfinite(p["se_A"])
        assert p["se_Ea"] is not None and math.isfinite(p["se_Ea"])
        assert p["se_A"] < 0.1 * p["A"]
        assert p["se_Ea"] < 0.1 * abs(p["Ea"])


def test_single_temperature_Ea_nonidentifiable():
    net = compile_network(ABC)
    batches = make_batches([300.0])
    res = calibrate(net, batches)
    assert res["non_identifiable"]
    keys = {(d["constant"], d["parameter"]) for d in res["non_identifiable"]}
    for label in ("r1.forward", "r2.forward"):
        assert (label, "Ea") in keys
        assert (label, "A") in keys
    for p in res["parameters"]:
        assert p["identifiable"]["Ea"] is False
        assert p["identifiable"]["A"] is False
        assert p["identifiable"]["ln_k_ref"] is True
        assert p["se_Ea"] is None
        assert p["se_A"] is None
    # k(T0) is still identified correctly
    for p, t in zip(res["parameters"], TRUE):
        k_true = t["A"] * math.exp(-t["Ea"] / (R * 300.0))
        assert p["k_at_Tref"] == pytest.approx(k_true, rel=1e-6)


def test_cold_hot_start_agree_1e_6():
    net = compile_network(ABC)
    batches = make_batches([290.0, 300.0, 310.0, 320.0])
    cold = calibrate(net, batches)
    beta_hot = np.array(cold["_beta"])
    beta_hot[0::2] *= 1.5
    beta_hot[1::2] += 50.0
    hot = calibrate(net, batches, initial_beta=beta_hot)
    b1 = np.array(cold["_beta"])
    b2 = np.array(hot["_beta"])
    rel = np.max(np.abs(b1 - b2) / np.maximum(np.abs(b1), 1e-30))
    assert rel < 1e-6


def test_stop_reason_reported():
    net = compile_network(ABC)
    batches = make_batches([290.0, 300.0, 310.0, 320.0])
    res = calibrate(net, batches, max_iter=200)
    assert res["iterations"] <= 200
    assert res["termination"] in ("converged", "max_iterations")
    assert res["stop_reason"] in (
        "relative_change_below_1e-10", "no_downhill_step", "max_iterations")


def test_iteration_cap_is_reported_as_not_converged():
    net = compile_network(ABC)
    batches = make_batches([290.0, 300.0, 310.0, 320.0])
    res = calibrate(net, batches, max_iter=1)
    assert res["iterations"] == 1
    assert res["termination"] == "max_iterations"
    assert res["stop_reason"] == "max_iterations"


def test_partial_observations_supported():
    """Measuring only B in every batch still determines the network."""
    net = compile_network(ABC)
    batches = make_batches([290.0, 300.0, 310.0, 320.0])
    for b in batches:
        for s in b["samples"]:
            s["concentrations"] = {"B": s["concentrations"]["B"]}
    res = calibrate(net, batches)
    for p, t in zip(res["parameters"], TRUE):
        assert abs(p["A"] / t["A"] - 1.0) < 1e-4
        assert abs(p["Ea"] / t["Ea"] - 1.0) < 1e-4


def test_residuals_length_matches_observations():
    net = compile_network(ABC)
    batches = make_batches([290.0, 300.0])
    ds = prepare_dataset(net, batches)
    res = calibrate(net, batches)
    assert len(res["residuals"]) == ds.n_obs
    assert res["n_observations"] == ds.n_obs


def test_reference_temperature_is_mean_of_data():
    net = compile_network(ABC)
    batches = make_batches([280.0, 300.0, 320.0])
    res = calibrate(net, batches)
    assert res["T_ref"] == pytest.approx(300.0, abs=1e-9)


def test_reversible_network_recovers_both_constants():
    net_def = {
        "species": ["A", "B"],
        "reactions": [
            {"name": "r", "stoichiometry": {"A": -1, "B": 1}, "reversible": True},
        ],
    }
    net = compile_network(net_def)
    assert net.n_constants == 2
    assert net.rate_constant_labels() == ["r.forward", "r.reverse"]
    true = [{"A": 4.0e4, "Ea": 2.5e4}, {"A": 9.0e4, "Ea": 3.5e4}]
    batches = []
    times = [0.2, 0.5, 1.0, 2.0, 4.0]
    for T in (290.0, 300.0, 310.0, 320.0):
        out = simulate(net, true, T, {"A": 1.0}, times)
        samples = [
            {"time": t,
             "concentrations": {s: out["concentrations"][s][i] for s in ("A", "B")}}
            for i, t in enumerate(times)
        ]
        batches.append({"temperature": T, "initial_concentrations": {"A": 1.0},
                        "samples": samples})
    res = calibrate(net, batches)
    assert res["termination"] == "converged"
    assert res["rmse"] < 1e-6
    assert [p["label"] for p in res["parameters"]] == ["r.forward", "r.reverse"]
    for p, t in zip(res["parameters"], true):
        assert abs(p["A"] / t["A"] - 1.0) < 1e-4
        assert abs(p["Ea"] / t["Ea"] - 1.0) < 1e-4
        assert p["identifiable"]["Ea"] is True
