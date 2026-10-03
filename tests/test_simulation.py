"""Tests of the chemistry/integrator/simulation core (no database)."""

import math

import numpy as np
import pytest

from app.chemistry import ValidationError, compile_network
from app.simulation import kvec_from_reparam, simulate


ABC = {
    "species": ["A", "B", "C"],
    "reactions": [
        {"name": "r1", "stoichiometry": {"A": -1, "B": 1}},
        {"name": "r2", "stoichiometry": {"B": -1, "C": 1}},
    ],
}


def test_mass_conservation_1e_8():
    net = compile_network(ABC)
    out = simulate(
        net, [{"A": 0.2, "Ea": 0.0}, {"A": 0.1, "Ea": 0.0}],
        300.0, {"A": 1.0}, np.linspace(0.0, 30.0, 61).tolist(),
    )
    C = np.array([out["concentrations"][s] for s in ("A", "B", "C")])
    total = C.sum(axis=0)
    assert np.max(np.abs(total - 1.0)) < 1e-8


def test_B_maximum_at_6_93():
    net = compile_network(ABC)
    out = simulate(
        net, [{"A": 0.2, "Ea": 0.0}, {"A": 0.1, "Ea": 0.0}],
        300.0, {"A": 1.0}, [20.0],
    )
    res = out["_result"]
    ts = np.linspace(0.0, 20.0, 400001)
    B = res.sample(ts)[:, 1]
    t_max = ts[int(np.argmax(B))]
    assert t_max == pytest.approx(math.log(2) / 0.1, abs=1e-3)


def test_analytic_curves_series_first_order():
    net = compile_network(ABC)
    k1, k2 = 0.2, 0.1
    tt = [0.25, 0.7, 1.5, 3.3, 8.0, 15.0]
    out = simulate(
        net, [{"A": k1, "Ea": 0.0}, {"A": k2, "Ea": 0.0}],
        300.0, {"A": 1.0}, tt,
    )
    t = np.array(tt)
    Aa = np.exp(-k1 * t)
    Ba = k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    Ca = 1 - Aa - Ba
    assert np.max(np.abs(np.array(out["concentrations"]["A"]) - Aa)) < 1e-7
    assert np.max(np.abs(np.array(out["concentrations"]["B"]) - Ba)) < 1e-7
    assert np.max(np.abs(np.array(out["concentrations"]["C"]) - Ca)) < 1e-7


def test_time_rescaling_k_c():
    """Multiplying every k by c compresses the time axis to t/c."""
    net = compile_network(ABC)
    c = 2.75
    base_t = [0.5, 1.0, 3.0, 7.0, 14.0]
    o1 = simulate(
        net, [{"A": 0.2, "Ea": 0.0}, {"A": 0.1, "Ea": 0.0}],
        300.0, {"A": 1.0}, base_t,
    )
    o2 = simulate(
        net, [{"A": 0.2 * c, "Ea": 0.0}, {"A": 0.1 * c, "Ea": 0.0}],
        300.0, {"A": 1.0}, [t / c for t in base_t],
    )
    for s in ("A", "B", "C"):
        assert np.max(np.abs(np.array(o1["concentrations"][s])
                             - np.array(o2["concentrations"][s]))) < 1e-10


def test_reversible_first_order_equilibrium():
    net_def = {
        "species": ["A", "B"],
        "reactions": [
            {"name": "r", "stoichiometry": {"A": -1, "B": 1}, "reversible": True},
        ],
    }
    net = compile_network(net_def)
    kf, kr = 0.3, 0.7
    out = simulate(
        net, [{"A": kf, "Ea": 0.0}, {"A": kr, "Ea": 0.0}],
        300.0, {"A": 1.0}, [0.0, 1.0, 10.0],
    )
    A = np.array(out["concentrations"]["A"])
    B = np.array(out["concentrations"]["B"])
    assert np.max(np.abs(A + B - 1.0)) < 1e-10
    t = np.array([0.0, 1.0, 10.0])
    A_analytic = (kr + kf * np.exp(-(kf + kr) * t)) / (kf + kr)
    assert np.max(np.abs(A - A_analytic)) < 1e-8
    assert A[-1] == pytest.approx(kr / (kf + kr), abs=1e-4)
    assert B[-1] == pytest.approx(kf / (kf + kr), abs=1e-4)


def test_steps_and_error_reported():
    net = compile_network(ABC)
    out = simulate(
        net, [{"A": 0.2, "Ea": 0.0}, {"A": 0.1, "Ea": 0.0}],
        300.0, {"A": 1.0}, [10.0],
    )
    assert out["steps"] > 0
    assert out["accepted_steps"] == out["steps"]
    assert out["estimated_error"] >= 0.0
    assert np.isfinite(out["estimated_error"])


def test_arrhenius_parameter_mapping():
    beta = np.array([math.log(0.2), 0.0, math.log(0.1), 0.0])
    k = kvec_from_reparam(beta, 300.0, T_ref=300.0)
    assert k == pytest.approx([0.2, 0.1])
    k2 = kvec_from_reparam(beta, np.array([300.0, 310.0]), T_ref=300.0)
    assert k2.shape == (2, 2)
    assert k2[0] == pytest.approx([0.2, 0.1])


# --------------------------------------------------------------------------
# field-level validation
# --------------------------------------------------------------------------

def test_negative_concentration_field():
    net = compile_network(ABC)
    with pytest.raises(ValidationError) as exc:
        simulate(
            net, [{"A": 0.2, "Ea": 0.0}, {"A": 0.1, "Ea": 0.0}],
            300.0, {"A": -0.5}, [1.0],
        )
    assert any("initial_concentrations.A" in f for f, _ in exc.value.errors)


def test_non_finite_concentration_field():
    net = compile_network(ABC)
    with pytest.raises(ValidationError) as exc:
        simulate(
            net, [{"A": 0.2, "Ea": 0.0}, {"A": 0.1, "Ea": 0.0}],
            300.0, {"A": float("nan")}, [1.0],
        )
    assert any("initial_concentrations.A" in f for f, _ in exc.value.errors)


def test_non_positive_temperature_field():
    net = compile_network(ABC)
    with pytest.raises(ValidationError) as exc:
        simulate(
            net, [{"A": 0.2, "Ea": 0.0}, {"A": 0.1, "Ea": 0.0}],
            0.0, {"A": 1.0}, [1.0],
        )
    assert any(f == "temperature" for f, _ in exc.value.errors)


def test_sample_times_not_increasing_field():
    net = compile_network(ABC)
    batch = {
        "temperature": 300.0,
        "initial_concentrations": {"A": 1.0},
        "samples": [
            {"time": 1.0, "concentrations": {"A": 0.8}},
            {"time": 1.0, "concentrations": {"A": 0.7}},
        ],
    }
    from app.simulation import validate_batch
    with pytest.raises(ValidationError) as exc:
        validate_batch(net, batch)
    assert any("samples[1].time" in f for f, _ in exc.value.errors)


def test_reaction_undefined_species_field():
    bad = {
        "species": ["A", "B"],
        "reactions": [
            {"name": "r1", "stoichiometry": {"A": -1, "Z": 1}},
        ],
    }
    with pytest.raises(ValidationError) as exc:
        compile_network(bad)
    assert any("reactions[0].stoichiometry.Z" in f for f, _ in exc.value.errors)


def test_orders_undefined_species_field():
    bad = {
        "species": ["A", "B"],
        "reactions": [
            {"name": "r1", "stoichiometry": {"A": -1, "B": 1},
             "orders": {"Q": 2.0}},
        ],
    }
    with pytest.raises(ValidationError) as exc:
        compile_network(bad)
    assert any("reactions[0].orders.Q" in f for f, _ in exc.value.errors)
