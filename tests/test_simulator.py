import math

import numpy as np

from app.integrator import IntegrationError, integrate
from app.kinetics import Network
from app.simulator import simulate

ABC = {
    "components": ["A", "B", "C"],
    "reactions": [
        {"id": "r1", "stoichiometry": {"A": -1, "B": 1}, "orders": {"A": 1}},
        {"id": "r2", "stoichiometry": {"B": -1, "C": 1}, "orders": {"B": 1}},
    ],
}
K1, K2 = 0.2, 0.1


def _net():
    return Network(ABC["components"], ABC["reactions"])


def _analytical(t, k1=K1, k2=K2, c0=1.0):
    a = c0 * math.exp(-k1 * t)
    b = c0 * k1 / (k2 - k1) * (math.exp(-k1 * t) - math.exp(-k2 * t))
    return a, b, c0 - a - b


def test_against_analytical_solution():
    net = _net()
    ln_a = np.array([math.log(K1), math.log(K2)])
    ea = np.zeros(2)
    times = np.linspace(0, 30, 3001).tolist()
    res = simulate(net, 300.0, [1.0, 0.0, 0.0], 30.0, ln_a, ea,
                   sample_times=times)
    for i, t in enumerate(res["times"]):
        a, b, c = _analytical(t)
        row = res["values"][i]
        assert abs(row[0] - a) < 1e-9
        assert abs(row[1] - b) < 1e-9
        assert abs(row[2] - c) < 1e-9


def test_mass_conservation_1e8():
    """A->B->C 1:1: sum of concentrations equals the initial total to 1e-8."""
    net = _net()
    ln_a = np.array([math.log(K1), math.log(K2)])
    ea = np.zeros(2)
    res = simulate(net, 300.0, [1.0, 0.0, 0.0], 40.0, ln_a, ea,
                   sample_times=np.linspace(0, 40, 500).tolist(),
                   rtol=1e-10, atol=1e-13)
    totals = res["values"].sum(axis=1)
    assert np.max(np.abs(totals - 1.0)) < 1e-8


def test_b_maximum_near_6_93_minutes():
    net = _net()
    ln_a = np.array([math.log(K1), math.log(K2)])
    ea = np.zeros(2)
    grid = np.arange(0.0, 20.0 + 1e-9, 0.01)
    res = simulate(net, 300.0, [1.0, 0.0, 0.0], 20.0, ln_a, ea,
                   sample_times=grid.tolist(), rtol=1e-11, atol=1e-14)
    i = int(np.argmax(res["values"][:, 1]))
    t_expected = math.log(K1 / K2) / (K1 - K2)
    assert abs(res["times"][i] - t_expected) < 0.02
    # textbook peak value 0.5
    assert abs(res["values"][i, 1] - 0.5) < 1e-4
    assert res["n_steps"] > 0


def test_rate_constant_scaling_compresses_time_axis():
    """Multiplying every rate constant by c compresses the time axis by 1/c:
    y(t; c*k) == y(c*t; k)."""
    net = _net()
    base_ln = np.array([math.log(K1), math.log(K2)])
    ea = np.zeros(2)
    c = 4.0
    t_end = 25.0
    ref = simulate(net, 300.0, [1.0, 0.0, 0.0], c * t_end, base_ln, ea,
                   sample_times=(c * np.arange(0, t_end + 1e-9, 0.05)).tolist(),
                   rtol=1e-11, atol=1e-14)
    scaled = simulate(net, 300.0, [1.0, 0.0, 0.0], t_end,
                      base_ln + math.log(c), ea,
                      sample_times=np.arange(0, t_end + 1e-9, 0.05).tolist(),
                      rtol=1e-11, atol=1e-14)
    assert np.max(np.abs(ref["values"] - scaled["values"])) < 1e-9


def test_reports_step_count_and_error_estimate():
    net = _net()
    ln_a = np.array([math.log(K1), math.log(K2)])
    res = simulate(net, 300.0, [1.0, 0.0, 0.0], 20.0, ln_a, np.zeros(2),
                   sample_times=[1.0, 5.0, 10.0])
    assert res["n_steps"] > 0
    assert res["n_rejected"] >= 0
    assert 0.0 <= res["max_scaled_local_error"] <= 1.0 + 1e-9
    # t0, the three requested samples, and t_end itself
    assert len(res["times"]) == 5
    assert res["times"][-1] == 20.0


def test_nonfinite_initial_state_is_rejected():
    net = _net()
    ln_a = np.array([math.log(K1), math.log(K2)])
    try:
        integrate(lambda t, y: net.rhs(y, 300.0, ln_a, np.zeros(2)),
                  np.array([1.0, np.nan, 0.0]), 0.0, 1.0)
        assert False, "expected IntegrationError"
    except IntegrationError:
        pass


def test_reversible_network_runs():
    net = Network(["A", "B"], [
        {"id": "r1", "stoichiometry": {"A": -1, "B": 1},
         "orders": {"A": 1}, "reversible": True}])
    assert net.n_channels == 2
    ln_a = np.array([math.log(0.3), math.log(0.1)])
    res = simulate(net, 300.0, [1.0, 0.0], 60.0, ln_a, np.zeros(2),
                   sample_times=[1.0, 5.0, 60.0])
    v = res["values"]
    assert np.all(v >= -1e-12)
    # approaches equilibrium kf/kr = 3 => B/A = 3 => B = 0.75
    assert abs(v[-1, 1] - 0.75) < 1e-6
    assert abs(v.sum(axis=1) - 1.0).max() < 1e-9


def test_second_order_against_analytical_solution():
    """A + B -> C, r = k*c_A*c_B, equal initial amounts a0:
    c_A(t) = a0 / (1 + a0*k*t)."""
    net = Network(
        ["A", "B", "C"],
        [{"id": "r", "stoichiometry": {"A": -1, "B": -1, "C": 1},
          "orders": {"A": 1, "B": 1}}])
    k = 0.3
    a0 = 1.0
    res = simulate(net, 300.0, [a0, a0, 0.0], 10.0,
                   np.array([math.log(k)]), np.zeros(1),
                   sample_times=np.linspace(0, 10, 201).tolist(),
                   rtol=1e-11, atol=1e-14)
    for i, t in enumerate(res["times"]):
        a = a0 / (1.0 + a0 * k * t)
        assert abs(res["values"][i, 0] - a) < 1e-10
        assert abs(res["values"][i, 1] - a) < 1e-10
        assert abs(res["values"][i, 2] - (a0 - a)) < 1e-10
    # A+B->C changes molecule count, so the sum is not conserved; the true
    # invariants are c_A - c_C-style elemental balances (here A = B and
    # c_A + c_C = a0).
    v = res["values"]
    assert abs(v[:, 0] - v[:, 1]).max() < 1e-10
    assert abs(v[:, 0] + v[:, 2] - a0).max() < 1e-10
