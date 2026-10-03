import math

import numpy as np

from app.calibration import calibrate
from app.kinetics import Network, R_GAS
from tests.helpers import (ABC_NETWORK, REVERSIBLE_NETWORK,
                           arrhenius_channels, make_batch)


def _net():
    return Network(ABC_NETWORK["components"], ABC_NETWORK["reactions"])


def _truth():
    t_ref = 320.0
    eas = [40000.0, 50000.0]
    k_refs = [0.2, 0.1]
    ln_a = [math.log(k) + ea / (R_GAS * t_ref)
            for k, ea in zip(k_refs, eas)]
    return ln_a, eas, t_ref


def test_multi_temperature_recovers_parameters_1e4():
    net = _net()
    ln_a, eas, t_ref = _truth()
    batches = [
        make_batch(net, T, ln_a, eas,
                   times=[2.0, 5.0, 8.0, 12.0, 20.0, 30.0])
        for T in (300.0, 310.0, 320.0, 330.0, 340.0)
    ]
    rep = calibrate(net, batches)
    assert rep["stop_reason"] == "converged"
    assert rep["converged"] is True
    assert rep["objective"]["sse"] < 1e-12
    true_a = [math.exp(ln_a[0]), math.exp(ln_a[1])]
    for j, ch in enumerate(rep["channels"]):
        assert ch["identifiable"] is True
        assert abs(ch["a"] - true_a[j]) / true_a[j] < 1e-4
        assert abs(ch["ea"] - eas[j]) / eas[j] < 1e-4
        assert ch["a_se"] is not None and ch["ea_se"] is not None


def test_partial_component_observation_still_fits():
    """No component must be measured in every sample; C is missing entirely
    from two of the five batches."""
    net = _net()
    ln_a, eas, _ = _truth()
    batches = []
    for i, T in enumerate((300.0, 310.0, 320.0, 330.0, 340.0)):
        observe = {"A", "B"} if i % 2 else {"A", "B", "C"}
        batches.append(make_batch(net, T, ln_a, eas,
                                  observe=observe,
                                  times=[2.0, 5.0, 8.0, 12.0, 20.0, 30.0]))
    rep = calibrate(net, batches)
    assert rep["objective"]["sse"] < 1e-12
    true_a = [math.exp(ln_a[0]), math.exp(ln_a[1])]
    for j, ch in enumerate(rep["channels"]):
        assert abs(ch["a"] - true_a[j]) / true_a[j] < 1e-4
        assert abs(ch["ea"] - eas[j]) / eas[j] < 1e-4


def test_single_temperature_reports_unidentifiable_ea():
    net = _net()
    ln_a, eas, _ = _truth()
    batches = [
        make_batch(net, 320.0, ln_a, eas,
                   times=[2.0, 5.0, 8.0, 12.0, 25.0]),
        make_batch(net, 320.0, ln_a, eas, y0=[0.8, 0.2, 0.0],
                   times=[2.0, 5.0, 8.0, 12.0, 25.0]),
    ]
    rep = calibrate(net, batches)
    assert rep["all_identifiable"] is False
    for ch in rep["channels"]:
        assert ch["identifiable"] is False
        assert ch["ea"] is None and ch["a"] is None
        assert ch["ea_se"] is None and ch["a_se"] is None
        assert "activation energy" in ch["unidentifiable_reason"]
        # ... but the rate constant at that temperature is still well found
        assert ch["k_at_reference_se"] is not None
    ks = [ch["k_at_reference"] for ch in rep["channels"]]
    assert abs(ks[0] - 0.2) / 0.2 < 1e-6
    assert abs(ks[1] - 0.1) / 0.1 < 1e-6


def test_cold_and_warm_start_agree_within_1e6():
    net = _net()
    ln_a, eas, _ = _truth()
    batches = [
        make_batch(net, T, ln_a, eas, times=[2.0, 5.0, 8.0, 12.0, 25.0])
        for T in (300.0, 320.0, 340.0)
    ]
    cold = calibrate(net, batches)
    bad_initial = {"q": np.array([math.log(5e-3), math.log(2.0)]),
                    "theta": np.array([500.0, 9000.0])}
    warm = calibrate(net, batches, initial=bad_initial)
    qc = np.array(cold["internal_parameters"]["q"])
    qw = np.array(warm["internal_parameters"]["q"])
    tc = np.array(cold["internal_parameters"]["theta"])
    tw = np.array(warm["internal_parameters"]["theta"])
    assert np.max(np.abs(qc - qw) / np.maximum(np.abs(qc), 1e-30)) < 1e-6
    assert np.max(np.abs(tc - tw) / np.maximum(np.abs(tc), 1e-30)) < 1e-6
    assert abs(cold["objective"]["sse"] - warm["objective"]["sse"]) < 1e-20


def test_iteration_cap_is_reported():
    net = _net()
    ln_a, eas, _ = _truth()
    batches = [
        make_batch(net, T, ln_a, eas, times=[2.0, 5.0, 8.0, 12.0, 25.0])
        for T in (300.0, 320.0, 340.0)
    ]
    rep = calibrate(net, batches, max_iters=1)
    assert rep["max_iterations"] == 1
    assert rep["iterations"] <= 1
    assert rep["stop_reason"] in {"converged", "max_iterations"}


def test_max_iteration_stop_on_unsolved_noisy_fit():
    """Noisy data need several iterations; a one-iteration budget must be
    reported as 'max_iterations', not as convergence."""
    net = _net()
    ln_a, eas, _ = _truth()
    batches = [
        make_batch(net, T, ln_a, eas, times=[2.0, 5.0, 8.0, 12.0, 25.0])
        for T in (300.0, 320.0, 340.0)
    ]
    rng = np.random.default_rng(0)
    for b in batches:
        for s in b["samples"]:
            for comp in s["observations"]:
                s["observations"][comp] = max(
                    0.0, s["observations"][comp] + rng.normal(0.0, 0.01))
    rep = calibrate(net, batches, max_iters=1)
    assert rep["iterations"] == 1
    assert rep["stop_reason"] == "max_iterations"
    assert rep["converged"] is False

    full = calibrate(net, batches, max_iters=200)
    assert full["stop_reason"] == "converged"
    assert full["objective"]["sse"] < rep["objective"]["sse"]


def test_residuals_are_returned_per_observation():
    net = _net()
    ln_a, eas, _ = _truth()
    batches = [make_batch(net, 300.0, ln_a, eas, times=[2.0, 5.0, 8.0])]
    rep = calibrate(net, batches)
    n_obs = 3 * 3
    assert len(rep["residuals"]) == n_obs
    for row in rep["residuals"]:
        assert set(row) == {"batch_id", "time", "component", "observed",
                            "predicted", "residual"}
        assert abs(row["predicted"] - row["observed"] - row["residual"]) < 1e-12


def test_reference_temperature_is_mean_reciprocal():
    net = _net()
    ln_a, eas, _ = _truth()
    temps = [300.0, 320.0, 340.0]
    batches = [make_batch(net, T, ln_a, eas, times=[2.0, 5.0, 8.0])
               for T in temps]
    rep = calibrate(net, batches)
    expected = 1.0 / np.mean([1.0 / T for T in temps])
    assert abs(rep["reference_temperature"] - expected) < 1e-9


def test_reversible_reaction_fits_both_channels():
    """A reversible elementary step has forward and reverse channels, each
    with its own A and Ea; both must be recovered."""
    net = Network(REVERSIBLE_NETWORK["components"],
                  REVERSIBLE_NETWORK["reactions"])
    assert net.n_channels == 2
    ln_a, ea = arrhenius_channels([0.3, 0.08], [30000.0, 25000.0], 320.0)
    batches = [
        make_batch(net, T, ln_a, ea, times=[1.0, 3.0, 6.0, 10.0, 20.0])
        for T in (300.0, 320.0, 340.0)
    ]
    rep = calibrate(net, batches)
    assert rep["stop_reason"] == "converged"
    true_a = [math.exp(v) for v in ln_a]
    assert len(rep["channels"]) == 2
    assert [c["direction"] for c in rep["channels"]] == ["forward", "reverse"]
    for j, ch in enumerate(rep["channels"]):
        assert abs(ch["a"] - true_a[j]) / true_a[j] < 1e-4
        assert abs(ch["ea"] - ea[j]) / ea[j] < 1e-4


def test_zero_time_initial_samples_are_handled():
    """A sample at t = 0 is an initial-state observation and must line up
    with y0 rather than shift the residual alignment."""
    net = _net()
    ln_a, eas, _ = _truth()
    batches = [
        make_batch(net, T, ln_a, eas, times=[2.0, 5.0, 8.0, 25.0])
        for T in (300.0, 320.0, 340.0)
    ]
    # Insert a t=0 measurement into each batch.
    for b, T in zip(batches, (300.0, 320.0, 340.0)):
        b["samples"].insert(0, {"time": 0.0, "observations": {
            "A": 1.0, "B": 0.0, "C": 0.0}})
    rep = calibrate(net, batches)
    assert rep["stop_reason"] == "converged"
    assert rep["objective"]["n_observations"] == 3 * 5 * 3
    zero_rows = [r for r in rep["residuals"] if r["time"] == 0.0]
    assert len(zero_rows) == 9
    for row in zero_rows:
        assert row["residual"] == 0.0
    for j, ch in enumerate(rep["channels"]):
        assert abs(ch["ea"] - eas[j]) / eas[j] < 1e-4
