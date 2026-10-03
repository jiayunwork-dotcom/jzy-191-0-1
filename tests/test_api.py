"""End-to-end API tests against PostgreSQL."""

import json

import numpy as np
import pytest

from app.simulation import simulate


def _make_batches(net_def, true, temperatures, times, ic=None):
    from app.chemistry import compile_network
    net = compile_network(net_def)
    batches = []
    for T in temperatures:
        out = simulate(net, true, T, ic or {"A": 1.0}, times)
        samples = [
            {"time": t,
             "concentrations": {s: out["concentrations"][s][i] for s in ("A", "B", "C")}}
            for i, t in enumerate(times)
        ]
        batches.append({
            "temperature": T,
            "initial_concentrations": ic or {"A": 1.0},
            "samples": samples,
        })
    return batches


@pytest.fixture()
def network(client, network_def):
    r = client.post("/api/networks", json={"name": "abc", "definition": network_def})
    assert r.status_code == 201, r.get_json()
    return r.get_json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_network_validation_field_errors(client):
    r = client.post("/api/networks", json={
        "name": "bad",
        "definition": {
            "species": ["A"],
            "reactions": [{"name": "r", "stoichiometry": {"A": -1, "Z": 1}}],
        },
    })
    assert r.status_code == 400
    details = r.get_json()["details"]
    assert any(d["field"] == "reactions[0].stoichiometry.Z" for d in details)


def test_dataset_versioning_add_and_remove(client, network):
    ds = client.post(f"/api/networks/{network['id']}/datasets",
                     json={"name": "d1"}).get_json()
    batches = _make_batches(network["definition"],
                            [{"A": 1e5, "Ea": 3e4}, {"A": 2e4, "Ea": 2e4}],
                            [300.0], [0.5, 1.0, 2.0])
    v1 = client.post(f"/api/datasets/{ds['id']}/batches",
                     json={"batch": batches[0]}).get_json()
    assert v1["version"] == 1
    v2 = client.post(f"/api/datasets/{ds['id']}/batches",
                     json={"batch": batches[0]}).get_json()
    assert v2["version"] == 2
    assert v2["parent_version_id"] == v1["id"]

    # v2 keeps two batches, v1 is immutable with one
    assert client.get(f"/api/dataset-versions/{v2['id']}").get_json()["batch_count"] == 2
    assert client.get(f"/api/dataset-versions/{v1['id']}").get_json()["batch_count"] == 1

    v3 = client.delete(f"/api/datasets/{ds['id']}/batches/0").get_json()
    assert v3["version"] == 3
    assert client.get(f"/api/dataset-versions/{v3['id']}").get_json()["batch_count"] == 1
    # old versions retained
    versions = client.get(f"/api/datasets/{ds['id']}/versions").get_json()["versions"]
    assert [v["version"] for v in versions] == [1, 2, 3]


def test_batch_validation_via_api(client, network):
    ds = client.post(f"/api/networks/{network['id']}/datasets",
                     json={"name": "d"}).get_json()
    r = client.post(f"/api/datasets/{ds['id']}/batches", json={"batch": {
        "temperature": -5,
        "initial_concentrations": {"A": 1.0},
        "samples": [{"time": 1.0, "concentrations": {"A": 0.8}}],
    }})
    assert r.status_code == 400
    assert any(d["field"] == "batches[0].temperature" for d in r.get_json()["details"])


def test_simulate_endpoint(client, network):
    r = client.post(f"/api/networks/{network['id']}/simulate", json={
        "parameters": [{"A": 0.2, "Ea": 0.0}, {"A": 0.1, "Ea": 0.0}],
        "temperature": 300.0,
        "initial_concentrations": {"A": 1.0},
        "times": [0.0, 1.0, 6.93, 20.0],
    })
    assert r.status_code == 200
    body = r.get_json()
    C = sum(np.array(body["concentrations"][s]) for s in ("A", "B", "C"))
    assert np.max(np.abs(C - 1.0)) < 1e-8
    assert body["steps"] > 0
    assert "estimated_error" in body


def test_full_calibration_flow_and_persistence(client, network):
    ds = client.post(f"/api/networks/{network['id']}/datasets",
                     json={"name": "d"}).get_json()
    true = [{"A": 1e5, "Ea": 3e4}, {"A": 2e4, "Ea": 2e4}]
    for b in _make_batches(network["definition"], true,
                           [290.0, 300.0, 310.0, 320.0],
                           [0.5, 1.0, 2.0, 4.0, 7.0, 12.0]):
        client.post(f"/api/datasets/{ds['id']}/batches", json={"batch": b})
    latest = client.get(f"/api/datasets/{ds['id']}/versions").get_json()["versions"][-1]

    r = client.post(f"/api/dataset-versions/{latest['id']}/calibrate", json={})
    assert r.status_code == 201
    body = r.get_json()
    result = body["result"]
    assert result["termination"] == "converged"
    for p, t in zip(result["parameters"], true):
        assert abs(p["A"] / t["A"] - 1) < 1e-4
        assert abs(p["Ea"] / t["Ea"] - 1) < 1e-4

    # result persisted and retrievable
    got = client.get(f"/api/calibrations/{body['calibration_id']}").get_json()
    assert got["version_id"] == latest["id"]
    assert got["result"]["T_ref"] == result["T_ref"]


def test_calibration_bound_to_version(client, network):
    ds = client.post(f"/api/networks/{network['id']}/datasets",
                     json={"name": "d"}).get_json()
    true = [{"A": 1e5, "Ea": 3e4}, {"A": 2e4, "Ea": 2e4}]
    b1 = _make_batches(network["definition"], true, [300.0],
                       [0.5, 1.0, 2.0])[0]
    v1 = client.post(f"/api/datasets/{ds['id']}/batches",
                     json={"batch": b1}).get_json()
    # add another batch -> new version
    b2 = _make_batches(network["definition"], true, [320.0],
                       [0.5, 1.0, 2.0])[0]
    v2 = client.post(f"/api/datasets/{ds['id']}/batches",
                     json={"batch": b2}).get_json()

    c1 = client.post(f"/api/dataset-versions/{v1['id']}/calibrate", json={}).get_json()
    # v2 calibration should warm-start from v1's calibration automatically
    c2 = client.post(f"/api/dataset-versions/{v2['id']}/calibrate", json={}).get_json()
    assert c2["parent_calibration_id"] == c1["calibration_id"]
    assert c1["version_id"] == v1["id"]
    assert c2["version_id"] == v2["id"]
    # v1 (single-T) says Ea non-identifiable; v2 (two-T) identifies it
    assert c1["result"]["non_identifiable"]
    assert all(p["identifiable"]["Ea"] for p in c2["result"]["parameters"])


def test_same_version_calibrations_consistent(client, network):
    ds = client.post(f"/api/networks/{network['id']}/datasets",
                     json={"name": "d"}).get_json()
    true = [{"A": 1e5, "Ea": 3e4}, {"A": 2e4, "Ea": 2e4}]
    for b in _make_batches(network["definition"], true,
                           [290.0, 300.0, 310.0], [0.5, 1.0, 2.0, 4.0]):
        client.post(f"/api/datasets/{ds['id']}/batches", json={"batch": b})
    v = client.get(f"/api/datasets/{ds['id']}/versions").get_json()["versions"][-1]
    r1 = client.post(f"/api/dataset-versions/{v['id']}/calibrate", json={}).get_json()
    r2 = client.post(f"/api/dataset-versions/{v['id']}/calibrate", json={}).get_json()
    for p1, p2 in zip(r1["result"]["parameters"], r2["result"]["parameters"]):
        assert abs(p1["A"] / p2["A"] - 1) < 1e-6
        assert abs(p1["Ea"] / p2["Ea"] - 1) < 1e-6


def test_restart_persistence(app, client, network):
    """A freshly created app against the same database sees prior data."""
    ds = client.post(f"/api/networks/{network['id']}/datasets",
                     json={"name": "d"}).get_json()
    true = [{"A": 1e5, "Ea": 3e4}, {"A": 2e4, "Ea": 2e4}]
    b = _make_batches(network["definition"], true, [290.0, 310.0],
                      [0.5, 1.0, 2.0, 4.0])
    for rec in b:
        client.post(f"/api/datasets/{ds['id']}/batches", json={"batch": rec})
    versions = client.get(f"/api/datasets/{ds['id']}/versions").get_json()["versions"]
    v = versions[-1]
    cal = client.post(f"/api/dataset-versions/{v['id']}/calibrate",
                      json={"cold_start": True}).get_json()

    # "restart": brand new app object, new connection pool, same database
    from app import create_app
    from tests.conftest import TestConfig
    app2 = create_app(TestConfig)
    client2 = app2.test_client()
    got_net = client2.get(f"/api/networks/{network['id']}")
    assert got_net.status_code == 200
    got_v = client2.get(f"/api/dataset-versions/{v['id']}").get_json()
    assert got_v["batch_count"] == 2
    got_cal = client2.get(f"/api/calibrations/{cal['calibration_id']}").get_json()
    assert got_cal["version_id"] == v["id"]
    for p, t in zip(got_cal["result"]["parameters"], true):
        assert abs(p["A"] / t["A"] - 1) < 1e-4


def test_cold_vs_warm_start_api_consistency(client, network):
    ds = client.post(f"/api/networks/{network['id']}/datasets",
                     json={"name": "d"}).get_json()
    true = [{"A": 1e5, "Ea": 3e4}, {"A": 2e4, "Ea": 2e4}]
    for rec in _make_batches(network["definition"], true,
                             [290.0, 300.0, 310.0, 320.0],
                             [0.5, 1.0, 2.0, 4.0, 7.0]):
        client.post(f"/api/datasets/{ds['id']}/batches", json={"batch": rec})
    v = client.get(f"/api/datasets/{ds['id']}/versions").get_json()["versions"][-1]
    cold = client.post(f"/api/dataset-versions/{v['id']}/calibrate",
                       json={"cold_start": True}).get_json()
    warm = client.post(f"/api/dataset-versions/{v['id']}/calibrate",
                       json={}).get_json()
    for p1, p2 in zip(cold["result"]["parameters"], warm["result"]["parameters"]):
        assert abs(p1["A"] / p2["A"] - 1) < 1e-6
        assert abs(p1["Ea"] / p2["Ea"] - 1) < 1e-6


def test_simulate_validation_fields(client, network):
    base = {
        "parameters": [{"A": 0.2, "Ea": 0.0}, {"A": 0.1, "Ea": 0.0}],
        "temperature": 300.0,
        "initial_concentrations": {"A": 1.0},
        "times": [0.0, 1.0],
    }

    def post(body):
        return client.post(f"/api/networks/{network['id']}/simulate", json=body)

    bad = json.loads(json.dumps(base)); bad["initial_concentrations"] = {"A": -1.0}
    assert any(d["field"] == "initial_concentrations.A" for d in post(bad).get_json()["details"])

    bad = json.loads(json.dumps(base)); bad["initial_concentrations"] = {"A": float("nan")}
    r = post(bad)
    assert r.status_code == 400

    bad = json.loads(json.dumps(base)); bad["temperature"] = 250.0
    # 250 K is fine; zero is not
    bad["temperature"] = 0.0
    assert any(d["field"] == "temperature" for d in post(bad).get_json()["details"])

    bad = json.loads(json.dumps(base)); bad["times"] = [1.0, 0.5]
    assert any("times[1]" in d["field"] for d in post(bad).get_json()["details"])

    bad = json.loads(json.dumps(base)); bad["parameters"][0]["A"] = -3.0
    assert any(d["field"].startswith("parameters[0].A") for d in post(bad).get_json()["details"])


def test_simulate_nonfinite_reports_field(client, network):
    # second-order with an astronomically large pre-factor -> overflow
    net_def = {
        "species": ["A", "B"],
        "reactions": [{"name": "r", "stoichiometry": {"A": -2, "B": 2},
                       "orders": {"A": 2}}],
    }
    net = client.post("/api/networks", json={"name": "blowup", "definition": net_def}).get_json()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = client.post(f"/api/networks/{net['id']}/simulate", json={
            "parameters": [{"A": 1e300, "Ea": 0.0}],
            "temperature": 300.0,
            "initial_concentrations": {"A": 1.0},
            "times": [1.0],
        })
    assert r.status_code == 400
    assert r.get_json()["error"] == "validation_error"


def test_batch_sample_undefined_species(client, network):
    ds = client.post(f"/api/networks/{network['id']}/datasets",
                     json={"name": "d"}).get_json()
    r = client.post(f"/api/datasets/{ds['id']}/batches", json={"batch": {
        "temperature": 300.0,
        "initial_concentrations": {"A": 1.0},
        "samples": [{"time": 1.0, "concentrations": {"Z": 0.5}}],
    }})
    assert r.status_code == 400
    assert any("concentrations.Z" in d["field"] for d in r.get_json()["details"])


def test_missing_entities_404(client):
    bad_id = "not-a-uuid"
    fake = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/networks/{bad_id}").status_code == 404
    assert client.get(f"/api/networks/{fake}").status_code == 404
    assert client.get(f"/api/datasets/{fake}").status_code == 404
    assert client.get(f"/api/dataset-versions/{fake}").status_code == 404
    assert client.get(f"/api/calibrations/{fake}").status_code == 404
