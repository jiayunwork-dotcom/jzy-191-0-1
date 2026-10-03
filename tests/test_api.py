import numpy as np

from app.kinetics import Network
from tests.helpers import (ABC_NETWORK, arrhenius_channels, make_batch,
                           make_dataset, post_network)


def _synthetic_batches():
    net = Network(ABC_NETWORK["components"], ABC_NETWORK["reactions"])
    t_ref = 320.0
    ln_a, ea = arrhenius_channels([0.2, 0.1], [40000.0, 50000.0], t_ref)
    batches = [
        make_batch(net, T, ln_a, ea,
                   times=[2.0, 5.0, 8.0, 12.0, 20.0, 30.0])
        for T in (300.0, 310.0, 320.0, 330.0, 340.0)
    ]
    return batches


def test_end_to_end_versioning_and_calibration(client):
    nid = post_network(client)
    batches = _synthetic_batches()

    # Version 1 with the first three batches.
    v1, ids = make_dataset(client, nid, batches[:3], name="campaign")
    assert v1["version_no"] == 1
    rv = client.post(f"/api/dataset-versions/{v1['id']}/calibrations", json={})
    assert rv.status_code == 201, rv.get_json()
    cal1 = rv.get_json()
    assert cal1["dataset_version_id"] == v1["id"]
    assert cal1["result"]["converged"] is True
    assert cal1["result"]["objective"]["sse"] < 1e-10
    for j, ch in enumerate(cal1["result"]["channels"]):
        assert abs(ch["ea"] - [40000.0, 50000.0][j]) / [40000., 50000.][j] < 1e-4

    # Version 2 adds two batches; v1 and its calibration must remain.
    rv = client.post(f"/api/datasets/{v1['dataset_id']}/versions",
                     json={"batch_ids": ids + ["__nonexistent__"]})
    assert rv.status_code == 404
    rv = client.post(f"/api/datasets/{v1['dataset_id']}/versions",
                     json={"batch_ids": []})
    # empty list is invalid
    assert rv.status_code == 400
    # add two new batches
    extra_ids = []
    for b in batches[3:]:
        r = client.post(f"/api/networks/{nid}/batches", json=b)
        extra_ids.append(r.get_json()["id"])
    rv = client.post(f"/api/datasets/{v1['dataset_id']}/versions",
                     json={"batch_ids": ids + extra_ids})
    assert rv.status_code == 201
    v2 = rv.get_json()
    assert v2["version_no"] == 2
    assert len(v2["batch_ids"]) == 5

    rv = client.post(f"/api/dataset-versions/{v2['id']}/calibrations",
                     json={})
    assert rv.status_code == 201
    cal2 = rv.get_json()
    assert cal2["result"]["objective"]["sse"] < 1e-10

    # Old version and old result survive.
    rv = client.get(f"/api/dataset-versions/{v1['id']}")
    assert rv.status_code == 200
    assert len(rv.get_json()["batch_ids"]) == 3
    rv = client.get(f"/api/calibrations/{cal1['id']}")
    assert rv.status_code == 200
    assert rv.get_json()["result"]["objective"]["sse"] == \
        cal1["result"]["objective"]["sse"]

    # Both calibrations are listed under their own versions.
    l1 = client.get(
        f"/api/dataset-versions/{v1['id']}/calibrations").get_json()
    l2 = client.get(
        f"/api/dataset-versions/{v2['id']}/calibrations").get_json()
    assert {c["id"] for c in l1} == {cal1["id"]}
    assert {c["id"] for c in l2} == {cal2["id"]}


def test_warm_start_uses_previous_version_result(client):
    nid = post_network(client)
    batches = _synthetic_batches()
    v1, ids = make_dataset(client, nid, batches[:3])
    cal1 = client.post(
        f"/api/dataset-versions/{v1['id']}/calibrations",
        json={}).get_json()

    extra = []
    for b in batches[3:]:
        extra.append(client.post(
            f"/api/networks/{nid}/batches", json=b).get_json()["id"])
    v2 = client.post(f"/api/datasets/{v1['dataset_id']}/versions",
                     json={"batch_ids": ids + extra}).get_json()
    # Warm and cold starts must agree to 1e-6.
    warm = client.post(
        f"/api/dataset-versions/{v2['id']}/calibrations",
        json={"use_previous_result": True}).get_json()
    cold = client.post(
        f"/api/dataset-versions/{v2['id']}/calibrations",
        json={"use_previous_result": False}).get_json()
    assert warm["started_from"] == "previous_version"
    assert cold["started_from"] == "grid"
    pw = np.array(warm["result"]["internal_parameters"]["q"])
    pc = np.array(cold["result"]["internal_parameters"]["q"])
    tw = np.array(warm["result"]["internal_parameters"]["theta"])
    tc = np.array(cold["result"]["internal_parameters"]["theta"])
    assert np.max(np.abs(pw - pc) / np.abs(pc)) < 1e-6
    assert np.max(np.abs(tw - tc) / np.abs(tc)) < 1e-6


def test_simulation_endpoint(client):
    nid = post_network(client)
    rv = client.post(f"/api/networks/{nid}/simulate", json={
        "temperature": 300.0,
        "initial_concentrations": {"A": 1.0},
        "t_end": 25.0,
        "sample_times": [6.93, 10.0],
        "parameters": {"per_channel": [{"a": 0.2}, {"a": 0.1}]},
    })
    assert rv.status_code == 200, rv.get_json()
    body = rv.get_json()
    assert body["times"] == [0.0, 6.93, 10.0, 25.0]
    assert abs(body["concentrations"][1]["B"] - 0.5) < 1e-3
    assert body["integration"]["n_steps"] > 0
    total = sum(body["concentrations"][1].values())
    assert abs(total - 1.0) < 1e-8


def test_single_temperature_api_reports_unidentifiable(client):
    nid = post_network(client)
    net = Network(ABC_NETWORK["components"], ABC_NETWORK["reactions"])
    ln_a, ea = arrhenius_channels([0.2, 0.1], [0.0, 0.0], 320.0)
    batches = [make_batch(net, 320.0, ln_a, ea,
                          times=[2.0, 5.0, 8.0, 12.0, 25.0])]
    v, _ = make_dataset(client, nid, batches)
    rv = client.post(f"/api/dataset-versions/{v['id']}/calibrations",
                     json={})
    body = rv.get_json()
    assert body["result"]["all_identifiable"] is False
    for ch in body["result"]["channels"]:
        assert ch["ea"] is None and ch["a"] is None
        assert ch["unidentifiable_reason"]


def test_dataset_rejects_batch_from_other_network(client):
    nid1 = post_network(client)
    nid2 = post_network(client)
    net = Network(ABC_NETWORK["components"], ABC_NETWORK["reactions"])
    ln_a, ea = arrhenius_channels([0.2, 0.1], [0.0, 0.0], 320.0)
    b = make_batch(net, 320.0, ln_a, ea, times=[2.0, 5.0])
    bid2 = client.post(f"/api/networks/{nid2}/batches", json=b).get_json()["id"]
    rv = client.post(f"/api/networks/{nid1}/datasets",
                     json={"name": "x", "batch_ids": [bid2]})
    assert rv.status_code == 400
