"""PostgreSQL-backed repository tests, including restart persistence.

Skipped unless TEST_POSTGRES_DSN points at a reachable PostgreSQL 16
instance (the compose stack provides one automatically).
"""

import numpy as np

from app.app import create_app
from app.kinetics import Network
from tests.helpers import ABC_NETWORK, arrhenius_channels, make_batch


def _make_campaign():
    net = Network(ABC_NETWORK["components"], ABC_NETWORK["reactions"])
    ln_a, ea = arrhenius_channels([0.2, 0.1], [40000.0, 50000.0], 320.0)
    batches = [
        make_batch(net, T, ln_a, ea, times=[2.0, 5.0, 8.0, 12.0, 25.0])
        for T in (300.0, 320.0, 340.0)
    ]
    return net, batches


def test_repository_roundtrip(any_repo):
    repo = any_repo
    spec = {"components": ABC_NETWORK["components"],
            "reactions": ABC_NETWORK["reactions"]}
    n = repo.create_network(spec)
    assert repo.get_network(n["id"])["spec"]["components"] == \
        ABC_NETWORK["components"]

    net, batches = _make_campaign()
    ids = [repo.create_batch(n["id"], b)["id"] for b in batches]
    assert repo.get_batch(ids[0])["temperature"] == 300.0

    v1 = repo.create_dataset(n["id"], "campaign", ids[:2])
    assert v1["version_no"] == 1
    v2 = repo.add_version(v1["dataset_id"], ids)
    assert v2["version_no"] == 2 and len(v2["batch_ids"]) == 3
    stored = repo.batches_for_version(v2["id"])
    assert [b["id"] for b in stored] == ids

    result = {"channels": [{"a": 1.0}], "converged": True}
    c = repo.save_calibration(n["id"], v1["dataset_id"], v2["id"], result)
    assert repo.get_calibration(c["id"])["result"]["converged"] is True
    assert repo.last_calibration_for_dataset(v1["dataset_id"])["id"] == c["id"]


def test_versions_and_results_survive_restart(pg_repo):
    repo = pg_repo
    net, batches = _make_campaign()
    n = repo.create_network(
        {"components": ABC_NETWORK["components"],
         "reactions": ABC_NETWORK["reactions"]})
    ids = [repo.create_batch(n["id"], b)["id"] for b in batches]
    v1 = repo.create_dataset(n["id"], "persist", ids)

    app = create_app(repository=repo)
    client = app.test_client()
    cal = client.post(
        f"/api/dataset-versions/{v1['id']}/calibrations", json={}).get_json()
    assert cal["result"]["converged"] is True

    # Simulate a full service restart: throw away the app and repository
    # objects and reconnect to the same database from scratch.
    del app, client, repo

    from app.repository import PostgresRepository
    import os
    fresh = PostgresRepository(os.environ["TEST_POSTGRES_DSN"])
    try:
        ds = fresh.get_dataset(v1["dataset_id"])
        assert ds is not None and ds["latest_version_no"] == 1
        v = fresh.get_version(v1["id"])
        assert v["batch_ids"] == ids
        got = fresh.get_calibration(cal["id"])
        assert got is not None
        assert got["result"]["objective"]["sse"] == \
            cal["result"]["objective"]["sse"]
        assert got["result"]["channels"][0]["ea"] is not None
    finally:
        fresh.close()


def test_repeatable_results_across_repository_backends(pg_repo):
    """The same dataset calibrated through an in-memory and a Postgres-backed
    service gives identical parameters."""
    from app.repository import MemoryRepository

    net, batches = _make_campaign()

    mem = MemoryRepository()
    nid = mem.create_network({"components": ABC_NETWORK["components"],
                              "reactions": ABC_NETWORK["reactions"]})["id"]
    bids = [mem.create_batch(nid, b)["id"] for b in batches]
    vm = mem.create_dataset(nid, "m", bids)
    cm = mem.save_calibration  # noqa: F841 (unused; service path used below)
    appm = create_app(repository=mem)
    rm = appm.test_client().post(
        f"/api/dataset-versions/{vm['id']}/calibrations", json={}).get_json()

    repo = pg_repo
    nid2 = repo.create_network({"components": ABC_NETWORK["components"],
                                "reactions": ABC_NETWORK["reactions"]})["id"]
    bids2 = [repo.create_batch(nid2, b)["id"] for b in batches]
    vp = repo.create_dataset(nid2, "p", bids2)
    appp = create_app(repository=repo)
    rp = appp.test_client().post(
        f"/api/dataset-versions/{vp['id']}/calibrations", json={}).get_json()

    qm = np.array(rm["result"]["internal_parameters"]["q"])
    qp = np.array(rp["result"]["internal_parameters"]["q"])
    tm = np.array(rm["result"]["internal_parameters"]["theta"])
    tp = np.array(rp["result"]["internal_parameters"]["theta"])
    assert np.max(np.abs(qm - qp) / np.abs(qm)) < 1e-10
    assert np.max(np.abs(tm - tp) / np.abs(tm)) < 1e-10
