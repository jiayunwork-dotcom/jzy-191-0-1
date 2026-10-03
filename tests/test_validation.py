from tests.helpers import post_network


def _fields(errors):
    return {d["field"] for d in errors}


def test_negative_concentration_is_rejected(client):
    nid = post_network(client)
    batch = {
        "temperature": 320.0,
        "initial_concentrations": {"A": -0.1},
        "samples": [{"time": 1.0, "observations": {"A": 0.5}}],
    }
    rv = client.post(f"/api/networks/{nid}/batches", json=batch)
    assert rv.status_code == 400
    fields = _fields(rv.get_json()["details"])
    assert "initial_concentrations.A" in fields


def test_nonfinite_concentration_is_rejected(client):
    nid = post_network(client)
    batch = {
        "temperature": 320.0,
        "initial_concentrations": {"A": 1.0},
        "samples": [{"time": 1.0, "observations": {"B": float("nan")}}],
    }
    rv = client.post(f"/api/networks/{nid}/batches", json=batch)
    assert rv.status_code == 400
    assert any("samples[0].observations.B" in f for f in
               _fields(rv.get_json()["details"]))


def test_non_positive_temperature_is_rejected(client):
    nid = post_network(client)
    batch = {
        "temperature": 0.0,
        "initial_concentrations": {"A": 1.0},
        "samples": [{"time": 1.0, "observations": {"A": 0.5}}],
    }
    rv = client.post(f"/api/networks/{nid}/batches", json=batch)
    assert rv.status_code == 400
    assert "temperature" in _fields(rv.get_json()["details"])

    batch["temperature"] = -5
    rv = client.post(f"/api/networks/{nid}/batches", json=batch)
    assert rv.status_code == 400


def test_non_increasing_sample_times_are_rejected(client):
    nid = post_network(client)
    batch = {
        "temperature": 320.0,
        "initial_concentrations": {"A": 1.0},
        "samples": [
            {"time": 2.0, "observations": {"A": 0.5}},
            {"time": 2.0, "observations": {"A": 0.4}},
            {"time": 1.5, "observations": {"A": 0.3}},
        ],
    }
    rv = client.post(f"/api/networks/{nid}/batches", json=batch)
    assert rv.status_code == 400
    fields = _fields(rv.get_json()["details"])
    assert "samples[1].time" in fields
    assert "samples[2].time" in fields


def test_reaction_undefined_component_is_rejected(client):
    bad = {
        "components": ["A", "B"],
        "reactions": [
            {"id": "r1", "stoichiometry": {"A": -1, "Z": 1},
             "orders": {"A": 1}},
        ],
    }
    rv = client.post("/api/networks", json=bad)
    assert rv.status_code == 400
    fields = _fields(rv.get_json()["details"])
    assert "reactions[0].stoichiometry.Z" in fields


def test_orders_undefined_component_is_rejected(client):
    bad = {
        "components": ["A", "B"],
        "reactions": [
            {"id": "r1", "stoichiometry": {"A": -1, "B": 1},
             "orders": {"Q": 1}},
        ],
    }
    rv = client.post("/api/networks", json=bad)
    assert rv.status_code == 400
    fields = _fields(rv.get_json()["details"])
    assert "reactions[0].orders.Q" in fields


def test_duplicate_components_and_reaction_ids(client):
    bad = {
        "components": ["A", "A", "B"],
        "reactions": [
            {"id": "x", "stoichiometry": {"A": -1, "B": 1}, "orders": {}},
            {"id": "x", "stoichiometry": {"B": -1}, "orders": {}},
        ],
    }
    rv = client.post("/api/networks", json=bad)
    assert rv.status_code == 400
    joined = " ".join(d["field"] for d in rv.get_json()["details"])
    assert "components" in joined and "reactions" in joined


def test_simulation_validation_errors(client):
    nid = post_network(client)
    rv = client.post(f"/api/networks/{nid}/simulate", json={
        "temperature": -1,
        "initial_concentrations": {"A": 1.0},
        "t_end": 10.0,
        "parameters": {"per_channel": [
            {"a": 0.2}, {"a": 0.1}]},
    })
    assert rv.status_code == 400
    assert "temperature" in _fields(rv.get_json()["details"])

    rv = client.post(f"/api/networks/{nid}/simulate", json={
        "temperature": 300,
        "initial_concentrations": {"A": 1.0},
        "t_end": 10.0,
        "sample_times": [5.0, 3.0],
        "parameters": {"per_channel": [
            {"a": 0.2}, {"a": 0.1}]},
    })
    assert rv.status_code == 400


def test_not_found_routes(client):
    assert client.get("/api/networks/nope").status_code == 404
    assert client.get("/api/batches/nope").status_code == 404
    assert client.get("/api/datasets/nope").status_code == 404
    assert client.get("/api/dataset-versions/nope").status_code == 404
    assert client.get("/api/calibrations/nope").status_code == 404


def test_simulation_accepts_ln_a_and_rejects_non_positive_a(client):
    import math
    nid = post_network(client)
    # ln_a form: ln(0.2), ln(0.1) with Ea = 0 gives the same rates.
    rv = client.post(f"/api/networks/{nid}/simulate", json={
        "temperature": 300,
        "initial_concentrations": {"A": 1.0},
        "t_end": 20.0,
        "sample_times": [6.93],
        "parameters": {"per_channel": [
            {"ln_a": math.log(0.2), "ea": 0.0},
            {"ln_a": math.log(0.1), "ea": 0.0}]},
    })
    assert rv.status_code == 200, rv.get_json()
    assert abs(rv.get_json()["concentrations"][1]["B"] - 0.5) < 1e-3

    rv = client.post(f"/api/networks/{nid}/simulate", json={
        "temperature": 300,
        "initial_concentrations": {"A": 1.0},
        "t_end": 20.0,
        "parameters": {"per_channel": [{"a": -1.0}, {"a": 0.1}]},
    })
    assert rv.status_code == 400
    assert "parameters.per_channel[0].a" in _fields(rv.get_json()["details"])

    # wrong number of channels
    rv = client.post(f"/api/networks/{nid}/simulate", json={
        "temperature": 300,
        "initial_concentrations": {"A": 1.0},
        "t_end": 20.0,
        "parameters": {"per_channel": [{"a": 0.2}]},
    })
    assert rv.status_code == 400
    assert "parameters.per_channel" in _fields(rv.get_json()["details"])
