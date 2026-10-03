"""Flask application factory and HTTP routes."""

from __future__ import annotations

import math
import os

import numpy as np
from flask import Flask, jsonify, request

from .calibration import calibrate
from .integrator import IntegrationError
from .kinetics import Network
from .repository import MemoryRepository, PostgresRepository
from .simulator import simulate
from .validation import (ValidationError, validate_batch, validate_network,
                         validate_simulation_request)


def create_app(repository=None):
    app = Flask(__name__)
    app.url_map.strict_slashes = False
    if repository is not None:
        app.repo = repository
    elif os.environ.get("DATABASE_URL"):
        app.repo = PostgresRepository(os.environ["DATABASE_URL"])
    else:
        app.repo = MemoryRepository()

    # ------------------------------------------------------------------ utils
    def _load_network(network_id):
        row = app.repo.get_network(network_id)
        if row is None:
            return None, (jsonify({"error": "network not found",
                                   "network_id": network_id}), 404)
        return Network(row["spec"]["components"], row["spec"]["reactions"]), None

    @app.errorhandler(ValidationError)
    def _on_validation_error(exc):
        return jsonify({"error": "validation_failed", "details": exc.errors}), 400

    @app.errorhandler(KeyError)
    def _on_key_error(exc):
        return jsonify({"error": "not_found", "message": str(exc.args[0])}), 404

    @app.errorhandler(ValueError)
    def _on_value_error(exc):
        return jsonify({"error": "invalid_request", "message": str(exc)}), 400

    @app.errorhandler(IntegrationError)
    def _on_integration_error(exc):
        return jsonify({"error": "integration_failed", "message": str(exc)}), 422

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    # --------------------------------------------------------------- networks
    @app.post("/api/networks")
    def create_network():
        body = request.get_json(silent=True)
        spec = validate_network(body or {})
        row = app.repo.create_network(spec)
        return jsonify(row), 201

    @app.get("/api/networks/<network_id>")
    def get_network(network_id):
        row = app.repo.get_network(network_id)
        if row is None:
            return jsonify({"error": "network not found"}), 404
        return jsonify(row)

    # ---------------------------------------------------------------- batches
    @app.post("/api/networks/<network_id>/batches")
    def create_batch(network_id):
        net, err = _load_network(network_id)
        if err:
            return err
        body = request.get_json(silent=True) or {}
        validate_batch(body, net.components)
        row = app.repo.create_batch(network_id, body)
        return jsonify(row), 201

    @app.get("/api/batches/<batch_id>")
    def get_batch(batch_id):
        row = app.repo.get_batch(batch_id)
        if row is None:
            return jsonify({"error": "batch not found"}), 404
        return jsonify(row)

    @app.get("/api/networks/<network_id>/batches")
    def list_batches(network_id):
        if app.repo.get_network(network_id) is None:
            return jsonify({"error": "network not found"}), 404
        return jsonify(app.repo.list_batches(network_id))

    # --------------------------------------------------------------- datasets
    @app.post("/api/networks/<network_id>/datasets")
    def create_dataset(network_id):
        if app.repo.get_network(network_id) is None:
            return jsonify({"error": "network not found"}), 404
        body = request.get_json(silent=True) or {}
        name = body.get("name")
        batch_ids = body.get("batch_ids")
        if not isinstance(name, str) or not name.strip():
            raise ValidationError([{"field": "name",
                                    "message": "must be a non-empty string"}])
        if not isinstance(batch_ids, list) or not batch_ids:
            raise ValidationError([{"field": "batch_ids",
                                    "message": "must be a non-empty list"}])
        row = app.repo.create_dataset(network_id, name, batch_ids)
        return jsonify(row), 201

    @app.post("/api/datasets/<dataset_id>/versions")
    def add_version(dataset_id):
        body = request.get_json(silent=True) or {}
        batch_ids = body.get("batch_ids")
        if not isinstance(batch_ids, list) or not batch_ids:
            raise ValidationError([{"field": "batch_ids",
                                    "message": "must be a non-empty list"}])
        row = app.repo.add_version(dataset_id, batch_ids)
        if row is None:
            return jsonify({"error": "dataset not found"}), 404
        return jsonify(row), 201

    @app.get("/api/datasets/<dataset_id>")
    def get_dataset(dataset_id):
        row = app.repo.get_dataset(dataset_id)
        if row is None:
            return jsonify({"error": "dataset not found"}), 404
        return jsonify(row)

    @app.get("/api/datasets/<dataset_id>/versions")
    def list_versions(dataset_id):
        rows = app.repo.list_versions(dataset_id)
        if rows is None:
            return jsonify({"error": "dataset not found"}), 404
        return jsonify(rows)

    @app.get("/api/dataset-versions/<version_id>")
    def get_version(version_id):
        row = app.repo.get_version(version_id)
        if row is None:
            return jsonify({"error": "dataset version not found"}), 404
        batches = app.repo.batches_for_version(version_id)
        return jsonify({**row, "batches": batches})

    # -------------------------------------------------------------- simulate
    @app.post("/api/networks/<network_id>/simulate")
    def run_simulation(network_id):
        net, err = _load_network(network_id)
        if err:
            return err
        body = request.get_json(silent=True) or {}
        parsed = validate_simulation_request(body, net.components,
                                             net.n_channels)
        rtol = body.get("rtol", None)
        atol = body.get("atol", None)
        kw = {}
        for name, val in (("rtol", rtol), ("atol", atol)):
            if val is not None:
                if not isinstance(val, (int, float)) \
                        or isinstance(val, bool) or not math.isfinite(val) \
                        or not 0.0 < val < 1.0:
                    raise ValidationError([{
                        "field": name,
                        "message": "must be a finite number in (0, 1)"}])
                kw[name] = float(val)
        res = simulate(net, parsed["temperature"],
                       parsed["initial_concentrations"], parsed["t_end"],
                       np.asarray(parsed["ln_a"]), np.asarray(parsed["ea"]),
                       sample_times=parsed["sample_times"], **kw)
        values = res["values"]
        return jsonify({
            "times": res["times"],
            "concentrations": [
                {comp: float(values[i, j])
                 for j, comp in enumerate(net.components)}
                for i in range(values.shape[0])
            ],
            "integration": {
                "n_steps": res["n_steps"],
                "n_rejected": res["n_rejected"],
                "max_scaled_local_error": res["max_scaled_local_error"],
            },
        })

    # ------------------------------------------------------------ calibrate
    @app.post("/api/dataset-versions/<version_id>/calibrations")
    def run_calibration(version_id):
        version = app.repo.get_version(version_id)
        if version is None:
            return jsonify({"error": "dataset version not found"}), 404
        dataset = app.repo.get_dataset(version["dataset_id"])
        net_row = app.repo.get_network(dataset["network_id"])
        net = Network(net_row["spec"]["components"],
                      net_row["spec"]["reactions"])
        batches = app.repo.batches_for_version(version_id)

        body = request.get_json(silent=True) or {}
        initial = None
        use_previous = bool(body.get("use_previous_result", True))
        if isinstance(body.get("initial_parameters"), dict):
            ip = body["initial_parameters"]
            if not (isinstance(ip.get("q"), list)
                    and isinstance(ip.get("theta"), list)
                    and len(ip["q"]) == 2 * net.n_channels
                    and len(ip["theta"]) == net.n_channels):
                raise ValidationError([{
                    "field": "initial_parameters",
                    "message": f"must contain numeric 'q' list of length "
                               f"{2 * net.n_channels} and 'theta' list of "
                               f"length {net.n_channels}"}])
            try:
                initial = {
                    "q": [float(x) for x in ip["q"]],
                    "theta": [float(x) for x in ip["theta"]],
                }
                if not all(math.isfinite(v) for v in
                           initial["q"] + initial["theta"]):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValidationError([{
                    "field": "initial_parameters",
                    "message": "all entries must be finite numbers"}])
        elif use_previous:
            prev = app.repo.last_calibration_for_dataset(dataset["id"])
            if prev is not None:
                initial = prev["result"].get("internal_parameters")

        max_iters = body.get("max_iterations", 200)
        if not isinstance(max_iters, int) or isinstance(max_iters, bool) \
                or not 1 <= max_iters <= 200:
            raise ValidationError([{
                "field": "max_iterations",
                "message": "must be an integer between 1 and 200"}])

        result = calibrate(net, batches, initial=initial,
                           max_iters=max_iters)
        saved = app.repo.save_calibration(
            dataset["network_id"], dataset["id"], version_id, result,
            started_from=("previous_version" if initial is not None
                          and not isinstance(body.get("initial_parameters"),
                                             dict)
                          else ("explicit"
                                if body.get("initial_parameters") else "grid")))
        return jsonify(saved), 201

    @app.get("/api/calibrations/<calibration_id>")
    def get_calibration(calibration_id):
        row = app.repo.get_calibration(calibration_id)
        if row is None:
            return jsonify({"error": "calibration not found"}), 404
        return jsonify(row)

    @app.get("/api/dataset-versions/<version_id>/calibrations")
    def list_version_calibrations(version_id):
        if app.repo.get_version(version_id) is None:
            return jsonify({"error": "dataset version not found"}), 404
        return jsonify(app.repo.list_calibrations(version_id=version_id))

    return app


# WSGI entry point (gunicorn uses this).
app = create_app()
