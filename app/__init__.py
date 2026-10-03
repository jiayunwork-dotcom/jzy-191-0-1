"""Flask application: reaction networks, versioned datasets, calibration."""

from __future__ import annotations

import uuid

from flask import Flask, jsonify, request

from . import db
from .calibration import (beta_from_previous, calibrate, prepare_dataset)
from .chemistry import ValidationError, compile_network
from .config import Config
from .simulation import simulate


def create_app(config: type | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config or Config)
    db.init_app(app)

    def bad_request(err: ValidationError):
        return jsonify({"error": "validation_error", "details": err.to_dict()}), 400

    def is_uuid(value: str) -> bool:
        try:
            uuid.UUID(str(value))
            return True
        except (ValueError, AttributeError):
            return False

    # ------------------------------------------------------------------
    @app.get("/health")
    def health():
        conn = db.get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return jsonify({"status": "ok"})

    # ------------------------- networks --------------------------------
    @app.post("/api/networks")
    def post_network():
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "invalid_json"}), 400
        try:
            compile_network(body.get("definition", body))
        except ValidationError as exc:
            return bad_request(exc)
        definition = body.get("definition", body)
        name = body.get("name") if "definition" in body else None
        if not name:
            name = "network"
        with db.transaction() as conn:
            row = db.create_network(conn, name, definition)
        return jsonify(_network_json(row)), 201

    @app.get("/api/networks/<network_id>")
    def get_network(network_id):
        if not is_uuid(network_id):
            return jsonify({"error": "not_found"}), 404
        row = db.get_network(db.get_db(), network_id)
        if row is None:
            return jsonify({"error": "not_found"}), 404
        return jsonify(_network_json(row))

    # ------------------------- datasets --------------------------------
    @app.post("/api/networks/<network_id>/datasets")
    def post_dataset(network_id):
        if not is_uuid(network_id):
            return jsonify({"error": "not_found"}), 404
        body = request.get_json(silent=True) or {}
        conn = db.get_db()
        if db.get_network(conn, network_id) is None:
            return jsonify({"error": "not_found", "field": "network_id"}), 404
        name = body.get("name") or "dataset"
        with db.transaction() as c:
            row = db.create_dataset(c, network_id, name)
        return jsonify(_dataset_json(row)), 201

    @app.get("/api/datasets/<dataset_id>")
    def get_dataset(dataset_id):
        if not is_uuid(dataset_id):
            return jsonify({"error": "not_found"}), 404
        row = db.get_dataset(db.get_db(), dataset_id)
        if row is None:
            return jsonify({"error": "not_found"}), 404
        return jsonify(_dataset_json(row))

    @app.get("/api/datasets/<dataset_id>/versions")
    def get_versions(dataset_id):
        if not is_uuid(dataset_id):
            return jsonify({"error": "not_found"}), 404
        conn = db.get_db()
        if db.get_dataset(conn, dataset_id) is None:
            return jsonify({"error": "not_found"}), 404
        versions = [_version_json(v) for v in db.list_versions(conn, dataset_id)]
        return jsonify({"dataset_id": dataset_id, "versions": versions})

    @app.get("/api/dataset-versions/<version_id>")
    def get_version(version_id):
        if not is_uuid(version_id):
            return jsonify({"error": "not_found"}), 404
        conn = db.get_db()
        version = db.get_version(conn, version_id)
        if version is None:
            return jsonify({"error": "not_found"}), 404
        batches = db.list_batches(conn, version_id)
        return jsonify({
            **_version_json(version),
            "batches": [_batch_json(b) for b in batches],
            "batch_count": len(batches),
        })

    @app.post("/api/datasets/<dataset_id>/batches")
    def post_batch(dataset_id):
        if not is_uuid(dataset_id):
            return jsonify({"error": "not_found"}), 404
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "invalid_json"}), 400
        batch = body.get("batch", body)
        conn = db.get_db()
        dataset = db.get_dataset(conn, dataset_id)
        if dataset is None:
            return jsonify({"error": "not_found"}), 404
        net_row = db.get_network(conn, dataset["network_id"])
        try:
            net = compile_network(net_row["definition"])
            prepare_dataset(net, [_batch_storage_to_api(batch)])
        except ValidationError as exc:
            return bad_request(exc)
        note = body.get("change_note") if "batch" in body else None
        with db.transaction() as c:
            version_id, number = db.add_batch(c, dataset_id, _normalise_batch_storage(batch), note)
            version = db.get_version(c, version_id)
        return jsonify(_version_json(version)), 201

    @app.delete("/api/datasets/<dataset_id>/batches/<int:index>")
    def delete_batch(dataset_id, index):
        if not is_uuid(dataset_id):
            return jsonify({"error": "not_found"}), 404
        conn = db.get_db()
        if db.get_dataset(conn, dataset_id) is None:
            return jsonify({"error": "not_found"}), 404
        try:
            with db.transaction() as c:
                version_id, number = db.remove_batch(c, dataset_id, index)
                version = db.get_version(c, version_id)
        except IndexError as exc:
            return jsonify({"error": "out_of_range", "message": str(exc)}), 404
        return jsonify(_version_json(version))

    # ------------------------- simulation ------------------------------
    @app.post("/api/networks/<network_id>/simulate")
    def post_simulate(network_id):
        if not is_uuid(network_id):
            return jsonify({"error": "not_found"}), 404
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "invalid_json"}), 400
        conn = db.get_db()
        row = db.get_network(conn, network_id)
        if row is None:
            return jsonify({"error": "not_found"}), 404
        try:
            net = compile_network(row["definition"])
            result = simulate(
                net, body.get("parameters"), body.get("temperature"),
                body.get("initial_concentrations"), body.get("times"),
            )
        except ValidationError as exc:
            return bad_request(exc)
        result.pop("_result", None)
        return jsonify(result)

    # ------------------------- calibration -----------------------------
    @app.post("/api/dataset-versions/<version_id>/calibrate")
    def post_calibrate(version_id):
        if not is_uuid(version_id):
            return jsonify({"error": "not_found"}), 404
        body = request.get_json(silent=True) or {}
        conn = db.get_db()
        version = db.get_version(conn, version_id)
        if version is None:
            return jsonify({"error": "not_found"}), 404
        dataset = db.get_dataset(conn, version["dataset_id"])
        net_row = db.get_network(conn, dataset["network_id"])
        batches_rows = db.list_batches(conn, version_id)
        if not batches_rows:
            return jsonify({"error": "empty_dataset",
                            "message": "version has no batches; nothing to calibrate"}), 400
        try:
            net = compile_network(net_row["definition"])
            batches = [_batch_storage_to_api(_batch_api_storage(b)) for b in batches_rows]
            prepared = prepare_dataset(net, batches)
        except ValidationError as exc:
            return bad_request(exc)

        initial_beta = None
        parent_calibration_id = None
        cold_start = bool(body.get("cold_start", False))
        prev_cal_id = body.get("previous_calibration_id")
        if prev_cal_id:
            prev = db.get_calibration(conn, prev_cal_id)
            if prev is None:
                return jsonify({"error": "not_found", "field": "previous_calibration_id"}), 404
            parent_calibration_id = prev_cal_id
            initial_beta = beta_from_previous(
                prev["result"]["parameters"], prev["result"]["T_ref"], prepared
            )
        elif not cold_start:
            # warm start by default: newest calibration on the previous version
            earlier = db.list_versions(conn, version["dataset_id"])
            earlier = [v for v in earlier if v["version"] < version["version"]]
            if earlier:
                cals = db.list_calibrations_for_version(conn, earlier[-1]["id"])
                if cals:
                    latest = cals[-1]
                    parent_calibration_id = latest["id"]
                    initial_beta = beta_from_previous(
                        latest["result"]["parameters"], latest["result"]["T_ref"], prepared
                    )

        result = calibrate(net, batches, initial_beta=initial_beta,
                           max_iter=int(body.get("max_iterations", 200)))
        with db.transaction() as c:
            saved = db.save_calibration(
                c, net_row["id"], version_id, result,
                parent_calibration_id=parent_calibration_id,
            )
        return jsonify({"calibration_id": saved["id"],
                        "version_id": version_id,
                        "parent_calibration_id": parent_calibration_id,
                        "created_at": saved["created_at"].isoformat(),
                        "result": _clean_result(result)}), 201

    @app.get("/api/calibrations/<calibration_id>")
    def get_calibration(calibration_id):
        if not is_uuid(calibration_id):
            return jsonify({"error": "not_found"}), 404
        row = db.get_calibration(db.get_db(), calibration_id)
        if row is None:
            return jsonify({"error": "not_found"}), 404
        return jsonify({
            "calibration_id": row["id"],
            "network_id": row["network_id"],
            "version_id": row["version_id"],
            "parent_calibration_id": row["parent_calibration_id"],
            "created_at": row["created_at"].isoformat(),
            "result": _clean_result(row["result"]),
        })

    @app.get("/api/dataset-versions/<version_id>/calibrations")
    def list_calibrations(version_id):
        if not is_uuid(version_id):
            return jsonify({"error": "not_found"}), 404
        conn = db.get_db()
        if db.get_version(conn, version_id) is None:
            return jsonify({"error": "not_found"}), 404
        rows = db.list_calibrations_for_version(conn, version_id)
        return jsonify({"calibrations": [{
            "calibration_id": r["id"],
            "version_id": r["version_id"],
            "parent_calibration_id": r["parent_calibration_id"],
            "created_at": r["created_at"].isoformat(),
        } for r in rows]})

    return app


# ----------------------------------------------------------------------
# (de)serialisation helpers
# ----------------------------------------------------------------------

def _network_json(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "definition": row["definition"],
        "created_at": row["created_at"].isoformat(),
    }


def _dataset_json(row):
    return {
        "id": row["id"],
        "network_id": row["network_id"],
        "name": row["name"],
        "created_at": row["created_at"].isoformat(),
    }


def _version_json(row):
    return {
        "id": row["id"],
        "dataset_id": row["dataset_id"],
        "version": row["version"],
        "parent_version_id": row["parent_version_id"],
        "change_note": row["change_note"],
        "created_at": row["created_at"].isoformat(),
    }


def _batch_json(row):
    return {
        "id": row["id"],
        "temperature": row["temperature"],
        "initial_concentrations": row["initial_concentrations"],
        "samples": row["samples"],
        "created_at": row["created_at"].isoformat(),
    }


def _normalise_batch_storage(batch: dict) -> dict:
    return {
        "temperature": batch["temperature"],
        "initial_concentrations": batch["initial_concentrations"],
        "samples": batch["samples"],
    }


# storage row -> API dict expected by prepare_dataset/validate_batch
def _batch_storage_to_api(b: dict) -> dict:
    return {
        "temperature": b["temperature"],
        "initial_concentrations": b["initial_concentrations"],
        "samples": b["samples"],
    }


def _batch_api_storage(row) -> dict:
    return {
        "temperature": row["temperature"],
        "initial_concentrations": row["initial_concentrations"],
        "samples": row["samples"],
    }


def _clean_result(result: dict) -> dict:
    out = {k: v for k, v in result.items() if not k.startswith("_")}
    return out
