"""PostgreSQL access layer: connection helpers and DAO functions.

Only psycopg2 is used -- no ORM.
"""

from __future__ import annotations

import json
import pathlib
import time
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from flask import current_app, g

_SCHEMA_PATH = pathlib.Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


def connect(database_url: str | None = None, *, retries: int = 30, delay: float = 1.0):
    """Open a connection, retrying while the database is starting up."""
    url = database_url or current_app.config["DATABASE_URL"]
    last = None
    for i in range(retries):
        try:
            conn = psycopg2.connect(url)
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as exc:  # pragma: no cover - startup race
            last = exc
            time.sleep(delay)
    raise last  # pragma: no cover


def get_db():
    if "db" not in g:
        g.db = connect()
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(conn=None):
    """Create tables if they do not exist (idempotent)."""
    owns = conn is None
    conn = conn or get_db()
    ddl = _SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    if owns:
        conn.close()


def init_app(app):
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()


@contextmanager
def transaction():
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# DAO
# ---------------------------------------------------------------------------

def _json(value):
    return json.dumps(value, ensure_ascii=False)


def create_network(conn, name: str, definition: dict) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO networks (name, definition) VALUES (%s, %s) "
            "RETURNING id, name, definition, created_at",
            (name, _json(definition)),
        )
        return dict(cur.fetchone())


def get_network(conn, network_id: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, name, definition, created_at FROM networks WHERE id = %s",
            (network_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def create_dataset(conn, network_id: str, name: str) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO datasets (network_id, name) VALUES (%s, %s) "
            "RETURNING id, network_id, name, created_at",
            (network_id, name),
        )
        return dict(cur.fetchone())


def get_dataset(conn, dataset_id: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, network_id, name, created_at FROM datasets WHERE id = %s",
            (dataset_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def latest_version(conn, dataset_id: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, dataset_id, version, parent_version_id, change_note, created_at "
            "FROM dataset_versions WHERE dataset_id = %s ORDER BY version DESC LIMIT 1",
            (dataset_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_version(conn, version_id: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, dataset_id, version, parent_version_id, change_note, created_at "
            "FROM dataset_versions WHERE id = %s",
            (version_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def list_versions(conn, dataset_id: str) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, dataset_id, version, parent_version_id, change_note, created_at "
            "FROM dataset_versions WHERE dataset_id = %s ORDER BY version",
            (dataset_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def _clone_batches(conn, src_version_id: str, dst_version_id: str):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO batches (version_id, temperature, initial_concentrations, samples) "
            "SELECT %s, temperature, initial_concentrations, samples FROM batches "
            "WHERE version_id = %s",
            (dst_version_id, src_version_id),
        )


def _new_version(conn, dataset_id: str, change_note: str):
    """Insert the next version, cloning parent batches. Caller controls tx."""
    parent = latest_version(conn, dataset_id)
    next_version = 1 if parent is None else parent["version"] + 1
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO dataset_versions (dataset_id, version, parent_version_id, change_note) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (dataset_id, next_version, parent["id"] if parent else None, change_note),
        )
        new_id = cur.fetchone()[0]
    if parent is not None:
        _clone_batches(conn, parent["id"], new_id)
    return new_id, next_version


def add_batch(conn, dataset_id: str, batch: dict, note: str | None = None) -> tuple[str, int]:
    """Create a new version = parent batches + one new batch. Returns (version_id, n)."""
    version_id, number = _new_version(
        conn, dataset_id, note or f"add batch at T={batch['temperature']}"
    )
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO batches (version_id, temperature, initial_concentrations, samples) "
            "VALUES (%s, %s, %s, %s)",
            (version_id, batch["temperature"],
             _json(batch["initial_concentrations"]), _json(batch["samples"])),
        )
    return str(version_id), number


def remove_batch(conn, dataset_id: str, batch_index: int, note: str | None = None) -> tuple[str, int]:
    """Create a new version with the batch at ``batch_index`` (0-based, current
    version order) omitted."""
    parent = latest_version(conn, dataset_id)
    if parent is None:
        raise LookupError("dataset has no versions")
    batches = list_batches(conn, parent["id"])
    if not (0 <= batch_index < len(batches)):
        raise IndexError(f"batch index {batch_index} out of range (0..{len(batches) - 1})")
    version_id, number = _new_version(conn, dataset_id, note or f"remove batch #{batch_index}")
    with conn.cursor() as cur:
        # clones have new ids; delete by ordinal position within the new version
        cur.execute(
            "DELETE FROM batches WHERE id = (SELECT id FROM batches "
            "WHERE version_id = %s ORDER BY created_at, id OFFSET %s LIMIT 1)",
            (version_id, batch_index),
        )
    return str(version_id), number


def list_batches(conn, version_id: str) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, version_id, temperature, initial_concentrations, samples, created_at "
            "FROM batches WHERE version_id = %s ORDER BY created_at, id",
            (version_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def save_calibration(conn, network_id: str, version_id: str, result: dict,
                     parent_calibration_id: str | None = None) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO calibrations (network_id, version_id, parent_calibration_id, result) "
            "VALUES (%s, %s, %s, %s) RETURNING id, version_id, parent_calibration_id, created_at",
            (network_id, version_id, parent_calibration_id, _json(result)),
        )
        row = dict(cur.fetchone())
    row["result"] = result
    return row


def get_calibration(conn, calibration_id: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, network_id, version_id, parent_calibration_id, result, created_at "
            "FROM calibrations WHERE id = %s",
            (calibration_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def list_calibrations_for_version(conn, version_id: str) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, network_id, version_id, parent_calibration_id, result, created_at "
            "FROM calibrations WHERE version_id = %s ORDER BY created_at",
            (version_id,),
        )
        return [dict(r) for r in cur.fetchall()]
