"""Storage layer.

Two interchangeable repositories implement the same interface:

* :class:`MemoryRepository` – thread-safe in-process storage (default, also
  used by the bulk of the test suite);
* :class:`PostgresRepository` – PostgreSQL 16 persistence that survives
  service restarts.

Dataset versioning model: a dataset is a named, ordered chain of immutable
versions.  Every version stores its full batch-id list (snapshot, not a
diff), and every calibration row binds to exactly one version.  Old versions
and old calibrations are never overwritten.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone


def new_id():
    return uuid.uuid4().hex


def utcnow():
    return datetime.now(timezone.utc)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS networks (
    id           TEXT PRIMARY KEY,
    spec         JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    id           TEXT PRIMARY KEY,
    network_id   TEXT NOT NULL REFERENCES networks(id),
    payload      JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id           TEXT PRIMARY KEY,
    network_id   TEXT NOT NULL REFERENCES networks(id),
    name         TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id           TEXT PRIMARY KEY,
    dataset_id   TEXT NOT NULL REFERENCES datasets(id),
    version_no   INTEGER NOT NULL,
    batch_ids    JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL,
    UNIQUE (dataset_id, version_no)
);

CREATE TABLE IF NOT EXISTS calibrations (
    id                    TEXT PRIMARY KEY,
    network_id            TEXT NOT NULL REFERENCES networks(id),
    dataset_id            TEXT NOT NULL REFERENCES datasets(id),
    dataset_version_id    TEXT NOT NULL REFERENCES dataset_versions(id),
    version_no            INTEGER NOT NULL,
    result                JSONB NOT NULL,
    started_from          JSONB,
    created_at            TIMESTAMPTZ NOT NULL
);
"""


def serialize_batch(batch):
    return json.loads(json.dumps(batch))  # normalize to plain JSON types


class MemoryRepository:
    """Deterministic in-process repository with the same semantics as the
    Postgres one.  A write lock makes concurrent Flask requests safe."""

    def __init__(self):
        self._lock = threading.RLock()
        self.networks = {}
        self.batches = {}
        self.datasets = {}
        self.versions = {}      # version id -> row
        self.dataset_versions = {}  # dataset_id -> {version_no: row}
        self.calibrations = {}

    # ---- networks ----------------------------------------------------------
    def create_network(self, spec):
        with self._lock:
            nid = new_id()
            row = {"id": nid, "spec": spec, "created_at": utcnow()}
            self.networks[nid] = row
            return self._network_row(row)

    def get_network(self, network_id):
        with self._lock:
            row = self.networks.get(network_id)
            return self._network_row(row) if row else None

    @staticmethod
    def _network_row(row):
        return {"id": row["id"], "spec": row["spec"],
                "created_at": row["created_at"].isoformat()}

    # ---- batches -----------------------------------------------------------
    def create_batch(self, network_id, batch):
        with self._lock:
            bid = batch.get("id") or new_id()
            payload = {
                "id": bid,
                "network_id": network_id,
                "temperature": float(batch["temperature"]),
                "initial_concentrations": batch["initial_concentrations"],
                "samples": [
                    {"time": float(s["time"]),
                     "observations": {k: float(v)
                                      for k, v in s["observations"].items()}}
                    for s in batch["samples"]
                ],
                "note": batch.get("note"),
            }
            row = {"id": bid, "network_id": network_id, "payload": payload,
                   "created_at": utcnow()}
            self.batches[bid] = row
            return self._batch_row(row)

    def get_batch(self, batch_id):
        with self._lock:
            row = self.batches.get(batch_id)
            return self._batch_row(row) if row else None

    def list_batches(self, network_id=None):
        with self._lock:
            return [self._batch_row(r) for r in self.batches.values()
                    if network_id is None or r["network_id"] == network_id]

    @staticmethod
    def _batch_row(row):
        p = row["payload"]
        return {"id": row["id"], "network_id": row["network_id"],
                "temperature": p["temperature"],
                "initial_concentrations": p["initial_concentrations"],
                "samples": p["samples"],
                "note": p.get("note"),
                "created_at": row["created_at"].isoformat()}

    # ---- datasets & versions ----------------------------------------------
    def create_dataset(self, network_id, name, batch_ids):
        with self._lock:
            did = new_id()
            self.datasets[did] = {"id": did, "network_id": network_id,
                                  "name": name, "created_at": utcnow()}
            self.dataset_versions[did] = {}
            return self._add_version(did, batch_ids)

    def add_version(self, dataset_id, batch_ids):
        with self._lock:
            if dataset_id not in self.datasets:
                return None
            return self._add_version(dataset_id, batch_ids)

    def _add_version(self, dataset_id, batch_ids):
        # Validate batch existence and network ownership; preserve order,
        # reject duplicates within one version.
        ds = self.datasets[dataset_id]
        clean, seen = [], set()
        for bid in batch_ids:
            if bid not in self.batches:
                raise KeyError(f"batch {bid!r} does not exist")
            if self.batches[bid]["network_id"] != ds["network_id"]:
                raise ValueError(
                    f"batch {bid!r} belongs to a different network")
            if bid in seen:
                raise ValueError(f"batch {bid!r} listed more than once")
            seen.add(bid)
            clean.append(bid)
        versions = self.dataset_versions[dataset_id]
        version_no = (max(versions, default=0) + 1) if versions else 1
        vid = new_id()
        row = {"id": vid, "dataset_id": dataset_id, "version_no": version_no,
               "batch_ids": clean, "created_at": utcnow()}
        versions[version_no] = row
        self.versions[vid] = row
        return self._version_row(row)

    def _version_row(self, row):
        return {"id": row["id"], "dataset_id": row["dataset_id"],
                "version_no": row["version_no"],
                "batch_ids": list(row["batch_ids"]),
                "created_at": row["created_at"].isoformat()}

    def get_version(self, version_id):
        with self._lock:
            row = self.versions.get(version_id)
            return self._version_row(row) if row else None

    def get_dataset(self, dataset_id):
        with self._lock:
            ds = self.datasets.get(dataset_id)
            if not ds:
                return None
            versions = sorted(self.dataset_versions[dataset_id].values(),
                              key=lambda r: r["version_no"])
            return {"id": ds["id"], "network_id": ds["network_id"],
                    "name": ds["name"],
                    "created_at": ds["created_at"].isoformat(),
                    "latest_version_no": versions[-1]["version_no"]
                    if versions else None,
                    "versions": [v["id"] for v in versions]}

    def list_versions(self, dataset_id):
        with self._lock:
            if dataset_id not in self.datasets:
                return None
            return [self._version_row(r)
                    for r in sorted(self.dataset_versions[dataset_id].values(),
                                    key=lambda r: r["version_no"])]

    def batches_for_version(self, version_id):
        with self._lock:
            v = self.versions.get(version_id)
            if not v:
                return None
            return [self._batch_row(self.batches[bid]) for bid in v["batch_ids"]]

    # ---- calibrations ------------------------------------------------------
    def save_calibration(self, network_id, dataset_id, version_id, result,
                         started_from=None):
        with self._lock:
            v = self.versions[version_id]
            cid = new_id()
            row = {"id": cid, "network_id": network_id,
                   "dataset_id": dataset_id, "dataset_version_id": version_id,
                   "version_no": v["version_no"], "result": result,
                   "started_from": started_from, "created_at": utcnow()}
            self.calibrations[cid] = row
            return self._calibration_row(row)

    def get_calibration(self, calibration_id):
        with self._lock:
            row = self.calibrations.get(calibration_id)
            return self._calibration_row(row) if row else None

    def list_calibrations(self, version_id=None, dataset_id=None):
        with self._lock:
            out = []
            for r in self.calibrations.values():
                if version_id and r["dataset_version_id"] != version_id:
                    continue
                if dataset_id and r["dataset_id"] != dataset_id:
                    continue
                out.append(self._calibration_row(r))
            out.sort(key=lambda r: r["created_at"])
            return out

    @staticmethod
    def _calibration_row(row):
        return {"id": row["id"], "network_id": row["network_id"],
                "dataset_id": row["dataset_id"],
                "dataset_version_id": row["dataset_version_id"],
                "version_no": row["version_no"], "result": row["result"],
                "started_from": row.get("started_from"),
                "created_at": row["created_at"].isoformat()}

    def last_calibration_for_dataset(self, dataset_id):
        with self._lock:
            rows = [r for r in self.calibrations.values()
                    if r["dataset_id"] == dataset_id]
            if not rows:
                return None
            rows.sort(key=lambda r: (r["version_no"], r["created_at"]))
            return self._calibration_row(rows[-1])


class PostgresRepository:
    """PostgreSQL-backed repository.  A small connection pool is shared by
    request threads; all multi-statement writes run in a transaction."""

    def __init__(self, conninfo, min_size=1, max_size=8, autocommit=False):
        from psycopg_pool import ConnectionPool
        self._pool = ConnectionPool(conninfo=conninfo, min_size=min_size,
                                    max_size=max_size, open=True,
                                    kwargs={"autocommit": autocommit})
        self.init_schema()

    def close(self):
        self._pool.close()

    def init_schema(self):
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)

    # ---- networks ----------------------------------------------------------
    def create_network(self, spec):
        nid = new_id()
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO networks (id, spec, created_at) VALUES (%s, %s, %s)",
                (nid, _json(spec), utcnow()))
        return self.get_network(nid)

    def get_network(self, network_id):
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, spec, created_at FROM networks WHERE id=%s",
                (network_id,)).fetchone()
        if not row:
            return None
        return {"id": row[0], "spec": row[1],
                "created_at": row[2].isoformat()}

    # ---- batches -----------------------------------------------------------
    def create_batch(self, network_id, batch):
        bid = batch.get("id") or new_id()
        payload = {
            "id": bid,
            "temperature": float(batch["temperature"]),
            "initial_concentrations": batch["initial_concentrations"],
            "samples": [
                {"time": float(s["time"]),
                 "observations": {k: float(v)
                                  for k, v in s["observations"].items()}}
                for s in batch["samples"]
            ],
            "note": batch.get("note"),
        }
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO batches (id, network_id, payload, created_at) "
                "VALUES (%s, %s, %s, %s)",
                (bid, network_id, _json(payload), utcnow()))
        return self.get_batch(bid)

    def get_batch(self, batch_id):
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, network_id, payload, created_at "
                "FROM batches WHERE id=%s", (batch_id,)).fetchone()
        return _batch_from_row(row) if row else None

    def list_batches(self, network_id=None):
        with self._pool.connection() as conn:
            if network_id:
                rows = conn.execute(
                    "SELECT id, network_id, payload, created_at FROM batches "
                    "WHERE network_id=%s ORDER BY created_at",
                    (network_id,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, network_id, payload, created_at "
                    "ORDER BY created_at").fetchall()
        return [_batch_from_row(r) for r in rows]

    # ---- datasets & versions ----------------------------------------------
    def create_dataset(self, network_id, name, batch_ids):
        did = new_id()
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO datasets (id, network_id, name, created_at) "
                "VALUES (%s, %s, %s, %s)",
                (did, network_id, name, utcnow()))
        return self.add_version(did, batch_ids)

    def add_version(self, dataset_id, batch_ids):
        with self._pool.connection() as conn:
            ds = conn.execute(
                "SELECT network_id FROM datasets WHERE id=%s FOR UPDATE",
                (dataset_id,)).fetchone()
            if not ds:
                return None
            network_id = ds[0]
            clean, seen = [], set()
            for bid in batch_ids:
                brow = conn.execute(
                    "SELECT network_id FROM batches WHERE id=%s",
                    (bid,)).fetchone()
                if not brow:
                    raise KeyError(f"batch {bid!r} does not exist")
                if brow[0] != network_id:
                    raise ValueError(
                        f"batch {bid!r} belongs to a different network")
                if bid in seen:
                    raise ValueError(f"batch {bid!r} listed more than once")
                seen.add(bid)
                clean.append(bid)
            last = conn.execute(
                "SELECT COALESCE(MAX(version_no), 0) FROM dataset_versions "
                "WHERE dataset_id=%s", (dataset_id,)).fetchone()[0]
            version_no = last + 1
            vid = new_id()
            conn.execute(
                "INSERT INTO dataset_versions "
                "(id, dataset_id, version_no, batch_ids, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (vid, dataset_id, version_no, _json(clean), utcnow()))
        return self.get_version(vid)

    def get_version(self, version_id):
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, dataset_id, version_no, batch_ids, created_at "
                "FROM dataset_versions WHERE id=%s", (version_id,)).fetchone()
        return _version_from_row(row) if row else None

    def get_dataset(self, dataset_id):
        with self._pool.connection() as conn:
            ds = conn.execute(
                "SELECT id, network_id, name, created_at FROM datasets "
                "WHERE id=%s", (dataset_id,)).fetchone()
            if not ds:
                return None
            vers = conn.execute(
                "SELECT id FROM dataset_versions WHERE dataset_id=%s "
                "ORDER BY version_no", (dataset_id,)).fetchall()
        return {"id": ds[0], "network_id": ds[1], "name": ds[2],
                "created_at": ds[3].isoformat(),
                "latest_version_no": len(vers),
                "versions": [v[0] for v in vers]}

    def list_versions(self, dataset_id):
        with self._pool.connection() as conn:
            ds = conn.execute(
                "SELECT 1 FROM datasets WHERE id=%s",
                (dataset_id,)).fetchone()
            if not ds:
                return None
            rows = conn.execute(
                "SELECT id, dataset_id, version_no, batch_ids, created_at "
                "FROM dataset_versions WHERE dataset_id=%s ORDER BY version_no",
                (dataset_id,)).fetchall()
        return [_version_from_row(r) for r in rows]

    def batches_for_version(self, version_id):
        with self._pool.connection() as conn:
            v = conn.execute(
                "SELECT batch_ids FROM dataset_versions WHERE id=%s",
                (version_id,)).fetchone()
            if not v:
                return None
            ids = v[0]
            rows = conn.execute(
                "SELECT id, network_id, payload, created_at FROM batches "
                "WHERE id = ANY(%s)", (ids,)).fetchall()
        by_id = {r[0]: r for r in rows}
        return [_batch_from_row(by_id[bid]) for bid in ids]

    # ---- calibrations ------------------------------------------------------
    def save_calibration(self, network_id, dataset_id, version_id, result,
                         started_from=None):
        v = self.get_version(version_id)
        cid = new_id()
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO calibrations "
                "(id, network_id, dataset_id, dataset_version_id, version_no, "
                " result, started_from, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (cid, network_id, dataset_id, version_id, v["version_no"],
                 _json(result),
                 _json(started_from) if started_from is not None else None,
                 utcnow()))
        return self.get_calibration(cid)

    def get_calibration(self, calibration_id):
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, network_id, dataset_id, dataset_version_id, "
                "version_no, result, started_from, created_at "
                "FROM calibrations WHERE id=%s", (calibration_id,)).fetchone()
        return _calibration_from_row(row) if row else None

    def list_calibrations(self, version_id=None, dataset_id=None):
        q = ("SELECT id, network_id, dataset_id, dataset_version_id, "
             "version_no, result, started_from, created_at FROM calibrations")
        cond, args = [], []
        if version_id:
            cond.append("dataset_version_id=%s"); args.append(version_id)
        if dataset_id:
            cond.append("dataset_id=%s"); args.append(dataset_id)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY created_at"
        with self._pool.connection() as conn:
            rows = conn.execute(q, args).fetchall()
        return [_calibration_from_row(r) for r in rows]

    def last_calibration_for_dataset(self, dataset_id):
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, network_id, dataset_id, dataset_version_id, "
                "version_no, result, started_from, created_at FROM calibrations "
                "WHERE dataset_id=%s ORDER BY version_no DESC, created_at DESC "
                "LIMIT 1", (dataset_id,)).fetchone()
        return _calibration_from_row(row) if row else None


# ---- row mappers -----------------------------------------------------------

def _json(obj):
    from psycopg.types.json import Jsonb
    return Jsonb(obj)


def _batch_from_row(row):
    p = row[2]
    return {"id": row[0], "network_id": row[1],
            "temperature": p["temperature"],
            "initial_concentrations": p["initial_concentrations"],
            "samples": p["samples"], "note": p.get("note"),
            "created_at": row[3].isoformat()}


def _version_from_row(row):
    return {"id": row[0], "dataset_id": row[1], "version_no": row[2],
            "batch_ids": list(row[3]), "created_at": row[4].isoformat()}


def _calibration_from_row(row):
    return {"id": row[0], "network_id": row[1], "dataset_id": row[2],
            "dataset_version_id": row[3], "version_no": row[4],
            "result": row[5], "started_from": row[6],
            "created_at": row[7].isoformat()}
