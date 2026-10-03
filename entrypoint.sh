#!/usr/bin/env bash
# Wait for PostgreSQL to accept connections, then start the app.
# Schema creation runs automatically when the repository is constructed.
set -euo pipefail

if [ -n "${DATABASE_URL:-}" ]; then
  python - <<'PY'
import os, sys, time
import psycopg

dsn = os.environ["DATABASE_URL"]
deadline = time.time() + 60
last = None
while time.time() < deadline:
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        print("database is ready", flush=True)
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last = exc
        time.sleep(1.0)
print(f"database did not become ready: {last}", file=sys.stderr)
sys.exit(1)
PY
fi

exec "$@"
