import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.app import create_app
from app.repository import MemoryRepository, PostgresRepository


@pytest.fixture
def repo():
    return MemoryRepository()


@pytest.fixture
def app(repo):
    app = create_app(repository=repo)
    app.config.update(TESTING=True)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def _maybe_postgres_dsn():
    return os.environ.get("TEST_POSTGRES_DSN")


@pytest.fixture
def pg_repo():
    dsn = _maybe_postgres_dsn()
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN not set; skipping PostgreSQL tests")
    repo = PostgresRepository(dsn)
    # Clean slate for the tables touched by tests.
    with repo._pool.connection() as conn:
        conn.execute("DELETE FROM calibrations")
        conn.execute("DELETE FROM dataset_versions")
        conn.execute("DELETE FROM datasets")
        conn.execute("DELETE FROM batches")
        conn.execute("DELETE FROM networks")
    yield repo
    repo.close()


@pytest.fixture(params=["memory", "postgres"])
def any_repo(request):
    if request.param == "memory":
        yield MemoryRepository()
        return
    dsn = _maybe_postgres_dsn()
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN not set; skipping PostgreSQL tests")
    repo = PostgresRepository(dsn)
    with repo._pool.connection() as conn:
        conn.execute("DELETE FROM calibrations")
        conn.execute("DELETE FROM dataset_versions")
        conn.execute("DELETE FROM datasets")
        conn.execute("DELETE FROM batches")
        conn.execute("DELETE FROM networks")
    yield repo
    repo.close()
