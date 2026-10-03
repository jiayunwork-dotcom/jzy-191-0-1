import os

import pytest

from app import create_app
from app.config import Config


class TestConfig(Config):
    DATABASE_URL = os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql://kinetics:kinetics@127.0.0.1:5433/kinetics",
    )


@pytest.fixture()
def app():
    application = create_app(TestConfig)
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def network_def():
    return {
        "species": ["A", "B", "C"],
        "reactions": [
            {"name": "r1", "stoichiometry": {"A": -1, "B": 1}},
            {"name": "r2", "stoichiometry": {"B": -1, "C": 1}},
        ],
    }


@pytest.fixture()
def network_def_rev():
    return {
        "species": ["A", "B"],
        "reactions": [
            {"name": "r", "stoichiometry": {"A": -1, "B": 1}, "reversible": True},
        ],
    }
