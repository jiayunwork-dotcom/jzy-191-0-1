"""Application configuration from environment variables."""

import os


class Config:
    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        "postgresql://kinetics:kinetics@localhost:5432/kinetics",
    )
    JSON_SORT_KEYS = False
