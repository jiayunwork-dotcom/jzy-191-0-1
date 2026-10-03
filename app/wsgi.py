"""WSGI entry point: ``python -m app.wsgi`` serves on $PORT (default 8000)."""

from . import create_app

app = create_app()

if __name__ == "__main__":
    import os

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
