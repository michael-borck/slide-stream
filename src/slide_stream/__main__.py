"""Enable `python -m slide_stream ...` (used by the web server's job runner)."""

from .cli import app

if __name__ == "__main__":
    app()
