"""Shared pytest config: $HOME isolation for the mocked suite + the live gate.

The mocked suite must never pick up a developer's real ``~/.slidestream.yaml``
(it would change which providers load and break deterministic tests). An
autouse fixture points ``Path.home()`` at an empty temp dir for every test
EXCEPT those marked ``live`` — the live e2e suite deliberately reads the real
home config to exercise whatever providers are actually configured.

Live tests live under ``tests/live/`` and only run with ``--run-live``.
"""

from pathlib import Path

import pytest

LIVE_DIR = Path(__file__).parent / "live"


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run live e2e tests against the real, configured providers",
    )


def pytest_collection_modifyitems(config, items):
    """Mark everything under tests/live as ``live`` and skip it unless
    ``--run-live`` is passed."""
    run_live = config.getoption("--run-live")
    skip_live = pytest.mark.skip(reason="live e2e test; pass --run-live to run")
    for item in items:
        try:
            in_live = LIVE_DIR in Path(str(item.fspath)).parents
        except Exception:
            in_live = False
        if in_live:
            item.add_marker(pytest.mark.live)
            if not run_live:
                item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _isolate_home(request, monkeypatch, tmp_path_factory):
    """Stub ``Path.home()`` to an empty dir so a real ~/.slidestream.yaml can't
    leak into the mocked suite. Live tests opt out — they need the real one."""
    if request.node.get_closest_marker("live"):
        return
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))


# Provider connection env vars: a developer (or ``uv run``, which auto-loads
# the project .env) may export tokens/base URLs for their real servers. The
# factory treats env-configured providers as deliberate choices, so a leaked
# var flips strict tests from the bare-install gTTS path into hard aborts.
_PROVIDER_ENV_VARS = (
    "VOICEBOX_BASE_URL",
    "VOICEBOX_TOKEN",
    "CHATTERBOX_BASE_URL",
    "CHATTERBOX_TOKEN",
    "SWARMUI_BASE_URL",
    "SWARMUI_TOKEN",
    "COMFYUI_TOKEN",
    "OLLAMA_BASE_URL",
    "OLLAMA_TOKEN",
    "OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def _isolate_provider_env(request, monkeypatch):
    """Scrub provider connection env vars for the mocked suite; live opt-out."""
    if request.node.get_closest_marker("live"):
        return
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
