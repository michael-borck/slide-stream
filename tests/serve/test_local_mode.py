"""Web UI local (desktop) mode: settings and host/origin guards."""


import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from slide_stream import serve  # noqa: E402


def test_local_mode_open_and_no_demo_limits(base_config):
    """Desktop mode: no token, demo limits off even if both are set."""
    client = TestClient(serve.create_app(
        config=base_config, token="secret", demo=True, local=True),
        base_url="http://127.0.0.1")
    cfg = client.get("/api/config").json()
    assert cfg["local"] is True
    assert cfg["auth_required"] is False
    assert cfg["demo"] is False
    assert client.get("/api/jobs/whatever").status_code == 404  # open


def test_settings_hidden_unless_local(base_config):
    client = TestClient(serve.create_app(config=base_config, local=False))
    assert client.get("/api/settings").status_code == 404


def test_settings_roundtrip(base_config, tmp_path, monkeypatch):
    from pathlib import Path as _P

    monkeypatch.setattr(_P, "home", staticmethod(lambda: tmp_path))
    # Local mode only accepts state-changing requests addressed to localhost.
    client = TestClient(serve.create_app(config=base_config, local=True),
                        base_url="http://127.0.0.1")

    s = client.get("/api/settings").json()
    assert s["yaml"] == ""  # nothing saved yet
    assert "providers:" in s["template"]

    ok = client.put("/api/settings",
                    json={"yaml": "providers:\n  tts:\n    provider: gtts\n"})
    assert ok.status_code == 200
    saved = tmp_path / ".slidestream.yaml"
    assert saved.exists()
    # Owner-only: the file typically holds API keys.
    assert (saved.stat().st_mode & 0o777) == 0o600
    assert "gtts" in client.get("/api/settings").json()["yaml"]


def test_settings_rejects_bad_yaml(base_config, tmp_path, monkeypatch):
    from pathlib import Path as _P

    monkeypatch.setattr(_P, "home", staticmethod(lambda: tmp_path))
    client = TestClient(serve.create_app(config=base_config, local=True),
                        base_url="http://127.0.0.1")
    bad = client.put("/api/settings", json={"yaml": "a: [unclosed"})
    assert bad.status_code == 400
    assert not (tmp_path / ".slidestream.yaml").exists()
    notdict = client.put("/api/settings", json={"yaml": "- just\n- a list\n"})
    assert notdict.status_code == 400


# --- upload limits ---------------------------------------------------------


def test_local_mode_rejects_foreign_host(base_config):
    app = serve.create_app(config=base_config, local=True)
    rebound = TestClient(app, base_url="http://evil.example.com")
    assert rebound.put("/api/settings", json={"yaml": ""}).status_code == 403
    assert rebound.post("/api/jobs", files={
        "deck": ("deck.md", b"# One\n- a\n")}).status_code == 403


def test_local_mode_rejects_foreign_origin(base_config, tmp_path, monkeypatch):
    from pathlib import Path as _P

    monkeypatch.setattr(_P, "home", staticmethod(lambda: tmp_path))
    client = TestClient(serve.create_app(config=base_config, local=True),
                        base_url="http://localhost")
    evil = client.put("/api/settings", json={"yaml": "a: 1\n"},
                      headers={"Origin": "https://evil.example.com"})
    assert evil.status_code == 403
    # Local pages and the Tauri shell's webview stay allowed.
    ok = client.put("/api/settings", json={"yaml": "a: 1\n"},
                    headers={"Origin": "http://127.0.0.1:8080"})
    assert ok.status_code == 200
    tauri = client.put("/api/settings", json={"yaml": "a: 1\n"},
                       headers={"Origin": "tauri://localhost"})
    assert tauri.status_code == 200


def test_local_origin_helper():
    assert serve._local_origin_ok("http://localhost:8080")
    assert serve._local_origin_ok("http://127.0.0.1")
    assert serve._local_origin_ok("tauri://localhost")
    assert serve._local_origin_ok("http://tauri.localhost")
    assert not serve._local_origin_ok("https://evil.example.com")
    assert not serve._local_origin_ok("null")
    assert not serve._local_origin_ok("http://localhost.evil.com")


# --- job hygiene: secrets and artifacts do not persist -------------------------
