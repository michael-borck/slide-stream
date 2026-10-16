"""Web UI upload limits and X-Forwarded-For handling."""

from io import BytesIO

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from slide_stream import serve  # noqa: E402


def _tiny_png() -> bytes:
    """A real (decodable) 4x4 PNG for photo uploads."""
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
    return buf.getvalue()


def test_deck_size_limit_413(base_config, monkeypatch, mocker):
    monkeypatch.setattr(serve, "MAX_DECK_BYTES", 1024)
    mocker.patch.object(serve, "_run_job")
    client = TestClient(serve.create_app(config=base_config, token=None))
    r = client.post("/api/jobs",
                    files={"deck": ("deck.md", b"# S\n" + b"x" * 4096)})
    assert r.status_code == 413
    assert "Deck" in r.json()["detail"]


def test_voice_size_limit_413(base_config, monkeypatch, mocker):
    monkeypatch.setattr(serve, "MAX_VOICE_BYTES", 1024)
    mocker.patch.object(serve, "_run_job")
    client = TestClient(serve.create_app(config=base_config, token=None))
    r = client.post("/api/jobs", files={
        "deck": ("deck.md", b"# One\n- a\n"),
        "voice": ("me.wav", b"x" * 4096, "audio/wav"),
    })
    assert r.status_code == 413
    assert "Voice" in r.json()["detail"]


def test_photo_upload_must_be_a_real_image(base_config, mocker):
    mocker.patch.object(serve, "_run_job")
    client = TestClient(serve.create_app(config=base_config, token=None))
    r = client.post("/api/jobs", files={
        "deck": ("deck.md", b"# One\n- a\n"),
        "photo": ("me.png", b"NOTANIMAGE", "image/png"),
    })
    assert r.status_code == 400


def test_photo_dimension_cap(base_config, monkeypatch, mocker):
    monkeypatch.setattr(serve, "MAX_IMAGE_DIM", 2)  # the test PNG is 4x4
    mocker.patch.object(serve, "_run_job")
    client = TestClient(serve.create_app(config=base_config, token=None))
    r = client.post("/api/jobs", files={
        "deck": ("deck.md", b"# One\n- a\n"),
        "photo": ("me.png", _tiny_png(), "image/png"),
    })
    assert r.status_code == 400
    assert "maximum" in r.json()["detail"]


# --- X-Forwarded-For handling ------------------------------------------------


def test_xff_ignored_without_trusted_proxy(base_config, mocker, monkeypatch):
    monkeypatch.delenv("SLIDESTREAM_TRUSTED_PROXY", raising=False)
    mocker.patch.object(serve, "_run_job")
    client = TestClient(serve.create_app(config=base_config, demo=True))
    deck = {"deck": ("deck.md", b"# One\n- a\n")}
    for i in range(serve.DEMO_JOBS_PER_HOUR):
        r = client.post("/api/jobs", files=deck,
                        headers={"X-Forwarded-For": f"10.0.0.{i}"})
        assert r.status_code == 200
    # A freshly spoofed XFF must not buy another job.
    over = client.post("/api/jobs", files=deck,
                       headers={"X-Forwarded-For": "10.99.99.99"})
    assert over.status_code == 429


def test_xff_rightmost_honored_behind_trusted_proxy(base_config, mocker,
                                                    monkeypatch):
    monkeypatch.setenv("SLIDESTREAM_TRUSTED_PROXIES", "10.0.0.0/24")
    mocker.patch.object(serve, "_run_job")
    client = TestClient(serve.create_app(config=base_config, demo=True),
                        client=("10.0.0.2", 1234))
    deck = {"deck": ("deck.md", b"# One\n- a\n")}
    # The proxy-appended (rightmost) value keys the limit; the client-chosen
    # left entries are ignored.
    for i in range(serve.DEMO_JOBS_PER_HOUR):
        r = client.post("/api/jobs", files=deck,
                        headers={"X-Forwarded-For": f"10.0.0.{i}, 9.9.9.9"})
        assert r.status_code == 200
    over = client.post("/api/jobs", files=deck,
                       headers={"X-Forwarded-For": "10.0.0.99, 9.9.9.9"})
    assert over.status_code == 429
    # A genuinely different client (per the proxy) gets its own bucket.
    ok = client.post("/api/jobs", files=deck,
                     headers={"X-Forwarded-For": "10.0.0.1, 8.8.8.8"})
    assert ok.status_code == 200


# --- demo hardening -----------------------------------------------------------
