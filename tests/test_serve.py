"""Tests for the web UI (slide-stream serve)."""

import copy

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from slide_stream import serve  # noqa: E402
from slide_stream.config_loader import DEFAULT_CONFIG  # noqa: E402


@pytest.fixture
def base_config():
    return copy.deepcopy(DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def _clear_jobs():
    serve._JOBS.clear()
    yield
    serve._JOBS.clear()


def test_index_served(base_config):
    client = TestClient(serve.create_app(config=base_config, token="secret"))
    r = client.get("/")
    assert r.status_code == 200
    assert "SlideStream" in r.text


def test_api_requires_token(base_config):
    client = TestClient(serve.create_app(config=base_config, token="secret"))
    # /api/config is public so the UI can bootstrap...
    cfg = client.get("/api/config")
    assert cfg.status_code == 200
    assert cfg.json()["auth_required"] is True
    # ...but protected endpoints reject a missing token.
    assert client.get("/api/jobs/whatever").status_code == 401
    ok = client.get("/api/jobs/whatever", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 404  # authorized, just no such job


def test_no_token_means_open(base_config):
    client = TestClient(serve.create_app(config=base_config, token=None))
    assert client.get("/api/config").status_code == 200
    assert client.get("/api/jobs/whatever").status_code == 404


def test_demo_flag_exposed(base_config):
    client = TestClient(serve.create_app(config=base_config, token=None, demo=True))
    assert client.get("/api/config").json()["demo"] is True
    off = TestClient(serve.create_app(config=base_config, token=None, demo=False))
    assert off.get("/api/config").json()["demo"] is False


def test_create_job_rejects_bad_deck_type(base_config):
    client = TestClient(serve.create_app(config=base_config, token=None))
    r = client.post("/api/jobs", files={"deck": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_job_lifecycle_and_download(base_config, tmp_path, monkeypatch):
    """Submit a deck; the (mocked) render produces a video that downloads."""

    def fake_run_job(job, deck_path, job_yaml, voice_path, photo_path):
        # Simulate a successful render writing output.mp4.
        assert job.workdir is not None
        out = job.workdir / "output.mp4"
        out.write_bytes(b"FAKEMP4")
        job.status = "done"
        job.output_path = out
        job.log = "rendered"
        # Ephemeral inputs are wiped by the real runner; emulate it.
        for p in (voice_path, photo_path):
            if p is not None:
                p.unlink(missing_ok=True)

    monkeypatch.setattr(serve, "_run_job", fake_run_job)
    # Run submitted work synchronously so the test is deterministic.
    monkeypatch.setattr(
        serve.ThreadPoolExecutor, "submit",
        lambda self, fn, *a, **k: fn(*a, **k),
    )

    client = TestClient(serve.create_app(config=base_config, token=None))
    r = client.post(
        "/api/jobs",
        files={
            "deck": ("deck.md", b"# One\n\n- a\n", "text/markdown"),
            "voice": ("me.wav", b"AUDIO", "audio/wav"),
            "photo": ("me.png", b"IMAGE", "image/png"),
        },
        data={"avatar": "true", "narration_seconds": "30"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "done"

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.content == b"FAKEMP4"


def test_job_config_ephemeral_voice_and_photo(base_config, tmp_path):
    """The per-job config wires the uploaded voice/photo into providers."""
    workdir = tmp_path / "job"
    workdir.mkdir()
    voice = workdir / "voice.wav"
    voice.write_bytes(b"a")
    photo = workdir / "photo.png"
    photo.write_bytes(b"b")

    job_yaml = serve._build_job_config(
        base_config, workdir,
        {"avatar": True, "narration_seconds": "25", "image_provider": "swarmui"},
        voice, photo,
    )
    import yaml

    cfg = yaml.safe_load(job_yaml.read_text())
    assert cfg["providers"]["tts"]["voice_sample"] == str(voice)
    assert cfg["providers"]["avatar"]["source_image"] == str(photo)
    assert cfg["providers"]["images"]["provider"] == "swarmui"
    assert cfg["settings"]["narration"]["target_seconds"] == 25.0


def test_job_config_avatar_off_when_unchecked(base_config, tmp_path):
    workdir = tmp_path / "job"
    workdir.mkdir()
    photo = workdir / "photo.png"
    photo.write_bytes(b"b")
    job_yaml = serve._build_job_config(
        base_config, workdir, {"avatar": False}, None, photo
    )
    import yaml

    cfg = yaml.safe_load(job_yaml.read_text())
    assert cfg["providers"]["avatar"]["provider"] == "none"


def test_download_accepts_token_query_param(base_config, tmp_path, monkeypatch):
    """A browser download link (no header) authenticates via ?t=."""
    monkeypatch.setattr(
        serve.ThreadPoolExecutor, "submit",
        lambda self, fn, *a, **k: None,  # don't run
    )
    client = TestClient(serve.create_app(config=base_config, token="secret"))
    # Seed a finished job directly.
    out = tmp_path / "o.mp4"
    out.write_bytes(b"VID")
    job = serve.Job(id="j1", status="done", workdir=tmp_path, output_path=out)
    serve._JOBS["j1"] = job

    # No auth at all -> 401.
    assert client.get("/api/jobs/j1/result").status_code == 401
    # Query-param token -> ok.
    r = client.get("/api/jobs/j1/result", params={"t": "secret"})
    assert r.status_code == 200
    assert r.content == b"VID"
