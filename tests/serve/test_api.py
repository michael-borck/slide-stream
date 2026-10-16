"""Web UI core: index, auth, jobs, job config and downloads."""

from io import BytesIO

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from slide_stream import serve  # noqa: E402


def _tiny_png() -> bytes:
    """A real (decodable) 4x4 PNG for photo uploads."""
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
    return buf.getvalue()


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
    cfg = client.get("/api/config").json()
    assert cfg["demo"] is True
    assert cfg["version"] == serve.__version__
    off = TestClient(serve.create_app(config=base_config, token=None, demo=False))
    assert off.get("/api/config").json()["demo"] is False


def test_create_job_rejects_bad_deck_type(base_config):
    client = TestClient(serve.create_app(config=base_config, token=None))
    r = client.post("/api/jobs", files={"deck": ("notes.docx", b"hi", "text/plain")})
    assert r.status_code == 400


def test_job_lifecycle_and_download(base_config, tmp_path, monkeypatch):
    """Submit a deck; the (mocked) render produces a video that downloads."""

    def fake_run_job(job, deck_path, job_yaml, voice_path, photo_path,
                     mode="video", notes=None):
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
            "photo": ("me.png", _tiny_png(), "image/png"),
        },
        data={"avatar": "true", "narration_seconds": "30"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    token = r.json()["token"]
    assert token  # per-job download token minted at creation

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "done"
    assert status["token"] == token  # non-demo status echoes it for the UI

    # The download needs the per-job token, not just the job id.
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 401
    result = client.get(f"/api/jobs/{job_id}/result", params={"t": token})
    assert result.status_code == 200
    assert result.content == b"FAKEMP4"


def test_api_check_returns_doctor_report(base_config):
    """The preflight endpoint parses the deck + config and returns findings."""
    client = TestClient(serve.create_app(config=base_config, token=None))
    r = client.post(
        "/api/check",
        files={"deck": ("deck.md", b"# One\n\n- a\n\n# Two\n\n- b\n", "text/markdown")},
        data={"output": "video"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "findings" in body and "estimates" in body
    assert any("2 slide(s)" in f["message"] for f in body["findings"])
    assert any("Video length" in e for e in body["estimates"])


def test_pptx_output_runs_enrich(base_config, monkeypatch):
    """Output=pptx routes the job through enrich (zip), not create (mp4)."""
    seen = {}

    def fake_run_job(job, deck_path, job_yaml, voice_path, photo_path,
                     mode="video", notes=None):
        seen["mode"] = mode
        seen["notes"] = notes
        assert job.workdir is not None
        out = job.workdir / "enriched.zip"
        out.write_bytes(b"PK\x03\x04zip")
        job.status = "done"
        job.output_path = out
        job.media_type = "application/zip"
        job.download_name = "slidestream-deck.zip"

    monkeypatch.setattr(serve, "_run_job", fake_run_job)
    monkeypatch.setattr(
        serve.ThreadPoolExecutor, "submit", lambda self, fn, *a, **k: fn(*a, **k)
    )
    client = TestClient(serve.create_app(config=base_config, token=None))
    r = client.post(
        "/api/jobs",
        files={"deck": ("deck.md", b"# One\n\n- a\n", "text/markdown")},
        data={"output": "pptx", "notes": "all"},
    )
    assert r.status_code == 200
    assert seen == {"mode": "pptx", "notes": "all"}
    token = r.json()["token"]
    job_id = r.json()["job_id"]
    result = client.get(f"/api/jobs/{job_id}/result", params={"t": token})
    assert result.status_code == 200
    assert result.content.startswith(b"PK")  # a zip, not an mp4


def test_pptx_rejects_bad_notes_mode(base_config):
    client = TestClient(serve.create_app(config=base_config, token=None))
    r = client.post(
        "/api/jobs",
        files={"deck": ("deck.md", b"# One\n\n- a\n", "text/markdown")},
        data={"output": "pptx", "notes": "bogus"},
    )
    assert r.status_code == 400


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

    cfg = yaml.safe_load(job_yaml.read_text())
    assert cfg["providers"]["tts"]["voice_sample"] == str(voice)
    assert cfg["providers"]["avatar"]["source_image"] == str(photo)
    assert cfg["providers"]["images"]["provider"] == "swarmui"
    assert cfg["settings"]["narration"]["target_seconds"] == 25.0


def test_job_config_photo_not_animated_becomes_static_self(base_config, tmp_path):
    """Photo + animate off = a still of themselves, not 'no head'."""
    workdir = tmp_path / "job"
    workdir.mkdir()
    photo = workdir / "photo.png"
    photo.write_bytes(b"b")
    job_yaml = serve._build_job_config(
        base_config, workdir, {"avatar": False}, None, photo
    )

    cfg = yaml.safe_load(job_yaml.read_text())
    assert cfg["providers"]["avatar"]["provider"] == "static"
    assert cfg["providers"]["avatar"]["source"] == str(photo)


def test_job_config_no_presenter_means_no_head(base_config, tmp_path):
    workdir = tmp_path / "job"
    workdir.mkdir()
    job_yaml = serve._build_job_config(base_config, workdir, {"avatar": True}, None, None)

    cfg = yaml.safe_load(job_yaml.read_text())
    assert cfg["providers"]["avatar"]["provider"] == "none"


def test_job_config_mascot_animate_toggle(base_config, tmp_path):
    """Mascot: without a usable animation engine, animate falls back to a
    static mascot — the web UI never produces the puppet mouth-flap."""

    workdir = tmp_path / "job"
    workdir.mkdir()
    on = yaml.safe_load(serve._build_job_config(
        base_config, workdir, {"avatar": True, "avatar_name": "teddy"}, None, None
    ).read_text())
    assert on["providers"]["avatar"]["provider"] == "static"
    assert on["providers"]["avatar"]["source"] == "teddy"
    off = yaml.safe_load(serve._build_job_config(
        base_config, workdir, {"avatar": False, "avatar_name": "teddy"}, None, None
    ).read_text())
    assert off["providers"]["avatar"]["provider"] == "static"


def test_job_config_mascot_animates_via_wan_s2v_when_configured(base_config, tmp_path):
    """With a detector-free engine configured, an animated mascot lip-syncs
    for real (wan-s2v) instead of falling back to the puppet mouth-flap."""

    base_config["providers"]["avatar"] = {
        "provider": "wan-s2v", "base_url": "https://comfy.example.org",
        "api_key": "tok",
    }
    workdir = tmp_path / "job"
    workdir.mkdir()
    cfg = yaml.safe_load(serve._build_job_config(
        base_config, workdir, {"avatar": True, "avatar_name": "owl"}, None, None
    ).read_text())
    av = cfg["providers"]["avatar"]
    assert av["provider"] == "wan-s2v"
    assert av["source"] == "owl"
    assert av["base_url"] == "https://comfy.example.org"  # server connection kept
    # Animate off still holds the mascot as a static image, engine or not.
    off = yaml.safe_load(serve._build_job_config(
        base_config, workdir, {"avatar": False, "avatar_name": "owl"}, None, None
    ).read_text())
    assert off["providers"]["avatar"]["provider"] == "static"


def test_job_config_mascot_falls_back_to_static_for_human_only_engine(base_config, tmp_path):
    """A human-only engine (sadtalker) can't animate a mascot, so an
    animated mascot degrades to a still image rather than the puppet
    mouth-flap (or a doomed SadTalker render)."""

    base_config["providers"]["avatar"] = {
        "provider": "sadtalker", "base_url": "https://comfy.example.org",
    }
    workdir = tmp_path / "job"
    workdir.mkdir()
    cfg = yaml.safe_load(serve._build_job_config(
        base_config, workdir, {"avatar": True, "avatar_name": "teddy"}, None, None
    ).read_text())
    assert cfg["providers"]["avatar"]["provider"] == "static"


def test_job_config_video_always_animates(base_config, tmp_path):
    """A video presenter uses the video engine even with animate off."""

    workdir = tmp_path / "job"
    workdir.mkdir()
    clip = workdir / "me.mp4"
    clip.write_bytes(b"v")
    cfg = yaml.safe_load(serve._build_job_config(
        base_config, workdir, {"avatar": False}, None, clip
    ).read_text())
    assert cfg["providers"]["avatar"].get("provider") != "static"
    assert cfg["providers"]["avatar"]["source_video"] == str(clip)


def test_job_config_video_routes_away_from_wan_s2v(base_config, tmp_path):
    """With the server engine wan-s2v (a still-image model), an uploaded
    video presenter routes through the auto provider (video -> Wav2Lip)."""

    base_config["providers"]["avatar"] = {
        "provider": "wan-s2v", "base_url": "https://comfy.example.org",
    }
    workdir = tmp_path / "job"
    workdir.mkdir()
    clip = workdir / "me.mp4"
    clip.write_bytes(b"v")
    cfg = yaml.safe_load(serve._build_job_config(
        base_config, workdir, {"avatar": True}, None, clip
    ).read_text())
    av = cfg["providers"]["avatar"]
    assert av["provider"] == "comfyui"
    assert av["source_video"] == str(clip)
    assert av["base_url"] == "https://comfy.example.org"  # same server kept


def test_download_uses_per_job_token(base_config, tmp_path):
    """A browser download link (no header) authenticates via a per-job ?t=."""
    client = TestClient(serve.create_app(config=base_config, token="secret"))
    # Seed a finished job directly.
    out = tmp_path / "o.mp4"
    out.write_bytes(b"VID")
    job = serve.Job(id="j1", status="done", workdir=tmp_path, output_path=out,
                    download_token="dltok")
    serve._JOBS["j1"] = job

    # No auth at all -> 401.
    assert client.get("/api/jobs/j1/result").status_code == 401
    # The long-lived instance token must never ride in a URL: rejected.
    assert client.get("/api/jobs/j1/result", params={"t": "secret"}).status_code == 401
    # The per-job download token -> ok.
    r = client.get("/api/jobs/j1/result", params={"t": "dltok"})
    assert r.status_code == 200
    assert r.content == b"VID"
    # API clients can still use the Authorization header.
    hdr = client.get("/api/jobs/j1/result",
                     headers={"Authorization": "Bearer secret"})
    assert hdr.status_code == 200


# --- demo mode: friction-free but guard-railed --------------------------------
