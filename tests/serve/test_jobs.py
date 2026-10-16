"""Web UI job hygiene, retention, live progress and the step trace."""

import copy
import io
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from slide_stream import serve  # noqa: E402


def test_job_yaml_written_owner_only(base_config, tmp_path):
    workdir = tmp_path / "job"
    workdir.mkdir()
    job_yaml = serve._build_job_config(base_config, workdir,
                                       {"avatar": True}, None, None)
    assert (job_yaml.stat().st_mode & 0o777) == 0o600


def test_voice_upload_overrides_stored_profile(base_config, tmp_path):
    """An uploaded clip must beat any inherited stored-voice keys."""
    base = copy.deepcopy(base_config)
    base["providers"]["tts"].update({
        "provider": "voicebox", "profile_id": "stored-id",
        "voice": "narrator", "reference_text": "server clip transcript",
    })
    workdir = tmp_path / "job"
    workdir.mkdir()
    voice = workdir / "voice.wav"
    voice.write_bytes(b"a")
    cfg = yaml.safe_load(serve._build_job_config(
        base, workdir, {"avatar": True}, voice, None).read_text())
    tts = cfg["providers"]["tts"]
    assert tts["voice_sample"] == str(voice)
    assert "profile_id" not in tts
    assert "voice" not in tts
    assert "reference_text" not in tts


def test_render_artifacts_wiped_after_job(base_config, tmp_path, monkeypatch):
    """After a render only output.mp4 remains: no deck, job.yaml, or tmp/."""
    workdir = tmp_path / "job"
    workdir.mkdir()
    deck = workdir / "deck.md"
    deck.write_text("# One\n- a\n")
    voice = workdir / "voice.wav"
    voice.write_bytes(b"a")
    job_yaml = serve._build_job_config(base_config, workdir,
                                       {"avatar": True}, voice, None)
    (workdir / "tmp").mkdir()

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.returncode = 0
            self.stdout = io.StringIO("rendered ok\n")
            Path(cmd[cmd.index("create") + 2]).write_bytes(b"MP4")

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(serve.subprocess, "Popen", FakePopen)
    job = serve.Job(id="j1", workdir=workdir, created_at=time.time())
    serve._run_job(job, deck, job_yaml, voice, None)

    assert job.status == "done"
    assert (workdir / "output.mp4").exists()
    assert not job_yaml.exists()  # held live API keys
    assert not deck.exists()
    assert not voice.exists()
    assert not (workdir / "tmp").exists()


def test_reaper_evicts_expired_jobs(tmp_path):
    workdir = tmp_path / "job"
    workdir.mkdir()
    now = time.time()
    serve._JOBS["old"] = serve.Job(
        id="old", status="done", workdir=workdir,
        created_at=now - serve.JOB_TTL_SECONDS - 10,
        completed_at=now - serve.JOB_TTL_SECONDS - 10)
    serve._JOBS["new"] = serve.Job(id="new", status="done", created_at=now, completed_at=now)
    serve._JOBS["run"] = serve.Job(
        id="run", status="running",
        created_at=now - serve.JOB_TTL_SECONDS - 10)
    serve._reap_expired_jobs(now=now)
    assert "old" not in serve._JOBS
    assert not workdir.exists()
    assert "new" in serve._JOBS
    assert "run" in serve._JOBS


@pytest.mark.parametrize("status", ["queued", "running"])
def test_reaper_never_removes_active_work(tmp_path, status):
    job = serve.Job(id="active", status=status, workdir=tmp_path, created_at=1)
    serve._JOBS[job.id] = job
    serve._reap_expired_jobs(now=serve.JOB_TTL_SECONDS + 20000)
    assert tmp_path.is_dir()
    assert job.id in serve._JOBS


@pytest.mark.parametrize("runner", ["_run_job", "_run_video_pipeline", "_run_project_enrich"])
@pytest.mark.parametrize("returncode", [0, 1])
def test_finished_jobs_get_full_retention_after_long_wait(tmp_path, monkeypatch, runner, returncode):
    workdir = tmp_path / "job"
    workdir.mkdir()
    deck = workdir / "deck.md"
    deck.write_text("# Slide\n")
    job_yaml = workdir / "job.yaml"
    job_yaml.write_text("{}")
    now = serve.JOB_TTL_SECONDS + 10000
    monkeypatch.setattr(serve.time, "time", lambda: now)

    def process(command, job, timeout):
        if "enrich" in command:
            out = Path(command[command.index("enrich") + 2])
            out.mkdir(exist_ok=True)
            (out / "deck.md").write_text("# Slide\n")
            (out.parent / f"{out.name}.zip").write_bytes(b"ZIP")
        else:
            Path(command[command.index("create") + 2]).write_bytes(b"MP4")
        return returncode

    monkeypatch.setattr(serve, "_stream_process", process)
    job = serve.Job(id="long", workdir=workdir, created_at=1)
    serve._JOBS[job.id] = job
    if runner == "_run_project_enrich":
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project = serve.Project(id="p", workdir=project_dir)
        serve._run_project_enrich(job, project, deck, job_yaml, None)
    elif runner == "_run_video_pipeline":
        serve._run_video_pipeline(job, deck, job_yaml, None, None)
    else:
        serve._run_job(job, deck, job_yaml, None, None)
    assert job.status == ("done" if returncode == 0 else "error")
    assert job.started_at == now
    assert job.completed_at == now
    serve._reap_expired_jobs(now=now + serve.JOB_TTL_SECONDS - 1)
    assert job.id in serve._JOBS and workdir.is_dir()
    serve._reap_expired_jobs(now=now + serve.JOB_TTL_SECONDS + 1)
    assert job.id not in serve._JOBS and not workdir.exists()


def test_settings_template_mentions_voicebox():
    assert "voicebox" in serve.SETTINGS_TEMPLATE
    for key in ("base_url", "profile_id", "voice_sample", "engine"):
        assert key in serve.SETTINGS_TEMPLATE


# --- live progress: streamed logs, heartbeat, demo-safe progress ---------------


def test_stream_process_updates_log_and_heartbeat(base_config, tmp_path, monkeypatch):
    """Output arrives line-by-line while the process runs, refreshing
    job.log and job.updated_at — the render looks alive from the outside."""
    workdir = tmp_path / "job"
    workdir.mkdir()
    deck = workdir / "deck.md"
    deck.write_text("# One\n- a\n")
    output = workdir / "output.mp4"
    output.write_bytes(b"MP4")

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            self.returncode = 0
            self.stdout = io.StringIO(
                "Slide 1/2: One\n  - Generated audio with gTTS\nSlide 2/2: Two\n"
            )

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(serve.subprocess, "Popen", FakePopen)
    job = serve.Job(id="j1", workdir=workdir, created_at=time.time())
    code = serve._stream_process(["create"], job, timeout=60)

    assert code == 0
    assert "Slide 2/2" in job.log
    assert job.updated_at > 0
    progress = serve._job_progress(job)
    assert progress is not None
    assert progress["slide"] == 2
    assert progress["slides"] == 2
    assert progress["stage"] == "generating voice"


def test_job_progress_stitching_stage():
    job = serve.Job(id="j", created_at=time.time())
    job.log = "Slide 1/3: A\nSlide 2/3: B\nSlide 3/3: C\n2. Combining Video Fragments...\n"
    progress = serve._job_progress(job)
    assert progress == {
        "slide": 3, "slides": 3, "stage": "stitching the final video",
    }


def test_job_progress_heartbeat_line():
    """A long GPU render prints heartbeats; the stage reads 'rendering the
    presenter' even mid-slide."""
    job = serve.Job(id="j", created_at=time.time())
    job.log = "Slide 4/10: Deep dive\n  - wan-s2v talking head: rendering… 190s elapsed\n"
    progress = serve._job_progress(job)
    assert progress is not None
    assert progress["slide"] == 4
    assert progress["stage"] == "rendering the presenter"


def test_job_progress_none_for_empty_or_unrecognised():
    job = serve.Job(id="j", created_at=time.time())
    assert serve._job_progress(job) is None
    job.log = "hello world\nnothing useful here\n"
    assert serve._job_progress(job) is None


def test_status_exposes_progress_and_idle_but_not_raw_log_in_demo(base_config):
    """Demo clients get derived progress + heartbeat, never raw log text."""
    serve._JOBS["j1"] = serve.Job(
        id="j1", status="running", log="Slide 1/5: A\n",
        created_at=time.time() - 30, updated_at=time.time() - 4,
        download_token="dltok",
    )
    client = TestClient(serve.create_app(config=base_config, demo=True))
    s = client.get("/api/jobs/j1").json()
    assert s["progress"] == {"slide": 1, "slides": 5}
    assert s["idle"] == 4
    assert s["log"] == ""  # raw text still withheld on the open demo
    assert "token" not in s


def test_status_idle_uses_created_at_before_output(base_config):
    """No output yet -> the heartbeat counts from job creation."""
    serve._JOBS["j1"] = serve.Job(
        id="j1", status="running", log="",
        created_at=time.time() - 12,
    )
    client = TestClient(serve.create_app(config=base_config, token=None))
    s = client.get("/api/jobs/j1").json()
    assert s["idle"] == 12
    assert "progress" not in s  # nothing recognisable printed yet


# --- demo-safe step trace -------------------------------------------------------


def test_job_trace_parses_providers_and_fallbacks():
    job = serve.Job(id="j", created_at=time.time())
    job.log = (
        "Found 2 slides.\n"
        "Slide 1/2: Coffee\n"
        "  - Generated SwarmUI image: coffee beans\n"
        "  - Generated audio with voicebox\n"
        "  - Generated wan-s2v talking head (slide 1)\n"
        "Slide 2/2: Tea\n"
        "  - SwarmUI error: timeout. Using text fallback.\n"
        "  - Generated audio with voicebox\n"
        "  - wan-s2v avatar error: GPU OOM\n"
        "2. Combining Video Fragments...\n"
    )
    trace = serve._job_trace(job)
    assert trace == [
        "slide 1: image via Swarmui",
        "slide 1: voice via voicebox",
        "slide 1: presenter animated (wan-s2v)",
        "slide 2: image failed — used a text card",
        "slide 2: voice via voicebox",
        "slide 2: presenter failed (wan-s2v)",
        "stitching slides together",
    ]


def test_job_warnings_include_text_fallback_count():
    job = serve.Job(id="j", created_at=time.time())
    job.log = (
        "Slide 1/3: A\n  - SwarmUI error: x. Using text fallback.\n"
        "Slide 2/3: B\n  - SwarmUI error: y. Using text fallback.\n"
        "Slide 3/3: C\n  - Generated Imagen image: z\n"
    )
    warnings = serve._job_warnings(job)
    assert any("2 of 3 slides used a plain text card" in w for w in warnings)


def test_status_exposes_trace_in_demo(base_config):
    serve._JOBS["j1"] = serve.Job(
        id="j1", status="running",
        log="Slide 1/2: A\n  - Generated SwarmUI image: x\n",
        created_at=time.time(), updated_at=time.time(),
    )
    client = TestClient(serve.create_app(config=base_config, demo=True))
    s = client.get("/api/jobs/j1").json()
    assert s["trace"] == ["slide 1: image via Swarmui"]
    assert s["log"] == ""  # raw text still withheld


def test_status_exposes_warnings_in_demo(base_config):
    serve._JOBS["j1"] = serve.Job(
        id="j1", status="done",
        log="Slide 1/1: A\n  - SwarmUI error: down. Using text fallback.\n",
        created_at=time.time(),
    )
    client = TestClient(serve.create_app(config=base_config, demo=True))
    s = client.get("/api/jobs/j1").json()
    assert any("text card" in w for w in s["warnings"])


# --- Project workflow (draft -> edit -> enrich -> render) ---------------------
