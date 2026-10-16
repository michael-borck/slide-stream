"""Web UI project workflow: draft, edit, enrich, render."""

import copy
import io
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from slide_stream import serve  # noqa: E402
from slide_stream.config_loader import DEFAULT_CONFIG  # noqa: E402


def _llm_config():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["providers"]["llm"]["provider"] = "openai"
    return cfg


def test_create_empty_project_and_state():
    client = TestClient(serve.create_app(config=_llm_config(), token=None))
    r = client.post("/api/projects")
    assert r.status_code == 200, r.text
    body = r.json()
    pid, token = body["project_id"], body["token"]
    assert token
    assert body["state"]["has_deck"] is False

    # State needs the per-project token, not just the id.
    assert client.get(f"/api/projects/{pid}").status_code == 401
    ok = client.get(f"/api/projects/{pid}", headers={"X-Project-Token": token})
    assert ok.status_code == 200
    assert ok.json()["has_deck"] is False


def test_create_project_with_deck_upload():
    client = TestClient(serve.create_app(config=_llm_config(), token=None))
    r = client.post(
        "/api/projects",
        files={"deck": ("deck.md", b"# One\n\n- a\n# Two\n\n- b\n", "text/markdown")},
    )
    assert r.status_code == 200, r.text
    state = r.json()["state"]
    assert state["has_deck"] is True
    assert state["slide_count"] == 2
    assert state["deck_format"] == "md"


def test_project_draft_writes_deck(monkeypatch):
    monkeypatch.setattr(
        serve, "_do_draft",
        lambda *a, **k: "# Intro\n\n- point\n# Details\n\n- more\n",
    )
    client = TestClient(serve.create_app(config=_llm_config(), token=None))
    pid = client.post("/api/projects").json()["project_id"]
    token = serve._PROJECTS[pid].token

    r = client.post(
        f"/api/projects/{pid}/draft",
        files={"source": ("report.pdf", b"%PDF-fake", "application/pdf")},
        data={"slides": "2"},
        headers={"X-Project-Token": token},
    )
    assert r.status_code == 200, r.text
    assert "# Intro" in r.json()["markdown"]
    assert r.json()["state"]["slide_count"] == 2


def test_project_draft_requires_llm_provider():
    # DEFAULT_CONFIG has llm provider 'none' -> draft must refuse cleanly.
    client = TestClient(serve.create_app(config=copy.deepcopy(DEFAULT_CONFIG), token=None))
    pid = client.post("/api/projects").json()["project_id"]
    token = serve._PROJECTS[pid].token
    r = client.post(
        f"/api/projects/{pid}/draft",
        files={"source": ("a.txt", b"hi", "text/plain")},
        headers={"X-Project-Token": token},
    )
    assert r.status_code == 400
    assert "LLM" in r.json()["detail"]


def test_project_save_deck_validates():
    client = TestClient(serve.create_app(config=_llm_config(), token=None))
    pid = client.post("/api/projects").json()["project_id"]
    token = serve._PROJECTS[pid].token

    # Prose with no headings is not a deck.
    bad = client.put(
        f"/api/projects/{pid}/deck",
        json={"markdown": "just prose"},
        headers={"X-Project-Token": token},
    )
    assert bad.status_code == 400

    good = client.put(
        f"/api/projects/{pid}/deck",
        json={"markdown": "# A\n\n- x\n# B\n\n- y\n"},
        headers={"X-Project-Token": token},
    )
    assert good.status_code == 200
    assert good.json()["state"]["slide_count"] == 2


def test_project_render_spawns_job(monkeypatch):
    def fake_run_job(job, deck_path, job_yaml, voice_path, photo_path,
                     mode="video", notes=None):
        assert job.workdir is not None
        assert deck_path.exists()  # the project deck was copied in
        out = job.workdir / "output.mp4"
        out.write_bytes(b"VID")
        job.status = "done"
        job.output_path = out

    monkeypatch.setattr(serve, "_run_job", fake_run_job)
    monkeypatch.setattr(
        serve.ThreadPoolExecutor, "submit", lambda self, fn, *a, **k: fn(*a, **k)
    )
    client = TestClient(serve.create_app(config=_llm_config(), token=None))
    pid = client.post(
        "/api/projects",
        files={"deck": ("deck.md", b"# One\n\n- a\n", "text/markdown")},
    ).json()["project_id"]
    token = serve._PROJECTS[pid].token

    r = client.post(
        f"/api/projects/{pid}/render",
        data={"avatar": "false"},
        headers={"X-Project-Token": token},
    )
    assert r.status_code == 200, r.text
    job_id, jtok = r.json()["job_id"], r.json()["token"]
    result = client.get(f"/api/jobs/{job_id}/result", params={"t": jtok})
    assert result.status_code == 200
    assert result.content == b"VID"
    # The canonical project deck survives the render (was copied, not consumed).
    assert (serve._PROJECTS[pid].workdir / "deck.md").exists()


def test_project_enrich_spawns_enrich_job(monkeypatch):
    captured = {}

    def fake_enrich(job, project, deck_path, job_yaml, notes):
        captured["notes"] = notes
        captured["project_id"] = project.id
        assert job.workdir is not None
        out = job.workdir / "enriched.zip"
        out.write_bytes(b"ZIP")
        job.status = "done"
        job.output_path = out

    monkeypatch.setattr(serve, "_run_project_enrich", fake_enrich)
    monkeypatch.setattr(
        serve.ThreadPoolExecutor, "submit", lambda self, fn, *a, **k: fn(*a, **k)
    )
    client = TestClient(serve.create_app(config=_llm_config(), token=None))
    pid = client.post(
        "/api/projects",
        files={"deck": ("deck.md", b"# One\n\n- a\n", "text/markdown")},
    ).json()["project_id"]
    token = serve._PROJECTS[pid].token

    r = client.post(
        f"/api/projects/{pid}/enrich",
        data={"image_provider": "text", "notes": "all"},
        headers={"X-Project-Token": token},
    )
    assert r.status_code == 200, r.text
    assert captured["notes"] == "all"
    assert captured["project_id"] == pid


def test_project_workflow_blocked_in_demo():
    client = TestClient(serve.create_app(config=_llm_config(), demo=True))
    assert client.post("/api/projects").status_code == 403


def test_reap_expired_projects(tmp_path):
    p = serve.Project(id="old", workdir=tmp_path, created_at=1.0, token="t")
    serve._PROJECTS["old"] = p
    marker = tmp_path / "deck.md"
    marker.write_text("# A\n")
    serve._reap_expired_projects(now=1.0 + serve.PROJECT_TTL_SECONDS + 1)
    assert "old" not in serve._PROJECTS


@pytest.mark.parametrize("edited", [False, True])
def test_run_project_enrich_populates_project_images(tmp_path, monkeypatch, edited):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    proj = serve.Project(id="p", workdir=proj_dir, created_at=time.time(), token="t")
    job = serve.Job(id="j", workdir=job_dir, created_at=time.time())
    deck = job_dir / "deck.md"
    deck.write_text("# A\n")
    (proj_dir / "deck.md").write_text("# A\n")
    jy = job_dir / "job.yaml"
    jy.write_text("{}")

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            self.returncode = 0
            self.stdout = io.StringIO("enriched\n")
            images = job_dir / "enriched" / "images"
            images.mkdir(parents=True)
            (images / "slide_1.png").write_bytes(b"PNG")
            (images.parent / "deck.md").write_text("# A\n![](images/slide_1.png)\n")
            (job_dir / "enriched.zip").write_bytes(b"ZIP")
            if edited:
                (proj_dir / "deck.md").write_text("# Changed during enrichment\n")

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(serve.subprocess, "Popen", FakePopen)
    serve._run_project_enrich(job, proj, deck, jy, None)
    assert job.status == "done"
    if edited:
        assert not (proj_dir / "images").exists()
    else:
        assert (proj_dir / "images" / "slide_1.png").read_bytes() == b"PNG"


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("suffix", [".md", ".pptx"])
def test_project_enrich_render_reuses_only_matching_revision(base_config, monkeypatch, tmp_path, edit, suffix):
    """Exercise the API chain and CLI image selection, preserving PPTX notes."""
    from PIL import Image
    from pptx import Presentation
    from typer.testing import CliRunner

    from slide_stream.cli import app as cli_app
    from slide_stream.export_pptx import export_pptx
    from slide_stream.providers.images import TextImageProvider

    monkeypatch.chdir(tmp_path)
    if suffix == ".pptx":
        original = tmp_path / "source.pptx"
        export_pptx([{"title": "A", "content": ["Bullet"], "notes": "Original notes."}], original)
        source = original.read_bytes()
    else:
        source = b"# A\n- Bullet\n"

    def fake_process(command, job, timeout):
        out = Path(command[command.index("enrich") + 2])
        (out / "images").mkdir(parents=True)
        Image.new("RGB", (16, 16), "red").save(out / "images" / "slide_1.png")
        (out / "deck.md").write_text("# A\n![](images/slide_1.png)\n- Bullet\n")
        (out.parent / f"{out.name}.zip").write_bytes(b"ZIP")
        return 0

    seen = {}

    def render(job, deck_path, job_yaml, *args):
        # PPTX output exercises the same prebuilt image selection as video.
        output = tmp_path / "rendered.pptx"
        result = CliRunner().invoke(cli_app, ["create", str(deck_path), str(output), "--config", str(job_yaml)])
        assert result.exit_code == 0, result.output
        prs = Presentation(str(output))
        if suffix == ".pptx" and not edit:
            notes = prs.slides[0].notes_slide.notes_text_frame
            assert notes is not None and "Original notes." in notes.text
        seen["rendered"] = True

    generated = []
    original_generate = TextImageProvider.generate_image

    def generate(self, query, filename, slide=None):
        generated.append(query)
        return original_generate(self, query, filename, slide)

    monkeypatch.setattr(TextImageProvider, "generate_image", generate)
    monkeypatch.setattr(serve, "_stream_process", fake_process)
    monkeypatch.setattr(serve, "_run_job", render)
    monkeypatch.setattr(serve.ThreadPoolExecutor, "submit", lambda self, fn, *a, **k: fn(*a, **k))
    client = TestClient(serve.create_app(config=base_config))
    project = client.post("/api/projects", files={"deck": (f"deck{suffix}", source)}).json()
    url = f"/api/projects/{project['project_id']}"
    headers = {"X-Project-Token": project["token"]}
    assert client.post(url + "/enrich", headers=headers).status_code == 200
    # The UI saves even unchanged Markdown before rendering.
    if suffix == ".md" or edit:
        markdown = "# Changed\n- New content\n" if edit else source.decode()
        assert client.put(url + "/deck", headers=headers, json={"markdown": markdown}).status_code == 200
    assert client.post(url + "/render", headers=headers).status_code == 200
    assert seen["rendered"]
    assert len(generated) == (1 if edit else 0)


def test_project_image_thumbnail_endpoint():
    client = TestClient(serve.create_app(config=_llm_config(), token=None))
    pid = client.post("/api/projects").json()["project_id"]
    project = serve._PROJECTS[pid]
    (project.workdir / "images").mkdir()
    (project.workdir / "images" / "slide_1.png").write_bytes(b"\x89PNGdata")

    # Wrong/absent token is rejected; the ?t= token works (for <img> tags).
    assert client.get(f"/api/projects/{pid}/images/slide_1.png").status_code == 401
    r = client.get(f"/api/projects/{pid}/images/slide_1.png", params={"t": project.token})
    assert r.status_code == 200
    assert r.content == b"\x89PNGdata"
    # Path traversal is neutralised.
    assert client.get(f"/api/projects/{pid}/images/nope.png",
                      params={"t": project.token}).status_code == 404


def test_project_render_includes_enriched_images(monkeypatch):
    seen = {}

    def fake_run_job(job, deck_path, job_yaml, voice_path, photo_path,
                     mode="video", notes=None):
        assert job.workdir is not None
        seen["has_images"] = (job.workdir / "images" / "slide_1.png").exists()
        out = job.workdir / "output.mp4"
        out.write_bytes(b"VID")
        job.status = "done"
        job.output_path = out

    monkeypatch.setattr(serve, "_run_job", fake_run_job)
    monkeypatch.setattr(
        serve.ThreadPoolExecutor, "submit", lambda self, fn, *a, **k: fn(*a, **k)
    )
    client = TestClient(serve.create_app(config=_llm_config(), token=None))
    pid = client.post(
        "/api/projects",
        files={"deck": ("deck.md", b"# One\n\n![](images/slide_1.png)\n\n- a\n", "text/markdown")},
    ).json()["project_id"]
    project = serve._PROJECTS[pid]
    (project.workdir / "images").mkdir()
    (project.workdir / "images" / "slide_1.png").write_bytes(b"PNG")

    r = client.post(f"/api/projects/{pid}/render", data={"avatar": "false"},
                    headers={"X-Project-Token": project.token})
    assert r.status_code == 200, r.text
    assert seen["has_images"] is True  # enriched images travelled with the deck


def test_build_job_config_applies_avatar_placement_and_transition(tmp_path):
    base = copy.deepcopy(DEFAULT_CONFIG)
    options = {
        "avatar_slides": "first,last",
        "reuse_avatar": True,
        "transition_seconds": "0.4",
    }
    p = serve._build_job_config(base, tmp_path, options, None, None)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert cfg["settings"]["avatar"]["slides"] == "first,last"
    assert cfg["settings"]["avatar"]["reuse_clip"] is True
    assert cfg["settings"]["video"]["transition_seconds"] == 0.4


def test_project_render_passes_placement_and_transition(monkeypatch):
    from pathlib import Path

    captured = {}

    def fake_run_job(job, deck_path, job_yaml, voice, photo, mode="video", notes=None):
        captured["cfg"] = yaml.safe_load(Path(job_yaml).read_text(encoding="utf-8"))
        assert job.workdir is not None
        out = job.workdir / "output.mp4"
        out.write_bytes(b"V")
        job.status = "done"
        job.output_path = out

    monkeypatch.setattr(serve, "_run_job", fake_run_job)
    monkeypatch.setattr(
        serve.ThreadPoolExecutor, "submit", lambda self, fn, *a, **k: fn(*a, **k)
    )
    client = TestClient(serve.create_app(config=_llm_config(), token=None))
    pid = client.post(
        "/api/projects",
        files={"deck": ("deck.md", b"# A\n\n- a\n", "text/markdown")},
    ).json()["project_id"]
    token = serve._PROJECTS[pid].token

    r = client.post(
        f"/api/projects/{pid}/render",
        data={"avatar": "false", "avatar_slides": "first",
              "reuse_avatar": "true", "transition": "0.5"},
        headers={"X-Project-Token": token},
    )
    assert r.status_code == 200, r.text
    cfg = captured["cfg"]
    assert cfg["settings"]["avatar"]["slides"] == "first"
    assert cfg["settings"]["avatar"]["reuse_clip"] is True
    assert cfg["settings"]["video"]["transition_seconds"] == 0.5


def test_job_trace_handles_enrich_flow():
    """The enrich flow prints the same slide markers plus a notes event."""
    job = serve.Job(id="j", created_at=time.time())
    job.log = (
        "Slide 1/2: Coffee\n"
        "  - Generated SwarmUI image: coffee beans\n"
        "  - Wrote AI speaker notes\n"
        "Slide 2/2: Tea\n"
        "  - Generated SwarmUI image: tea leaves\n"
    )
    assert serve._job_trace(job) == [
        "slide 1: image via Swarmui",
        "slide 1: notes written",
        "slide 2: image via Swarmui",
    ]
