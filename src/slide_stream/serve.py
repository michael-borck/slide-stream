# FastAPI route handlers are registered via decorators, so pyright's
# unused-function check is a false positive for them.
# pyright: reportUnusedFunction=false
"""Web UI for slide-stream (optional ``[serve]`` extra).

`slide-stream serve` starts a small FastAPI app: upload a deck (+ optional
voice sample and photo), render it as a background job, and download the video.
Token-authenticated so it can run locally or on a VPS.

Design notes:
- The server is **stateless about biometric data**: an uploaded voice sample /
  photo is used only for that render and deleted afterwards. The lecturer's
  browser remembers them (IndexedDB) so they need not re-pick each job — the
  data stays on their laptop, never stored on the server at rest.
- Each render runs as a subprocess (``python -m slide_stream create``) so a
  crash can't take down the server and ffmpeg/moviepy memory is reclaimed.
"""

import copy
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from .config_loader import load_config


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    log: str = ""
    error: str = ""
    workdir: Path | None = None
    output_path: Path | None = None
    created_at: float = field(default_factory=lambda: 0.0)
    # Per-job download secret: knowing the job UUID alone must not be enough
    # to fetch the video, and the long-lived instance token never goes in a URL.
    download_token: str = ""
    # Output media (video render vs enriched-deck zip) for the download response.
    media_type: str = "video/mp4"
    download_name: str = "slidestream.mp4"


# In-memory job registry (single-process v1). job_id -> Job.
_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


@dataclass
class Project:
    """A workflow session: a server-side workdir holding the deck (and, once
    enriched, its images) so the draft -> enrich -> render steps chain without
    re-uploading between each. Unlike a Job, a Project persists across requests;
    it is reaped on a TTL like jobs. Voice/photo inputs are NEVER stored on a
    project — those stay ephemeral, attached per render job."""
    id: str
    workdir: Path
    created_at: float = field(default_factory=lambda: 0.0)
    # Per-project secret: the project id alone must not authorize edits/reads.
    token: str = ""


# In-memory project registry (single-process v1). project_id -> Project.
_PROJECTS: dict[str, Project] = {}


def _shutdown() -> None:  # pragma: no cover - kills the process
    os._exit(0)


# Demo-mode guardrails: friction-free (no token) but bounded.
DEMO_MAX_SLIDES = 5
DEMO_JOBS_PER_HOUR = 3
_DEMO_HITS: dict[str, list[float]] = {}  # client ip -> job timestamps


def _env_int(name: str, default: int) -> int:
    """Integer from the environment, falling back on missing/garbage values."""
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


# Upload caps (bytes), overridable via SLIDESTREAM_MAX_{DECK,VOICE,PHOTO}_MB.
MAX_DECK_BYTES = _env_int("SLIDESTREAM_MAX_DECK_MB", 30) * 1024 * 1024
MAX_VOICE_BYTES = _env_int("SLIDESTREAM_MAX_VOICE_MB", 30) * 1024 * 1024
MAX_PHOTO_BYTES = _env_int("SLIDESTREAM_MAX_PHOTO_MB", 15) * 1024 * 1024
# Largest photo we will hand to Pillow/the avatar engines (per side).
MAX_IMAGE_DIM = 8000
# How long a job (and its workdir) may live before it is reaped.
JOB_TTL_SECONDS = _env_int("SLIDESTREAM_JOB_TTL_MIN", 60) * 60


def _reap_expired_jobs(now: float | None = None) -> None:
    """Evict jobs past the TTL and delete their workdirs.

    Called lazily from the job endpoints so 'nothing stored' holds without a
    dedicated reaper thread. Jobs that are still queued/running get an extra
    hour of grace (the render subprocess timeout) so a live render's files
    are never deleted out from under it.
    """
    t = now if now is not None else time.time()
    expired: list[Job] = []
    with _LOCK:
        for job in list(_JOBS.values()):
            grace = 0 if job.status in ("done", "error") else 3600
            if job.created_at and t - job.created_at > JOB_TTL_SECONDS + grace:
                expired.append(job)
                del _JOBS[job.id]
    for job in expired:
        if job.workdir is not None:
            shutil.rmtree(job.workdir, ignore_errors=True)


# Projects live at least as long as jobs; a workflow spans several requests.
PROJECT_TTL_SECONDS = _env_int("SLIDESTREAM_PROJECT_TTL_MIN", 240) * 60


def _reap_expired_projects(now: float | None = None) -> None:
    """Evict projects past the TTL and delete their workdirs (lazy, like jobs)."""
    t = now if now is not None else time.time()
    expired: list[Project] = []
    with _LOCK:
        for project in list(_PROJECTS.values()):
            if project.created_at and t - project.created_at > PROJECT_TTL_SECONDS:
                expired.append(project)
                del _PROJECTS[project.id]
    for project in expired:
        shutil.rmtree(project.workdir, ignore_errors=True)


def _project_deck(project: Project) -> Path | None:
    """The project's canonical deck file (.md preferred), or None if unset."""
    for name in ("deck.md", "deck.pptx"):
        candidate = project.workdir / name
        if candidate.exists():
            return candidate
    return None


def _project_state(project: Project) -> dict[str, Any]:
    """A JSON-able snapshot of what the project currently holds."""
    deck = _project_deck(project)
    images_dir = project.workdir / "images"
    images = (
        sorted(p.name for p in images_dir.iterdir() if p.is_file())
        if images_dir.is_dir()
        else []
    )
    slide_count = None
    if deck is not None:
        try:
            slide_count = len(_parse_deck_slides(deck))
        except Exception:
            slide_count = None
    return {
        "project_id": project.id,
        "has_deck": deck is not None,
        "deck_format": deck.suffix.lstrip(".") if deck else None,
        "slide_count": slide_count,
        "images": images,
    }


# Origins the local (desktop) server accepts state-changing requests from:
# this machine's own pages, or the Tauri shell's webview.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
_TAURI_ORIGINS = ("tauri://localhost", "http://tauri.localhost",
                  "https://tauri.localhost")


def _local_origin_ok(origin: str) -> bool:
    """True if an Origin header value belongs to this machine (or Tauri)."""
    if origin.lower() in _TAURI_ORIGINS:
        return True
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and (
        (parts.hostname or "") in _LOCAL_HOSTS
    )


def _validate_photo_upload(path: Path) -> str | None:
    """Error message if an uploaded image is undecodable or hostile, else None.

    Pillow's MAX_IMAGE_PIXELS bomb guard stays active; on top of it we bound
    the dimensions so downstream avatar engines get something reasonable.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            width, height = im.size
            if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
                return (
                    f"Image is {width}x{height}; the maximum is "
                    f"{MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}"
                )
            im.verify()
    except Image.DecompressionBombError:
        return "Image is too large to decode safely"
    except (UnidentifiedImageError, OSError, ValueError):
        return "Could not decode the image"
    return None


def _demo_rate_ok(ip: str, now: float | None = None) -> bool:
    """True if this IP may start another demo job (and record the hit)."""
    t = now if now is not None else time.time()
    with _LOCK:
        hits = [h for h in _DEMO_HITS.get(ip, []) if t - h < 3600]
        if len(hits) >= DEMO_JOBS_PER_HOUR:
            _DEMO_HITS[ip] = hits
            return False
        hits.append(t)
        _DEMO_HITS[ip] = hits
        return True


def _count_slides(deck_path: Path) -> int | None:
    """Best-effort slide count for the demo cap; None if unparseable."""
    try:
        if deck_path.suffix.lower() == ".pptx":
            from pptx import Presentation  # type: ignore[import-untyped]

            return len(Presentation(str(deck_path)).slides)
        from .parser import parse_markdown

        return len(parse_markdown(deck_path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _parse_deck_slides(deck_path: Path) -> list[dict[str, Any]]:
    """Parse a deck to slide dicts for the doctor preflight (.md or .pptx)."""
    if deck_path.suffix.lower() == ".pptx":
        from .powerpoint import parse_powerpoint

        return parse_powerpoint(deck_path)
    from .parser import parse_markdown

    return parse_markdown(deck_path.read_text(encoding="utf-8"))


# Remote engines that animate a stylized mascot (no human face to detect):
# Wan2.2-S2V has no detector at all, and D-ID handles stylized faces. The
# others (sadtalker/wav2lip/comfyui-auto) start with a human face detector and
# fail on a mascot, so a mascot presenter falls back to the no-GPU puppet there.
_MASCOT_ANIMATE_ENGINES = ("wan-s2v", "d-id")


def _server_animation_engine(base: dict[str, Any]) -> str | None:
    """The server-configured animated-avatar engine, if one is usable.

    Returns the base config's avatar provider name when it selects a remote
    engine that has its connection details set (a ComfyUI ``base_url``, or a
    D-ID ``api_key``), else None. Used to decide whether an animated presenter
    can lip-sync for real or must fall back to the no-GPU puppet mouth-flap.
    """
    av = base.get("providers", {}).get("avatar", {})
    provider = av.get("provider")
    if provider in ("wan-s2v", "sadtalker", "wav2lip", "comfyui"):
        return provider if av.get("base_url") else None
    if provider == "d-id":
        return provider if av.get("api_key") else None
    return None


def _build_job_config(base: dict[str, Any], workdir: Path, options: dict[str, Any],
                      voice_path: Path | None, photo_path: Path | None) -> Path:
    """Write a per-job config YAML: server base + this job's overrides."""
    cfg = copy.deepcopy(base)
    cfg.setdefault("providers", {}).setdefault("tts", {})
    cfg["providers"].setdefault("images", {})
    cfg["providers"].setdefault("avatar", {})
    cfg.setdefault("settings", {})

    # Keep renders self-contained inside the job dir.
    cfg["settings"]["temp_dir"] = str(workdir / "tmp")

    if options.get("narration_seconds"):
        cfg["settings"].setdefault("narration", {})["target_seconds"] = float(
            options["narration_seconds"]
        )
    if options.get("image_provider"):
        cfg["providers"]["images"]["provider"] = options["image_provider"]
    if options.get("accent"):
        cfg["providers"]["tts"]["accent"] = options["accent"]

    # A per-job voice sample turns on ephemeral cloning. Chatterbox and
    # Voicebox both accept just the clip: Chatterbox needs no transcript, and
    # Voicebox transcribes it server-side when reference_text is absent.
    # The upload must win: drop inherited stored-voice keys that providers
    # would otherwise prefer (voicebox picks profile_id over voice_sample),
    # and the server config's reference_text describes ITS clip, not this one.
    if voice_path is not None:
        tts = cfg["providers"]["tts"]
        for key in ("profile_id", "voice", "reference_text"):
            tts.pop(key, None)
        tts["voice_sample"] = str(voice_path)

    # Presenter: a built-in mascot wins over an uploaded file. The 'animate'
    # toggle then picks the engine per source:
    #   mascot  + animate -> the server's detector-free engine (wan-s2v/d-id)
    #                        for real lip-sync if configured, else the no-GPU
    #                        puppet mouth-flap; animate off -> static mascot
    #   photo   + animate -> server's engine (wan-s2v/sadtalker/d-id/comfyui);
    #                        else static photo (a still of themselves)
    #   video             -> always the video engine (a clip is inherently
    #                        animated; wav2lip/comfyui)
    #   nothing           -> no head
    av = cfg["providers"]["avatar"]
    animate = options.get("avatar", True)
    engine = _server_animation_engine(base)
    if options.get("avatar_name"):
        name = options["avatar_name"]
        av["source"] = name
        if animate and engine in _MASCOT_ANIMATE_ENGINES:
            # A real engine can lip-sync the mascot from the narration audio.
            av["provider"] = engine
            if engine == "d-id":
                av["source_image"] = name
        else:
            av["provider"] = "puppet" if animate else "static"
    elif photo_path is not None:
        from .providers.avatar import _source_kind

        av["source"] = str(photo_path)
        if _source_kind(str(photo_path)) == "video":
            av["source_video"] = str(photo_path)
        elif animate:
            av["source_image"] = str(photo_path)
        else:
            av["provider"] = "static"
    else:
        av["provider"] = "none"

    # Owner-only from the first byte: the expanded config holds live API keys.
    job_yaml = workdir / "job.yaml"
    fd = os.open(job_yaml, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(cfg))
    return job_yaml


def _run_job(job: Job, deck_path: Path, job_yaml: Path,
             voice_path: Path | None, photo_path: Path | None,
             mode: str = "video", notes: str | None = None) -> None:
    """Run one job in a subprocess, then wipe the biometric inputs.

    ``mode`` 'video' renders an MP4 (``create``); 'pptx' produces an enriched
    deck zip (``enrich --pptx --zip``), optionally with AI presenter notes.
    """
    assert job.workdir is not None
    if mode == "pptx":
        out_dir = job.workdir / "enriched"
        output = job.workdir / "enriched.zip"  # enrich --zip writes <dir>.zip
        command = [sys.executable, "-m", "slide_stream", "enrich",
                   str(deck_path), str(out_dir), "--config", str(job_yaml),
                   "--pptx", "--zip"]
        if notes in ("fill", "all"):
            command += ["--notes", notes]
        media_type, download_name = "application/zip", "slidestream-deck.zip"
    else:
        output = job.workdir / "output.mp4"
        command = [sys.executable, "-m", "slide_stream", "create",
                   str(deck_path), str(output), "--config", str(job_yaml)]
        media_type, download_name = "video/mp4", "slidestream.mp4"

    with _LOCK:
        job.status = "running"
        job.media_type = media_type
        job.download_name = download_name
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=3600,
        )
        log = (proc.stdout or "") + (proc.stderr or "")
        with _LOCK:
            job.log = log[-8000:]
            if proc.returncode == 0 and output.exists():
                job.status = "done"
                job.output_path = output
            else:
                job.status = "error"
                job.error = f"{'enrich' if mode == 'pptx' else 'render'} exited {proc.returncode}"
    except subprocess.TimeoutExpired:
        with _LOCK:
            job.status = "error"
            job.error = "render timed out"
    except Exception as e:  # pragma: no cover - defensive
        with _LOCK:
            job.status = "error"
            job.error = str(e)
    finally:
        # Ephemeral: inputs and the key-bearing job config never persist past
        # the render. Only output.mp4 remains, until it is downloaded (demo)
        # or the TTL reaper removes the whole workdir.
        for p in (voice_path, photo_path, deck_path, job_yaml):
            if p is not None:
                Path(p).unlink(missing_ok=True)
        shutil.rmtree(job.workdir / "tmp", ignore_errors=True)


def _do_draft(source_path: Path, slides: int | None, provider: str,
              model: str | None, base_url: str | None) -> str:
    """Extract a document and draft deck Markdown from it (blocking: offload to
    a threadpool). Raises DraftError / ValueError with a user-facing message."""
    from .draft import (
        build_draft_prompt,
        clamp_source,
        clean_llm_markdown,
        extract_source_text,
        validate_deck_markdown,
    )
    from .llm import get_llm_client, query_llm

    source_text = extract_source_text(source_path)
    if not source_text.strip():
        from .draft import DraftError

        raise DraftError(
            "No extractable text was found in the document "
            "(a scanned/image-only PDF, perhaps?)."
        )
    source_text, _ = clamp_source(source_text)

    import io

    from rich.console import Console

    client = get_llm_client(provider, base_url=base_url)
    # Swallow query_llm's progress prints — they'd land in the server log.
    quiet_console = Console(file=io.StringIO())
    result = query_llm(
        client, provider, build_draft_prompt(source_text, slides),
        quiet_console, model,
    )
    if not result:
        from .draft import DraftError

        raise DraftError("The LLM returned no content. Try again.")
    deck_markdown = clean_llm_markdown(result)
    validate_deck_markdown(deck_markdown)  # raises DraftError if unusable
    return deck_markdown.rstrip() + "\n"


SETTINGS_TEMPLATE = """\
# SlideStream settings (~/.slidestream.yaml)
# Uncomment and edit what you use. Keys can reference environment variables
# as ${VAR}. Full docs: https://github.com/michael-borck/slide-stream
providers:
  llm:
    provider: gemini            # gemini | openai | claude | groq | ollama | none
    model: gemini-2.0-flash
  tts:
    provider: voicebox          # default; or gtts (free), kokoro (offline),
                                # chatterbox, elevenlabs, openai
    # base_url: https://voice.example.org        # voicebox/chatterbox server
    # api_key: "${VOICEBOX_TOKEN}"               # if the server needs auth
    # engine: kokoro            # voicebox: kokoro|chatterbox|qwen|luxtts|tada
    # profile_id: "<id from POST /profiles>"     # a stored voicebox voice...
    # voice_sample: /path/to/you.wav  # ...or clone this clip per run, then
    #                                 # delete it from the server
    # accent: australian        # gtts: australian|british|american|...
  images:
    provider: text              # text (no AI) | gemini | dalle3 | swarmui |
                                # local | pexels | unsplash
    # base_url: https://swarmui.example.org
    # model: juggernautXL_v9
  avatar:
    provider: none              # none | static | puppet | wan-s2v |
                                # sadtalker | wav2lip | comfyui | d-id
    # source: teddy             # built-in mascot, or a photo/video path
    # base_url: https://comfyui.example.org
    # api_key: "${COMFYUI_TOKEN}"   # if the ComfyUI server needs auth
    # wan-s2v animates mascots AND human head shots (no face detector);
    # sadtalker/wav2lip/comfyui are human-faces-only.
settings:
  strict: false
  narration:
    target_seconds: 45
"""


def create_app(config: dict[str, Any] | None = None, token: str | None = None,
               max_workers: int = 1, demo: bool | None = None,
               local: bool | None = None):
    """Build the FastAPI app. Requires the ``[serve]`` extra.

    ``demo`` (or the ``SLIDESTREAM_DEMO`` env var) shows a banner in the UI
    inviting users to install locally for full control over the LLM, image,
    and video generation — used on the hosted VPS instance.

    ``local`` (or ``SLIDESTREAM_LOCAL=1``) is desktop/laptop mode: no token,
    no demo limits, and a Settings page that edits ~/.slidestream.yaml. Used
    by the Tauri desktop shell.
    """
    try:
        from fastapi import (
            Depends,
            FastAPI,
            File,
            Form,
            Header,
            HTTPException,
            Request,
            UploadFile,
        )
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from starlette.background import BackgroundTask
        from starlette.responses import Response
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The web UI needs extra packages. Install with: "
            'pip install "slide-stream[serve]"'
        ) from e

    base_config = config if config is not None else load_config()
    if local is None:
        local = os.getenv("SLIDESTREAM_LOCAL", "").lower() in ("1", "true", "yes")
    local_mode = bool(local)
    auth_token = "" if local_mode else (token or os.getenv("SLIDESTREAM_TOKEN") or "")
    if demo is None:
        demo = os.getenv("SLIDESTREAM_DEMO", "").lower() in ("1", "true", "yes")
    demo_mode = bool(demo) and not local_mode
    # Only honor X-Forwarded-For when the operator says a reverse proxy sits
    # in front (deploy/docker-compose.yml sets this); otherwise the header is
    # client-supplied and would let anyone dodge the demo rate limit.
    trusted_proxy = os.getenv(
        "SLIDESTREAM_TRUSTED_PROXY", ""
    ).lower() in ("1", "true", "yes")
    executor = ThreadPoolExecutor(max_workers=max_workers)
    jobs_root = Path(tempfile.mkdtemp(prefix="slidestream_serve_"))

    app = FastAPI(title="SlideStream")

    if local_mode:

        @app.middleware("http")
        async def local_guard(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            # Local (desktop) mode has no token, so block CSRF/DNS-rebinding:
            # state-changing requests must target localhost and come from a
            # local page (or the Tauri shell's webview).
            if request.method in ("POST", "PUT", "PATCH", "DELETE"):
                host = (
                    (request.headers.get("host") or "")
                    .rsplit(":", 1)[0].strip("[]").lower()
                )
                if host not in _LOCAL_HOSTS:
                    return JSONResponse(
                        {"detail": "Requests must be addressed to localhost"},
                        status_code=403,
                    )
                origin = request.headers.get("origin")
                if origin and not _local_origin_ok(origin):
                    return JSONResponse(
                        {"detail": "Cross-origin requests are not allowed"},
                        status_code=403,
                    )
            return await call_next(request)

    def require_token(authorization: str | None = Header(default=None)) -> None:
        # Demo mode is friction-free: no token, guarded by rate/slide limits
        # instead. A token only gates private/full instances.
        if demo_mode or not auth_token:
            return
        expected = f"Bearer {auth_token}"
        if authorization is None or not secrets.compare_digest(
            authorization.encode(), expected.encode()
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing token")

    def client_ip(request: Request) -> str:
        # Behind a trusted reverse proxy the real IP is the value the proxy
        # appended to X-Forwarded-For (the rightmost one — earlier entries
        # are whatever the client chose to send).
        if trusted_proxy:
            fwd = request.headers.get("x-forwarded-for")
            if fwd:
                return fwd.split(",")[-1].strip()
        return request.client.host if request.client else "unknown"

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        # Public so the UI can bootstrap: token/demo, and the choices this
        # server actually supports (built-in avatars; accents only if the
        # configured TTS provider offers them — currently gTTS).
        from .avatars import avatar_names
        from .providers.tts import GTTS_ACCENTS

        tts_provider = base_config.get("providers", {}).get("tts", {}).get("provider")
        return {
            "auth_required": bool(auth_token) and not demo_mode,
            "demo": demo_mode,
            "local": local_mode,
            "limits": (
                {"max_slides": DEMO_MAX_SLIDES, "jobs_per_hour": DEMO_JOBS_PER_HOUR}
                if demo_mode
                else None
            ),
            "avatars": avatar_names(),
            "accents": list(GTTS_ACCENTS) if tts_provider == "gtts" else [],
            # The UI offers a PowerPoint output; AI presenter notes need an LLM.
            "llm": base_config.get("providers", {}).get("llm", {}).get("provider", "none") != "none",
        }

    settings_path = Path.home() / ".slidestream.yaml"

    @app.post("/api/quit")
    def quit_app() -> dict[str, Any]:
        # Desktop mode only: the Tauri shell calls this when the window
        # closes so the sidecar server (and its render subprocesses' parent)
        # exits cleanly even if the process kill only reaches the launcher.
        if not local_mode:
            raise HTTPException(status_code=404, detail="Not available")
        threading.Timer(0.2, _shutdown).start()
        return {"ok": True}

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        # Desktop mode only: read the user's config for the Settings page.
        if not local_mode:
            raise HTTPException(status_code=404, detail="Not available")
        text = ""
        if settings_path.exists():
            text = settings_path.read_text(encoding="utf-8")
        return {"path": str(settings_path), "yaml": text,
                "template": SETTINGS_TEMPLATE}

    @app.put("/api/settings")
    async def put_settings(request: Request) -> dict[str, Any]:
        if not local_mode:
            raise HTTPException(status_code=404, detail="Not available")
        body = await request.json()
        text = body.get("yaml", "")
        try:
            parsed = yaml.safe_load(text) if text.strip() else None
            if parsed is not None and not isinstance(parsed, dict):
                raise ValueError("top level must be a mapping")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}") from e
        # Owner-only: the settings file typically holds API keys. os.open's
        # mode only applies on creation, so chmod covers pre-existing files.
        fd = os.open(settings_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(settings_path, 0o600)
        return {"ok": True, "path": str(settings_path)}

    async def save_upload(
        upload: UploadFile, dest: Path, max_bytes: int, kind: str
    ) -> None:
        # Stream to disk in chunks with a hard cap — never buffer in RAM.
        size = 0
        with open(dest, "wb") as f:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{kind} exceeds the "
                        f"{max_bytes // (1024 * 1024)} MB limit",
                    )
                f.write(chunk)

    @app.post("/api/jobs")
    async def create_job(
        request: Request,
        deck: UploadFile = File(...),
        voice: UploadFile | None = File(default=None),
        photo: UploadFile | None = File(default=None),
        narration_seconds: str | None = Form(default=None),
        image_provider: str | None = Form(default=None),
        avatar: str | None = Form(default=None),
        avatar_name: str | None = Form(default=None),
        accent: str | None = Form(default=None),
        output: str | None = Form(default=None),
        notes: str | None = Form(default=None),
        _: None = Depends(require_token),
    ) -> JSONResponse:
        _reap_expired_jobs()
        suffix = Path(deck.filename or "deck.md").suffix.lower()
        if suffix not in (".md", ".pptx"):
            raise HTTPException(status_code=400, detail="Deck must be .md or .pptx")
        mode = "pptx" if (output or "video").lower() == "pptx" else "video"
        notes_mode = (notes or "").lower() if notes else None
        if notes_mode not in (None, "fill", "all"):
            raise HTTPException(status_code=400, detail="notes must be 'fill' or 'all'")

        # Reject obviously oversized requests before touching the body.
        declared = request.headers.get("content-length")
        max_request = MAX_DECK_BYTES + MAX_VOICE_BYTES + MAX_PHOTO_BYTES + 1024 * 1024
        if declared and declared.isdigit() and int(declared) > max_request:
            raise HTTPException(status_code=413, detail="Upload too large")

        if demo_mode and not _demo_rate_ok(client_ip(request)):
            raise HTTPException(
                status_code=429,
                detail=f"Demo limit: {DEMO_JOBS_PER_HOUR} videos per hour. "
                "Install locally for unlimited renders: pip install slide-stream",
            )

        job_id = uuid.uuid4().hex
        workdir = jobs_root / job_id
        workdir.mkdir(parents=True)
        try:
            deck_path = workdir / f"deck{suffix}"
            await save_upload(deck, deck_path, MAX_DECK_BYTES, "Deck")

            if demo_mode:
                n = _count_slides(deck_path)
                if n is None:
                    # Fail closed: an unparseable deck must not dodge the cap.
                    raise HTTPException(
                        status_code=400, detail="Could not parse the deck"
                    )
                if n > DEMO_MAX_SLIDES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Demo limit: {DEMO_MAX_SLIDES} slides per deck "
                        f"(yours has {n}). Install locally for unlimited decks: "
                        "pip install slide-stream",
                    )

            voice_path = None
            if voice is not None and voice.filename:
                voice_path = workdir / f"voice{Path(voice.filename).suffix or '.wav'}"
                await save_upload(voice, voice_path, MAX_VOICE_BYTES, "Voice sample")
            photo_path = None
            if photo is not None and photo.filename:
                from .providers.avatar import _source_kind

                photo_path = workdir / f"photo{Path(photo.filename).suffix or '.png'}"
                await save_upload(photo, photo_path, MAX_PHOTO_BYTES, "Photo")
                if _source_kind(str(photo_path)) == "image":
                    problem = _validate_photo_upload(photo_path)
                    if problem:
                        raise HTTPException(status_code=400, detail=problem)

            options = {
                "narration_seconds": narration_seconds,
                "image_provider": image_provider,
                "avatar": (avatar or "true").lower() != "false",
                "avatar_name": avatar_name,
                "accent": accent,
            }
            # Desktop mode re-reads ~/.slidestream.yaml per job so Settings
            # edits apply without restarting the app.
            job_base = load_config() if local_mode else base_config
            job_yaml = _build_job_config(
                job_base, workdir, options, voice_path, photo_path
            )
        except Exception:
            # Never leave a rejected job's uploads (or its key-bearing
            # job.yaml) on disk.
            shutil.rmtree(workdir, ignore_errors=True)
            raise

        job = Job(id=job_id, workdir=workdir, created_at=time.time(),
                  download_token=secrets.token_urlsafe(24))
        with _LOCK:
            _JOBS[job_id] = job
        executor.submit(_run_job, job, deck_path, job_yaml, voice_path,
                        photo_path, mode, notes_mode)
        return JSONResponse({"job_id": job_id, "status": job.status,
                             "token": job.download_token})

    @app.post("/api/check")
    async def check_deck(
        deck: UploadFile = File(...),
        voice: UploadFile | None = File(default=None),
        photo: UploadFile | None = File(default=None),
        narration_seconds: str | None = Form(default=None),
        image_provider: str | None = Form(default=None),
        avatar: str | None = Form(default=None),
        avatar_name: str | None = Form(default=None),
        accent: str | None = Form(default=None),
        output: str | None = Form(default=None),
        _: None = Depends(require_token),
    ) -> JSONResponse:
        """Offline preflight (the 'doctor'): assess the deck + resolved config
        and return warnings + estimates as JSON, without rendering anything."""
        suffix = Path(deck.filename or "deck.md").suffix.lower()
        if suffix not in (".md", ".pptx"):
            raise HTTPException(status_code=400, detail="Deck must be .md or .pptx")

        workdir = jobs_root / ("check_" + uuid.uuid4().hex)
        workdir.mkdir(parents=True)
        try:
            deck_path = workdir / f"deck{suffix}"
            await save_upload(deck, deck_path, MAX_DECK_BYTES, "Deck")
            voice_path = None
            if voice is not None and voice.filename:
                voice_path = workdir / f"voice{Path(voice.filename).suffix or '.wav'}"
                await save_upload(voice, voice_path, MAX_VOICE_BYTES, "Voice sample")
            photo_path = None
            if photo is not None and photo.filename:
                photo_path = workdir / f"photo{Path(photo.filename).suffix or '.png'}"
                await save_upload(photo, photo_path, MAX_PHOTO_BYTES, "Photo")

            options = {
                "narration_seconds": narration_seconds,
                "image_provider": image_provider,
                "avatar": (avatar or "true").lower() != "false",
                "avatar_name": avatar_name,
                "accent": accent,
            }
            job_base = load_config() if local_mode else base_config
            job_yaml = _build_job_config(job_base, workdir, options, voice_path, photo_path)
            cfg = yaml.safe_load(job_yaml.read_text(encoding="utf-8")) or {}

            slides = _parse_deck_slides(deck_path)
            if not slides:
                raise HTTPException(status_code=400, detail="Could not parse the deck")

            from .doctor import run_doctor

            avatar_enabled = (
                cfg.get("providers", {}).get("avatar", {}).get("provider", "none") != "none"
            )
            report = run_doctor(slides, cfg, {
                "mode": "pptx" if (output or "video").lower() == "pptx" else "create",
                "input_ext": suffix,
                "verbatim_notes": False,
                "script_blocks": None,
                "avatar_enabled": avatar_enabled,
                "narration_seconds": float(narration_seconds) if narration_seconds else None,
                "output_path": None,
            })
            return JSONResponse({
                "blockers": report.blockers,
                "warnings": report.warnings,
                "findings": [
                    {"group": f.group, "severity": f.severity, "message": f.message}
                    for f in report.findings
                ],
                "estimates": report.estimates,
            })
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, _: None = Depends(require_token)) -> dict[str, Any]:
        _reap_expired_jobs()
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        out: dict[str, Any] = {"job_id": job.id, "status": job.status,
                               "error": job.error}
        if demo_mode:
            # The open demo gets only the coarse status the UI shows anyway:
            # raw render logs can leak paths and exception text, and the
            # download token must stay knowable only to the job's creator.
            out["log"] = ""
        else:
            out["log"] = job.log[-4000:]
            out["token"] = job.download_token
        return out

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str, t: str | None = None,
                   authorization: str | None = Header(default=None)):
        _reap_expired_jobs()
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Result not ready")
        # A browser download link can't set an Authorization header, so accept
        # the per-job download token via ?t= — never the long-lived instance
        # token, which must not land in proxy logs or browser history.
        header_ok = bool(
            auth_token
            and authorization is not None
            and secrets.compare_digest(
                authorization.encode(), f"Bearer {auth_token}".encode()
            )
        )
        token_ok = bool(
            t and job.download_token
            and secrets.compare_digest(t.encode(), job.download_token.encode())
        )
        if not (header_ok or token_ok):
            raise HTTPException(
                status_code=401, detail="Invalid or missing download token"
            )
        if job.status != "done" or job.output_path is None:
            raise HTTPException(status_code=404, detail="Result not ready")
        background: BackgroundTask | None = None
        if demo_mode:
            # Demo promises "nothing stored": one download, then the whole
            # job dir goes away.
            workdir = job.workdir

            def _wipe() -> None:
                with _LOCK:
                    _JOBS.pop(job_id, None)
                if workdir is not None:
                    shutil.rmtree(workdir, ignore_errors=True)

            background = BackgroundTask(_wipe)
        return FileResponse(str(job.output_path), media_type=job.media_type,
                            filename=job.download_name, background=background)

    # ---- Project workflow: draft -> edit -> enrich -> render --------------
    # A stateful session (unlike the one-shot jobs above) so a document can
    # become a deck, be edited, enriched, and rendered without re-uploading
    # between steps. Gated to non-demo mode: it needs an LLM and persists state,
    # so it's the desktop / self-hosted experience, not the open demo.
    projects_root = jobs_root / "projects"
    projects_root.mkdir(exist_ok=True)

    def _project_or_401(project_id: str, token: str | None) -> Project:
        if demo_mode:
            raise HTTPException(
                status_code=403,
                detail="The project workflow is available in the app / "
                "self-hosted mode, not the open demo.",
            )
        _reap_expired_projects()
        project = _PROJECTS.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Unknown project")
        if not (
            token
            and secrets.compare_digest(token.encode(), project.token.encode())
        ):
            raise HTTPException(
                status_code=401, detail="Invalid or missing project token"
            )
        return project

    @app.post("/api/projects")
    async def create_project(
        deck: UploadFile | None = File(default=None),
        _: None = Depends(require_token),
    ) -> JSONResponse:
        if demo_mode:
            raise HTTPException(
                status_code=403,
                detail="The project workflow is available in the app / "
                "self-hosted mode, not the open demo.",
            )
        _reap_expired_projects()
        pid = uuid.uuid4().hex
        workdir = projects_root / pid
        workdir.mkdir(parents=True)
        try:
            if deck is not None and deck.filename:
                suffix = Path(deck.filename).suffix.lower()
                if suffix not in (".md", ".pptx"):
                    raise HTTPException(
                        status_code=400, detail="Deck must be .md or .pptx"
                    )
                await save_upload(
                    deck, workdir / f"deck{suffix}", MAX_DECK_BYTES, "Deck"
                )
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        project = Project(
            id=pid, workdir=workdir, created_at=time.time(),
            token=secrets.token_urlsafe(24),
        )
        with _LOCK:
            _PROJECTS[pid] = project
        return JSONResponse(
            {"project_id": pid, "token": project.token,
             "state": _project_state(project)}
        )

    @app.get("/api/projects/{project_id}")
    def project_state_endpoint(
        project_id: str,
        x_project_token: str | None = Header(default=None),
        _: None = Depends(require_token),
    ) -> dict[str, Any]:
        return _project_state(_project_or_401(project_id, x_project_token))

    @app.post("/api/projects/{project_id}/draft")
    async def project_draft(
        project_id: str,
        source: UploadFile = File(...),
        slides: str | None = Form(default=None),
        x_project_token: str | None = Header(default=None),
        _: None = Depends(require_token),
    ) -> JSONResponse:
        project = _project_or_401(project_id, x_project_token)
        job_base = load_config() if local_mode else base_config
        llm = job_base.get("providers", {}).get("llm", {})
        provider = llm.get("provider", "none")
        if provider == "none":
            raise HTTPException(
                status_code=400,
                detail="draft needs an LLM provider configured in Settings "
                "(e.g. claude, openai, gemini).",
            )
        n: int | None = None
        if slides:
            try:
                n = int(slides)
            except ValueError:
                raise HTTPException(status_code=400, detail="slides must be a number")
            if n < 1:
                raise HTTPException(status_code=400, detail="slides must be positive")

        from .draft import SUPPORTED_SUFFIXES

        suffix = Path(source.filename or "source").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source type. Use: {', '.join(SUPPORTED_SUFFIXES)}",
            )
        src_path = project.workdir / f"source{suffix}"
        await save_upload(source, src_path, MAX_DECK_BYTES, "Document")

        from starlette.concurrency import run_in_threadpool

        from .draft import DraftError

        try:
            markdown = await run_in_threadpool(
                _do_draft, src_path, n, provider, llm.get("model"),
                llm.get("base_url"),
            )
        except (DraftError, ValueError, ImportError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            src_path.unlink(missing_ok=True)

        (project.workdir / "deck.md").write_text(markdown, encoding="utf-8")
        # A freshly drafted deck supersedes any previously uploaded .pptx.
        (project.workdir / "deck.pptx").unlink(missing_ok=True)
        return JSONResponse(
            {"markdown": markdown, "state": _project_state(project)}
        )

    @app.put("/api/projects/{project_id}/deck")
    async def save_deck(
        project_id: str,
        request: Request,
        x_project_token: str | None = Header(default=None),
        _: None = Depends(require_token),
    ) -> JSONResponse:
        project = _project_or_401(project_id, x_project_token)
        body = await request.json()
        markdown = body.get("markdown", "")
        if not isinstance(markdown, str) or not markdown.strip():
            raise HTTPException(
                status_code=400, detail="markdown must be a non-empty string"
            )
        if len(markdown.encode("utf-8")) > MAX_DECK_BYTES:
            raise HTTPException(status_code=413, detail="Deck too large")

        from .parser import parse_markdown

        if not parse_markdown(markdown):
            raise HTTPException(
                status_code=400,
                detail="No slides found (need at least one '# ' heading).",
            )
        (project.workdir / "deck.md").write_text(markdown, encoding="utf-8")
        (project.workdir / "deck.pptx").unlink(missing_ok=True)
        return JSONResponse({"state": _project_state(project)})

    def _spawn_project_job(
        project: Project, mode: str, options: dict[str, Any],
        voice_path: Path | None, photo_path: Path | None, notes: str | None,
    ) -> JSONResponse:
        """Copy the project deck into a fresh job workdir and run it, so the
        canonical project deck is never consumed/deleted by the render."""
        deck = _project_deck(project)
        if deck is None:
            raise HTTPException(
                status_code=400, detail="Project has no deck yet"
            )
        job_id = uuid.uuid4().hex
        workdir = jobs_root / job_id
        workdir.mkdir(parents=True)
        deck_copy = workdir / deck.name
        shutil.copyfile(deck, deck_copy)
        # Voice/photo were uploaded into a temp area; move them under the job.
        moved_voice = moved_photo = None
        if voice_path is not None:
            moved_voice = workdir / voice_path.name
            shutil.move(str(voice_path), moved_voice)
        if photo_path is not None:
            moved_photo = workdir / photo_path.name
            shutil.move(str(photo_path), moved_photo)
        job_base = load_config() if local_mode else base_config
        job_yaml = _build_job_config(
            job_base, workdir, options, moved_voice, moved_photo
        )
        job = Job(
            id=job_id, workdir=workdir, created_at=time.time(),
            download_token=secrets.token_urlsafe(24),
        )
        with _LOCK:
            _JOBS[job_id] = job
        executor.submit(
            _run_job, job, deck_copy, job_yaml, moved_voice, moved_photo,
            mode, notes,
        )
        return JSONResponse(
            {"job_id": job_id, "status": job.status, "token": job.download_token}
        )

    @app.post("/api/projects/{project_id}/enrich")
    async def enrich_project(
        project_id: str,
        image_provider: str | None = Form(default=None),
        notes: str | None = Form(default=None),
        x_project_token: str | None = Header(default=None),
        _: None = Depends(require_token),
    ) -> JSONResponse:
        _reap_expired_jobs()
        project = _project_or_401(project_id, x_project_token)
        notes_mode = (notes or "").lower() if notes else None
        if notes_mode not in (None, "fill", "all"):
            raise HTTPException(status_code=400, detail="notes must be 'fill' or 'all'")
        options = {"image_provider": image_provider, "avatar": False}
        return _spawn_project_job(
            project, "pptx", options, None, None, notes_mode
        )

    @app.post("/api/projects/{project_id}/render")
    async def render_project(
        project_id: str,
        voice: UploadFile | None = File(default=None),
        photo: UploadFile | None = File(default=None),
        narration_seconds: str | None = Form(default=None),
        image_provider: str | None = Form(default=None),
        avatar: str | None = Form(default=None),
        avatar_name: str | None = Form(default=None),
        accent: str | None = Form(default=None),
        x_project_token: str | None = Header(default=None),
        _: None = Depends(require_token),
    ) -> JSONResponse:
        _reap_expired_jobs()
        project = _project_or_401(project_id, x_project_token)
        if _project_deck(project) is None:
            raise HTTPException(
                status_code=400, detail="Project has no deck to render yet"
            )
        staging = project.workdir / ("render_" + uuid.uuid4().hex)
        staging.mkdir()
        voice_path = photo_path = None
        try:
            if voice is not None and voice.filename:
                voice_path = staging / f"voice{Path(voice.filename).suffix or '.wav'}"
                await save_upload(voice, voice_path, MAX_VOICE_BYTES, "Voice sample")
            if photo is not None and photo.filename:
                photo_path = staging / f"photo{Path(photo.filename).suffix or '.png'}"
                await save_upload(photo, photo_path, MAX_PHOTO_BYTES, "Photo")
                from .providers.avatar import _source_kind

                if _source_kind(str(photo_path)) == "image":
                    problem = _validate_photo_upload(photo_path)
                    if problem:
                        raise HTTPException(status_code=400, detail=problem)
            options = {
                "narration_seconds": narration_seconds,
                "image_provider": image_provider,
                "avatar": (avatar or "true").lower() != "false",
                "avatar_name": avatar_name,
                "accent": accent,
            }
            return _spawn_project_job(
                project, "video", options, voice_path, photo_path, None
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    return app


# Single-page UI. Remembers the voice sample + photo in the browser (IndexedDB)
# so the lecturer never re-picks them; the server stores neither at rest.
INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SlideStream — slides in, narrated video out</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#faf9f6;--ink:#1e1b18;--muted:#6b645c;--line:#e7e2d9;--accent:#c2410c;
 --accent-soft:#c2410c14;--card:#ffffff;--radius:14px}
@media (prefers-color-scheme: dark){
 :root{--bg:#17150f;--ink:#f2ede3;--muted:#a89f92;--line:#2c2820;--card:#201d16;
  --accent-soft:#c2410c26}}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--ink);line-height:1.55}
.wrap{max-width:660px;margin:0 auto;padding:2.2rem 1.25rem 3rem}
h1{font-family:Fraunces,Georgia,serif;font-weight:600;font-size:1.9rem;margin:0;letter-spacing:-.01em}
h1 em{font-style:italic;color:var(--accent)}
.tag{color:var(--muted);margin:.3rem 0 1.6rem;font-size:.98rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
 padding:1.4rem 1.5rem;box-shadow:0 1px 2px rgba(0,0,0,.03)}
label{display:block;margin:1.05rem 0 .3rem;font-weight:600;font-size:.92rem}
label:first-child{margin-top:0}
input,select{font:inherit;color:inherit;width:100%;padding:.55rem .7rem;border:1px solid var(--line);
 border-radius:9px;background:var(--bg)}
input[type=file]{padding:.45rem .5rem;cursor:pointer}
input[type=file]::file-selector-button{font:inherit;font-weight:600;margin-right:.8rem;
 padding:.4rem .9rem;border:0;border-radius:7px;background:var(--accent-soft);color:var(--accent);cursor:pointer}
input:focus,select:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent}
button{font:inherit;font-weight:600;width:100%;margin-top:1.4rem;padding:.75rem 1.2rem;
 border:0;border-radius:10px;background:var(--accent);color:#fff;cursor:pointer;transition:transform .06s}
button:active{transform:translateY(1px)}
.row{display:flex;gap:.55rem;align-items:center;margin-top:1rem}
.row input{width:auto;accent-color:var(--accent)} .row label{margin:0}
.muted{color:var(--muted);font-size:.85rem;margin:.35rem 0 0}
.banner{display:none;background:var(--accent-soft);border:1px solid var(--accent);
 border-radius:var(--radius);padding:.85rem 1.1rem;margin-bottom:1.2rem;font-size:.92rem}
.banner a{color:var(--accent);font-weight:600;text-decoration:none}
.banner code{font-family:ui-monospace,Menlo,monospace;font-size:.88em;
 background:rgba(120,120,120,.14);padding:.08em .4em;border-radius:5px}
.paths{display:grid;gap:.45rem;margin:0 0 1.3rem;font-size:.88rem;color:var(--muted)}
.paths div{display:flex;gap:.6rem;align-items:baseline}
.paths span{flex:none;font-weight:600;font-size:.72rem;text-transform:uppercase;
 letter-spacing:.08em;color:var(--accent);min-width:6.5em}
.req{font-size:.72rem;font-weight:600;color:var(--accent);background:var(--accent-soft);
 padding:.1rem .5rem;border-radius:99px;vertical-align:middle}
details{margin-top:1.3rem;border-top:1px solid var(--line);padding-top:1rem}
details summary{cursor:pointer;font-weight:600;font-size:.95rem;color:var(--muted);list-style:none}
details summary::before{content:"▸ ";color:var(--accent)}
details[open] summary::before{content:"▾ "}
details summary:hover{color:var(--ink)}
#status{margin:1rem 0 .4rem;font-weight:500}
#status a{color:var(--accent);font-weight:600}
.badge{display:inline-block;padding:.12rem .65rem;border-radius:99px;font-size:.8rem;
 background:var(--accent-soft);color:var(--accent);font-weight:600}
#log{display:none;white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);
 padding:.7rem .8rem;border-radius:9px;font-family:ui-monospace,Menlo,monospace;
 font-size:.75rem;max-height:220px;overflow:auto;color:var(--muted)}
#log.on{display:block}
#report{display:none;margin-top:1rem;border:1px solid var(--line);border-radius:9px;
 padding:.7rem 1rem .9rem;font-size:.88rem}
#report.on{display:block}
#report h4{margin:.7rem 0 .25rem;font-size:.72rem;text-transform:uppercase;
 letter-spacing:.07em;color:var(--accent)}
#report h4:first-child{margin-top:.2rem}
#report .f{margin:.16rem 0} #report .est{color:var(--muted)}
#report .sum{margin-top:.7rem;padding-top:.5rem;border-top:1px solid var(--line);font-weight:600}
#check{background:transparent;border:1px solid var(--line);color:var(--ink);margin-top:.7rem}
#check:hover{border-color:var(--accent);color:var(--accent)}
footer{margin-top:2rem;padding-top:1.1rem;border-top:1px solid var(--line);
 font-size:.85rem;color:var(--muted);display:flex;gap:1.2rem;flex-wrap:wrap}
footer a{color:var(--muted);text-decoration:none;font-weight:500}
footer a:hover{color:var(--accent)}
.head{display:flex;align-items:baseline;justify-content:space-between}
#gear{display:none;font:inherit;width:auto;margin:0;padding:.4rem .8rem;
 background:transparent;border:1px solid var(--line);border-radius:9px;color:var(--muted)}
#gear:hover{color:var(--accent);border-color:var(--accent)}
#settings{display:none;position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:20}
#settings.on{display:block}
.panel{position:absolute;top:0;right:0;bottom:0;width:min(560px,94vw);background:var(--card);
 border-left:1px solid var(--line);padding:1.3rem 1.4rem;overflow:auto;display:flex;flex-direction:column}
.panel h2{font-family:Fraunces,Georgia,serif;font-weight:600;margin:0 0 .2rem;font-size:1.3rem}
.panel textarea{flex:1;min-height:320px;font-family:ui-monospace,Menlo,monospace;font-size:.8rem;
 line-height:1.5;width:100%;padding:.7rem;border:1px solid var(--line);border-radius:9px;
 background:var(--bg);color:inherit;resize:vertical;margin:.8rem 0}
.panel .btns{display:flex;gap:.6rem}
.panel .btns button{margin:0;width:auto;padding:.55rem 1rem}
.panel .ghost{background:transparent;border:1px solid var(--line);color:var(--ink)}
#saveMsg{font-size:.85rem;margin:.5rem 0 0}
</style></head><body><div class="wrap">
<div class="head"><h1>🎬 Slide<em>Stream</em></h1>
<button id="gear" title="Providers &amp; settings">⚙ Settings</button></div>
<p class="tag">Slides in, narrated video out — in your voice, or a friendly mascot's.</p>
<div id="settings"><div class="panel">
<h2>Settings</h2>
<p class="muted" id="setPath">Edits your ~/.slidestream.yaml — providers, servers, keys.</p>
<textarea id="setYaml" spellcheck="false" placeholder="# empty — click 'Insert template' to start"></textarea>
<div class="btns">
 <button id="setSave">Save</button>
 <button id="setTpl" class="ghost">Insert template</button>
 <button id="setClose" class="ghost">Close</button>
</div>
<p id="saveMsg"></p>
</div></div>
<div id="demo" class="banner">
 <strong>Hosted demo</strong> — <span id="limits">limited</span>, nothing stored.
 Want unlimited renders, your own AI providers and full privacy?
 <a id="dlBanner" href="https://github.com/michael-borck/slide-stream/releases/latest">⬇ Get the desktop app</a> &middot;
 <code>pip install slide-stream</code> &middot;
 <a href="https://slidestream.eduserver.au">learn more</a>
</div>
<div class="paths">
 <div><span>Minimal</span>Just a slide deck → narrated video with a stock voice.</div>
 <div><span>Your voice</span>Add a 10–30s voice sample → narration in your voice.</div>
 <div><span>Presenter</span>Pick a mascot, or add your photo/video → a talking head in the corner.</div>
</div>
<div class="card">
<div id="tokrow" style="display:none"><label>Access token</label>
 <input id="token" type="password" placeholder="paste your token">
 <p class="muted">Stored in this browser only.</p></div>
<label>Slide deck <span class="req">required</span> <span style="font-weight:400;color:var(--muted)">(.md or .pptx)</span></label>
<input id="deck" type="file" accept=".md,.pptx">
<p class="muted">This is all you need — everything below is optional.</p>
<label>Output</label>
<select id="output">
 <option value="video">🎬 Narrated video (.mp4)</option>
 <option value="pptx">🖼️ PowerPoint deck (.pptx + images, no video)</option>
</select>
<div id="notesRow" style="display:none">
 <label>AI presenter notes</label>
 <select id="notes">
  <option value="">None</option>
  <option value="all">Write for every slide</option>
  <option value="fill">Only where notes are missing</option>
 </select>
 <p class="muted">Adds speaker notes to the PowerPoint.</p>
</div>
<details id="extras">
<summary>Voice &amp; presenter <span style="font-weight:400">(optional)</span></summary>
<label>Your voice <span style="font-weight:400;color:var(--muted)">(a 10–30s sample clones it for this render only)</span></label>
<input id="voice" type="file" accept="audio/*">
<label>Mascot presenter</label>
<select id="avatarName"><option value="">None</option></select>
<p class="muted">A friendly character presents in the corner — or upload yourself below.</p>
<label>Your photo or short video <span style="font-weight:400;color:var(--muted)">(front-facing)</span></label>
<input id="photo" type="file" accept="image/*,video/*">
<p class="muted" id="remembered"></p>
<div class="row"><input id="avatar" type="checkbox" checked><label>Animate the presenter</label></div>
<p class="muted">On: a mascot gets a cartoon mouth-flap; your photo becomes an AI talking head.
Off: the presenter appears as a still image in the corner.</p>
<label id="accentRow" style="display:none">Accent</label>
<select id="accent" style="display:none"><option value="">— default —</option></select>
<label>Seconds of narration per slide</label>
<input id="secs" type="number" min="10" placeholder="e.g. 30">
</details>
<button id="check">Check deck first</button>
<button id="go">Create video</button>
<p id="status"></p><div id="report"></div><div id="log"></div>
</div>
<footer>
 <a href="https://slidestream.eduserver.au">About</a>
 <span style="color:var(--muted)">Desktop app:
  <a href="https://github.com/michael-borck/slide-stream/releases/latest/download/SlideStream-macos-apple-silicon.dmg">macOS</a> ·
  <a href="https://github.com/michael-borck/slide-stream/releases/latest/download/SlideStream-macos-intel.dmg">macOS Intel</a> ·
  <a href="https://github.com/michael-borck/slide-stream/releases/latest/download/SlideStream-windows-setup.exe">Windows</a> ·
  <a href="https://github.com/michael-borck/slide-stream/releases/latest/download/SlideStream-linux.AppImage">Linux</a></span>
 <a href="https://pypi.org/project/slide-stream/">pip install slide-stream</a>
 <a href="https://github.com/michael-borck/slide-stream">GitHub</a>
</footer>
</div>
<script>
const $=id=>document.getElementById(id);
// Platform-detected desktop download in the demo banner (stable asset names).
(()=>{const ua=(navigator.userAgent||"").toLowerCase();
 const f=ua.includes("mac")?"SlideStream-macos-apple-silicon.dmg":
   ua.includes("win")?"SlideStream-windows-setup.exe":
   ua.includes("linux")?"SlideStream-linux.AppImage":null;
 if(f)$("dlBanner").href=
  "https://github.com/michael-borck/slide-stream/releases/latest/download/"+f})();
$("token").value=localStorage.getItem("ss_token")||"";
$("token").oninput=e=>localStorage.setItem("ss_token",e.target.value);
// Bootstrap: show the token field only if required, and the demo banner if on.
let hasLLM=false;
fetch("/api/config").then(r=>r.json()).then(c=>{
 if(c.auth_required)$("tokrow").style.display="block";
 if(c.demo){$("demo").style.display="block";
  if(c.limits)$("limits").textContent=
   c.limits.max_slides+" slides per deck, "+c.limits.jobs_per_hour+" videos per hour";}
 if(c.local)$("gear").style.display="inline-block";
 hasLLM=!!c.llm;
 (c.avatars||[]).forEach(a=>{const o=document.createElement("option");o.value=a;o.textContent=a;$("avatarName").appendChild(o)});
 if((c.accents||[]).length){$("accentRow").style.display="block";$("accent").style.display="block";
  c.accents.forEach(a=>{const o=document.createElement("option");o.value=a;o.textContent=a;$("accent").appendChild(o)})}
}).catch(()=>{});
// Output mode: video (create) vs PowerPoint (enrich). PowerPoint hides the
// voice/presenter extras (unused) and, with an LLM, offers AI notes.
$("output").onchange=()=>{const pptx=$("output").value==="pptx";
 $("notesRow").style.display=(pptx&&hasLLM)?"block":"none";
 $("extras").style.display=pptx?"none":"";
 $("go").textContent=pptx?"Create PowerPoint":"Create video";
 $("report").classList.remove("on")};
// IndexedDB: remember voice + photo across jobs (client-side only).
let db;const openDB=()=>new Promise(r=>{const q=indexedDB.open("ss",1);
 q.onupgradeneeded=()=>q.result.createObjectStore("files");q.onsuccess=()=>{db=q.result;r()}});
const put=(k,v)=>new Promise(r=>{db.transaction("files","readwrite").objectStore("files").put(v,k).onsuccess=r});
const get=k=>new Promise(r=>{const q=db.transaction("files").objectStore("files").get(k);q.onsuccess=()=>r(q.result)});
let savedVoice,savedPhoto;
openDB().then(async()=>{savedVoice=await get("voice");savedPhoto=await get("photo");
 const b=[];if(savedVoice)b.push("voice: "+savedVoice.name);if(savedPhoto)b.push("photo: "+savedPhoto.name);
 $("remembered").textContent=b.length?("Remembered "+b.join(", ")+" — leave the fields empty to reuse."):"";
 if(b.length)$("extras").open=true});
const auth=()=>({Authorization:"Bearer "+$("token").value});
async function fileOrSaved(input,key,saved){const f=input.files[0];
 if(f){await put(key,f);return f}return saved||null}
async function buildFD(){
 const deck=$("deck").files[0];
 if(!deck){$("status").textContent="Pick a deck first.";return null}
 const fd=new FormData();fd.append("deck",deck);fd.append("output",$("output").value);
 if($("output").value==="pptx"){if($("notes").value)fd.append("notes",$("notes").value)}
 else{
  const voice=await fileOrSaved($("voice"),"voice",savedVoice);
  const photo=await fileOrSaved($("photo"),"photo",savedPhoto);
  if(voice)fd.append("voice",voice);if(photo)fd.append("photo",photo);
  fd.append("avatar",$("avatar").checked?"true":"false");
  if($("avatarName").value)fd.append("avatar_name",$("avatarName").value);
  if($("accent").value)fd.append("accent",$("accent").value);
 }
 if($("secs").value)fd.append("narration_seconds",$("secs").value);
 return fd}
const ICON={ok:"✅",warn:"⚠️",blocker:"❌"};
const esc=s=>{const d=document.createElement("div");d.textContent=s;return d.innerHTML};
function renderReport(rep){
 const groups={};rep.findings.forEach(f=>{(groups[f.group]=groups[f.group]||[]).push(f)});
 let h="";Object.keys(groups).forEach(g=>{h+="<h4>"+esc(g)+"</h4>";
  groups[g].forEach(f=>{h+='<div class="f">'+(ICON[f.severity]||"")+" "+esc(f.message)+"</div>"})});
 if((rep.estimates||[]).length){h+="<h4>Estimates</h4>";
  rep.estimates.forEach(e=>{h+='<div class="f est">• '+esc(e)+"</div>"})}
 h+='<div class="sum">'+(rep.blockers?("❌ "+rep.blockers+" blocker(s) · "):"")+
  (rep.warnings?("⚠️ "+rep.warnings+" warning(s)"):"✅ no warnings")+"</div>";
 $("report").innerHTML=h;$("report").classList.add("on")}
$("check").onclick=async()=>{
 const fd=await buildFD();if(!fd)return;
 $("check").disabled=true;$("status").textContent="Checking…";$("report").classList.remove("on");
 try{const r=await fetch("/api/check",{method:"POST",headers:auth(),body:fd});
  if(!r.ok){$("status").textContent="Error: "+(await r.text());return}
  $("status").textContent="";renderReport(await r.json())}
 finally{$("check").disabled=false}};
$("go").onclick=async()=>{
 const fd=await buildFD();if(!fd)return;
 $("status").textContent="Uploading…";$("log").textContent="";$("report").classList.remove("on");
 let res=await fetch("/api/jobs",{method:"POST",headers:auth(),body:fd});
 if(!res.ok){$("status").textContent="Error: "+(await res.text());return}
 const {job_id,token}=await res.json();poll(job_id,token)};
// Settings (desktop/local mode): edit ~/.slidestream.yaml in-app.
let setTemplate="";
$("gear").onclick=async()=>{
 const r=await fetch("/api/settings");if(!r.ok)return;
 const s=await r.json();setTemplate=s.template||"";
 $("setYaml").value=s.yaml||"";$("setPath").textContent="Edits "+s.path+" — providers, servers, keys.";
 $("saveMsg").textContent="";$("settings").classList.add("on")};
$("setClose").onclick=()=>$("settings").classList.remove("on");
$("settings").onclick=e=>{if(e.target.id==="settings")$("settings").classList.remove("on")};
$("setTpl").onclick=()=>{if(!$("setYaml").value.trim()||confirm("Replace current contents with the template?"))
 $("setYaml").value=setTemplate};
$("setSave").onclick=async()=>{
 const r=await fetch("/api/settings",{method:"PUT",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({yaml:$("setYaml").value})});
 const j=await r.json().catch(()=>({}));
 $("saveMsg").textContent=r.ok?"✓ Saved — applies to your next video.":("✗ "+(j.detail||"Save failed"));
 $("saveMsg").style.color=r.ok?"":"var(--accent)"};
async function poll(id,tok){
 const r=await fetch("/api/jobs/"+id,{headers:auth()});const j=await r.json();
 $("status").innerHTML='<span class="badge">'+j.status+'</span>';
 $("log").textContent=j.log||"";$("log").classList.toggle("on",!!j.log);
 if(j.status==="done"){const lbl=$("output").value==="pptx"?"download deck (.zip)":"download video";
  $("status").innerHTML+=' <a href="/api/jobs/'+id+'/result?t='+
   encodeURIComponent(tok||j.token||"")+'" download>⬇ '+lbl+'</a>';return}
 if(j.status==="error"){$("status").textContent="Failed: "+(j.error||"see log");return}
 setTimeout(()=>poll(id,tok),2500)}
</script></body></html>"""
