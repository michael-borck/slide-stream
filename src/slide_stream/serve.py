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
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


# In-memory job registry (single-process v1). job_id -> Job.
_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


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

    # A per-job voice sample turns on ephemeral Chatterbox cloning.
    if voice_path is not None:
        cfg["providers"]["tts"]["voice_sample"] = str(voice_path)

    # A per-job photo drives the avatar (provider comes from the server config,
    # e.g. sadtalker/d-id). No photo, or avatar off -> no head.
    if photo_path is not None and options.get("avatar", True):
        cfg["providers"]["avatar"]["source_image"] = str(photo_path)
    else:
        cfg["providers"]["avatar"]["provider"] = "none"

    job_yaml = workdir / "job.yaml"
    job_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return job_yaml


def _run_job(job: Job, deck_path: Path, job_yaml: Path,
             voice_path: Path | None, photo_path: Path | None) -> None:
    """Render one job in a subprocess, then wipe the biometric inputs."""
    assert job.workdir is not None
    output = job.workdir / "output.mp4"
    with _LOCK:
        job.status = "running"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "slide_stream", "create",
             str(deck_path), str(output), "--config", str(job_yaml)],
            capture_output=True, text=True, timeout=3600,
        )
        log = (proc.stdout or "") + (proc.stderr or "")
        with _LOCK:
            job.log = log[-8000:]
            if proc.returncode == 0 and output.exists():
                job.status = "done"
                job.output_path = output
            else:
                job.status = "error"
                job.error = f"render exited {proc.returncode}"
    except subprocess.TimeoutExpired:
        with _LOCK:
            job.status = "error"
            job.error = "render timed out"
    except Exception as e:  # pragma: no cover - defensive
        with _LOCK:
            job.status = "error"
            job.error = str(e)
    finally:
        # Ephemeral: the voice sample and photo never persist past the render.
        for p in (voice_path, photo_path):
            if p is not None:
                Path(p).unlink(missing_ok=True)


def create_app(config: dict[str, Any] | None = None, token: str | None = None,
               max_workers: int = 1, demo: bool | None = None):
    """Build the FastAPI app. Requires the ``[serve]`` extra.

    ``demo`` (or the ``SLIDESTREAM_DEMO`` env var) shows a banner in the UI
    inviting users to install locally for full control over the LLM, image,
    and video generation — used on the hosted VPS instance.
    """
    try:
        from fastapi import (
            Depends,
            FastAPI,
            File,
            Form,
            Header,
            HTTPException,
            UploadFile,
        )
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The web UI needs extra packages. Install with: "
            'pip install "slide-stream[serve]"'
        ) from e

    base_config = config if config is not None else load_config()
    auth_token = token or os.getenv("SLIDESTREAM_TOKEN") or ""
    if demo is None:
        demo = os.getenv("SLIDESTREAM_DEMO", "").lower() in ("1", "true", "yes")
    demo_mode = bool(demo)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    jobs_root = Path(tempfile.mkdtemp(prefix="slidestream_serve_"))

    app = FastAPI(title="SlideStream")

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if not auth_token:
            return  # no token configured (local, trusted) -> open
        expected = f"Bearer {auth_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing token")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        # Public so the UI can bootstrap (whether a token is needed, demo mode).
        return {"auth_required": bool(auth_token), "demo": demo_mode}

    @app.post("/api/jobs")
    async def create_job(
        deck: UploadFile = File(...),
        voice: UploadFile | None = File(default=None),
        photo: UploadFile | None = File(default=None),
        narration_seconds: str | None = Form(default=None),
        image_provider: str | None = Form(default=None),
        avatar: str | None = Form(default=None),
        _: None = Depends(require_token),
    ) -> JSONResponse:
        suffix = Path(deck.filename or "deck.md").suffix.lower()
        if suffix not in (".md", ".pptx"):
            raise HTTPException(status_code=400, detail="Deck must be .md or .pptx")

        job_id = uuid.uuid4().hex
        workdir = jobs_root / job_id
        workdir.mkdir(parents=True)
        deck_path = workdir / f"deck{suffix}"
        deck_path.write_bytes(await deck.read())

        voice_path = None
        if voice is not None and voice.filename:
            voice_path = workdir / f"voice{Path(voice.filename).suffix or '.wav'}"
            voice_path.write_bytes(await voice.read())
        photo_path = None
        if photo is not None and photo.filename:
            photo_path = workdir / f"photo{Path(photo.filename).suffix or '.png'}"
            photo_path.write_bytes(await photo.read())

        options = {
            "narration_seconds": narration_seconds,
            "image_provider": image_provider,
            "avatar": (avatar or "true").lower() != "false",
        }
        job_yaml = _build_job_config(base_config, workdir, options, voice_path, photo_path)

        job = Job(id=job_id, workdir=workdir, created_at=time.time())
        with _LOCK:
            _JOBS[job_id] = job
        executor.submit(_run_job, job, deck_path, job_yaml, voice_path, photo_path)
        return JSONResponse({"job_id": job_id, "status": job.status})

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, _: None = Depends(require_token)) -> dict[str, Any]:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return {"job_id": job.id, "status": job.status, "error": job.error,
                "log": job.log[-4000:]}

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str, t: str | None = None,
                   authorization: str | None = Header(default=None)):
        # A browser download link can't set an Authorization header, so accept
        # the token via the ?t= query param here as well.
        if auth_token and authorization != f"Bearer {auth_token}" and t != auth_token:
            raise HTTPException(status_code=401, detail="Invalid or missing token")
        job = _JOBS.get(job_id)
        if job is None or job.status != "done" or job.output_path is None:
            raise HTTPException(status_code=404, detail="Result not ready")
        return FileResponse(str(job.output_path), media_type="video/mp4",
                            filename="slidestream.mp4")

    return app


# Single-page UI. Remembers the voice sample + photo in the browser (IndexedDB)
# so the lecturer never re-picks them; the server stores neither at rest.
INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SlideStream</title>
<style>
 body{font:16px system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem;color:#222}
 h1{font-size:1.5rem} label{display:block;margin:.8rem 0 .2rem;font-weight:600}
 input,select,button{font:inherit;padding:.5rem;width:100%;box-sizing:border-box}
 button{background:#2563eb;color:#fff;border:0;border-radius:6px;margin-top:1rem;cursor:pointer}
 .row{display:flex;gap:.5rem;align-items:center} .row input{width:auto}
 .muted{color:#666;font-size:.85rem} #log{white-space:pre-wrap;background:#f4f4f4;padding:.6rem;border-radius:6px;font-size:.8rem;max-height:220px;overflow:auto}
 .badge{display:inline-block;padding:.1rem .5rem;border-radius:99px;font-size:.8rem}
</style></head><body>
<h1>🎬 SlideStream</h1>
<div id="demo" style="display:none;background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:.7rem 1rem;margin-bottom:1rem;font-size:.9rem">
 <strong>Hosted demo.</strong> This instance runs a limited set of free tools.
 For full control over voice, image and video generation, install locally:
 <code>pip install slide-stream</code> — see
 <a href="https://github.com/michael-borck/slide-stream">the docs</a>.
</div>
<div id="tokrow" style="display:none"><label>Access token</label>
 <input id="token" type="password" placeholder="paste your token">
 <p class="muted">Stored in this browser only.</p></div>
<label>Deck (.md or .pptx)</label><input id="deck" type="file" accept=".md,.pptx">
<label>Your voice sample (optional, 10–30s)</label><input id="voice" type="file" accept="audio/*">
<label>Your photo (optional, front-facing)</label><input id="photo" type="file" accept="image/*">
<p class="muted" id="remembered"></p>
<div class="row"><input id="avatar" type="checkbox"><label style="margin:0">Talking-head avatar</label></div>
<label>Narration seconds per slide (optional)</label><input id="secs" type="number" min="10" placeholder="e.g. 30">
<button id="go">Create video</button>
<p id="status"></p><div id="log"></div>
<script>
const $=id=>document.getElementById(id);
$("token").value=localStorage.getItem("ss_token")||"";
$("token").oninput=e=>localStorage.setItem("ss_token",e.target.value);
// Bootstrap: show the token field only if required, and the demo banner if on.
fetch("/api/config").then(r=>r.json()).then(c=>{
 if(c.auth_required)$("tokrow").style.display="block";
 if(c.demo)$("demo").style.display="block";}).catch(()=>{});
// IndexedDB: remember voice + photo across jobs (client-side only).
let db;const openDB=()=>new Promise(r=>{const q=indexedDB.open("ss",1);
 q.onupgradeneeded=()=>q.result.createObjectStore("files");q.onsuccess=()=>{db=q.result;r()}});
const put=(k,v)=>new Promise(r=>{db.transaction("files","readwrite").objectStore("files").put(v,k).onsuccess=r});
const get=k=>new Promise(r=>{const q=db.transaction("files").objectStore("files").get(k);q.onsuccess=()=>r(q.result)});
let savedVoice,savedPhoto;
openDB().then(async()=>{savedVoice=await get("voice");savedPhoto=await get("photo");
 const b=[];if(savedVoice)b.push("voice: "+savedVoice.name);if(savedPhoto)b.push("photo: "+savedPhoto.name);
 $("remembered").textContent=b.length?("Remembered "+b.join(", ")+" — leave the fields empty to reuse."):""});
const auth=()=>({Authorization:"Bearer "+$("token").value});
async function fileOrSaved(input,key,saved){const f=input.files[0];
 if(f){await put(key,f);return f}return saved||null}
$("go").onclick=async()=>{
 const deck=$("deck").files[0];if(!deck){$("status").textContent="Pick a deck first.";return}
 const voice=await fileOrSaved($("voice"),"voice",savedVoice);
 const photo=await fileOrSaved($("photo"),"photo",savedPhoto);
 const fd=new FormData();fd.append("deck",deck);
 if(voice)fd.append("voice",voice);if(photo)fd.append("photo",photo);
 fd.append("avatar",$("avatar").checked?"true":"false");
 if($("secs").value)fd.append("narration_seconds",$("secs").value);
 $("status").textContent="Uploading…";$("log").textContent="";
 let res=await fetch("/api/jobs",{method:"POST",headers:auth(),body:fd});
 if(!res.ok){$("status").textContent="Error: "+(await res.text());return}
 const {job_id}=await res.json();poll(job_id)};
async function poll(id){
 const r=await fetch("/api/jobs/"+id,{headers:auth()});const j=await r.json();
 $("status").innerHTML='<span class="badge">'+j.status+'</span>';$("log").textContent=j.log||"";
 if(j.status==="done"){$("status").innerHTML+=' <a href="/api/jobs/'+id+'/result?t='+
   encodeURIComponent($("token").value)+'" download>⬇ download video</a>';return}
 if(j.status==="error"){$("status").textContent="Failed: "+(j.error||"see log");return}
 setTimeout(()=>poll(id),2500)}
</script></body></html>"""
