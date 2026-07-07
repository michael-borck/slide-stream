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

# Demo-mode guardrails: friction-free (no token) but bounded.
DEMO_MAX_SLIDES = 5
DEMO_JOBS_PER_HOUR = 3
_DEMO_HITS: dict[str, list[float]] = {}  # client ip -> job timestamps


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

    # A per-job voice sample turns on ephemeral Chatterbox cloning.
    if voice_path is not None:
        cfg["providers"]["tts"]["voice_sample"] = str(voice_path)

    # Presenter: a built-in mascot wins over an uploaded file. The 'animate'
    # toggle then picks the engine per source:
    #   mascot  + animate -> puppet (mouth-flap, no GPU);  else static mascot
    #   photo   + animate -> server's engine (d-id/sadtalker/comfyui);
    #                        else static photo (a still of themselves)
    #   video             -> always the video engine (a clip is inherently
    #                        animated; wav2lip/comfyui)
    #   nothing           -> no head
    av = cfg["providers"]["avatar"]
    animate = options.get("avatar", True)
    if options.get("avatar_name"):
        av["provider"] = "puppet" if animate else "static"
        av["source"] = options["avatar_name"]
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
            Request,
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
        # Demo mode is friction-free: no token, guarded by rate/slide limits
        # instead. A token only gates private/full instances.
        if demo_mode or not auth_token:
            return
        expected = f"Bearer {auth_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing token")

    def client_ip(request: Request) -> str:
        # Behind a reverse proxy (NPM) the real IP is in X-Forwarded-For.
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
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
            "limits": (
                {"max_slides": DEMO_MAX_SLIDES, "jobs_per_hour": DEMO_JOBS_PER_HOUR}
                if demo_mode
                else None
            ),
            "avatars": avatar_names(),
            "accents": list(GTTS_ACCENTS) if tts_provider == "gtts" else [],
        }

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
        _: None = Depends(require_token),
    ) -> JSONResponse:
        suffix = Path(deck.filename or "deck.md").suffix.lower()
        if suffix not in (".md", ".pptx"):
            raise HTTPException(status_code=400, detail="Deck must be .md or .pptx")

        if demo_mode and not _demo_rate_ok(client_ip(request)):
            raise HTTPException(
                status_code=429,
                detail=f"Demo limit: {DEMO_JOBS_PER_HOUR} videos per hour. "
                "Install locally for unlimited renders: pip install slide-stream",
            )

        job_id = uuid.uuid4().hex
        workdir = jobs_root / job_id
        workdir.mkdir(parents=True)
        deck_path = workdir / f"deck{suffix}"
        deck_path.write_bytes(await deck.read())

        if demo_mode:
            n = _count_slides(deck_path)
            if n is not None and n > DEMO_MAX_SLIDES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Demo limit: {DEMO_MAX_SLIDES} slides per deck "
                    f"(yours has {n}). Install locally for unlimited decks: "
                    "pip install slide-stream",
                )

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
            "avatar_name": avatar_name,
            "accent": accent,
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
        if (not demo_mode and auth_token
                and authorization != f"Bearer {auth_token}" and t != auth_token):
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
footer{margin-top:2rem;padding-top:1.1rem;border-top:1px solid var(--line);
 font-size:.85rem;color:var(--muted);display:flex;gap:1.2rem;flex-wrap:wrap}
footer a{color:var(--muted);text-decoration:none;font-weight:500}
footer a:hover{color:var(--accent)}
</style></head><body><div class="wrap">
<h1>🎬 Slide<em>Stream</em></h1>
<p class="tag">Slides in, narrated video out — in your voice, or a friendly mascot's.</p>
<div id="demo" class="banner">
 <strong>Hosted demo</strong> — <span id="limits">limited</span>, nothing stored.
 Want unlimited renders, your own AI providers and full privacy?
 <code>pip install slide-stream</code> &middot;
 <a href="https://slidestream.eduserver.au">learn more</a> &middot;
 <a href="https://github.com/michael-borck/slide-stream">GitHub</a>
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
<button id="go">Create video</button>
<p id="status"></p><div id="log"></div>
</div>
<footer>
 <a href="https://slidestream.eduserver.au">About</a>
 <a href="https://pypi.org/project/slide-stream/">pip install slide-stream</a>
 <a href="https://github.com/michael-borck/slide-stream">GitHub</a>
</footer>
</div>
<script>
const $=id=>document.getElementById(id);
$("token").value=localStorage.getItem("ss_token")||"";
$("token").oninput=e=>localStorage.setItem("ss_token",e.target.value);
// Bootstrap: show the token field only if required, and the demo banner if on.
fetch("/api/config").then(r=>r.json()).then(c=>{
 if(c.auth_required)$("tokrow").style.display="block";
 if(c.demo){$("demo").style.display="block";
  if(c.limits)$("limits").textContent=
   c.limits.max_slides+" slides per deck, "+c.limits.jobs_per_hour+" videos per hour";}
 (c.avatars||[]).forEach(a=>{const o=document.createElement("option");o.value=a;o.textContent=a;$("avatarName").appendChild(o)});
 if((c.accents||[]).length){$("accentRow").style.display="block";$("accent").style.display="block";
  c.accents.forEach(a=>{const o=document.createElement("option");o.value=a;o.textContent=a;$("accent").appendChild(o)})}
}).catch(()=>{});
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
$("go").onclick=async()=>{
 const deck=$("deck").files[0];if(!deck){$("status").textContent="Pick a deck first.";return}
 const voice=await fileOrSaved($("voice"),"voice",savedVoice);
 const photo=await fileOrSaved($("photo"),"photo",savedPhoto);
 const fd=new FormData();fd.append("deck",deck);
 if(voice)fd.append("voice",voice);if(photo)fd.append("photo",photo);
 fd.append("avatar",$("avatar").checked?"true":"false");
 if($("avatarName").value)fd.append("avatar_name",$("avatarName").value);
 if($("accent").value)fd.append("accent",$("accent").value);
 if($("secs").value)fd.append("narration_seconds",$("secs").value);
 $("status").textContent="Uploading…";$("log").textContent="";
 let res=await fetch("/api/jobs",{method:"POST",headers:auth(),body:fd});
 if(!res.ok){$("status").textContent="Error: "+(await res.text());return}
 const {job_id}=await res.json();poll(job_id)};
async function poll(id){
 const r=await fetch("/api/jobs/"+id,{headers:auth()});const j=await r.json();
 $("status").innerHTML='<span class="badge">'+j.status+'</span>';
 $("log").textContent=j.log||"";$("log").classList.toggle("on",!!j.log);
 if(j.status==="done"){$("status").innerHTML+=' <a href="/api/jobs/'+id+'/result?t='+
   encodeURIComponent($("token").value)+'" download>⬇ download video</a>';return}
 if(j.status==="error"){$("status").textContent="Failed: "+(j.error||"see log");return}
 setTimeout(()=>poll(id),2500)}
</script></body></html>"""
