"""Talking-head avatar provider implementations."""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from rich.console import Console

from ..avatars import mouth_box, resolve_avatar
from .base import AvatarProvider

# Proven SadTalker ComfyUI workflow (see docs referenced by sadtalker-api.md):
# LoadImage + LoadAudio -> SadTalker -> ShowVideo (terminal node surfaces the
# output path). Only the image/audio filenames and a few tuned params vary.
SADTALKER_WORKFLOW: dict[str, Any] = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "portrait.png"}},
    "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}},
    "3": {
        "class_type": "SadTalker",
        "inputs": {
            "image": ["1", 0],
            "audio": ["2", 0],
            "poseStyle": 0,
            "faceModelResolution": "256",
            "preprocess": "full",
            "stillMode": True,
            "batchSizeInGeneration": 2,
            "gfpganAsFaceEnhancer": False,
            "useIdleMode": False,
            "idleModeTime": 5,
            "useRefVideo": False,
            "refInfo": "pose",
        },
    },
    "4": {"class_type": "ShowVideo", "inputs": {"show_video_path": ["3", 1]}},
}

# Proven Wav2Lip ComfyUI workflow (see docs/wav2lip-api.md): load a reference
# video -> re-sync its mouth to new audio -> encode MP4. Family 2 (video mouth
# swap); the source frames loop under longer narration.
WAV2LIP_WORKFLOW: dict[str, Any] = {
    "1": {"class_type": "VHS_LoadVideo", "inputs": {
        "video": "reference.mp4", "force_rate": 0, "custom_width": 0,
        "custom_height": 0, "frame_load_cap": 0, "skip_first_frames": 0,
        "select_every_nth": 1}},
    "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}},
    "3": {"class_type": "Wav2Lip", "inputs": {
        "images": ["1", 0], "mode": "repetitive",
        "face_detect_batch": 8, "audio": ["2", 0]}},
    "4": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["3", 0], "audio": ["3", 1], "frame_rate": 25,
        "loop_count": 0, "filename_prefix": "wav2lip_out",
        "format": "video/h264-mp4", "pingpong": True, "save_output": True}},
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def _source_kind(path: str) -> str:
    """'video' if the path looks like a video file, else 'image'."""
    return "video" if Path(path).suffix.lower() in VIDEO_EXTS else "image"

console = Console()
err_console = Console(stderr=True, style="bold red")


class NoneAvatarProvider(AvatarProvider):
    """Default provider: avatar feature disabled, never produces a head."""

    @property
    def name(self) -> str:
        return "none"

    def is_available(self) -> bool:
        return True

    def generate(
        self, audio_path: str, output_path: str, slide_num: int
    ) -> str | None:
        return None


class PrecomputedAvatarProvider(AvatarProvider):
    """Use pre-supplied head clips instead of generating them.

    Maps slide N to ``{assets_dir}/head_N.mp4``. This needs no GPU or
    lip-sync service: record (or render) the clips once and drop them in a
    directory. Slides without a matching clip render without a head.
    """

    def _assets_dir(self) -> Path | None:
        avatar_config = self.config.get("providers", {}).get("avatar", {})
        assets_dir = avatar_config.get("assets_dir")
        return Path(assets_dir) if assets_dir else None

    @property
    def name(self) -> str:
        return "precomputed"

    def is_available(self) -> bool:
        """Available when the configured assets directory exists."""
        assets_dir = self._assets_dir()
        return assets_dir is not None and assets_dir.is_dir()

    def generate(
        self, audio_path: str, output_path: str, slide_num: int
    ) -> str | None:
        assets_dir = self._assets_dir()
        if assets_dir is None:
            err_console.print("  - Precomputed avatar: no assets_dir configured.")
            return None
        head = assets_dir / f"head_{slide_num}.mp4"
        if not head.is_file():
            err_console.print(f"  - Precomputed avatar: {head.name} not found.")
            return None
        console.print(f"  - Using precomputed head clip: {head.name}")
        return str(head)


class StaticAvatarProvider(AvatarProvider):
    """A static mascot image held in the corner — no lip-sync, no GPU.

    Composites the avatar image (a built-in character name like ``teddy`` or a
    file path) as the corner circle for each slide. Reliable and free: ideal
    for the fun character defaults, which stylized-face lip-sync engines can't
    animate. An obviously-not-human mascot plus a fun accent dodges the
    uncanny valley by design.

    Config: ``provider: static``, ``source: teddy`` (or a path/URL-less file).
    """

    @property
    def name(self) -> str:
        return "static"

    def _source(self) -> str | None:
        cfg = self.config.get("providers", {}).get("avatar", {})
        return resolve_avatar(cfg.get("source") or cfg.get("source_image"))

    def is_available(self) -> bool:
        source = self._source()
        return bool(source and Path(source).is_file())

    def generate(
        self, audio_path: str, output_path: str, slide_num: int
    ) -> str | None:
        try:
            from moviepy import ImageClip

            source = self._source()
            if not source or not Path(source).is_file():
                raise FileNotFoundError(f"avatar source not found: {source}")
            # A short still clip; media.py freezes its last frame to fill the
            # slide, so the mascot simply sits in the corner while audio plays.
            clip = ImageClip(source, duration=1.0)
            clip.write_videofile(
                output_path, fps=24, codec="libx264", logger=None
            )
            clip.close()
            console.print(f"  - Static avatar: {Path(source).name}")
            return output_path
        except Exception as e:
            err_console.print(f"  - Static avatar error: {e}")
            return None


class PuppetAvatarProvider(AvatarProvider):
    """Non-AI cartoon mouth-flap: opens the mouth with the audio loudness.

    Draws a simple mouth over the avatar's mouth region, its openness driven by
    the audio's loudness envelope (closed on silence). No AI, no face detection,
    no GPU — works on any mascot. Deliberately crude, which reads as charming
    for a cartoon (South Park / Character Animator amplitude lip-sync).

    Config: ``provider: puppet``, ``source: teddy`` (built-in name or image
    path). For a custom image, set the mouth region with
    ``mouth: [cx, cy, w, h]`` (fractions of the image).
    """

    FPS = 12

    @property
    def name(self) -> str:
        return "puppet"

    def _source(self) -> str | None:
        cfg = self.config.get("providers", {}).get("avatar", {})
        return cfg.get("source") or cfg.get("source_image")

    def is_available(self) -> bool:
        resolved = resolve_avatar(self._source())
        return bool(resolved and Path(resolved).is_file())

    def _mouth_box(self, source: str | None) -> tuple[float, float, float, float]:
        cfg = self.config.get("providers", {}).get("avatar", {})
        override = cfg.get("mouth")
        if isinstance(override, list | tuple) and len(override) == 4:
            return tuple(float(v) for v in override)  # type: ignore[return-value]
        return mouth_box(source)

    def generate(
        self, audio_path: str, output_path: str, slide_num: int
    ) -> str | None:
        try:
            import wave

            import numpy as np
            from moviepy import ImageSequenceClip
            from PIL import Image, ImageDraw

            raw_source = self._source()
            source = resolve_avatar(raw_source)
            if not source or not Path(source).is_file():
                raise FileNotFoundError(f"avatar source not found: {source}")

            # Read the audio as 16kHz mono PCM (ffmpeg handles mp3/wav/...),
            # then compute a per-frame RMS loudness envelope.
            wav_path, is_temp = _to_16k_wav(audio_path)
            try:
                with wave.open(wav_path, "rb") as wf:
                    sr = wf.getframerate()
                    raw = wf.readframes(wf.getnframes())
            finally:
                if is_temp:
                    Path(wav_path).unlink(missing_ok=True)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            duration = len(samples) / sr if sr else 0.0

            n_frames = max(1, int(duration * self.FPS))
            step = sr / self.FPS
            env = np.zeros(n_frames)
            for i in range(n_frames):
                seg = samples[int(i * step): int((i + 1) * step)]
                if len(seg):
                    env[i] = float(np.sqrt(np.mean(seg**2)))
            peak = float(env.max()) or 1.0
            openness = np.clip(env / (peak * 0.5), 0.0, 1.0)
            openness[env < peak * 0.10] = 0.0  # rest the mouth during silence

            base = Image.open(source).convert("RGB")
            width, height = base.size
            cx, cy, mw, mh = self._mouth_box(raw_source)

            frames = []
            for value in openness:
                frame = base.copy()
                if value > 0.12:
                    draw = ImageDraw.Draw(frame)
                    hh = value * mh * height
                    ww = mw * width
                    draw.ellipse(
                        [cx * width - ww / 2, cy * height - hh / 2,
                         cx * width + ww / 2, cy * height + hh / 2],
                        fill=(45, 18, 22),
                    )
                frames.append(np.asarray(frame))

            clip = ImageSequenceClip(frames, fps=self.FPS)
            clip.write_videofile(
                output_path, fps=self.FPS, codec="libx264", logger=None
            )
            clip.close()
            console.print(f"  - Puppet mouth-flap: {Path(source).name}")
            return output_path
        except Exception as e:
            err_console.print(f"  - Puppet avatar error: {e}")
            return None


class DIDAvatarProvider(AvatarProvider):
    """Generate a lip-synced talking head via the D-ID Talks API (BYOK).

    For each slide, the narration audio is driven onto a single source image
    (the lecturer's photo) to produce a talking-head clip, which media.py
    composites over the slide. No GPU needed — you bring a D-ID API key.

    Config (``providers.avatar``):
      provider: d-id
      source_image: ./lecturer.jpg   # local path or a public image URL
      api_key: "${DID_API_KEY}"      # D-ID Basic auth key from studio.d-id.com
    Optional: ``timeout`` (per-slide, default 300s), ``poll_interval`` (3s).
    """

    BASE_URL = "https://api.d-id.com"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        # The source image is uploaded once per run; cache the resulting URL.
        self._source_url: str | None = None

    @property
    def name(self) -> str:
        return "d-id"

    def _avatar_config(self) -> dict[str, Any]:
        return self.config.get("providers", {}).get("avatar", {})

    def _api_key(self) -> str | None:
        return self._avatar_config().get("api_key") or os.getenv("DID_API_KEY")

    def _source_image(self) -> str | None:
        return self._avatar_config().get("source_image")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Basic {self._api_key()}",
            "accept": "application/json",
        }

    def is_available(self) -> bool:
        """Available when an API key and a source image are configured."""
        return bool(self._api_key() and self._source_image())

    def _ensure_source_url(self) -> str:
        """Return a D-ID-usable source image URL, uploading a local file once."""
        if self._source_url:
            return self._source_url
        source = resolve_avatar(self._source_image())
        if source is None:
            raise ValueError("no source_image configured")
        if source.startswith(("http://", "https://")):
            self._source_url = source
            return source
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"source_image not found: {source}")
        with open(path, "rb") as f:
            response = requests.post(
                f"{self.BASE_URL}/images",
                headers=self._headers(),
                files={"image": (path.name, f)},
                timeout=60,
            )
        response.raise_for_status()
        url: str = response.json()["url"]
        self._source_url = url
        console.print("  - Uploaded avatar source image to D-ID")
        return url

    def _upload_audio(self, audio_path: str) -> str:
        path = Path(audio_path)
        with open(path, "rb") as f:
            response = requests.post(
                f"{self.BASE_URL}/audios",
                headers=self._headers(),
                files={"audio": (path.name, f)},
                timeout=60,
            )
        response.raise_for_status()
        return response.json()["url"]

    def generate(
        self, audio_path: str, output_path: str, slide_num: int
    ) -> str | None:
        try:
            avatar_config = self._avatar_config()
            timeout = float(avatar_config.get("timeout") or 300)
            poll_interval = float(avatar_config.get("poll_interval") or 3)
            headers = {**self._headers(), "content-type": "application/json"}

            source_url = self._ensure_source_url()
            audio_url = self._upload_audio(audio_path)

            # 1. Create the talk (audio-driven).
            create = requests.post(
                f"{self.BASE_URL}/talks",
                headers=headers,
                json={
                    "source_url": source_url,
                    "script": {"type": "audio", "audio_url": audio_url},
                },
                timeout=60,
            )
            create.raise_for_status()
            talk_id = create.json()["id"]

            # 2. Poll until the render is done.
            deadline = time.monotonic() + timeout
            result_url: str | None = None
            while time.monotonic() < deadline:
                status_resp = requests.get(
                    f"{self.BASE_URL}/talks/{talk_id}", headers=headers, timeout=30
                )
                status_resp.raise_for_status()
                data = status_resp.json()
                status = data.get("status")
                if status == "done":
                    result_url = data.get("result_url")
                    break
                if status == "error":
                    raise ValueError(f"D-ID render failed: {data.get('error', data)}")
                time.sleep(poll_interval)

            if not result_url:
                raise TimeoutError(f"D-ID render timed out after {timeout:.0f}s")

            # 3. Download the finished clip.
            clip = requests.get(result_url, timeout=timeout)
            clip.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(clip.content)
            console.print(f"  - Generated D-ID talking head (slide {slide_num})")
            return output_path

        except Exception as e:
            err_console.print(f"  - D-ID avatar error: {e}")
            return None


class _ComfyUIAvatar(AvatarProvider):
    """Shared base for ComfyUI-driven talking heads (SadTalker / Wav2Lip).

    Both drive a ComfyUI server the same way — upload the source and audio,
    POST a workflow to ``/prompt``, poll ``/history``, download via ``/view``.
    They differ only in the source type (photo vs video) and the workflow
    graph, both keyed off ``_kind()`` ('image' -> SadTalker, 'video' ->
    Wav2Lip). Subclasses set the provider name and the source.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        # The source is uploaded once per run; cache the input filename.
        self._source_name: str | None = None

    def _avatar_config(self) -> dict[str, Any]:
        return self.config.get("providers", {}).get("avatar", {})

    def _base_url(self) -> str | None:
        base_url = self._avatar_config().get("base_url") or os.getenv("COMFYUI_BASE_URL")
        return base_url.rstrip("/") if base_url else None

    def _headers(self) -> dict[str, str]:
        api_key = self._avatar_config().get("api_key") or os.getenv("COMFYUI_TOKEN")
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _source(self) -> str | None:
        """Path/name of the source photo or video (subclass-specific)."""
        raise NotImplementedError

    def _kind(self) -> str:
        """'image' or 'video' — detected from the source by default."""
        return _source_kind(self._source() or "")

    def is_available(self) -> bool:
        """Available when a ComfyUI URL and a source are configured."""
        return bool(self._base_url() and self._source())

    def _upload(self, path: Path) -> str:
        """Upload a file to ComfyUI's input/ dir; return its stored name."""
        with open(path, "rb") as f:
            response = requests.post(
                f"{self._base_url()}/upload/image",
                headers=self._headers(),
                files={"image": (path.name, f)},
                timeout=120,
            )
        response.raise_for_status()
        return response.json()["name"]

    def _ensure_source(self) -> str:
        """Return the input/ filename for the source, uploading it once."""
        if self._source_name:
            return self._source_name
        source = resolve_avatar(self._source())
        if source is None:
            raise ValueError("no avatar source configured")
        path = Path(source)
        # A bare filename that is not a local file is assumed to already live in
        # ComfyUI's input/ directory.
        self._source_name = self._upload(path) if path.is_file() else source
        if path.is_file():
            console.print("  - Uploaded avatar source to ComfyUI")
        return self._source_name

    def _build_workflow(self, kind: str, source_name: str, audio_name: str) -> dict[str, Any]:
        import copy

        cfg = self._avatar_config()
        if kind == "video":
            wf = copy.deepcopy(WAV2LIP_WORKFLOW)
            wf["1"]["inputs"]["video"] = source_name
            wf["2"]["inputs"]["audio"] = audio_name
            wf["3"]["inputs"]["mode"] = cfg.get("wav2lip_mode") or "repetitive"
            wf["3"]["inputs"]["face_detect_batch"] = int(cfg.get("face_detect_batch") or 8)
            wf["4"]["inputs"]["frame_rate"] = int(cfg.get("frame_rate") or 25)
            wf["4"]["inputs"]["pingpong"] = bool(cfg.get("pingpong", True))
            return wf
        wf = copy.deepcopy(SADTALKER_WORKFLOW)
        wf["1"]["inputs"]["image"] = source_name
        wf["2"]["inputs"]["audio"] = audio_name
        sad = wf["3"]["inputs"]
        sad["stillMode"] = bool(cfg.get("still_mode", True))
        sad["preprocess"] = cfg.get("preprocess") or "full"
        sad["faceModelResolution"] = str(cfg.get("face_resolution") or "256")
        sad["poseStyle"] = int(cfg.get("pose_style") or 0)
        sad["gfpganAsFaceEnhancer"] = bool(cfg.get("face_enhancer", False))
        return wf

    def generate(
        self, audio_path: str, output_path: str, slide_num: int
    ) -> str | None:
        try:
            base_url = self._base_url()
            if not base_url:
                raise ValueError("ComfyUI base_url not configured")
            cfg = self._avatar_config()
            timeout = float(cfg.get("timeout") or 600)
            poll_interval = float(cfg.get("poll_interval") or 5)
            headers = self._headers()

            source_name = self._ensure_source()
            kind = self._kind()
            # These nodes load audio at 16kHz; some installs crash resampling
            # from other rates (librosa/resampy/numba mismatch). Feed 16kHz
            # mono WAV so they never resample.
            audio_16k, is_temp = _to_16k_wav(audio_path)
            try:
                audio_name = self._upload(Path(audio_16k))
            finally:
                if is_temp:
                    Path(audio_16k).unlink(missing_ok=True)
            workflow = self._build_workflow(kind, source_name, audio_name)

            # 1. Submit the workflow.
            submit = requests.post(
                f"{base_url}/prompt",
                headers={**headers, "content-type": "application/json"},
                json={"prompt": workflow},
                timeout=60,
            )
            submit.raise_for_status()
            prompt_id = submit.json()["prompt_id"]

            # 2. Poll history until success.
            deadline = time.monotonic() + timeout
            video_path: str | None = None
            while time.monotonic() < deadline:
                hist = requests.get(
                    f"{base_url}/history/{prompt_id}", headers=headers, timeout=30
                )
                hist.raise_for_status()
                data = hist.json()
                entry = data.get(prompt_id)
                if entry:
                    status_str = entry.get("status", {}).get("status_str")
                    if status_str == "success":
                        video_path = _find_output_path(kind, entry.get("outputs", {}))
                        break
                    if status_str == "error":
                        raise ValueError(f"ComfyUI workflow error: {entry.get('status')}")
                time.sleep(poll_interval)

            if not video_path:
                raise TimeoutError(f"{self.name} render timed out after {timeout:.0f}s")

            # 3. Download the result via /view.
            clip = requests.get(
                f"{base_url}/view",
                headers=headers,
                params={"filename": Path(video_path).name, "type": "output"},
                timeout=timeout,
            )
            clip.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(clip.content)
            console.print(f"  - Generated {self.name} talking head (slide {slide_num})")
            return output_path

        except Exception as e:
            err_console.print(f"  - {self.name} avatar error: {e}")
            return None


class SadTalkerAvatarProvider(_ComfyUIAvatar):
    """SadTalker (photo -> talking head) via ComfyUI. Family 1.

    Config (``providers.avatar``): ``provider: sadtalker``, ``base_url``,
    ``source_image`` (photo), optional ``api_key`` (Bearer). Tuning:
    ``still_mode``, ``preprocess``, ``face_resolution``, ``pose_style``,
    ``face_enhancer``. ``still_mode`` on by default to reduce corner-circle jank.
    """

    @property
    def name(self) -> str:
        return "sadtalker"

    def _source(self) -> str | None:
        return self._avatar_config().get("source_image")

    def _kind(self) -> str:
        return "image"


class Wav2LipAvatarProvider(_ComfyUIAvatar):
    """Wav2Lip (video -> lip-synced) via ComfyUI. Family 2.

    Config (``providers.avatar``): ``provider: wav2lip``, ``base_url``,
    ``source_video`` (a short idle clip of the lecturer), optional ``api_key``.
    The source loops (pingpong) under longer narration. Tuning: ``wav2lip_mode``,
    ``face_detect_batch``, ``frame_rate``, ``pingpong``.
    """

    @property
    def name(self) -> str:
        return "wav2lip"

    def _source(self) -> str | None:
        return self._avatar_config().get("source_video")

    def _kind(self) -> str:
        return "video"


class ComfyUIAvatarProvider(_ComfyUIAvatar):
    """Auto-routing talking head: photo -> SadTalker, video -> Wav2Lip.

    Detects the source type from its extension, so one provider handles both.
    Config (``providers.avatar``): ``provider: comfyui``, ``base_url``, and a
    ``source`` (or ``source_image`` / ``source_video``) that may be either a
    photo or a video. Ideal for the web UI, where the user uploads either.
    """

    @property
    def name(self) -> str:
        return "comfyui"

    def _source(self) -> str | None:
        cfg = self._avatar_config()
        return cfg.get("source") or cfg.get("source_image") or cfg.get("source_video")


def _to_16k_wav(audio_path: str) -> tuple[str, bool]:
    """Return a 16kHz mono WAV path for the audio, converting via ffmpeg.

    Returns (path, is_temporary). If ffmpeg is unavailable, returns the input
    unchanged so upload is still attempted.
    """
    if not shutil.which("ffmpeg"):
        return audio_path, False
    fd, converted = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", audio_path,
             "-ar", "16000", "-ac", "1", converted],
            check=True,
        )
    except Exception:
        Path(converted).unlink(missing_ok=True)
        return audio_path, False
    return converted, True


def _find_output_path(kind: str, outputs: dict[str, Any]) -> str | None:
    """Pull the output video path from a ComfyUI history entry.

    SadTalker's ShowVideo node reports ``show_video_path``; Wav2Lip's
    VHS_VideoCombine reports ``gifs`` (with filename/fullpath).
    """
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        if kind == "video":
            gifs = node_output.get("gifs")
            if gifs:
                return gifs[0].get("filename") or Path(gifs[0]["fullpath"]).name
        else:
            paths = node_output.get("show_video_path")
            if paths:
                return paths[0]
    return None
