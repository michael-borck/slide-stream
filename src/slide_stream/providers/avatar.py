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

from ..avatars import avatar_prompt, mouth_box, resolve_avatar
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

# Proven Wan2.2-S2V ComfyUI workflow (see docs/wan-s2v-api.md): still image +
# audio -> lip-synced video, with NO face detector anywhere in the path, so it
# animates non-human mascots and human head shots alike. Node 8/9 hold the
# positive/negative prompts, 10 the frame size + length (4n+1 @ 16fps), 11 the
# sampler; node 14 (SaveVideo) surfaces the MP4 under gifs[0].
WAN_S2V_WORKFLOW: dict[str, Any] = {
    "1": {"class_type": "UNETLoader", "inputs": {
        "unet_name": "wan2.2_s2v_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {
        "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
    "4": {"class_type": "AudioEncoderLoader", "inputs": {
        "audio_encoder_name": "wav2vec2_large_english_fp16.safetensors"}},
    "5": {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}},
    "6": {"class_type": "AudioEncoderEncode", "inputs": {
        "audio_encoder": ["4", 0], "audio": ["5", 0]}},
    "7": {"class_type": "LoadImage", "inputs": {"image": "character.png"}},
    "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
    "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
    "10": {"class_type": "WanSoundImageToVideo", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "vae": ["3", 0],
        "audio_encoder_output": ["6", 0], "ref_image": ["7", 0],
        "width": 448, "height": 608, "length": 49}},
    "11": {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["10", 0], "negative": ["10", 1],
        "latent_image": ["10", 2], "seed": 42, "steps": 20, "cfg": 6.0,
        "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0}},
    "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
    "13": {"class_type": "CreateVideo", "inputs": {
        "images": ["12", 0], "fps": 16, "audio": ["5", 0]}},
    "14": {"class_type": "SaveVideo", "inputs": {
        "video": ["13", 0], "filename_prefix": "s2v", "format": "mp4", "codec": "h264"}},
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def _s2v_length(seconds: float, fps: int = 16) -> int:
    """Frame count for a Wan2.2-S2V run: the nearest valid ``4n+1`` at 16fps.

    Wan runs at 16fps and requires ``length == 4n+1`` (n>=1), so 3s->49,
    4s->65, 5s->81. Sub-second audio still yields a minimal 5-frame clip.
    """
    frames = max(1, round(seconds * fps))
    n = max(1, round((frames - 1) / 4))
    return n * 4 + 1


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

    def _output_style(self) -> str:
        """Which history field holds the result: 'image' -> show_video_path,
        'video' -> gifs. Defaults to the source kind (SadTalker vs Wav2Lip);
        S2V overrides it because it takes an image but emits a gifs video."""
        return self._kind()

    def _pre_generate(self, base_url: str, headers: dict[str, str]) -> None:
        """Hook run just before the workflow is queued. No-op by default;
        S2V overrides it to free VRAM (POST /free)."""
        return None

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

    def _build_workflow(
        self, kind: str, source_name: str, audio_name: str, audio_seconds: float = 0.0
    ) -> dict[str, Any]:
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
                audio_seconds = _wav_seconds(audio_16k)
            finally:
                if is_temp:
                    Path(audio_16k).unlink(missing_ok=True)
            workflow = self._build_workflow(
                kind, source_name, audio_name, audio_seconds
            )

            # Provider-specific pre-flight (e.g. S2V frees VRAM before queueing).
            self._pre_generate(base_url, headers)

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
            subfolder = ""
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
                        found = _find_output_path(
                            self._output_style(), entry.get("outputs", {})
                        )
                        if found:
                            video_path, subfolder = found
                        break
                    if status_str == "error":
                        raise ValueError(f"ComfyUI workflow error: {entry.get('status')}")
                time.sleep(poll_interval)

            if not video_path:
                raise TimeoutError(f"{self.name} render timed out after {timeout:.0f}s")

            # 3. Download the result via /view. History entries may place the
            # file in an output subfolder; pass it through when present.
            params = {"filename": Path(video_path).name, "type": "output"}
            if subfolder:
                params["subfolder"] = subfolder
            clip = requests.get(
                f"{base_url}/view",
                headers=headers,
                params=params,
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


class WanS2VAvatarProvider(_ComfyUIAvatar):
    """Wan2.2-S2V (still image + audio -> talking video) via ComfyUI.

    Unlike SadTalker/Wav2Lip/LivePortrait, Wan2.2-S2V has **no face detector**
    anywhere in its path, so it animates non-human mascots (teddy, owl, robot,
    wizard) as well as human head shots — the narration audio drives the mouth
    and a text prompt describes the mouth anatomy. See docs/wan-s2v-api.md.

    Config (``providers.avatar``): ``provider: wan-s2v``, ``base_url``,
    ``source`` / ``source_image`` (a still image or a built-in avatar name),
    optional ``api_key`` (Bearer, or ``COMFYUI_TOKEN``).

    Duration: renders a short ``clip_seconds`` (default 4) clip that media.py
    loops under longer narration — S2V is ~4 min per 3s of video, so full-
    length is costly. Set ``full_length: true`` to render the whole narration
    with exact lip-sync instead. Tuning: ``prompt``, ``negative_prompt``,
    ``width`` (448), ``height`` (608), ``steps`` (20), ``cfg`` (6.0), ``seed``.
    """

    @property
    def name(self) -> str:
        return "wan-s2v"

    def _source(self) -> str | None:
        cfg = self._avatar_config()
        return cfg.get("source") or cfg.get("source_image")

    def _kind(self) -> str:
        return "image"

    def _output_style(self) -> str:
        # S2V's SaveVideo node reports the MP4 under gifs[0], like the VHS path.
        return "video"

    def _pre_generate(self, base_url: str, headers: dict[str, str]) -> None:
        # The 16.4GB unet stays resident after a run, so the next run's text
        # encoder OOMs unless models are freed first (docs/wan-s2v-api.md).
        # Best-effort: a fresh server simply has nothing to free.
        try:
            requests.post(
                f"{base_url}/free",
                headers=headers,
                json={"unload_models": True, "free_memory": True},
                timeout=60,
            )
        except Exception as e:
            err_console.print(f"  - wan-s2v: /free failed (continuing): {e}")

    def _build_workflow(
        self, kind: str, source_name: str, audio_name: str, audio_seconds: float = 0.0
    ) -> dict[str, Any]:
        import copy

        cfg = self._avatar_config()
        full = bool(cfg.get("full_length", False))
        clip_seconds = float(cfg.get("clip_seconds") or 4.0)
        # Never render longer than the audio; short mode caps at clip_seconds.
        seconds = audio_seconds if (full and audio_seconds > 0) else clip_seconds
        if audio_seconds > 0:
            seconds = min(seconds, audio_seconds)
        length = _s2v_length(seconds)

        prompt = cfg.get("prompt") or avatar_prompt(self._source())
        negative = cfg.get("negative_prompt") or "blurry, distorted, static, still"

        wf = copy.deepcopy(WAN_S2V_WORKFLOW)
        wf["5"]["inputs"]["audio"] = audio_name
        wf["7"]["inputs"]["image"] = source_name
        wf["8"]["inputs"]["text"] = prompt
        wf["9"]["inputs"]["text"] = negative
        wf["10"]["inputs"]["width"] = int(cfg.get("width") or 448)
        wf["10"]["inputs"]["height"] = int(cfg.get("height") or 608)
        wf["10"]["inputs"]["length"] = length
        wf["11"]["inputs"]["steps"] = int(cfg.get("steps") or 20)
        wf["11"]["inputs"]["cfg"] = float(cfg.get("cfg") or 6.0)
        wf["11"]["inputs"]["seed"] = int(cfg.get("seed") or 42)
        return wf


def _wav_seconds(path: str) -> float:
    """Duration of a WAV file in seconds, or 0.0 if it can't be read.

    Used to size a Wan2.2-S2V run to the narration; a 0.0 result makes S2V
    fall back to its configured ``clip_seconds``.
    """
    import wave

    try:
        with wave.open(path, "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() / rate if rate else 0.0
    except Exception:
        return 0.0


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


def _find_output_path(kind: str, outputs: dict[str, Any]) -> tuple[str, str] | None:
    """Pull (video path, subfolder) from a ComfyUI history entry.

    SadTalker's ShowVideo node reports ``show_video_path``; Wav2Lip's
    VHS_VideoCombine reports ``gifs`` (with filename/subfolder/fullpath).
    The subfolder is "" when the file sits directly in output/.
    """
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        if kind == "video":
            gifs = node_output.get("gifs")
            if gifs:
                name = gifs[0].get("filename") or Path(gifs[0]["fullpath"]).name
                return name, str(gifs[0].get("subfolder") or "")
        else:
            paths = node_output.get("show_video_path")
            if paths:
                return paths[0], ""
    return None
