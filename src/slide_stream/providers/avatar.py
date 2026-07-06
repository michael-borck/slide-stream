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
        source = self._source_image()
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


class SadTalkerAvatarProvider(AvatarProvider):
    """Self-hosted talking head via SadTalker running as a ComfyUI node.

    Drives a ComfyUI server's native /prompt API with the proven SadTalker
    workflow: the lecturer photo + each slide's narration audio -> a lip-synced
    clip, composited by media.py. Family 1 (photo -> full-head synthesis);
    ``stillMode`` defaults on to minimise head-bob jank in the corner circle.

    Config (``providers.avatar``):
      provider: sadtalker
      base_url: https://comfyui.example.org
      source_image: ./lecturer.png   # local path (uploaded once) or input/ name
      api_key: "${COMFYUI_TOKEN}"    # optional Bearer, if fronted by auth
    Optional SadTalker tuning: ``still_mode`` (bool), ``preprocess``,
    ``face_resolution`` ("256"/"512"), ``pose_style`` (int),
    ``face_enhancer`` (bool). Plus ``timeout`` (default 600s), ``poll_interval``.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        # The source image is uploaded once per run; cache the input filename.
        self._source_name: str | None = None

    @property
    def name(self) -> str:
        return "sadtalker"

    def _avatar_config(self) -> dict[str, Any]:
        return self.config.get("providers", {}).get("avatar", {})

    def _base_url(self) -> str | None:
        base_url = self._avatar_config().get("base_url") or os.getenv("COMFYUI_BASE_URL")
        return base_url.rstrip("/") if base_url else None

    def _headers(self) -> dict[str, str]:
        api_key = self._avatar_config().get("api_key") or os.getenv("COMFYUI_TOKEN")
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _source_image(self) -> str | None:
        return self._avatar_config().get("source_image")

    def is_available(self) -> bool:
        """Available when a ComfyUI URL and a source image are configured."""
        return bool(self._base_url() and self._source_image())

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
        """Return the input/ filename for the source image, uploading once."""
        if self._source_name:
            return self._source_name
        source = self._source_image()
        if source is None:
            raise ValueError("no source_image configured")
        path = Path(source)
        # A bare filename that is not a local file is assumed to already live in
        # ComfyUI's input/ directory.
        self._source_name = self._upload(path) if path.is_file() else source
        if path.is_file():
            console.print("  - Uploaded avatar source image to ComfyUI")
        return self._source_name

    def _build_workflow(self, image_name: str, audio_name: str) -> dict[str, Any]:
        import copy

        cfg = self._avatar_config()
        wf = copy.deepcopy(SADTALKER_WORKFLOW)
        wf["1"]["inputs"]["image"] = image_name
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

            image_name = self._ensure_source()
            # SadTalker loads audio at 16kHz; some installs crash resampling
            # from other rates (librosa/resampy/numba mismatch). Feed 16kHz
            # mono WAV so it never resamples.
            audio_16k, is_temp = _to_16k_wav(audio_path)
            try:
                audio_name = self._upload(Path(audio_16k))
            finally:
                if is_temp:
                    Path(audio_16k).unlink(missing_ok=True)
            workflow = self._build_workflow(image_name, audio_name)

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
                        video_path = _find_show_video_path(entry.get("outputs", {}))
                        break
                    if status_str == "error":
                        raise ValueError(f"ComfyUI workflow error: {entry.get('status')}")
                time.sleep(poll_interval)

            if not video_path:
                raise TimeoutError(f"SadTalker render timed out after {timeout:.0f}s")

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
            console.print(f"  - Generated SadTalker talking head (slide {slide_num})")
            return output_path

        except Exception as e:
            err_console.print(f"  - SadTalker avatar error: {e}")
            return None


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


def _find_show_video_path(outputs: dict[str, Any]) -> str | None:
    """Pull the ShowVideo node's output path out of a ComfyUI history entry."""
    for node_output in outputs.values():
        paths = node_output.get("show_video_path") if isinstance(node_output, dict) else None
        if paths:
            return paths[0]
    return None
