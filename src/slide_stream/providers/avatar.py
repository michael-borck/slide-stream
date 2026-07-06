"""Talking-head avatar provider implementations."""

import os
import time
from pathlib import Path
from typing import Any

import requests
from rich.console import Console

from .base import AvatarProvider

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
