"""Image provider implementations."""

import base64
import os
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

from .base import ImageProvider, StrictModeError, is_strict

console = Console()
err_console = Console(stderr=True, style="bold red")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "are", "was", "were", "be", "been", "have", "has", "do",
    "does", "this", "that", "it", "its", "by", "as", "from",
}


def extract_keywords(text: str) -> set[str]:
    """Content words (lowercased, >2 chars, stopwords removed) from text."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}

# Pillow only searches a handful of directories by name, and on macOS Arial
# lives in /System/Library/Fonts/Supplemental where it does not look, so
# absolute paths are included alongside plain names.
FONT_CANDIDATES = [
    "arial.ttf",
    "Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "Helvetica.ttc",  # macOS
    "DejaVuSans.ttf",  # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a scalable font at the requested size.

    Falls back to Pillow's built-in scalable font (Pillow >= 10.1) rather than
    the unsized bitmap default, which renders ~10px glyphs on a 1080p canvas.
    """
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


class TextImageProvider(ImageProvider):
    """Generate text-based images for slides."""

    @property
    def name(self) -> str:
        return "text"

    def is_available(self) -> bool:
        return True

    def generate_image(
        self, query: str, filename: str, slide: dict[str, Any] | None = None
    ) -> str:
        """Create a text-based slide image showing the slide's title and content.

        This is the default image source, so the on-screen text must reflect the
        actual slide. When ``slide`` is provided its title and bullet points are
        rendered; otherwise the query is drawn as a fallback.
        """
        settings = self.config.get("settings", {}).get("image", {})
        video_settings = self.config.get("settings", {}).get("video", {})

        resolution = tuple(video_settings.get("resolution", [1920, 1080]))
        bg_color = settings.get("bg_color", "black")
        font_color = settings.get("font_color", "white")
        title_font_size = settings.get("title_font_size", 100)
        content_font_size = settings.get("content_font_size", 60)
        max_line_width = settings.get("max_line_width", 50)

        img = Image.new("RGB", resolution, color=bg_color)
        draw = ImageDraw.Draw(img)

        title_font = load_font(title_font_size)
        content_font = load_font(content_font_size)

        if slide is not None:
            title = (slide.get("title") or "").strip() or "Slide"
            content_items = [
                str(item).strip()
                for item in slide.get("content", [])
                if str(item).strip()
            ]
        else:
            title = "Slide"
            content_items = [f"Topic: {query}"] if query else []

        # Draw the title (wrapped to fit the frame).
        y_pos = resolution[1] * 0.1
        for line in textwrap.wrap(title, width=max_line_width) or [title]:
            draw.text(
                (resolution[0] * 0.1, y_pos),
                line,
                font=title_font,
                fill=font_color,
            )
            y_pos += title_font_size + 20

        # Draw the content bullets (wrapped to fit).
        y_pos += 40
        for item in content_items:
            for line in textwrap.wrap(f"• {item}", width=max_line_width):
                draw.text(
                    (resolution[0] * 0.1, y_pos),
                    line,
                    font=content_font,
                    fill=font_color,
                )
                y_pos += content_font_size + 20
            y_pos += 20

        img.save(filename)
        console.print(f"  - Generated text slide: {title}")
        return filename


class LocalImageProvider(ImageProvider):
    """Pick images from a local folder by keyword-matching filenames.

    Ported from slide-vision. Each slide gets the highest-scoring image whose
    filename shares the most keywords with the slide's title/content. Every
    image is used at most once per run, so slides don't collapse onto one
    popular image. Slides with no keyword match fall back to a text image.
    Pairs well with the ``scan`` command, which AI-renames images to keyword
    slugs first.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._all_images: list[Path] | None = None
        self._used: set[Path] = set()
        # Whether the most recent generate_image() found a real folder match
        # (vs. falling back to a text card). enrich uses this to list gaps.
        self.matched_last: bool = False

    @property
    def name(self) -> str:
        return "local"

    def _folder(self) -> Path | None:
        folder = self.config.get("providers", {}).get("images", {}).get("folder")
        return Path(folder) if folder else None

    def is_available(self) -> bool:
        folder = self._folder()
        return folder is not None and folder.is_dir()

    def _images(self) -> list[Path]:
        if self._all_images is None:
            folder = self._folder()
            if folder is not None and folder.is_dir():
                self._all_images = sorted(
                    p
                    for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
            else:
                self._all_images = []
        return self._all_images

    def generate_image(
        self, query: str, filename: str, slide: dict[str, Any] | None = None
    ) -> str:
        available = [p for p in self._images() if p not in self._used]
        if available:
            text = query
            if slide is not None:
                text = f"{slide.get('title', '')} {' '.join(str(c) for c in slide.get('content', []))} {query}"
            keywords = extract_keywords(text)

            best: Path | None = None
            best_score = 0
            for img in available:
                score = len(extract_keywords(img.stem) & keywords)
                if score > best_score:
                    best, best_score = img, score

            if best is not None and best_score > 0:
                self._used.add(best)
                try:
                    shutil.copy2(best, filename)
                    console.print(f"  - Using local image: {best.name}")
                    self.matched_last = True
                    return filename
                except Exception as e:
                    err_console.print(f"  - Local image copy error: {e}")

        self.matched_last = False
        return self._fallback_to_text(query, filename, slide=slide)

    def _fallback_to_text(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Fall back to a text image, preserving slide content when known."""
        return TextImageProvider(self.config).generate_image(query, filename, slide=slide)


class SwarmUIImageProvider(ImageProvider):
    """Generate images via a self-hosted SwarmUI server's native API.

    SwarmUI does not speak the OpenAI /v1/images shape, so this provider uses
    its GetNewSession -> GenerateText2Image -> fetch-path flow. Point
    ``providers.images.base_url`` at the SwarmUI server (e.g.
    https://image.example.org). Optional settings: ``model`` (SwarmUI model
    name; omit to use the server's loaded model), ``steps``, ``width``,
    ``height``, ``timeout``, and ``api_key`` (Bearer, if fronted by auth).
    Falls back to a text image on any error.
    """

    @property
    def name(self) -> str:
        return "swarmui"

    def _settings(self) -> dict[str, Any]:
        return self.config.get("providers", {}).get("images", {})

    def _base_url(self) -> str | None:
        base_url = self._settings().get("base_url") or os.getenv("SWARMUI_BASE_URL")
        return base_url.rstrip("/") if base_url else None

    def _headers(self) -> dict[str, str]:
        api_key = self._settings().get("api_key") or os.getenv("SWARMUI_TOKEN")
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def is_available(self) -> bool:
        return bool(self._base_url())

    def generate_image(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Generate an image via SwarmUI's native text-to-image API."""
        try:
            base_url = self._base_url()
            if not base_url:
                raise ValueError("SwarmUI base_url not configured")
            settings = self._settings()
            headers = self._headers()
            timeout = float(settings.get("timeout") or 180)

            # 1. Open a session.
            session = requests.post(
                f"{base_url}/API/GetNewSession", json={}, headers=headers, timeout=30
            )
            session.raise_for_status()
            session_id = session.json()["session_id"]

            # 2. Generate. Default to a 16:9 size the video pipeline can scale.
            prompt = (
                f"A professional, clean image for a presentation slide about: "
                f"{query}. High quality, suitable for business presentation, "
                f"no text overlay."
            )
            payload: dict[str, Any] = {
                "session_id": session_id,
                "prompt": prompt,
                "images": 1,
                "steps": int(settings.get("steps") or 20),
                "width": int(settings.get("width") or 1024),
                "height": int(settings.get("height") or 576),
            }
            if settings.get("model"):
                payload["model"] = settings["model"]

            gen = requests.post(
                f"{base_url}/API/GenerateText2Image",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            gen.raise_for_status()
            data = gen.json()
            images = data.get("images")
            if not images:
                raise ValueError(f"SwarmUI returned no image ({data})")

            # 3. Fetch the generated image by its server path.
            image_path = images[0]
            img_response = requests.get(
                f"{base_url}/{image_path.lstrip('/')}", headers=headers, timeout=timeout
            )
            img_response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(img_response.content)

            console.print(f"  - Generated SwarmUI image: {query}")
            return filename

        except Exception as e:
            err_console.print(f"  - SwarmUI error: {e}. Using text fallback.")
            return self._fallback_to_text(query, filename, slide=slide)

    def _fallback_to_text(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Fall back to a text image, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            raise StrictModeError(
                f"Strict mode: image provider '{self.name}' failed and fallback "
                "to text images is disabled."
            )
        return TextImageProvider(self.config).generate_image(query, filename, slide=slide)


class GeminiImageProvider(ImageProvider):
    """Generate images with Google Imagen via the google-genai SDK.

    Cheap cloud generation (Imagen 4 Fast is ~$0.02/image). Set
    ``providers.images.model`` to pick the Imagen model (default the Fast
    tier). Requires ``pip install 'slide-stream[gemini]'`` and a
    ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``). Falls back to a text image on
    any error.
    """

    @property
    def name(self) -> str:
        return "gemini"

    def _api_key(self) -> str | None:
        api_keys = self.config.get("api_keys", {})
        return (
            api_keys.get("gemini")
            or api_keys.get("google")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

    def is_available(self) -> bool:
        return bool(self._api_key())

    def generate_image(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Generate an image with Google Imagen."""
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types  # type: ignore[import-not-found]

            api_key = self._api_key()
            if not api_key:
                raise ValueError("Gemini API key not found")

            settings = self.config.get("providers", {}).get("images", {})
            model = settings.get("model") or "imagen-4.0-fast-generate-001"
            video_settings = self.config.get("settings", {}).get("video", {})
            resolution = video_settings.get("resolution", [1920, 1080])
            aspect_ratio = "16:9" if resolution[0] >= resolution[1] else "9:16"

            client = genai.Client(api_key=api_key)
            prompt = (
                f"A professional, clean image for a presentation slide about: "
                f"{query}. High quality, suitable for business presentation, "
                f"no text overlay."
            )
            response = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1, aspect_ratio=aspect_ratio
                ),
            )
            generated = getattr(response, "generated_images", None)
            if not generated:
                raise ValueError("Imagen returned no image")
            image_bytes = generated[0].image.image_bytes
            with open(filename, "wb") as f:
                f.write(image_bytes)

            console.print(f"  - Generated Imagen image: {query}")
            return filename

        except ImportError:
            err_console.print(
                "  - google-genai not installed. Install with: "
                "pip install 'slide-stream[gemini]'"
            )
            return self._fallback_to_text(query, filename, slide=slide)
        except Exception as e:
            err_console.print(f"  - Imagen error: {e}. Using text fallback.")
            return self._fallback_to_text(query, filename, slide=slide)

    def _fallback_to_text(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Fall back to a text image, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            raise StrictModeError(
                f"Strict mode: image provider '{self.name}' failed and fallback "
                "to text images is disabled."
            )
        return TextImageProvider(self.config).generate_image(query, filename, slide=slide)


class DalleImageProvider(ImageProvider):
    """Generate images using DALL-E 3 via OpenAI API."""

    @property
    def name(self) -> str:
        return "dalle3"

    def is_available(self) -> bool:
        """Check if OpenAI API key is available."""
        api_keys = self.config.get("api_keys", {})
        openai_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
        return bool(openai_key)

    def generate_image(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Generate image using DALL-E 3."""
        try:
            from openai import OpenAI

            api_keys = self.config.get("api_keys", {})
            api_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError("OpenAI API key not found")

            client = OpenAI(api_key=api_key)

            # Create a descriptive prompt for presentation images
            prompt = f"A professional, clean image for a presentation slide about: {query}. High quality, suitable for business presentation, no text overlay."

            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1792x1024",  # Close to 1920x1080 aspect ratio
                quality="standard",
                n=1,
            )

            if not response.data or not response.data[0].url:
                raise ValueError("DALL-E returned no image URL")
            image_url = response.data[0].url

            # Download the generated image
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()

            with open(filename, 'wb') as f:
                f.write(img_response.content)

            console.print(f"  - Generated DALL-E image: {query}")
            return filename

        except ImportError:
            err_console.print("  - OpenAI library not installed. Install with: pip install openai")
            return self._fallback_to_text(query, filename, slide=slide)
        except Exception as e:
            err_console.print(f"  - DALL-E error: {e}. Using text fallback.")
            return self._fallback_to_text(query, filename, slide=slide)

    def _fallback_to_text(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Fall back to a text image, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            raise StrictModeError(
                f"Strict mode: image provider '{self.name}' failed and fallback "
                "to text images is disabled."
            )
        text_provider = TextImageProvider(self.config)
        return text_provider.generate_image(query, filename, slide=slide)


class PexelsImageProvider(ImageProvider):
    """Download images from Pexels stock photo service."""

    @property
    def name(self) -> str:
        return "pexels"

    def is_available(self) -> bool:
        """Check if Pexels API key is available."""
        api_keys = self.config.get("api_keys", {})
        pexels_key = api_keys.get("pexels") or os.getenv("PEXELS_API_KEY")
        return bool(pexels_key)

    def generate_image(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Download image from Pexels."""
        try:
            api_keys = self.config.get("api_keys", {})
            api_key = api_keys.get("pexels") or os.getenv("PEXELS_API_KEY")

            if not api_key:
                raise ValueError("Pexels API key not found")

            settings = self.config.get("settings", {}).get("image", {})
            timeout = settings.get("download_timeout", 15)

            headers = {"Authorization": api_key}

            # Search for images
            search_url = "https://api.pexels.com/v1/search"
            params = {
                "query": query,
                "per_page": 1,
                "orientation": "landscape"
            }

            response = requests.get(search_url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()

            data = response.json()
            if not data.get("photos"):
                raise ValueError("No images found")

            # Get the first image
            photo = data["photos"][0]
            image_url = photo["src"]["large2x"]  # High resolution

            # Download the image
            img_response = requests.get(image_url, timeout=timeout)
            img_response.raise_for_status()

            with open(filename, 'wb') as f:
                f.write(img_response.content)

            console.print(f"  - Downloaded Pexels image: {query}")
            return filename

        except Exception as e:
            err_console.print(f"  - Pexels error: {e}. Using text fallback.")
            return self._fallback_to_text(query, filename, slide=slide)

    def _fallback_to_text(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Fall back to a text image, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            raise StrictModeError(
                f"Strict mode: image provider '{self.name}' failed and fallback "
                "to text images is disabled."
            )
        text_provider = TextImageProvider(self.config)
        return text_provider.generate_image(query, filename, slide=slide)


class UnsplashImageProvider(ImageProvider):
    """Download images from Unsplash stock photo service."""

    @property
    def name(self) -> str:
        return "unsplash"

    def is_available(self) -> bool:
        """Check if Unsplash API key is available."""
        api_keys = self.config.get("api_keys", {})
        unsplash_key = api_keys.get("unsplash") or os.getenv("UNSPLASH_ACCESS_KEY")
        return bool(unsplash_key)

    def generate_image(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Download image from Unsplash."""
        try:
            api_keys = self.config.get("api_keys", {})
            api_key = api_keys.get("unsplash") or os.getenv("UNSPLASH_ACCESS_KEY")

            if not api_key:
                raise ValueError("Unsplash API key not found")

            settings = self.config.get("settings", {}).get("image", {})
            timeout = settings.get("download_timeout", 15)
            video_settings = self.config.get("settings", {}).get("video", {})
            resolution = video_settings.get("resolution", [1920, 1080])

            headers = {"Authorization": f"Client-ID {api_key}"}

            # Search for images
            search_url = "https://api.unsplash.com/search/photos"
            params = {
                "query": query,
                "per_page": 1,
                "orientation": "landscape"
            }

            response = requests.get(search_url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()

            data = response.json()
            if not data.get("results"):
                raise ValueError("No images found")

            # Get the first image
            photo = data["results"][0]
            image_url = f"{photo['urls']['raw']}&w={resolution[0]}&h={resolution[1]}&fit=crop"

            # Download the image
            img_response = requests.get(image_url, timeout=timeout)
            img_response.raise_for_status()

            with open(filename, 'wb') as f:
                f.write(img_response.content)

            console.print(f"  - Downloaded Unsplash image: {query}")
            return filename

        except Exception as e:
            err_console.print(f"  - Unsplash error: {e}. Using text fallback.")
            return self._fallback_to_text(query, filename, slide=slide)

    def _fallback_to_text(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Fall back to a text image, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            raise StrictModeError(
                f"Strict mode: image provider '{self.name}' failed and fallback "
                "to text images is disabled."
            )
        text_provider = TextImageProvider(self.config)
        return text_provider.generate_image(query, filename, slide=slide)


class OpenAICompatImageProvider(ImageProvider):
    """Image generation via any OpenAI-compatible /v1/images endpoint.

    The backend is selected by ``base_url`` in config, so this works against a
    local server (LocalAI, an Automatic1111 OpenAI shim, ...) or a hosted one
    without vendor-specific code. Handles responses that return either a URL
    or inline base64 (``b64_json``), since local servers commonly do the
    latter. Falls back to a text image on any failure.
    """

    @property
    def name(self) -> str:
        return "openai-compatible"

    def _settings(self) -> dict:
        return self.config.get("providers", {}).get("images", {})

    def _base_url(self) -> str | None:
        return self._settings().get("base_url") or os.getenv("OPENAI_BASE_URL")

    def is_available(self) -> bool:
        """Available only when a base_url is configured.

        An OpenAI key alone is not enough: without a base_url this provider
        would silently talk to the real OpenAI API instead of the intended
        local/self-hosted server. Use the ``dalle3`` provider for real OpenAI.
        """
        return bool(self._base_url())

    def generate_image(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Generate an image via an OpenAI-compatible endpoint."""
        try:
            from openai import OpenAI

            api_keys = self.config.get("api_keys", {})
            settings = self._settings()

            base_url = self._base_url()
            api_key = (
                settings.get("api_key")
                or api_keys.get("openai")
                or os.getenv("OPENAI_API_KEY")
                or "not-needed"
            )
            model = settings.get("model") or "dall-e-3"
            size = settings.get("size") or "1792x1024"
            timeout = settings.get("download_timeout", 30)

            client = OpenAI(base_url=base_url, api_key=api_key)

            prompt = (
                f"A professional, clean image for a presentation slide about: "
                f"{query}. High quality, suitable for business presentation, "
                f"no text overlay."
            )
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                n=1,
            )

            if not response.data:
                raise ValueError("Endpoint returned no image data")

            item = response.data[0]
            b64 = getattr(item, "b64_json", None)
            url = getattr(item, "url", None)
            if b64:
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(b64))
            elif url:
                img_response = requests.get(url, timeout=timeout)
                img_response.raise_for_status()
                with open(filename, "wb") as f:
                    f.write(img_response.content)
            else:
                raise ValueError("Endpoint returned neither a URL nor b64_json")

            console.print(f"  - Generated image via OpenAI-compatible endpoint: {query}")
            return filename

        except ImportError:
            err_console.print("  - OpenAI library not installed. Install with: pip install openai")
            return self._fallback_to_text(query, filename, slide=slide)
        except Exception as e:
            err_console.print(f"  - OpenAI-compatible image error: {e}. Using text fallback.")
            return self._fallback_to_text(query, filename, slide=slide)

    def _fallback_to_text(self, query: str, filename: str, slide: dict[str, Any] | None = None) -> str:
        """Fall back to a text image, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            raise StrictModeError(
                f"Strict mode: image provider '{self.name}' failed and fallback "
                "to text images is disabled."
            )
        text_provider = TextImageProvider(self.config)
        return text_provider.generate_image(query, filename, slide=slide)
