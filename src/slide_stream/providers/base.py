"""Abstract base classes for providers."""

from abc import ABC, abstractmethod
from typing import Any


class StrictModeError(Exception):
    """Raised in strict mode when a provider fails and fallback is disabled."""


def is_strict(config: dict[str, Any]) -> bool:
    """Whether strict mode is enabled (fail instead of falling back)."""
    return bool(config.get("settings", {}).get("strict", False))


class ImageProvider(ABC):
    """Abstract base class for image providers."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config

    @abstractmethod
    def generate_image(
        self, query: str, filename: str, slide: dict[str, Any] | None = None
    ) -> str:
        """Generate or download an image based on query.

        Args:
            query: Search query or prompt for image generation
            filename: Target filename to save the image
            slide: Optional slide dict (title/content). The text provider renders
                it; other providers ignore it and use ``query``.

        Returns:
            Path to the saved image file
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured and available."""
        pass


class TTSProvider(ABC):
    """Abstract base class for text-to-speech providers."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
        # Incremented each time a synthesis quietly falls back to gTTS, so the
        # CLI can warn the user afterwards that some slides do not use the
        # configured voice. Read via getattr(provider, "fallback_count", 0).
        self.fallback_count: int = 0

    @abstractmethod
    def synthesize(self, text: str, filename: str) -> str | None:
        """Convert text to speech and save to file.

        Args:
            text: Text to convert to speech
            filename: Target filename to save the audio

        Returns:
            Path to the saved audio file, or None if failed
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured and available."""
        pass

    def close(self) -> None:
        """Release any per-run server-side resources.

        Called once at the end of a run, on success and failure alike.
        Providers that upload a voice clone override this to delete it.
        """
        return None


class AvatarProvider(ABC):
    """Abstract base class for talking-head avatar providers.

    An avatar provider turns a slide's narration audio into a short
    head-and-shoulders video (lip-synced by real engines; precomputed clips
    for the no-GPU provider), which media.py composites over the slide.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config

    @abstractmethod
    def generate(
        self, audio_path: str, output_path: str, slide_num: int
    ) -> str | None:
        """Produce a head video for one slide's narration audio.

        Args:
            audio_path: Path to the slide's narration audio file
            output_path: Suggested target path for a generated video; providers
                that already have a file (e.g. precomputed) may return a
                different existing path instead
            slide_num: 1-based slide number, used by providers that map slides
                to pre-supplied clips

        Returns:
            Path to the head video to composite, or None if unavailable/failed.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured and available."""
        pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config

    @abstractmethod
    def generate_text(self, prompt: str, model: str | None = None) -> str | None:
        """Generate text based on prompt.

        Args:
            prompt: Input prompt for text generation
            model: Optional specific model to use

        Returns:
            Generated text or None if failed
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured and available."""
        pass
