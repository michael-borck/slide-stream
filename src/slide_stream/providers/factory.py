"""Provider factory for creating and managing providers."""

from typing import Any

from rich.console import Console

from .avatar import DIDAvatarProvider, NoneAvatarProvider, PrecomputedAvatarProvider
from .base import (
    AvatarProvider,
    ImageProvider,
    StrictModeError,
    TTSProvider,
    is_strict,
)
from .images import (
    DalleImageProvider,
    GeminiImageProvider,
    LocalImageProvider,
    OpenAICompatImageProvider,
    PexelsImageProvider,
    SwarmUIImageProvider,
    TextImageProvider,
    UnsplashImageProvider,
)
from .tts import (
    ChatterboxTTSProvider,
    ElevenLabsTTSProvider,
    GTTSProvider,
    KokoroTTSProvider,
    OpenAICompatTTSProvider,
    OpenAITTSProvider,
)

console = Console()
err_console = Console(stderr=True, style="bold red")


class ProviderFactory:
    """Factory for creating provider instances."""

    # Registry of available providers
    IMAGE_PROVIDERS: dict[str, type[ImageProvider]] = {
        "text": TextImageProvider,
        "local": LocalImageProvider,
        "dalle3": DalleImageProvider,
        "gemini": GeminiImageProvider,
        "swarmui": SwarmUIImageProvider,
        "openai-compatible": OpenAICompatImageProvider,
        "pexels": PexelsImageProvider,
        "unsplash": UnsplashImageProvider,
    }

    TTS_PROVIDERS: dict[str, type[TTSProvider]] = {
        "gtts": GTTSProvider,
        "kokoro": KokoroTTSProvider,
        "chatterbox": ChatterboxTTSProvider,
        "elevenlabs": ElevenLabsTTSProvider,
        "openai": OpenAITTSProvider,
        "openai-compatible": OpenAICompatTTSProvider,
    }

    AVATAR_PROVIDERS: dict[str, type[AvatarProvider]] = {
        "none": NoneAvatarProvider,
        "precomputed": PrecomputedAvatarProvider,
        "d-id": DIDAvatarProvider,
    }

    @classmethod
    def create_image_provider(cls, config: dict[str, Any]) -> ImageProvider:
        """Create an image provider based on configuration."""
        providers_config = config.get("providers", {})
        image_config = providers_config.get("images", {})

        provider_name = image_config.get("provider", "text")
        fallback_name = image_config.get("fallback", "text")

        # Try primary provider
        if provider_name in cls.IMAGE_PROVIDERS:
            provider_class = cls.IMAGE_PROVIDERS[provider_name]
            provider = provider_class(config)

            if provider.is_available():
                console.print(f"✅ Image Provider: {provider.name}")
                return provider
            else:
                err_console.print(f"❌ {provider.name} not available (missing API key?)")
        else:
            err_console.print(f"❌ Unknown image provider: {provider_name}")

        if is_strict(config):
            raise StrictModeError(
                f"Strict mode: image provider '{provider_name}' is not usable "
                "and fallback is disabled."
            )

        # Fallback to backup provider
        if fallback_name in cls.IMAGE_PROVIDERS:
            fallback_class = cls.IMAGE_PROVIDERS[fallback_name]
            fallback_provider = fallback_class(config)
            console.print(f"🔄 Falling back to: {fallback_provider.name}")
            return fallback_provider

        # Final fallback to text
        return TextImageProvider(config)

    @classmethod
    def create_tts_provider(cls, config: dict[str, Any]) -> TTSProvider:
        """Create a TTS provider based on configuration."""
        providers_config = config.get("providers", {})
        tts_config = providers_config.get("tts", {})

        provider_name = tts_config.get("provider", "gtts")

        # Try requested provider
        if provider_name in cls.TTS_PROVIDERS:
            provider_class = cls.TTS_PROVIDERS[provider_name]
            provider = provider_class(config)

            if provider.is_available():
                console.print(f"✅ TTS Provider: {provider.name}")
                return provider
            else:
                err_console.print(f"❌ {provider.name} not available (missing API key?)")
        else:
            err_console.print(f"❌ Unknown TTS provider: {provider_name}")

        if is_strict(config):
            raise StrictModeError(
                f"Strict mode: TTS provider '{provider_name}' is not usable "
                "and fallback is disabled."
            )

        # Fallback to gTTS (always available)
        console.print("🔄 Falling back to: gtts")
        return GTTSProvider(config)

    @classmethod
    def create_avatar_provider(cls, config: dict[str, Any]) -> AvatarProvider:
        """Create an avatar provider based on configuration."""
        providers_config = config.get("providers", {})
        avatar_config = providers_config.get("avatar", {})

        provider_name = avatar_config.get("provider", "none")

        # Try requested provider
        if provider_name in cls.AVATAR_PROVIDERS:
            provider_class = cls.AVATAR_PROVIDERS[provider_name]
            provider = provider_class(config)

            if provider.is_available():
                if provider.name != "none":
                    console.print(f"✅ Avatar Provider: {provider.name}")
                return provider
            else:
                err_console.print(
                    f"❌ {provider.name} avatar not available (missing assets/service?)"
                )
        else:
            err_console.print(f"❌ Unknown avatar provider: {provider_name}")

        if is_strict(config):
            raise StrictModeError(
                f"Strict mode: avatar provider '{provider_name}' is not usable "
                "and fallback is disabled."
            )

        # Fallback: disable the avatar feature for this run.
        console.print("🔄 Falling back to: no avatar")
        return NoneAvatarProvider(config)

    @classmethod
    def list_image_providers(cls) -> dict[str, str]:
        """Get list of available image providers."""
        return {
            "text": "Text-based images (always available)",
            "local": "Local folder images matched by filename keywords (set providers.images.folder)",
            "dalle3": "DALL-E 3 AI image generation (requires OpenAI API key)",
            "gemini": "Google Imagen generation, cheap (~$0.02/image; requires GEMINI_API_KEY and slide-stream[gemini])",
            "swarmui": "Self-hosted SwarmUI server (set base_url; native SwarmUI API, no OpenAI shim needed)",
            "openai-compatible": "Any OpenAI-compatible image endpoint (set base_url; local or hosted)",
            "pexels": "Pexels stock photos (requires Pexels API key)",
            "unsplash": "Unsplash stock photos (requires Unsplash API key)",
        }

    @classmethod
    def list_tts_providers(cls) -> dict[str, str]:
        """Get list of available TTS providers."""
        return {
            "gtts": "Google Text-to-Speech (free, no API key; requires an internet connection)",
            "kokoro": "Kokoro local TTS — fully offline (pip install 'slide-stream[local-tts]'; ~340MB one-time model download)",
            "chatterbox": "Self-hosted Chatterbox voice cloning (set base_url; voice_sample uploads ephemerally per run)",
            "elevenlabs": "ElevenLabs premium TTS (requires ElevenLabs API key)",
            "openai": "OpenAI TTS (requires OpenAI API key)",
            "openai-compatible": "Any OpenAI-compatible speech endpoint (set base_url; local or hosted)",
        }

    @classmethod
    def list_avatar_providers(cls) -> dict[str, str]:
        """Get list of available avatar providers."""
        return {
            "none": "Avatar disabled (default)",
            "precomputed": "Pre-supplied head clips: assets_dir/head_N.mp4 (no GPU or service needed)",
            "d-id": "D-ID lip-synced talking head from a source image (BYOK; requires DID_API_KEY + source_image)",
        }

    @classmethod
    def check_provider_availability(cls, config: dict[str, Any]) -> dict[str, dict[str, bool]]:
        """Check availability of all providers."""
        availability = {
            "images": {},
            "tts": {},
            "avatar": {}
        }

        # Check image providers
        for name, provider_class in cls.IMAGE_PROVIDERS.items():
            provider = provider_class(config)
            availability["images"][name] = provider.is_available()

        # Check TTS providers
        for name, provider_class in cls.TTS_PROVIDERS.items():
            provider = provider_class(config)
            availability["tts"][name] = provider.is_available()

        # Check avatar providers
        for name, provider_class in cls.AVATAR_PROVIDERS.items():
            provider = provider_class(config)
            availability["avatar"][name] = provider.is_available()

        return availability
