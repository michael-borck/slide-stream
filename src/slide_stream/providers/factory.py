"""Provider factory for creating and managing providers."""

import os
from typing import Any

from rich.console import Console

from .avatar import (
    ComfyUIAvatarProvider,
    DIDAvatarProvider,
    NoneAvatarProvider,
    PrecomputedAvatarProvider,
    PuppetAvatarProvider,
    SadTalkerAvatarProvider,
    StaticAvatarProvider,
    WanS2VAvatarProvider,
    Wav2LipAvatarProvider,
)
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
    VoiceboxTTSProvider,
)

console = Console()
err_console = Console(stderr=True, style="bold red")

# TTS providers that need a self-hosted server URL to do anything. When one of
# these is selected (e.g. the default, voicebox) but no connection details are
# set, that just means "not configured yet" — a quiet gTTS fallback, not an
# error. Providers keyed on an API key (elevenlabs/openai) or a local package
# (kokoro) are the user's explicit choice, so their unavailability is an error.
_SERVER_TTS_PROVIDERS = ("voicebox", "chatterbox", "openai-compatible")
_TTS_CONNECTION_KEYS = ("base_url", "api_key", "profile_id", "voice_sample")
# The env vars each server provider reads for its connection details (see the
# provider implementations in tts.py). A user configured purely via env vars
# is configured, and must not be silently downgraded to gTTS.
_SERVER_TTS_ENV_VARS = {
    "voicebox": ("VOICEBOX_BASE_URL", "VOICEBOX_TOKEN"),
    "chatterbox": ("CHATTERBOX_BASE_URL", "CHATTERBOX_TOKEN"),
    "openai-compatible": ("OPENAI_BASE_URL",),
}


def _tts_unconfigured(provider_name: str, tts_config: dict[str, Any]) -> bool:
    """True when a server TTS provider has no connection details set at all,
    neither in config keys nor in the env vars the provider itself reads."""
    if provider_name not in _SERVER_TTS_PROVIDERS:
        return False
    if any(tts_config.get(key) for key in _TTS_CONNECTION_KEYS):
        return False
    return not any(
        os.getenv(var) for var in _SERVER_TTS_ENV_VARS.get(provider_name, ())
    )


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
        "voicebox": VoiceboxTTSProvider,
        "elevenlabs": ElevenLabsTTSProvider,
        "openai": OpenAITTSProvider,
        "openai-compatible": OpenAICompatTTSProvider,
    }

    AVATAR_PROVIDERS: dict[str, type[AvatarProvider]] = {
        "none": NoneAvatarProvider,
        "static": StaticAvatarProvider,
        "puppet": PuppetAvatarProvider,
        "precomputed": PrecomputedAvatarProvider,
        "d-id": DIDAvatarProvider,
        "sadtalker": SadTalkerAvatarProvider,
        "wav2lip": Wav2LipAvatarProvider,
        "wan-s2v": WanS2VAvatarProvider,
        "comfyui": ComfyUIAvatarProvider,
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

        # Fallback to backup provider — only if it is actually usable.
        if fallback_name in cls.IMAGE_PROVIDERS:
            fallback_class = cls.IMAGE_PROVIDERS[fallback_name]
            fallback_provider = fallback_class(config)
            if fallback_provider.is_available():
                console.print(f"🔄 Falling back to: {fallback_provider.name}")
                return fallback_provider
            err_console.print(
                f"❌ Fallback image provider '{fallback_provider.name}' "
                "not available either."
            )

        # Final fallback to text
        console.print("🔄 Falling back to: text")
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

            # A server provider with no connection details at all was never
            # really chosen — it's just the default (voicebox) on a bare
            # install. There is nothing for strict mode to protect, so use the
            # free default rather than erroring or aborting.
            if _tts_unconfigured(provider_name, tts_config):
                console.print(
                    f"🔊 No {provider.name} server configured — using free gTTS. "
                    f"Set providers.tts.base_url to enable {provider.name}."
                )
                return GTTSProvider(config)

            # Configured but unusable: a genuine error the user should see.
            err_console.print(
                f"❌ {provider.name} not available "
                "(check base_url / API key / voice settings)."
            )
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
            "voicebox": "Self-hosted Voicebox studio (base_url + profile_id, or voice_sample+reference_text for an ephemeral clone deleted after the run; multi-engine: kokoro/chatterbox/qwen/...)",
            "elevenlabs": "ElevenLabs premium TTS (requires ElevenLabs API key)",
            "openai": "OpenAI TTS (requires OpenAI API key)",
            "openai-compatible": "Any OpenAI-compatible speech endpoint (set base_url; local or hosted)",
        }

    @classmethod
    def list_avatar_providers(cls) -> dict[str, str]:
        """Get list of available avatar providers."""
        return {
            "none": "Avatar disabled (default)",
            "static": "A static mascot image in the corner — no lip-sync, no GPU (source: a built-in name like 'teddy', or a path)",
            "puppet": "Cartoon mouth-flap driven by audio loudness — no AI/GPU, works on any mascot (source: 'teddy' or a path)",
            "precomputed": "Pre-supplied head clips: assets_dir/head_N.mp4 (no GPU or service needed)",
            "d-id": "D-ID lip-synced talking head from a source image (BYOK; requires DID_API_KEY + source_image)",
            "sadtalker": "Self-hosted SadTalker (photo) via a ComfyUI server (base_url + source_image)",
            "wav2lip": "Self-hosted Wav2Lip (video) via a ComfyUI server (base_url + source_video)",
            "wan-s2v": "Self-hosted Wan2.2-S2V (still image + audio) via a ComfyUI server — no face detector, so it animates mascots AND human head shots (base_url + source)",
            "comfyui": "Auto: photo -> SadTalker, video -> Wav2Lip, via a ComfyUI server (base_url + source)",
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
