"""Provider factory for creating and managing providers."""

from typing import Any, Dict, Type

from rich.console import Console

from .base import ImageProvider, TTSProvider
from .images import DalleImageProvider, PexelsImageProvider, TextImageProvider, UnsplashImageProvider
from .tts import ElevenLabsTTSProvider, GTTSProvider, OpenAITTSProvider

console = Console()
err_console = Console(stderr=True, style="bold red")


class ProviderFactory:
    """Factory for creating provider instances."""
    
    # Registry of available providers
    IMAGE_PROVIDERS: Dict[str, Type[ImageProvider]] = {
        "text": TextImageProvider,
        "dalle3": DalleImageProvider,
        "pexels": PexelsImageProvider,
        "unsplash": UnsplashImageProvider,
    }
    
    TTS_PROVIDERS: Dict[str, Type[TTSProvider]] = {
        "gtts": GTTSProvider,
        "elevenlabs": ElevenLabsTTSProvider,
        "openai": OpenAITTSProvider,
    }
    
    @classmethod
    def create_image_provider(cls, config: Dict[str, Any]) -> ImageProvider:
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
        
        # Fallback to backup provider
        if fallback_name in cls.IMAGE_PROVIDERS:
            fallback_class = cls.IMAGE_PROVIDERS[fallback_name]
            fallback_provider = fallback_class(config)
            console.print(f"🔄 Falling back to: {fallback_provider.name}")
            return fallback_provider
        
        # Final fallback to text
        return TextImageProvider(config)
    
    @classmethod
    def create_tts_provider(cls, config: Dict[str, Any]) -> TTSProvider:
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
        
        # Fallback to gTTS (always available)
        console.print("🔄 Falling back to: gtts")
        return GTTSProvider(config)
    
    @classmethod
    def list_image_providers(cls) -> Dict[str, str]:
        """Get list of available image providers."""
        return {
            "text": "Text-based images (always available)",
            "dalle3": "DALL-E 3 AI image generation (requires OpenAI API key)",
            "pexels": "Pexels stock photos (requires Pexels API key)",
            "unsplash": "Unsplash stock photos (requires Unsplash API key)",
        }
    
    @classmethod
    def list_tts_providers(cls) -> Dict[str, str]:
        """Get list of available TTS providers."""
        return {
            "gtts": "Google Text-to-Speech (free, always available)",
            "elevenlabs": "ElevenLabs premium TTS (requires ElevenLabs API key)",
            "openai": "OpenAI TTS (requires OpenAI API key)",
        }
    
    @classmethod
    def check_provider_availability(cls, config: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
        """Check availability of all providers."""
        availability = {
            "images": {},
            "tts": {}
        }
        
        # Check image providers
        for name, provider_class in cls.IMAGE_PROVIDERS.items():
            provider = provider_class(config)
            availability["images"][name] = provider.is_available()
        
        # Check TTS providers
        for name, provider_class in cls.TTS_PROVIDERS.items():
            provider = provider_class(config)
            availability["tts"][name] = provider.is_available()
        
        return availability