"""Text-to-speech provider implementations."""

import os
from typing import Any, Dict, Optional

from rich.console import Console

from .base import TTSProvider

console = Console()
err_console = Console(stderr=True, style="bold red")


class GTTSProvider(TTSProvider):
    """Google Text-to-Speech provider (free)."""
    
    @property
    def name(self) -> str:
        return "gtts"
    
    def is_available(self) -> bool:
        """gTTS is always available."""
        return True
    
    def synthesize(self, text: str, filename: str) -> Optional[str]:
        """Convert text to speech using Google TTS."""
        try:
            from gtts import gTTS
            
            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            console.print(f"  - Generated audio with gTTS")
            return filename
            
        except Exception as e:
            err_console.print(f"  - gTTS error: {e}")
            return None


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs premium text-to-speech provider."""
    
    @property
    def name(self) -> str:
        return "elevenlabs"
    
    def is_available(self) -> bool:
        """Check if ElevenLabs API key is available."""
        api_keys = self.config.get("api_keys", {})
        elevenlabs_key = api_keys.get("elevenlabs") or os.getenv("ELEVENLABS_API_KEY")
        return bool(elevenlabs_key)
    
    def synthesize(self, text: str, filename: str) -> Optional[str]:
        """Convert text to speech using ElevenLabs."""
        try:
            from elevenlabs import generate, save
            
            api_keys = self.config.get("api_keys", {})
            api_key = api_keys.get("elevenlabs") or os.getenv("ELEVENLABS_API_KEY")
            
            if not api_key:
                raise ValueError("ElevenLabs API key not found")
            
            # Set API key
            os.environ["ELEVENLABS_API_KEY"] = api_key
            
            # Get voice from config
            tts_config = self.config.get("providers", {}).get("tts", {})
            voice = tts_config.get("voice", "Rachel")  # Default to Rachel
            
            # Generate audio
            audio = generate(
                text=text,
                voice=voice,
                model="eleven_monolingual_v1"
            )
            
            # Save to file
            save(audio, filename)
            console.print(f"  - Generated audio with ElevenLabs ({voice})")
            return filename
            
        except ImportError:
            err_console.print("  - ElevenLabs library not installed. Install with: pip install elevenlabs")
            return self._fallback_to_gtts(text, filename)
        except Exception as e:
            err_console.print(f"  - ElevenLabs error: {e}. Using gTTS fallback.")
            return self._fallback_to_gtts(text, filename)
    
    def _fallback_to_gtts(self, text: str, filename: str) -> Optional[str]:
        """Fallback to gTTS."""
        gtts_provider = GTTSProvider(self.config)
        return gtts_provider.synthesize(text, filename)


class OpenAITTSProvider(TTSProvider):
    """OpenAI text-to-speech provider."""
    
    @property
    def name(self) -> str:
        return "openai"
    
    def is_available(self) -> bool:
        """Check if OpenAI API key is available."""
        api_keys = self.config.get("api_keys", {})
        openai_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
        return bool(openai_key)
    
    def synthesize(self, text: str, filename: str) -> Optional[str]:
        """Convert text to speech using OpenAI TTS."""
        try:
            from openai import OpenAI
            
            api_keys = self.config.get("api_keys", {})
            api_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            client = OpenAI(api_key=api_key)
            
            # Get voice from config
            tts_config = self.config.get("providers", {}).get("tts", {})
            voice = tts_config.get("voice", "nova")  # Default to nova
            
            # Generate audio
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            
            # Save to file
            response.stream_to_file(filename)
            console.print(f"  - Generated audio with OpenAI TTS ({voice})")
            return filename
            
        except ImportError:
            err_console.print("  - OpenAI library not installed. Install with: pip install openai")
            return self._fallback_to_gtts(text, filename)
        except Exception as e:
            err_console.print(f"  - OpenAI TTS error: {e}. Using gTTS fallback.")
            return self._fallback_to_gtts(text, filename)
    
    def _fallback_to_gtts(self, text: str, filename: str) -> Optional[str]:
        """Fallback to gTTS."""
        gtts_provider = GTTSProvider(self.config)
        return gtts_provider.synthesize(text, filename)