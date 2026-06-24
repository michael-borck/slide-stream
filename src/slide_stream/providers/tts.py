"""Text-to-speech provider implementations."""

import os

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

    def synthesize(self, text: str, filename: str) -> str | None:
        """Convert text to speech using Google TTS."""
        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            console.print("  - Generated audio with gTTS")
            return filename

        except Exception as e:
            err_console.print(f"  - gTTS error: {e}")
            return None


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs premium text-to-speech provider."""

    # Voice IDs for the common built-in voices, so users can configure a
    # familiar name instead of an opaque ID. Anything not listed here is
    # passed through to the API as a voice_id verbatim.
    VOICE_IDS = {
        "rachel": "21m00Tcm4TlvDq8ikWAM",
        "drew": "29vD33N1CtxCmqQRPOHJ",
        "clyde": "2EiwWnXFnvU5JabPnv8n",
        "domi": "AZnzlk1XvdvUeBnXmlld",
        "dave": "CYw3kZ02Hs0563khs1Fj",
        "bella": "EXAVITQu4vr4xnSDxMaL",
        "antoni": "ErXwobaYiN019PkySvjV",
        "josh": "TxGEqnHWrfWFTfGW9XjX",
        "arnold": "VR6AewLTigWG4xSOukaG",
        "adam": "pNInz6obpgDQGcFmaJgB",
        "sam": "yoZ06aMxZJJ28mfd3POQ",
    }

    @property
    def name(self) -> str:
        return "elevenlabs"

    def is_available(self) -> bool:
        """Check if ElevenLabs API key is available."""
        api_keys = self.config.get("api_keys", {})
        elevenlabs_key = api_keys.get("elevenlabs") or os.getenv("ELEVENLABS_API_KEY")
        return bool(elevenlabs_key)

    def synthesize(self, text: str, filename: str) -> str | None:
        """Convert text to speech using ElevenLabs."""
        try:
            from elevenlabs import save
            from elevenlabs.client import ElevenLabs

            api_keys = self.config.get("api_keys", {})
            api_key = api_keys.get("elevenlabs") or os.getenv("ELEVENLABS_API_KEY")

            if not api_key:
                raise ValueError("ElevenLabs API key not found")

            client = ElevenLabs(api_key=api_key)

            # Get voice from config and resolve a friendly name to its ID.
            tts_config = self.config.get("providers", {}).get("tts", {})
            voice = tts_config.get("voice") or "Rachel"  # Default to Rachel
            voice_id = self.VOICE_IDS.get(voice.lower(), voice)

            # Generate audio (1.x+ client API)
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
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

    def _fallback_to_gtts(self, text: str, filename: str) -> str | None:
        """Fallback to gTTS."""
        gtts_provider = GTTSProvider(self.config)
        return gtts_provider.synthesize(text, filename)


class OpenAICompatTTSProvider(TTSProvider):
    """Text-to-speech via any OpenAI-compatible /v1/audio/speech endpoint.

    Works against a local server (LocalAI, openedai-speech, Kokoro-FastAPI,
    ...) or a hosted one. The backend is chosen entirely by ``base_url`` in
    config, so no vendor-specific SDK or code path is needed. Local servers
    usually need no API key.
    """

    @property
    def name(self) -> str:
        return "openai-compatible"

    def _base_url(self) -> str | None:
        tts_config = self.config.get("providers", {}).get("tts", {})
        return tts_config.get("base_url") or os.getenv("OPENAI_BASE_URL")

    def is_available(self) -> bool:
        """Available only when a base_url is configured.

        An OpenAI key alone is not enough: without a base_url this provider
        would silently talk to the real OpenAI API instead of the intended
        local/self-hosted server. Use the ``openai`` TTS provider for real OpenAI.
        """
        return bool(self._base_url())

    def synthesize(self, text: str, filename: str) -> str | None:
        """Convert text to speech via an OpenAI-compatible endpoint."""
        try:
            from openai import OpenAI

            api_keys = self.config.get("api_keys", {})
            tts_config = self.config.get("providers", {}).get("tts", {})

            base_url = self._base_url()
            # Local servers typically ignore the key; send a placeholder so the
            # SDK doesn't refuse to construct a client.
            api_key = (
                tts_config.get("api_key")
                or api_keys.get("openai")
                or os.getenv("OPENAI_API_KEY")
                or "not-needed"
            )
            model = tts_config.get("model") or "tts-1"
            voice = tts_config.get("voice") or "alloy"

            client = OpenAI(base_url=base_url, api_key=api_key)

            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
            )
            response.write_to_file(filename)
            console.print(
                f"  - Generated audio with OpenAI-compatible endpoint ({voice})"
            )
            return filename

        except ImportError:
            err_console.print("  - OpenAI library not installed. Install with: pip install openai")
            return self._fallback_to_gtts(text, filename)
        except Exception as e:
            err_console.print(f"  - OpenAI-compatible TTS error: {e}. Using gTTS fallback.")
            return self._fallback_to_gtts(text, filename)

    def _fallback_to_gtts(self, text: str, filename: str) -> str | None:
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

    def synthesize(self, text: str, filename: str) -> str | None:
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

    def _fallback_to_gtts(self, text: str, filename: str) -> str | None:
        """Fallback to gTTS."""
        gtts_provider = GTTSProvider(self.config)
        return gtts_provider.synthesize(text, filename)
