"""Text-to-speech provider implementations."""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from rich.console import Console

from .base import TTSProvider, is_strict

console = Console()
err_console = Console(stderr=True, style="bold red")

# Kokoro-82M ONNX model files, downloaded once on first use.
KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)


# gTTS renders English accents by routing through a Google TLD. Free accents.
GTTS_ACCENTS = {
    "australian": "com.au",
    "british": "co.uk",
    "american": "us",
    "canadian": "ca",
    "indian": "co.in",
    "irish": "ie",
    "south-african": "co.za",
}


def _warn_if_sample_too_short(sample_path: Path) -> None:
    """Voice-clone engines fail on very short references; warn before uploading."""
    if sample_path.suffix.lower() != ".wav":
        return
    try:
        import wave

        with wave.open(str(sample_path), "rb") as w:
            seconds = w.getnframes() / (w.getframerate() or 1)
        if seconds < 5:
            err_console.print(
                f"  - Warning: voice sample is only {seconds:.1f}s; "
                "references under ~5s usually fail. Use 10-30s."
            )
    except Exception:
        pass


def _prepare_sample_as_wav(sample_path: Path) -> tuple[Path, bool]:
    """Return a WAV path for the sample, converting via ffmpeg if needed.

    Voice Memos and phone recordings arrive as .m4a/.mp3; clone engines are
    happiest with WAV, so convert to 24kHz mono before uploading. Returns
    (path, is_temporary).
    """
    if sample_path.suffix.lower() == ".wav":
        return sample_path, False
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required to convert a non-WAV voice_sample "
            f"({sample_path.suffix}). Install it or supply a .wav file."
        )
    fd, converted = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(sample_path),
                "-ar", "24000", "-ac", "1",
                converted,
            ],
            check=True,
        )
    except Exception:
        # Don't leak the mkstemp file when ffmpeg fails.
        Path(converted).unlink(missing_ok=True)
        raise
    console.print(
        f"  - Converted voice sample {sample_path.name} to WAV for upload"
    )
    return Path(converted), True


class GTTSProvider(TTSProvider):
    """Google Text-to-Speech provider (free), with English accent selection."""

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

            tts_config = self.config.get("providers", {}).get("tts", {})
            # An accent name (australian/british/...) picks the Google TLD;
            # a raw tld also works. Unknown -> default 'com'.
            accent = str(tts_config.get("accent") or "").lower()
            tld = tts_config.get("tld") or GTTS_ACCENTS.get(accent, "com")

            tts = gTTS(text=text, lang="en", tld=tld)
            tts.save(filename)
            label = f" ({accent})" if accent in GTTS_ACCENTS else ""
            console.print(f"  - Generated audio with gTTS{label}")
            return filename

        except Exception as e:
            err_console.print(f"  - gTTS error: {e}")
            return None


class KokoroTTSProvider(TTSProvider):
    """Fully offline TTS using the Kokoro-82M model via ONNX (CPU-friendly).

    Requires the optional dependencies: pip install "slide-stream[local-tts]".
    Model files (~340 MB total) are downloaded once to
    ~/.cache/slide-stream/kokoro/ on first use; set
    ``providers.tts.model_path`` and ``voices_path`` to use existing files.
    After that, synthesis needs no network and no API key.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        # Loading the ~310 MB model is expensive; keep one engine per provider
        # instance so a multi-slide run pays the cost once.
        self._engine: Any = None

    @property
    def name(self) -> str:
        return "kokoro"

    def is_available(self) -> bool:
        """Available when the optional local-tts dependencies are installed."""
        return (
            importlib.util.find_spec("kokoro_onnx") is not None
            and importlib.util.find_spec("soundfile") is not None
        )

    def _tts_config(self) -> dict[str, Any]:
        return self.config.get("providers", {}).get("tts", {})

    def _cache_dir(self) -> Path:
        return Path(
            os.getenv("SLIDE_STREAM_CACHE") or Path.home() / ".cache" / "slide-stream"
        )

    def _espeak_config(self) -> Any:
        """Build an espeak config whose data path fits espeak-ng's limits.

        espeak-ng stores its data path in a fixed ~160-byte buffer; the
        espeakng_loader data dir inside a deeply nested site-packages can
        exceed that, which makes the C library fall back to a nonexistent
        compiled-in path and exit(1) the whole process. When the real path is
        too long, copy the data (~19MB, one-time) to our short cache path.
        A symlink is not enough: phonemizer resolves it back to the long path.
        """
        import shutil

        import espeakng_loader  # type: ignore[import-not-found]
        from kokoro_onnx import EspeakConfig  # type: ignore[import-not-found]

        data_path = Path(espeakng_loader.get_data_path())
        if len(str(data_path)) > 140:
            short_copy = self._cache_dir() / "espeak-ng-data"
            if not (short_copy / "phontab").exists():
                short_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(data_path, short_copy, dirs_exist_ok=True)
            data_path = short_copy
        return EspeakConfig(
            lib_path=espeakng_loader.get_library_path(), data_path=str(data_path)
        )

    def _model_files(self) -> tuple[str, str]:
        """Return (model_path, voices_path), downloading them if needed."""
        tts_config = self._tts_config()
        model_path = tts_config.get("model_path")
        voices_path = tts_config.get("voices_path")
        if model_path and voices_path:
            return str(model_path), str(voices_path)

        cache_dir = self._cache_dir() / "kokoro"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model = cache_dir / "kokoro-v1.0.onnx"
        voices = cache_dir / "voices-v1.0.bin"
        for target, url in ((model, KOKORO_MODEL_URL), (voices, KOKORO_VOICES_URL)):
            if not target.exists():
                self._download(url, target)
        return str(model), str(voices)

    @staticmethod
    def _download(url: str, target: Path) -> None:
        console.print(f"  - Downloading {target.name} (one-time setup)...")
        # Stream to a .part file and rename, so an interrupted download can
        # never be mistaken for a complete model file on the next run.
        partial = target.with_suffix(target.suffix + ".part")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(partial, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        partial.rename(target)

    def synthesize(self, text: str, filename: str) -> str | None:
        """Convert text to speech locally with Kokoro."""
        try:
            import soundfile as sf  # type: ignore[import-not-found]
            from kokoro_onnx import Kokoro  # type: ignore[import-not-found]

            tts_config = self._tts_config()
            voice = tts_config.get("voice") or "af_sarah"
            speed = float(tts_config.get("speed") or 1.0)
            lang = tts_config.get("lang") or "en-us"

            if self._engine is None:
                model_path, voices_path = self._model_files()
                self._engine = Kokoro(
                    model_path, voices_path, espeak_config=self._espeak_config()
                )

            samples, sample_rate = self._engine.create(
                text, voice=voice, speed=speed, lang=lang
            )
            # The pipeline names audio files .mp3; force real WAV output
            # regardless. soundfile would otherwise pick the encoder from the
            # extension, and MP3 needs a lame-enabled libsndfile that many
            # platforms lack. Callers reuse the path they passed in, and
            # downstream ffmpeg (via moviepy) probes content, not extension,
            # so WAV data behind an .mp3 name plays fine.
            sf.write(filename, samples, sample_rate, format="WAV")
            console.print(f"  - Generated audio with Kokoro ({voice})")
            return filename

        except ImportError:
            err_console.print(
                '  - Kokoro not installed. Install with: pip install "slide-stream[local-tts]"'
            )
            return self._fallback_to_gtts(text, filename)
        except Exception as e:
            err_console.print(f"  - Kokoro error: {e}. Using gTTS fallback.")
            return self._fallback_to_gtts(text, filename)

    def _fallback_to_gtts(self, text: str, filename: str) -> str | None:
        """Fallback to gTTS, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            err_console.print(
                f"  - Strict mode: not falling back to gTTS after {self.name} failed."
            )
            return None
        self.fallback_count += 1
        gtts_provider = GTTSProvider(self.config)
        return gtts_provider.synthesize(text, filename)


class ChatterboxTTSProvider(TTSProvider):
    """Voice cloning via a self-hosted Chatterbox TTS server (devnen).

    Privacy-first flow: when ``voice_sample`` points at a local reference
    recording, it is uploaded once per run under a random UUID filename and
    used for every slide, so no lecturer-recognisable voice name ever exists
    on the server, and a server-side cleanup job (see contrib/chatterbox/)
    removes UUID files afterwards. Alternatively set ``voice`` to a stock
    voice or an existing server-side reference file.

    References shorter than ~5 seconds are rejected by the engine; 10-30
    seconds of clean speech is recommended.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        # One uploaded sample per provider instance (i.e. per run).
        self._session_voice: str | None = None

    @property
    def name(self) -> str:
        return "chatterbox"

    def _tts_config(self) -> dict[str, Any]:
        return self.config.get("providers", {}).get("tts", {})

    def _base_url(self) -> str | None:
        base_url = self._tts_config().get("base_url") or os.getenv("CHATTERBOX_BASE_URL")
        if not base_url:
            return None
        # Accept a base_url with or without the OpenAI-style /v1 suffix.
        return base_url.rstrip("/").removesuffix("/v1")

    def _api_key(self) -> str | None:
        return (
            self._tts_config().get("api_key")
            or self.config.get("api_keys", {}).get("chatterbox")
            or os.getenv("CHATTERBOX_TOKEN")
        )

    def _headers(self) -> dict[str, str]:
        api_key = self._api_key()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def is_available(self) -> bool:
        """Available when a server URL and a voice source are configured."""
        tts_config = self._tts_config()
        return bool(
            self._base_url()
            and (tts_config.get("voice") or tts_config.get("voice_sample"))
        )

    def _ensure_voice(self) -> str:
        """Return the server-side voice name, uploading the sample if needed."""
        tts_config = self._tts_config()
        voice_sample = tts_config.get("voice_sample")
        if not voice_sample:
            return tts_config.get("voice") or "default"

        if self._session_voice:
            return self._session_voice

        import uuid

        sample_path = Path(voice_sample)
        if not sample_path.is_file():
            raise FileNotFoundError(f"voice_sample not found: {voice_sample}")

        upload_path, is_temporary = _prepare_sample_as_wav(sample_path)
        try:
            _warn_if_sample_too_short(upload_path)
            session_voice = f"{uuid.uuid4()}.wav"
            with open(upload_path, "rb") as f:
                response = requests.post(
                    f"{self._base_url()}/upload_reference",
                    files={"files": (session_voice, f)},
                    headers=self._headers(),
                    timeout=60,
                )
            response.raise_for_status()
        finally:
            if is_temporary:
                upload_path.unlink(missing_ok=True)
        console.print(
            f"  - Uploaded voice sample as ephemeral reference {session_voice}"
        )
        self._session_voice = session_voice
        return session_voice

    def synthesize(self, text: str, filename: str) -> str | None:
        """Convert text to speech via the Chatterbox server."""
        try:
            tts_config = self._tts_config()
            voice = self._ensure_voice()

            payload: dict[str, Any] = {
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": "wav",
                "speed": float(tts_config.get("speed") or 1.0),
            }
            if tts_config.get("language"):
                payload["language"] = tts_config["language"]

            response = requests.post(
                f"{self._base_url()}/v1/audio/speech",
                json=payload,
                headers=self._headers(),
                timeout=float(tts_config.get("timeout") or 300),
            )
            response.raise_for_status()

            with open(filename, "wb") as f:
                f.write(response.content)
            console.print(f"  - Generated audio with Chatterbox ({voice})")
            return filename

        except Exception as e:
            err_console.print(f"  - Chatterbox error: {e}. Using gTTS fallback.")
            return self._fallback_to_gtts(text, filename)

    def _fallback_to_gtts(self, text: str, filename: str) -> str | None:
        """Fallback to gTTS, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            err_console.print(
                f"  - Strict mode: not falling back to gTTS after {self.name} failed."
            )
            return None
        self.fallback_count += 1
        gtts_provider = GTTSProvider(self.config)
        return gtts_provider.synthesize(text, filename)


class VoiceboxTTSProvider(TTSProvider):
    """Self-hosted Voicebox voice studio as a TTS provider.

    Two modes, selected by config:

    *Persistent profile* — point at a profile you created on the server and
    reuse it across runs. Nothing is created or deleted::

        provider: voicebox
        base_url: https://voice.example.org   # or VOICEBOX_BASE_URL
        profile_id: "<id from POST /profiles>"
        engine: kokoro

    *Ephemeral clone* — supply a local reference recording. A throwaway profile
    is created for the run, cloned from the sample, and deleted afterwards, so
    no clone of the speaker is left on the server::

        provider: voicebox
        base_url: https://voice.example.org
        voice_sample: ./my_voice.wav
        engine: chatterbox
        # reference_text: "The exact words spoken in the clip."

    Cloning needs a transcript of the reference clip. Give ``reference_text``
    to supply it yourself; omit it and Voicebox transcribes the clip via its
    own Whisper endpoint (optionally pinned with ``transcribe_model``), so just
    the audio is enough.

    Because Voicebox exposes ``DELETE /profiles/{id}`` (which also removes the
    profile's samples and reference audio from disk), the clone can be removed
    over the API — unlike the standalone Chatterbox server, which needs the
    server-side cron in contrib/chatterbox/.

    Rendered narration is *not* removed by profile deletion: generation history
    rows outlive their profile. Each generation is therefore deleted via
    ``DELETE /history/{id}`` as soon as its audio is downloaded. Set
    ``delete_generations: false`` to keep them in the server's history.

    ``engine`` selects the model rendering the voice: qwen | qwen_custom_voice |
    luxtts | chatterbox | chatterbox_turbo | tada | kokoro. Cloning quality
    depends on the engine; deletion does not.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        # Set for the lifetime of one run, only when we created the profile.
        self._session_profile_id: str | None = None
        self._profile_ready: bool = False

    @property
    def name(self) -> str:
        return "voicebox"

    def _tts_config(self) -> dict[str, Any]:
        return self.config.get("providers", {}).get("tts", {})

    def _base_url(self) -> str | None:
        base_url = self._tts_config().get("base_url") or os.getenv("VOICEBOX_BASE_URL")
        return base_url.rstrip("/") if base_url else None

    def _headers(self) -> dict[str, str]:
        api_key = self._tts_config().get("api_key") or os.getenv("VOICEBOX_TOKEN")
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def is_available(self) -> bool:
        """Available with a server URL plus either a profile or a voice sample."""
        tts_config = self._tts_config()
        return bool(
            self._base_url()
            and (tts_config.get("profile_id") or tts_config.get("voice_sample"))
        )

    def _transcribe_sample(self, wav_path: Path) -> str:
        """Transcribe the reference clip via Voicebox's own Whisper endpoint.

        Voicebox clones from audio *plus* a transcript of that audio. When the
        user supplies only the clip, we ask the server to transcribe it, so no
        local Whisper install and no hand-typed transcript are needed. The
        first call may return 202 while Voicebox downloads its Whisper model;
        that is a "wait and retry", not a failure.
        """
        tts_config = self._tts_config()
        base_url = self._base_url()
        timeout = float(tts_config.get("timeout") or 300)
        deadline = time.monotonic() + timeout

        data: dict[str, str] = {}
        if tts_config.get("language"):
            data["language"] = tts_config["language"]
        if tts_config.get("transcribe_model"):
            data["model"] = tts_config["transcribe_model"]

        while True:
            with open(wav_path, "rb") as f:
                response = requests.post(
                    f"{base_url}/transcribe",
                    headers=self._headers(),
                    files={"file": (wav_path.name, f)},
                    data=data,
                    timeout=timeout,
                )
            if response.status_code == 202:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Voicebox is still downloading its transcription model. "
                        "Try again once it has finished, or set reference_text."
                    )
                console.print(
                    "  - Waiting for Voicebox to download its transcription model..."
                )
                time.sleep(max(float(tts_config.get("poll_interval") or 1), 2))
                continue
            response.raise_for_status()
            text = (response.json().get("text") or "").strip()
            if not text:
                raise ValueError(
                    "Voicebox transcription returned no text. Supply reference_text "
                    "for the voice_sample instead."
                )
            preview = text if len(text) <= 80 else text[:77] + "..."
            console.print(f'  - Transcribed voice sample: "{preview}"')
            return text

    def _create_ephemeral_profile(self) -> str:
        """Create a throwaway cloned-voice profile from ``voice_sample``."""
        import uuid

        tts_config = self._tts_config()
        base_url = self._base_url()

        sample_path = Path(tts_config["voice_sample"])
        if not sample_path.is_file():
            raise FileNotFoundError(f"voice_sample not found: {sample_path}")

        upload_path, is_temporary = _prepare_sample_as_wav(sample_path)
        try:
            _warn_if_sample_too_short(upload_path)

            # A transcript of the clip is required. Use the one the user gave
            # us, else have Voicebox transcribe the audio. Do this before
            # creating the profile so a failure here leaks nothing.
            reference_text = tts_config.get("reference_text")
            if not reference_text:
                reference_text = self._transcribe_sample(upload_path)

            # A random name, so no speaker-recognisable label exists server-side
            # even in the window before the profile is deleted.
            response = requests.post(
                f"{base_url}/profiles",
                headers=self._headers(),
                json={
                    "name": f"slide-stream-{uuid.uuid4()}",
                    "description": "Ephemeral SlideStream run; deleted after render.",
                    "language": tts_config.get("language") or "en",
                    "voice_type": "cloned",
                },
                timeout=60,
            )
            response.raise_for_status()
            profile_id = response.json()["id"]
            # Record before uploading the sample: if the upload fails, close()
            # must still delete the half-built profile rather than leak it.
            self._session_profile_id = profile_id

            with open(upload_path, "rb") as f:
                sample = requests.post(
                    f"{base_url}/profiles/{profile_id}/samples",
                    headers=self._headers(),
                    files={"file": (upload_path.name, f)},
                    data={"reference_text": reference_text},
                    timeout=float(tts_config.get("timeout") or 300),
                )
            sample.raise_for_status()
        finally:
            if is_temporary:
                upload_path.unlink(missing_ok=True)

        self._profile_ready = True
        console.print(f"  - Cloned voice into ephemeral profile {profile_id}")
        return profile_id

    def _ensure_profile(self) -> str:
        """Return the profile to generate against, cloning one if needed.

        An explicit ``voice_sample`` wins: the user handed us a recording to
        clone (e.g. an upload in serve mode), so a configured ``profile_id``
        must not silently override it. The profile_id is used only when no
        voice_sample is set — mirroring Chatterbox's ``_ensure_voice``.
        """
        tts_config = self._tts_config()
        if not tts_config.get("voice_sample"):
            configured = tts_config.get("profile_id")
            if configured:
                return configured
        if self._session_profile_id and self._profile_ready:
            return self._session_profile_id
        if self._session_profile_id:
            # An earlier attempt died between create and sample upload. Remove
            # that profile instead of generating against a voiceless clone.
            self._delete_profile(self._session_profile_id)
            self._session_profile_id = None
        return self._create_ephemeral_profile()

    def _delete_generation(self, generation_id: str) -> None:
        """Remove a generation and its rendered audio from the server."""
        try:
            response = requests.delete(
                f"{self._base_url()}/history/{generation_id}",
                headers=self._headers(),
                timeout=30,
            )
            response.raise_for_status()
        except Exception as e:
            err_console.print(
                f"  - Warning: could not delete Voicebox generation "
                f"{generation_id}: {e}. Rendered audio remains on the server."
            )

    def _delete_profile(self, profile_id: str) -> None:
        """Remove a profile, its samples, and its cloned reference audio."""
        try:
            response = requests.delete(
                f"{self._base_url()}/profiles/{profile_id}",
                headers=self._headers(),
                timeout=60,
            )
            response.raise_for_status()
            console.print(f"  - Deleted ephemeral voice profile {profile_id}")
        except Exception as e:
            err_console.print(
                f"  - Warning: could not delete Voicebox profile {profile_id}: {e}. "
                "The cloned voice may still exist on the server."
            )

    def close(self) -> None:
        """Delete the run's ephemeral profile, if we created one."""
        if self._session_profile_id:
            self._delete_profile(self._session_profile_id)
            self._session_profile_id = None
            self._profile_ready = False

    def synthesize(self, text: str, filename: str) -> str | None:
        """Convert text to speech via a Voicebox profile."""
        try:
            tts_config = self._tts_config()
            base_url = self._base_url()
            if not base_url:
                raise ValueError("Voicebox base_url is required")
            profile_id = self._ensure_profile()
            headers = self._headers()
            timeout = float(tts_config.get("timeout") or 300)
            poll_interval = float(tts_config.get("poll_interval") or 1)

            # 1. Request generation.
            gen = requests.post(
                f"{base_url}/generate",
                headers=headers,
                json={
                    "profile_id": profile_id,
                    "text": text,
                    "language": tts_config.get("language") or "en",
                    "engine": tts_config.get("engine") or "kokoro",
                    "normalize": True,
                },
                timeout=timeout,
            )
            gen.raise_for_status()
            data = gen.json()
            gen_id = data["id"]

            timed_out = False
            try:
                # 2. Poll until the server reports a terminal status
                # (generation may be async). Unknown in-progress statuses
                # (pending/queued/...) are treated like "generating"; a
                # missing status means the server rendered synchronously.
                deadline = time.monotonic() + timeout
                status = data.get("status")
                status_data: dict[str, Any] = data
                while status and status not in ("completed", "failed"):
                    if time.monotonic() >= deadline:
                        timed_out = True
                        raise TimeoutError(
                            f"Voicebox generation {gen_id} still '{status}' "
                            f"after {timeout:.0f}s"
                        )
                    time.sleep(poll_interval)
                    st = requests.get(
                        f"{base_url}/generate/{gen_id}/status",
                        headers=headers,
                        timeout=30,
                    )
                    st.raise_for_status()
                    status_data = st.json()
                    status = status_data.get("status")

                if status == "failed":
                    detail = (
                        status_data.get("error")
                        or status_data.get("error_message")
                        or status_data.get("detail")
                        or status_data
                    )
                    raise RuntimeError(f"Voicebox generation failed: {detail}")

                # 3. Fetch the audio.
                audio = requests.get(
                    f"{base_url}/audio/{gen_id}", headers=headers, timeout=timeout
                )
                audio.raise_for_status()
                with open(filename, "wb") as f:
                    f.write(audio.content)
            finally:
                # 4. Drop the server-side copy of the narration, even if the
                # download failed part-way through — but not on a poll
                # timeout, when the server may still be rendering it (the
                # sweep script will reap it later).
                if not timed_out and tts_config.get("delete_generations", True):
                    self._delete_generation(gen_id)

            engine = tts_config.get("engine") or "kokoro"
            console.print(f"  - Generated audio with Voicebox ({engine})")
            return filename

        except Exception as e:
            err_console.print(f"  - Voicebox error: {e}. Using gTTS fallback.")
            return self._fallback_to_gtts(text, filename)

    def _fallback_to_gtts(self, text: str, filename: str) -> str | None:
        """Fallback to gTTS, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            err_console.print(
                f"  - Strict mode: not falling back to gTTS after {self.name} failed."
            )
            return None
        self.fallback_count += 1
        gtts_provider = GTTSProvider(self.config)
        return gtts_provider.synthesize(text, filename)


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
        """Fallback to gTTS, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            err_console.print(
                f"  - Strict mode: not falling back to gTTS after {self.name} failed."
            )
            return None
        self.fallback_count += 1
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
        """Fallback to gTTS, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            err_console.print(
                f"  - Strict mode: not falling back to gTTS after {self.name} failed."
            )
            return None
        self.fallback_count += 1
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

            # Save to file (stream_to_file is deprecated in the OpenAI SDK)
            response.write_to_file(filename)
            console.print(f"  - Generated audio with OpenAI TTS ({voice})")
            return filename

        except ImportError:
            err_console.print("  - OpenAI library not installed. Install with: pip install openai")
            return self._fallback_to_gtts(text, filename)
        except Exception as e:
            err_console.print(f"  - OpenAI TTS error: {e}. Using gTTS fallback.")
            return self._fallback_to_gtts(text, filename)

    def _fallback_to_gtts(self, text: str, filename: str) -> str | None:
        """Fallback to gTTS, unless strict mode disables fallbacks."""
        if is_strict(self.config):
            err_console.print(
                f"  - Strict mode: not falling back to gTTS after {self.name} failed."
            )
            return None
        self.fallback_count += 1
        gtts_provider = GTTSProvider(self.config)
        return gtts_provider.synthesize(text, filename)
