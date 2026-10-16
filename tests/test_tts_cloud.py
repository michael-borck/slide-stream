"""Tests for cloud and OpenAI-compatible TTS providers."""

import pytest

from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.providers.factory import ProviderFactory
from slide_stream.providers.tts import (
    ElevenLabsTTSProvider,
    OpenAICompatTTSProvider,
)


@pytest.fixture
def config():
    """A minimal valid config (a fresh copy of the defaults)."""
    import copy

    return copy.deepcopy(DEFAULT_CONFIG)


# --- TextImageProvider (no network) ----------------------------------------


def test_elevenlabs_availability_follows_api_key(config, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config["api_keys"] = {}
    assert ElevenLabsTTSProvider(config).is_available() is False

    config["api_keys"] = {"elevenlabs": "el-test"}
    assert ElevenLabsTTSProvider(config).is_available() is True


# --- Mocked SDK behaviour ---------------------------------------------------


def test_elevenlabs_voice_name_resolves_to_id(config, tmp_path, mocker):
    """A friendly voice name is mapped to its voice_id before the API call."""
    out = tmp_path / "audio.mp3"
    config["api_keys"] = {"elevenlabs": "el-test"}
    config["providers"]["tts"]["voice"] = "Rachel"

    fake_client = mocker.MagicMock()
    fake_client.text_to_speech.convert.return_value = b"audio-bytes"
    mocker.patch("elevenlabs.client.ElevenLabs", return_value=fake_client)
    saved = {}

    def fake_save(audio, filename):
        saved["filename"] = filename
        with open(filename, "wb") as f:
            f.write(b"audio-bytes")

    mocker.patch("elevenlabs.save", side_effect=fake_save)

    result = ElevenLabsTTSProvider(config).synthesize("Hello", str(out))

    assert result == str(out)
    # Rachel resolved to her canonical voice_id.
    _, kwargs = fake_client.text_to_speech.convert.call_args
    assert kwargs["voice_id"] == ElevenLabsTTSProvider.VOICE_IDS["rachel"]


def test_elevenlabs_falls_back_to_gtts_on_error(config, tmp_path, mocker):
    out = tmp_path / "audio.mp3"
    config["api_keys"] = {"elevenlabs": "el-test"}

    fake_client = mocker.MagicMock()
    fake_client.text_to_speech.convert.side_effect = RuntimeError("api down")
    mocker.patch("elevenlabs.client.ElevenLabs", return_value=fake_client)

    # gTTS fallback is also mocked, so still no network.
    def fake_save(filename):
        with open(filename, "wb") as f:
            f.write(b"fallback-audio")

    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = fake_save
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    result = ElevenLabsTTSProvider(config).synthesize("Hello", str(out))

    assert result == str(out)
    assert out.exists()


# --- OpenAI-compatible providers (local / hosted, base_url driven) ----------


def test_openai_compat_tts_available_with_base_url_no_key(config, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config["api_keys"] = {}
    config["providers"]["tts"] = {"provider": "openai-compatible"}
    assert OpenAICompatTTSProvider(config).is_available() is False

    # A configured base_url alone makes it available (local servers need no key).
    config["providers"]["tts"]["base_url"] = "http://localhost:8000/v1"
    assert OpenAICompatTTSProvider(config).is_available() is True


def test_factory_selects_openai_compat_tts(config):
    config["providers"]["tts"] = {
        "provider": "openai-compatible",
        "base_url": "http://localhost:8000/v1",
    }
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, OpenAICompatTTSProvider)


def test_openai_compat_tts_writes_file(config, tmp_path, mocker):
    out = tmp_path / "audio.mp3"
    config["providers"]["tts"] = {
        "provider": "openai-compatible",
        "base_url": "http://localhost:8000/v1",
        "voice": "en_US-amy",
        "model": "tts-1",
    }

    def fake_write(filename):
        with open(filename, "wb") as f:
            f.write(b"local-audio")

    fake_response = mocker.MagicMock()
    fake_response.write_to_file.side_effect = fake_write
    fake_client = mocker.MagicMock()
    fake_client.audio.speech.create.return_value = fake_response
    openai_ctor = mocker.patch("openai.OpenAI", return_value=fake_client)

    result = OpenAICompatTTSProvider(config).synthesize("Hello", str(out))

    assert result == str(out)
    assert out.exists()
    # base_url was forwarded to the SDK client.
    assert openai_ctor.call_args.kwargs["base_url"] == "http://localhost:8000/v1"


def test_openai_compat_tts_falls_back_to_gtts_on_error(config, tmp_path, mocker):
    out = tmp_path / "audio.mp3"
    config["providers"]["tts"] = {
        "provider": "openai-compatible",
        "base_url": "http://localhost:8000/v1",
    }

    fake_client = mocker.MagicMock()
    fake_client.audio.speech.create.side_effect = RuntimeError("server down")
    mocker.patch("openai.OpenAI", return_value=fake_client)

    def fake_save(filename):
        with open(filename, "wb") as f:
            f.write(b"fallback")

    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = fake_save
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    result = OpenAICompatTTSProvider(config).synthesize("Hello", str(out))
    assert result == str(out)
    assert out.exists()


def test_openai_compat_tts_requires_base_url_not_just_key(config, monkeypatch):
    """A bare OpenAI key must NOT make the compat TTS provider available."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config["api_keys"] = {"openai": "sk-test"}
    config["providers"]["tts"] = {"provider": "openai-compatible"}
    assert OpenAICompatTTSProvider(config).is_available() is False

    config["providers"]["tts"]["base_url"] = "http://localhost:8000/v1"
    assert OpenAICompatTTSProvider(config).is_available() is True


def test_tts_strict_suppresses_gtts_fallback(config, monkeypatch):
    """A failing premium TTS provider returns None in strict mode rather than
    synthesizing with the wrong (gTTS) voice."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config["settings"]["strict"] = True

    provider = ElevenLabsTTSProvider(config)
    # No API key: synthesize fails internally and must NOT fall back.
    assert provider.synthesize("hello", "unused.mp3") is None


def test_openai_tts_uses_write_to_file(config, tmp_path, mocker):
    from slide_stream.providers.tts import OpenAITTSProvider

    config["api_keys"] = {"openai": "sk-test"}

    def fake_write(filename):
        with open(filename, "wb") as f:
            f.write(b"openai-audio")

    fake_response = mocker.MagicMock()
    fake_response.write_to_file.side_effect = fake_write
    fake_client = mocker.MagicMock()
    fake_client.audio.speech.create.return_value = fake_response
    mocker.patch("openai.OpenAI", return_value=fake_client)

    out = tmp_path / "audio.mp3"
    result = OpenAITTSProvider(config).synthesize("Hello", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"openai-audio"
    fake_response.write_to_file.assert_called_once_with(str(out))
    fake_response.stream_to_file.assert_not_called()
