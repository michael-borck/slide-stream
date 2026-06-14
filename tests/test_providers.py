"""Tests for image/TTS providers and the provider factory."""

import pytest
from PIL import Image

from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.providers.factory import ProviderFactory
from slide_stream.providers.images import (
    DalleImageProvider,
    OpenAICompatImageProvider,
    TextImageProvider,
)
from slide_stream.providers.tts import (
    ElevenLabsTTSProvider,
    GTTSProvider,
    OpenAICompatTTSProvider,
)


@pytest.fixture
def config():
    """A minimal valid config (a fresh copy of the defaults)."""
    import copy

    return copy.deepcopy(DEFAULT_CONFIG)


# --- TextImageProvider (no network) ----------------------------------------


def test_text_image_provider_writes_file(config, tmp_path):
    out = tmp_path / "slide.png"
    provider = TextImageProvider(config)

    result = provider.generate_image("Quantum computing", str(out))

    assert result == str(out)
    assert out.exists()
    # Image is created at the configured resolution.
    with Image.open(out) as img:
        assert img.size == tuple(config["settings"]["video"]["resolution"])


def test_text_image_provider_always_available(config):
    assert TextImageProvider(config).is_available() is True
    assert TextImageProvider(config).name == "text"


# --- Provider factory selection / fallback ---------------------------------


def test_factory_returns_configured_image_provider(config):
    config["providers"]["images"]["provider"] = "text"
    provider = ProviderFactory.create_image_provider(config)
    assert isinstance(provider, TextImageProvider)


def test_factory_falls_back_when_primary_unavailable(config, monkeypatch):
    # dalle3 needs an OpenAI key; without one it must fall back to text.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config["providers"]["images"]["provider"] = "dalle3"
    config["providers"]["images"]["fallback"] = "text"
    config["api_keys"] = {}

    provider = ProviderFactory.create_image_provider(config)
    assert isinstance(provider, TextImageProvider)


def test_factory_unknown_image_provider_falls_back_to_text(config):
    config["providers"]["images"]["provider"] = "does-not-exist"
    config["providers"]["images"]["fallback"] = "also-missing"
    provider = ProviderFactory.create_image_provider(config)
    assert isinstance(provider, TextImageProvider)


def test_factory_default_tts_is_gtts(config):
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)


def test_factory_unknown_tts_falls_back_to_gtts(config):
    config["providers"]["tts"]["provider"] = "nope"
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)


def test_check_provider_availability_reports_always_on_providers(config):
    availability = ProviderFactory.check_provider_availability(config)
    assert availability["images"]["text"] is True
    assert availability["tts"]["gtts"] is True


def test_provider_listings_contain_known_providers():
    assert "text" in ProviderFactory.list_image_providers()
    assert "dalle3" in ProviderFactory.list_image_providers()
    assert "gtts" in ProviderFactory.list_tts_providers()
    assert "elevenlabs" in ProviderFactory.list_tts_providers()


# --- Availability gating on API keys ---------------------------------------


def test_dalle_availability_follows_api_key(config, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config["api_keys"] = {}
    assert DalleImageProvider(config).is_available() is False

    config["api_keys"] = {"openai": "sk-test"}
    assert DalleImageProvider(config).is_available() is True


def test_elevenlabs_availability_follows_api_key(config, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config["api_keys"] = {}
    assert ElevenLabsTTSProvider(config).is_available() is False

    config["api_keys"] = {"elevenlabs": "el-test"}
    assert ElevenLabsTTSProvider(config).is_available() is True


# --- Mocked SDK behaviour ---------------------------------------------------


def test_gtts_provider_saves_file(config, tmp_path, mocker):
    """gTTS is mocked so the test makes no network call."""
    out = tmp_path / "audio.mp3"

    def fake_save(filename):
        with open(filename, "wb") as f:
            f.write(b"fake-audio")

    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = fake_save
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    result = GTTSProvider(config).synthesize("Hello world", str(out))

    assert result == str(out)
    assert out.exists()


def test_dalle_falls_back_to_text_on_api_error(config, tmp_path, mocker):
    """If the DALL-E call fails, a text image is produced instead."""
    out = tmp_path / "img.png"
    config["api_keys"] = {"openai": "sk-test"}

    fake_client = mocker.MagicMock()
    fake_client.images.generate.side_effect = RuntimeError("boom")
    mocker.patch("openai.OpenAI", return_value=fake_client)

    result = DalleImageProvider(config).generate_image("topic", str(out))

    assert result == str(out)
    assert out.exists()  # produced by the text fallback


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


def test_factory_selects_openai_compat_image(config):
    config["providers"]["images"] = {
        "provider": "openai-compatible",
        "fallback": "text",
        "base_url": "http://localhost:8080/v1",
    }
    provider = ProviderFactory.create_image_provider(config)
    assert isinstance(provider, OpenAICompatImageProvider)


def test_openai_compat_image_handles_b64(config, tmp_path, mocker):
    """Local servers commonly return inline base64 rather than a URL."""
    import base64

    out = tmp_path / "img.png"
    config["providers"]["images"] = {
        "provider": "openai-compatible",
        "base_url": "http://localhost:8080/v1",
    }

    item = mocker.MagicMock()
    item.b64_json = base64.b64encode(b"raw-image-bytes").decode()
    item.url = None
    fake_client = mocker.MagicMock()
    fake_client.images.generate.return_value.data = [item]
    mocker.patch("openai.OpenAI", return_value=fake_client)

    result = OpenAICompatImageProvider(config).generate_image("topic", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"raw-image-bytes"


def test_openai_compat_image_handles_url(config, tmp_path, mocker):
    out = tmp_path / "img.png"
    config["providers"]["images"] = {
        "provider": "openai-compatible",
        "base_url": "http://localhost:8080/v1",
    }

    item = mocker.MagicMock()
    item.b64_json = None
    item.url = "http://localhost:8080/image.png"
    fake_client = mocker.MagicMock()
    fake_client.images.generate.return_value.data = [item]
    mocker.patch("openai.OpenAI", return_value=fake_client)

    fake_get = mocker.MagicMock()
    fake_get.content = b"downloaded-bytes"
    mocker.patch(
        "slide_stream.providers.images.requests.get", return_value=fake_get
    )

    result = OpenAICompatImageProvider(config).generate_image("topic", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"downloaded-bytes"


def test_openai_compat_image_falls_back_to_text_on_error(config, tmp_path, mocker):
    out = tmp_path / "img.png"
    config["providers"]["images"] = {
        "provider": "openai-compatible",
        "base_url": "http://localhost:8080/v1",
    }

    fake_client = mocker.MagicMock()
    fake_client.images.generate.side_effect = RuntimeError("boom")
    mocker.patch("openai.OpenAI", return_value=fake_client)

    result = OpenAICompatImageProvider(config).generate_image("topic", str(out))

    assert result == str(out)
    assert out.exists()  # text fallback produced an image
