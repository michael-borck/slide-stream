"""Tests for image/TTS providers and the provider factory."""

import pytest
from PIL import Image

from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.providers.base import StrictModeError
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


def test_dalle_fallback_renders_slide_content(config, tmp_path, mocker):
    """When DALL-E fails, the text fallback renders the slide's own content
    (not a generic 'Topic: ...'): two slides sharing a query but differing in
    content produce different fallback images."""
    config["api_keys"] = {"openai": "sk-test"}
    fake_client = mocker.MagicMock()
    fake_client.images.generate.side_effect = RuntimeError("boom")
    mocker.patch("openai.OpenAI", return_value=fake_client)

    provider = DalleImageProvider(config)
    out_a = tmp_path / "a.png"
    out_b = tmp_path / "b.png"
    provider.generate_image(
        "same query", str(out_a), slide={"title": "T", "content": ["Apple", "Banana"]}
    )
    provider.generate_image(
        "same query", str(out_b), slide={"title": "T", "content": ["Carrot", "Daikon"]}
    )

    assert out_a.exists() and out_b.exists()
    assert out_a.read_bytes() != out_b.read_bytes()


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


def test_openai_compat_image_requires_base_url_not_just_key(config, monkeypatch):
    """A bare OpenAI key must NOT make the compat image provider available:
    without a base_url it would silently bill real OpenAI instead of the local
    server. Only an explicit base_url enables it."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config["api_keys"] = {"openai": "sk-test"}
    config["providers"]["images"] = {"provider": "openai-compatible"}
    assert OpenAICompatImageProvider(config).is_available() is False

    config["providers"]["images"]["base_url"] = "http://localhost:8080/v1"
    assert OpenAICompatImageProvider(config).is_available() is True


def test_openai_compat_tts_requires_base_url_not_just_key(config, monkeypatch):
    """A bare OpenAI key must NOT make the compat TTS provider available."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config["api_keys"] = {"openai": "sk-test"}
    config["providers"]["tts"] = {"provider": "openai-compatible"}
    assert OpenAICompatTTSProvider(config).is_available() is False

    config["providers"]["tts"]["base_url"] = "http://localhost:8000/v1"
    assert OpenAICompatTTSProvider(config).is_available() is True


def test_text_image_provider_renders_slide_content(config, tmp_path):
    """The text provider renders the slide's own content, not a generic
    placeholder. Two slides sharing a query but differing in content produce
    different images (the old code ignored content entirely)."""
    slide_a = {"title": "Same Title", "content": ["Apple", "Banana"]}
    slide_b = {"title": "Same Title", "content": ["Carrot", "Daikon"]}
    out_a = tmp_path / "a.png"
    out_b = tmp_path / "b.png"

    TextImageProvider(config).generate_image("same query", str(out_a), slide=slide_a)
    TextImageProvider(config).generate_image("same query", str(out_b), slide=slide_b)

    assert out_a.exists() and out_b.exists()
    assert out_a.read_bytes() != out_b.read_bytes()


# --- Strict mode (no silent fallbacks) --------------------------------------


def test_factory_strict_raises_for_unavailable_tts(config, monkeypatch):
    """In strict mode an unusable TTS provider aborts instead of using gTTS."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config["settings"]["strict"] = True
    config["providers"]["tts"]["provider"] = "elevenlabs"

    with pytest.raises(StrictModeError):
        ProviderFactory.create_tts_provider(config)


def test_factory_strict_raises_for_unknown_image_provider(config):
    config["settings"]["strict"] = True
    config["providers"]["images"]["provider"] = "no-such-provider"

    with pytest.raises(StrictModeError):
        ProviderFactory.create_image_provider(config)


def test_factory_non_strict_still_falls_back(config, monkeypatch):
    """Default behaviour is unchanged: unavailable providers fall back."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config["providers"]["tts"]["provider"] = "elevenlabs"

    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)


def test_tts_strict_suppresses_gtts_fallback(config, monkeypatch):
    """A failing premium TTS provider returns None in strict mode rather than
    synthesizing with the wrong (gTTS) voice."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config["settings"]["strict"] = True

    provider = ElevenLabsTTSProvider(config)
    # No API key: synthesize fails internally and must NOT fall back.
    assert provider.synthesize("hello", "unused.mp3") is None


def test_image_strict_raises_instead_of_text_fallback(config, tmp_path, monkeypatch):
    """A failing image provider raises in strict mode rather than silently
    rendering a text slide."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config["settings"]["strict"] = True
    out = tmp_path / "slide.png"

    with pytest.raises(StrictModeError):
        DalleImageProvider(config).generate_image("query", str(out))
    assert not out.exists()


def test_image_non_strict_falls_back_to_text(config, tmp_path, monkeypatch):
    """Default behaviour is unchanged: a failing image provider writes a text
    slide image instead."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "slide.png"

    result = DalleImageProvider(config).generate_image("query", str(out))
    assert result == str(out)
    assert out.exists()
