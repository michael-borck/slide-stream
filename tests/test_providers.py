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
    KokoroTTSProvider,
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


def _delenv_voicebox(monkeypatch):
    monkeypatch.delenv("VOICEBOX_BASE_URL", raising=False)
    monkeypatch.delenv("VOICEBOX_TOKEN", raising=False)


def test_factory_default_tts_falls_back_to_gtts_without_a_server(config, monkeypatch):
    """The default is voicebox, but with no server configured a bare install
    must still render — so the factory quietly returns gTTS."""
    _delenv_voicebox(monkeypatch)
    assert config["providers"]["tts"]["provider"] == "voicebox"
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)


def test_factory_unconfigured_default_does_not_error(config, capsys, monkeypatch):
    """A zero-config user must not see a scary ❌; just a friendly hint."""
    _delenv_voicebox(monkeypatch)
    ProviderFactory.create_tts_provider(config)
    captured = capsys.readouterr()
    assert "not available" not in (captured.out + captured.err)


def test_factory_configured_but_broken_tts_is_an_error(config, capsys):
    """A voicebox with a base_url but a real failure is a genuine error."""
    config["providers"]["tts"]["base_url"] = "https://vb.org"
    # base_url set but no profile_id/voice_sample -> is_available() False, and
    # this counts as "set up but unusable", not "never configured".
    ProviderFactory.create_tts_provider(config)
    captured = capsys.readouterr()
    assert "not available" in (captured.out + captured.err)


def test_factory_unconfigured_default_uses_gtts_even_in_strict(config, monkeypatch):
    """Strict protects an explicitly-chosen provider; the unconfigured default
    voicebox was never chosen, so a bare strict install still renders on gTTS
    instead of aborting."""
    _delenv_voicebox(monkeypatch)
    config["settings"]["strict"] = True
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)


def test_factory_env_configured_voicebox_is_not_silently_downgraded(
    config, monkeypatch, capsys
):
    """A voicebox configured purely via env vars counts as configured: a
    broken setup is reported as an error instead of quietly using gTTS."""
    monkeypatch.setenv("VOICEBOX_BASE_URL", "https://vb.org")
    # base_url alone (no profile_id/voice_sample) -> unavailable, a real error.
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)  # non-strict still falls back
    captured = capsys.readouterr()
    assert "not available" in (captured.out + captured.err)


def test_factory_env_configured_voicebox_aborts_in_strict(config, monkeypatch):
    """Strict mode must trigger for an env-var-configured (but broken)
    provider rather than silently downgrading to gTTS."""
    monkeypatch.setenv("VOICEBOX_BASE_URL", "https://vb.org")
    config["settings"]["strict"] = True
    with pytest.raises(StrictModeError):
        ProviderFactory.create_tts_provider(config)


def test_factory_configured_broken_voicebox_still_aborts_in_strict(config):
    """But a voicebox the user did set up (base_url present) must still abort
    under strict rather than silently swapping to gTTS."""
    config["settings"]["strict"] = True
    config["providers"]["tts"]["base_url"] = "https://vb.org"
    with pytest.raises(StrictModeError):
        ProviderFactory.create_tts_provider(config)


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


# --- Font loading (text slides must not use the tiny bitmap fallback) -------


def test_load_font_returns_scalable_font_at_requested_size():
    """load_font must yield a real scalable font at the asked-for size; the
    old code silently fell back to Pillow's unsized ~10px bitmap font on
    systems without arial.ttf/DejaVuSans.ttf on Pillow's search path (macOS)."""
    from slide_stream.providers.images import load_font

    font = load_font(72)
    assert getattr(font, "size", None) == 72


# --- Kokoro local TTS --------------------------------------------------------


def test_kokoro_unavailable_without_optional_deps(config, monkeypatch):
    import importlib.util as ilu

    monkeypatch.setattr(ilu, "find_spec", lambda name: None)
    assert KokoroTTSProvider(config).is_available() is False


def test_kokoro_available_when_deps_installed(config, monkeypatch):
    import importlib.util as ilu

    monkeypatch.setattr(ilu, "find_spec", lambda name: object())
    assert KokoroTTSProvider(config).is_available() is True


def test_kokoro_uses_configured_model_paths_without_download(config):
    """Explicit model_path/voices_path must be used as-is (no network, no
    cache directory)."""
    config["providers"]["tts"]["model_path"] = "/models/kokoro.onnx"
    config["providers"]["tts"]["voices_path"] = "/models/voices.bin"

    provider = KokoroTTSProvider(config)
    assert provider._model_files() == ("/models/kokoro.onnx", "/models/voices.bin")


def test_kokoro_strict_failure_returns_none(config, monkeypatch):
    """If kokoro_onnx is not importable, strict mode must refuse the gTTS
    fallback and return None."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("kokoro_onnx") or name == "soundfile":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    config["settings"]["strict"] = True

    assert KokoroTTSProvider(config).synthesize("hello", "unused.mp3") is None


def test_factory_strict_raises_for_unavailable_kokoro(config, monkeypatch):
    import importlib.util as ilu

    monkeypatch.setattr(ilu, "find_spec", lambda name: None)
    config["settings"]["strict"] = True
    config["providers"]["tts"]["provider"] = "kokoro"

    with pytest.raises(StrictModeError):
        ProviderFactory.create_tts_provider(config)


# --- Chatterbox provider (network mocked) ------------------------------------


def _chatterbox_config(config, **tts_overrides):
    config["providers"]["tts"] = {
        "provider": "chatterbox",
        "base_url": "https://cb.example.org",
        **tts_overrides,
    }
    return config


def test_chatterbox_availability_needs_url_and_voice(config, monkeypatch):
    monkeypatch.delenv("CHATTERBOX_BASE_URL", raising=False)
    monkeypatch.delenv("CHATTERBOX_TOKEN", raising=False)
    from slide_stream.providers.tts import ChatterboxTTSProvider

    config["providers"]["tts"] = {"provider": "chatterbox"}
    assert ChatterboxTTSProvider(config).is_available() is False

    _chatterbox_config(config)  # url but no voice source
    assert ChatterboxTTSProvider(config).is_available() is False

    _chatterbox_config(config, voice="Emily.wav")
    assert ChatterboxTTSProvider(config).is_available() is True

    _chatterbox_config(config, voice_sample="/tmp/me.wav")
    assert ChatterboxTTSProvider(config).is_available() is True


def test_chatterbox_stock_voice_synthesis(config, tmp_path, mocker):
    from slide_stream.providers.tts import ChatterboxTTSProvider

    _chatterbox_config(config, voice="Emily.wav", api_key="sekret")
    out = tmp_path / "audio.mp3"

    fake_response = mocker.MagicMock(content=b"RIFFfake")
    post = mocker.patch(
        "slide_stream.providers.tts.requests.post", return_value=fake_response
    )

    result = ChatterboxTTSProvider(config).synthesize("Hello", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"RIFFfake"
    url = post.call_args[0][0]
    kwargs = post.call_args[1]
    assert url == "https://cb.example.org/v1/audio/speech"
    assert kwargs["json"]["voice"] == "Emily.wav"
    assert kwargs["headers"]["Authorization"] == "Bearer sekret"


def test_chatterbox_base_url_v1_suffix_stripped(config, tmp_path, mocker):
    from slide_stream.providers.tts import ChatterboxTTSProvider

    _chatterbox_config(config, voice="Emily.wav", base_url="https://cb.example.org/v1/")
    fake_response = mocker.MagicMock(content=b"RIFFfake")
    post = mocker.patch(
        "slide_stream.providers.tts.requests.post", return_value=fake_response
    )

    ChatterboxTTSProvider(config).synthesize("Hello", str(tmp_path / "a.mp3"))

    assert post.call_args[0][0] == "https://cb.example.org/v1/audio/speech"


def test_chatterbox_voice_sample_uploads_uuid_once(config, tmp_path, mocker):
    """voice_sample mode: uploads under a UUID name once, reuses it for every
    slide in the run, and never sends the user's original filename."""
    import uuid as uuid_mod

    from slide_stream.providers.tts import ChatterboxTTSProvider

    sample = tmp_path / "michael-voice.wav"
    sample.write_bytes(b"fake-sample")
    _chatterbox_config(config, voice_sample=str(sample))

    fake_response = mocker.MagicMock(content=b"RIFFfake")
    post = mocker.patch(
        "slide_stream.providers.tts.requests.post", return_value=fake_response
    )

    provider = ChatterboxTTSProvider(config)
    provider.synthesize("Slide one", str(tmp_path / "1.mp3"))
    provider.synthesize("Slide two", str(tmp_path / "2.mp3"))

    upload_calls = [c for c in post.call_args_list if c[0][0].endswith("/upload_reference")]
    speech_calls = [c for c in post.call_args_list if c[0][0].endswith("/v1/audio/speech")]
    assert len(upload_calls) == 1  # one upload for the whole run
    assert len(speech_calls) == 2

    uploaded_name = upload_calls[0][1]["files"]["files"][0]
    assert uploaded_name.endswith(".wav")
    uuid_mod.UUID(uploaded_name.removesuffix(".wav"))  # valid UUID or raises
    assert "michael" not in uploaded_name
    # Both speech calls used the ephemeral UUID voice.
    for call in speech_calls:
        assert call[1]["json"]["voice"] == uploaded_name


def test_chatterbox_strict_failure_returns_none(config, tmp_path, mocker):
    from slide_stream.providers.tts import ChatterboxTTSProvider

    _chatterbox_config(config, voice="Emily.wav")
    config["settings"]["strict"] = True
    mocker.patch(
        "slide_stream.providers.tts.requests.post",
        side_effect=RuntimeError("server down"),
    )

    assert ChatterboxTTSProvider(config).synthesize("x", str(tmp_path / "a.mp3")) is None


def test_chatterbox_missing_sample_fails_cleanly(config, tmp_path):
    from slide_stream.providers.tts import ChatterboxTTSProvider

    _chatterbox_config(config, voice_sample=str(tmp_path / "nope.wav"))
    config["settings"]["strict"] = True

    assert ChatterboxTTSProvider(config).synthesize("x", str(tmp_path / "a.mp3")) is None


# --- UUID voice filtering -----------------------------------------------------


def test_uuid_voice_filter():
    from slide_stream.cli import _is_uuid_voice

    assert _is_uuid_voice("c7e37659-03f8-4781-8bd9-f6ec70ca9f5a.wav") is True
    assert _is_uuid_voice("AA6B8FDE-C9AF-4B0A-A023-E7D5F6294EE7.WAV") is True
    assert _is_uuid_voice("c7e37659-03f8-4781-8bd9-f6ec70ca9f5a") is True
    assert _is_uuid_voice("michael.wav") is False
    assert _is_uuid_voice("Emily.wav") is False
    assert _is_uuid_voice("deadbeef.wav") is False


def test_chatterbox_converts_m4a_sample_to_wav(config, tmp_path, mocker):
    """Voice Memos-style .m4a samples are converted to WAV (real ffmpeg)
    before the UUID upload; the temp WAV is cleaned up afterwards."""
    import subprocess
    import wave

    from slide_stream.providers.tts import ChatterboxTTSProvider

    # Build a real m4a: 1s of silence, wav -> m4a via ffmpeg.
    src_wav = tmp_path / "memo.wav"
    with wave.open(str(src_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 24000)
    sample = tmp_path / "memo.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_wav), str(sample)],
        check=True,
    )

    _chatterbox_config(config, voice_sample=str(sample))
    uploaded = {}

    def fake_post(url, **kwargs):
        if url.endswith("/upload_reference"):
            name, fileobj = kwargs["files"]["files"]
            uploaded["name"] = name
            uploaded["head"] = fileobj.read(4)
        response = mocker.MagicMock(content=b"RIFFfake")
        return response

    mocker.patch("slide_stream.providers.tts.requests.post", side_effect=fake_post)

    out = tmp_path / "a.mp3"
    result = ChatterboxTTSProvider(config).synthesize("hello", str(out))

    assert result == str(out)
    assert uploaded["name"].endswith(".wav")
    assert uploaded["head"] == b"RIFF"  # genuinely converted to WAV


# --- Gemini/Imagen image provider --------------------------------------------


def test_gemini_image_availability_follows_key(config, monkeypatch):
    from slide_stream.providers.images import GeminiImageProvider

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config["api_keys"] = {}
    assert GeminiImageProvider(config).is_available() is False

    config["api_keys"] = {"gemini": "g-test"}
    assert GeminiImageProvider(config).is_available() is True


def test_gemini_image_falls_back_to_text_without_sdk(config, tmp_path, monkeypatch):
    """No google-genai installed -> text fallback (still produces an image)."""
    from slide_stream.providers.images import GeminiImageProvider

    config["api_keys"] = {"gemini": "g-test"}
    out = tmp_path / "img.png"
    # google-genai is not installed in the test env, so the import fails.
    result = GeminiImageProvider(config).generate_image(
        "Quantum", str(out), slide={"title": "Quantum", "content": ["a"]}
    )
    assert result == str(out)
    assert out.exists()  # text card


def test_gemini_image_success_with_mocked_sdk(config, tmp_path, monkeypatch, mocker):
    """Inject a fake google-genai and verify the Imagen bytes are written."""
    import sys

    from slide_stream.providers.images import GeminiImageProvider

    config["api_keys"] = {"gemini": "g-test"}
    config["providers"]["images"]["model"] = "imagen-4.0-fast-generate-001"

    fake_image = mocker.MagicMock()
    fake_image.image.image_bytes = b"\x89PNG-imagen"
    fake_response = mocker.MagicMock(generated_images=[fake_image])
    fake_client = mocker.MagicMock()
    fake_client.models.generate_images.return_value = fake_response

    genai_mod = mocker.MagicMock()
    genai_mod.Client.return_value = fake_client
    types_mod = mocker.MagicMock()
    google_mod = mocker.MagicMock()
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)

    out = tmp_path / "img.png"
    result = GeminiImageProvider(config).generate_image("Neurons", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"\x89PNG-imagen"
    # Configured model + landscape aspect ratio were passed through.
    kwargs = fake_client.models.generate_images.call_args.kwargs
    assert kwargs["model"] == "imagen-4.0-fast-generate-001"


def test_factory_registers_gemini_image_provider():
    assert "gemini" in ProviderFactory.list_image_providers()


# --- SwarmUI image provider (network mocked) ---------------------------------


def test_swarmui_availability_needs_base_url(config, monkeypatch):
    from slide_stream.providers.images import SwarmUIImageProvider

    monkeypatch.delenv("SWARMUI_BASE_URL", raising=False)
    config["providers"]["images"] = {"provider": "swarmui"}
    assert SwarmUIImageProvider(config).is_available() is False

    config["providers"]["images"]["base_url"] = "https://image.example.org"
    assert SwarmUIImageProvider(config).is_available() is True


def test_swarmui_generate_flow(config, tmp_path, mocker):
    """session -> generate -> fetch path -> save bytes; model passed through."""
    from slide_stream.providers.images import SwarmUIImageProvider

    config["providers"]["images"] = {
        "provider": "swarmui",
        "base_url": "https://image.example.org",
        "model": "juggernautXL_v9",
        "steps": 20,
    }

    session_resp = mocker.MagicMock()
    session_resp.json.return_value = {"session_id": "sess-123"}
    gen_resp = mocker.MagicMock()
    gen_resp.json.return_value = {"images": ["View/local/0.png"]}
    img_resp = mocker.MagicMock(content=b"\x89PNG-swarm")

    post = mocker.patch(
        "slide_stream.providers.images.requests.post",
        side_effect=[session_resp, gen_resp],
    )
    get = mocker.patch(
        "slide_stream.providers.images.requests.get", return_value=img_resp
    )

    out = tmp_path / "img.png"
    result = SwarmUIImageProvider(config).generate_image("a red bicycle", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"\x89PNG-swarm"
    # GenerateText2Image got the session id, model, and steps.
    gen_call = post.call_args_list[1]
    assert gen_call[0][0].endswith("/API/GenerateText2Image")
    body = gen_call[1]["json"]
    assert body["session_id"] == "sess-123"
    assert body["model"] == "juggernautXL_v9"
    assert body["steps"] == 20
    # Image fetched from the returned server path.
    assert get.call_args[0][0] == "https://image.example.org/View/local/0.png"


def test_swarmui_no_model_falls_back_to_text(config, tmp_path, mocker):
    from slide_stream.providers.images import SwarmUIImageProvider

    config["providers"]["images"] = {
        "provider": "swarmui",
        "base_url": "https://image.example.org",
    }
    session_resp = mocker.MagicMock()
    session_resp.json.return_value = {"session_id": "s"}
    gen_resp = mocker.MagicMock()
    gen_resp.json.return_value = {"error": "No model input given."}
    mocker.patch(
        "slide_stream.providers.images.requests.post",
        side_effect=[session_resp, gen_resp],
    )

    out = tmp_path / "img.png"
    result = SwarmUIImageProvider(config).generate_image(
        "topic", str(out), slide={"title": "T", "content": ["a"]}
    )
    assert result == str(out)
    assert out.exists()  # text fallback rendered


def test_factory_registers_swarmui_provider():
    assert "swarmui" in ProviderFactory.list_image_providers()


# --- Voicebox TTS provider (network mocked) ----------------------------------


def test_voicebox_availability_needs_url_and_profile(config, monkeypatch):
    from slide_stream.providers.tts import VoiceboxTTSProvider

    monkeypatch.delenv("VOICEBOX_BASE_URL", raising=False)
    config["providers"]["tts"] = {"provider": "voicebox"}
    assert VoiceboxTTSProvider(config).is_available() is False

    config["providers"]["tts"] = {"provider": "voicebox", "base_url": "https://vb.org"}
    assert VoiceboxTTSProvider(config).is_available() is False

    config["providers"]["tts"]["profile_id"] = "p1"
    assert VoiceboxTTSProvider(config).is_available() is True


def test_voicebox_sync_generation(config, tmp_path, mocker):
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config["providers"]["tts"] = {
        "provider": "voicebox", "base_url": "https://vb.org",
        "profile_id": "p1", "engine": "kokoro",
    }
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "completed"}
    post = mocker.patch("slide_stream.providers.tts.requests.post", return_value=gen)
    audio = mocker.MagicMock(content=b"AUDIO")
    get = mocker.patch("slide_stream.providers.tts.requests.get", return_value=audio)

    out = tmp_path / "a.mp3"
    result = VoiceboxTTSProvider(config).synthesize("hello", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"AUDIO"
    body = post.call_args[1]["json"]
    assert body["profile_id"] == "p1"
    assert body["engine"] == "kokoro"
    assert get.call_args[0][0] == "https://vb.org/audio/g1"


def test_voicebox_polls_until_done(config, tmp_path, mocker):
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config["providers"]["tts"] = {
        "provider": "voicebox", "base_url": "https://vb.org",
        "profile_id": "p1", "poll_interval": 0,
    }
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "generating"}
    mocker.patch("slide_stream.providers.tts.requests.post", return_value=gen)

    generating = mocker.MagicMock()
    generating.json.return_value = {"status": "generating"}
    done = mocker.MagicMock()
    done.json.return_value = {"status": "completed"}
    audio = mocker.MagicMock(content=b"OK")
    get = mocker.patch(
        "slide_stream.providers.tts.requests.get",
        side_effect=[generating, done, audio],
    )
    mocker.patch("slide_stream.providers.tts.time.sleep")

    out = tmp_path / "a.mp3"
    assert VoiceboxTTSProvider(config).synthesize("hi", str(out)) == str(out)
    assert out.read_bytes() == b"OK"
    # Last GET fetched the audio after polling status twice.
    assert get.call_args_list[-1][0][0] == "https://vb.org/audio/g1"


def test_voicebox_strict_failure_returns_none(config, tmp_path, mocker):
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config["providers"]["tts"] = {
        "provider": "voicebox", "base_url": "https://vb.org", "profile_id": "p1",
    }
    config["settings"]["strict"] = True
    mocker.patch(
        "slide_stream.providers.tts.requests.post",
        side_effect=RuntimeError("down"),
    )
    assert VoiceboxTTSProvider(config).synthesize("x", str(tmp_path / "a.mp3")) is None


def test_factory_registers_voicebox():
    assert "voicebox" in ProviderFactory.list_tts_providers()


# --- Voicebox ephemeral voice clone (create -> generate -> delete) ------------


def _voicebox_clone_config(config, tmp_path):
    sample = tmp_path / "voice.wav"
    sample.write_bytes(b"RIFF")
    config["providers"]["tts"] = {
        "provider": "voicebox",
        "base_url": "https://vb.org",
        "voice_sample": str(sample),
        "reference_text": "The exact words.",
        "engine": "chatterbox",
    }
    return config


def test_voicebox_available_with_voice_sample_and_no_profile(config, tmp_path):
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config = _voicebox_clone_config(config, tmp_path)
    assert VoiceboxTTSProvider(config).is_available() is True


def test_voicebox_transcribes_sample_when_no_reference_text(config, tmp_path, mocker):
    """Given only audio, the clip is transcribed via Voicebox and that text is
    used as the clone's reference transcript."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config = _voicebox_clone_config(config, tmp_path)
    del config["providers"]["tts"]["reference_text"]

    transcribe = mocker.MagicMock(status_code=200)
    transcribe.json.return_value = {"text": "words from the clip", "duration": 6.0}
    profile = mocker.MagicMock(status_code=200)
    profile.json.return_value = {"id": "prof1"}
    sample = mocker.MagicMock(status_code=200)
    gen = mocker.MagicMock(status_code=200)
    gen.json.return_value = {"id": "g1", "status": "completed"}
    post = mocker.patch(
        "slide_stream.providers.tts.requests.post",
        side_effect=[transcribe, profile, sample, gen],
    )
    mocker.patch(
        "slide_stream.providers.tts.requests.get",
        return_value=mocker.MagicMock(content=b"AUDIO"),
    )
    mocker.patch("slide_stream.providers.tts.requests.delete")

    out = tmp_path / "a.mp3"
    assert VoiceboxTTSProvider(config).synthesize("hi", str(out)) == str(out)

    # First call transcribes the clip...
    assert post.call_args_list[0][0][0] == "https://vb.org/transcribe"
    # ...and the transcript is what gets uploaded as the sample's reference.
    assert post.call_args_list[2][0][0] == "https://vb.org/profiles/prof1/samples"
    assert post.call_args_list[2][1]["data"]["reference_text"] == "words from the clip"


def test_voicebox_waits_for_whisper_model_download(config, tmp_path, mocker):
    """A 202 means the transcription model is downloading: wait and retry."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config = _voicebox_clone_config(config, tmp_path)
    del config["providers"]["tts"]["reference_text"]

    downloading = mocker.MagicMock(status_code=202)
    ready = mocker.MagicMock(status_code=200)
    ready.json.return_value = {"text": "clip text", "duration": 5.0}
    profile = mocker.MagicMock(status_code=200)
    profile.json.return_value = {"id": "prof1"}
    sample = mocker.MagicMock(status_code=200)
    gen = mocker.MagicMock(status_code=200)
    gen.json.return_value = {"id": "g1", "status": "completed"}
    post = mocker.patch(
        "slide_stream.providers.tts.requests.post",
        side_effect=[downloading, ready, profile, sample, gen],
    )
    mocker.patch(
        "slide_stream.providers.tts.requests.get",
        return_value=mocker.MagicMock(content=b"A"),
    )
    mocker.patch("slide_stream.providers.tts.requests.delete")
    sleep = mocker.patch("slide_stream.providers.tts.time.sleep")

    out = tmp_path / "a.mp3"
    assert VoiceboxTTSProvider(config).synthesize("hi", str(out)) == str(out)
    # Transcribe was retried after the 202, and we slept in between.
    transcribes = [c for c in post.call_args_list if c[0][0] == "https://vb.org/transcribe"]
    assert len(transcribes) == 2
    sleep.assert_called()


def test_voicebox_no_profile_created_if_transcription_fails(config, tmp_path, mocker):
    """Transcription runs before profile creation, so a failure leaks nothing."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config = _voicebox_clone_config(config, tmp_path)
    del config["providers"]["tts"]["reference_text"]
    config["settings"]["strict"] = True

    failed = mocker.MagicMock(status_code=500)
    failed.raise_for_status.side_effect = RuntimeError("transcribe failed")
    post = mocker.patch("slide_stream.providers.tts.requests.post", return_value=failed)
    delete = mocker.patch("slide_stream.providers.tts.requests.delete")

    provider = VoiceboxTTSProvider(config)
    assert provider.synthesize("hi", str(tmp_path / "a.mp3")) is None
    # Only the transcribe endpoint was hit; no profile to leak, nothing to delete.
    assert all(c[0][0] == "https://vb.org/transcribe" for c in post.call_args_list)
    provider.close()
    delete.assert_not_called()


def test_voicebox_reference_text_skips_transcription(config, tmp_path, mocker):
    """When reference_text is given, we must not call the transcribe endpoint."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config = _voicebox_clone_config(config, tmp_path)  # includes reference_text
    profile = mocker.MagicMock(status_code=200)
    profile.json.return_value = {"id": "prof1"}
    gen = mocker.MagicMock(status_code=200)
    gen.json.return_value = {"id": "g1", "status": "completed"}
    post = mocker.patch(
        "slide_stream.providers.tts.requests.post",
        side_effect=[profile, mocker.MagicMock(status_code=200), gen],
    )
    mocker.patch(
        "slide_stream.providers.tts.requests.get",
        return_value=mocker.MagicMock(content=b"A"),
    )
    mocker.patch("slide_stream.providers.tts.requests.delete")

    VoiceboxTTSProvider(config).synthesize("hi", str(tmp_path / "a.mp3"))
    assert not any("/transcribe" in c[0][0] for c in post.call_args_list)


def test_voicebox_clones_generates_then_deletes_everything(config, tmp_path, mocker):
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config = _voicebox_clone_config(config, tmp_path)

    profile = mocker.MagicMock()
    profile.json.return_value = {"id": "prof1"}
    sample = mocker.MagicMock()
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "completed"}
    post = mocker.patch(
        "slide_stream.providers.tts.requests.post",
        side_effect=[profile, sample, gen],
    )
    audio = mocker.MagicMock(content=b"AUDIO")
    mocker.patch("slide_stream.providers.tts.requests.get", return_value=audio)
    delete = mocker.patch("slide_stream.providers.tts.requests.delete")

    provider = VoiceboxTTSProvider(config)
    out = tmp_path / "a.mp3"
    assert provider.synthesize("hello", str(out)) == str(out)
    assert out.read_bytes() == b"AUDIO"

    # Profile created as a cloned voice, then the sample uploaded to it.
    assert post.call_args_list[0][0][0] == "https://vb.org/profiles"
    assert post.call_args_list[0][1]["json"]["voice_type"] == "cloned"
    assert post.call_args_list[1][0][0] == "https://vb.org/profiles/prof1/samples"
    assert post.call_args_list[1][1]["data"]["reference_text"] == "The exact words."
    # Generation ran against the freshly cloned profile.
    assert post.call_args_list[2][1]["json"]["profile_id"] == "prof1"

    # The rendered narration is dropped as soon as it is downloaded...
    assert delete.call_args_list[0][0][0] == "https://vb.org/history/g1"
    # ...and the clone itself only at end of run.
    assert delete.call_count == 1
    provider.close()
    assert delete.call_args_list[1][0][0] == "https://vb.org/profiles/prof1"


def test_voicebox_clone_reused_across_slides(config, tmp_path, mocker):
    """The profile is created once per run, not once per slide."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config = _voicebox_clone_config(config, tmp_path)
    profile = mocker.MagicMock()
    profile.json.return_value = {"id": "prof1"}
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "completed"}
    post = mocker.patch(
        "slide_stream.providers.tts.requests.post",
        side_effect=[profile, mocker.MagicMock(), gen, gen],
    )
    mocker.patch(
        "slide_stream.providers.tts.requests.get",
        return_value=mocker.MagicMock(content=b"A"),
    )
    mocker.patch("slide_stream.providers.tts.requests.delete")

    provider = VoiceboxTTSProvider(config)
    provider.synthesize("one", str(tmp_path / "1.mp3"))
    provider.synthesize("two", str(tmp_path / "2.mp3"))

    profile_creates = [
        c for c in post.call_args_list if c[0][0] == "https://vb.org/profiles"
    ]
    assert len(profile_creates) == 1


def test_voicebox_deletes_profile_if_sample_upload_fails(config, tmp_path, mocker):
    """A half-built profile must not be left behind on the server."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config = _voicebox_clone_config(config, tmp_path)
    config["settings"]["strict"] = True
    profile = mocker.MagicMock()
    profile.json.return_value = {"id": "prof1"}
    mocker.patch(
        "slide_stream.providers.tts.requests.post",
        side_effect=[profile, RuntimeError("upload failed")],
    )
    delete = mocker.patch("slide_stream.providers.tts.requests.delete")

    provider = VoiceboxTTSProvider(config)
    assert provider.synthesize("hi", str(tmp_path / "a.mp3")) is None
    provider.close()
    assert delete.call_args_list[-1][0][0] == "https://vb.org/profiles/prof1"


def test_voicebox_persistent_profile_is_never_deleted(config, tmp_path, mocker):
    """profile_id names a voice the user owns; close() must not delete it."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config["providers"]["tts"] = {
        "provider": "voicebox", "base_url": "https://vb.org", "profile_id": "keepme",
    }
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "completed"}
    mocker.patch("slide_stream.providers.tts.requests.post", return_value=gen)
    mocker.patch(
        "slide_stream.providers.tts.requests.get",
        return_value=mocker.MagicMock(content=b"A"),
    )
    delete = mocker.patch("slide_stream.providers.tts.requests.delete")

    provider = VoiceboxTTSProvider(config)
    provider.synthesize("hi", str(tmp_path / "a.mp3"))
    provider.close()

    deleted = [c[0][0] for c in delete.call_args_list]
    assert "https://vb.org/profiles/keepme" not in deleted
    assert "https://vb.org/history/g1" in deleted


def test_voicebox_delete_generations_can_be_disabled(config, tmp_path, mocker):
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config["providers"]["tts"] = {
        "provider": "voicebox", "base_url": "https://vb.org",
        "profile_id": "p1", "delete_generations": False,
    }
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "completed"}
    mocker.patch("slide_stream.providers.tts.requests.post", return_value=gen)
    mocker.patch(
        "slide_stream.providers.tts.requests.get",
        return_value=mocker.MagicMock(content=b"A"),
    )
    delete = mocker.patch("slide_stream.providers.tts.requests.delete")

    VoiceboxTTSProvider(config).synthesize("hi", str(tmp_path / "a.mp3"))
    delete.assert_not_called()


def test_voicebox_deletes_generation_even_if_download_fails(config, tmp_path, mocker):
    """A failed download must not strand rendered narration on the server."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config["providers"]["tts"] = {
        "provider": "voicebox", "base_url": "https://vb.org", "profile_id": "p1",
    }
    config["settings"]["strict"] = True
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "completed"}
    mocker.patch("slide_stream.providers.tts.requests.post", return_value=gen)
    mocker.patch(
        "slide_stream.providers.tts.requests.get",
        side_effect=RuntimeError("connection reset"),
    )
    delete = mocker.patch("slide_stream.providers.tts.requests.delete")

    assert VoiceboxTTSProvider(config).synthesize("hi", str(tmp_path / "a.mp3")) is None
    assert delete.call_args_list[0][0][0] == "https://vb.org/history/g1"


def test_voicebox_cleanup_failure_warns_but_does_not_raise(config, tmp_path, mocker):
    """close() runs in a finally; it must never mask the original error."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config = _voicebox_clone_config(config, tmp_path)
    provider = VoiceboxTTSProvider(config)
    provider._session_profile_id = "prof1"
    mocker.patch(
        "slide_stream.providers.tts.requests.delete",
        side_effect=RuntimeError("server down"),
    )
    provider.close()  # must not raise
    assert provider._session_profile_id is None


def test_voicebox_voice_sample_beats_profile_id(config, tmp_path, mocker):
    """An explicit voice_sample wins over a configured profile_id: the user's
    uploaded recording must not be silently ignored (serve use-case)."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config = _voicebox_clone_config(config, tmp_path)
    config["providers"]["tts"]["profile_id"] = "configured-profile"

    profile = mocker.MagicMock()
    profile.json.return_value = {"id": "ephemeral-prof"}
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "completed"}
    post = mocker.patch(
        "slide_stream.providers.tts.requests.post",
        side_effect=[profile, mocker.MagicMock(), gen],
    )
    mocker.patch(
        "slide_stream.providers.tts.requests.get",
        return_value=mocker.MagicMock(content=b"A"),
    )
    mocker.patch("slide_stream.providers.tts.requests.delete")

    provider = VoiceboxTTSProvider(config)
    out = tmp_path / "a.mp3"
    assert provider.synthesize("hi", str(out)) == str(out)

    # The clone was created and generation ran against it, not the
    # configured persistent profile.
    gen_calls = [c for c in post.call_args_list if c[0][0] == "https://vb.org/generate"]
    assert gen_calls[0][1]["json"]["profile_id"] == "ephemeral-prof"


# --- Voicebox polling: failed / timeout / unknown in-progress statuses --------


def _voicebox_profile_config(config, **overrides):
    config["providers"]["tts"] = {
        "provider": "voicebox", "base_url": "https://vb.org",
        "profile_id": "p1", "poll_interval": 0,
        **overrides,
    }
    return config


def test_voicebox_failed_generation_raises_instead_of_fetching_audio(
    config, tmp_path, mocker
):
    """A terminal 'failed' status must surface the server's error, not fetch
    /audio/{id} as if it had succeeded."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    _voicebox_profile_config(config)
    config["settings"]["strict"] = True
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "generating"}
    mocker.patch("slide_stream.providers.tts.requests.post", return_value=gen)
    failed = mocker.MagicMock()
    failed.json.return_value = {"status": "failed", "error": "engine crashed"}
    get = mocker.patch("slide_stream.providers.tts.requests.get", return_value=failed)
    delete = mocker.patch("slide_stream.providers.tts.requests.delete")
    mocker.patch("slide_stream.providers.tts.time.sleep")

    out = tmp_path / "a.mp3"
    assert VoiceboxTTSProvider(config).synthesize("hi", str(out)) is None
    # The audio endpoint was never hit...
    assert not any("/audio/" in c[0][0] for c in get.call_args_list)
    # ...but the failed generation row is still cleaned up (terminal state).
    assert delete.call_args_list[0][0][0] == "https://vb.org/history/g1"
    assert not out.exists()


def test_voicebox_poll_timeout_raises_and_keeps_generation(config, tmp_path, mocker):
    """When the deadline expires mid-render, raise TimeoutError and do NOT
    delete the generation — the server may still be writing it."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    _voicebox_profile_config(config)
    config["settings"]["strict"] = True
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "generating"}
    mocker.patch("slide_stream.providers.tts.requests.post", return_value=gen)
    get = mocker.patch("slide_stream.providers.tts.requests.get")
    delete = mocker.patch("slide_stream.providers.tts.requests.delete")
    # First call computes the deadline; the loop check then sees it expired.
    mocker.patch(
        "slide_stream.providers.tts.time.monotonic", side_effect=[0.0, 1000.0]
    )
    mocker.patch("slide_stream.providers.tts.time.sleep")

    assert VoiceboxTTSProvider(config).synthesize("hi", str(tmp_path / "a.mp3")) is None
    delete.assert_not_called()
    get.assert_not_called()


def test_voicebox_pending_and_queued_poll_like_generating(config, tmp_path, mocker):
    """Unknown in-progress statuses (pending/queued) keep polling until a
    terminal status arrives."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    _voicebox_profile_config(config)
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "pending"}
    mocker.patch("slide_stream.providers.tts.requests.post", return_value=gen)
    queued = mocker.MagicMock()
    queued.json.return_value = {"status": "queued"}
    done = mocker.MagicMock()
    done.json.return_value = {"status": "completed"}
    audio = mocker.MagicMock(content=b"OK")
    get = mocker.patch(
        "slide_stream.providers.tts.requests.get",
        side_effect=[queued, done, audio],
    )
    mocker.patch("slide_stream.providers.tts.requests.delete")
    mocker.patch("slide_stream.providers.tts.time.sleep")

    out = tmp_path / "a.mp3"
    assert VoiceboxTTSProvider(config).synthesize("hi", str(out)) == str(out)
    assert out.read_bytes() == b"OK"
    assert get.call_args_list[-1][0][0] == "https://vb.org/audio/g1"


# --- Strict mode for SwarmUI / Gemini image fallbacks -------------------------


def test_swarmui_strict_raises_instead_of_text_fallback(config, tmp_path, mocker):
    from slide_stream.providers.images import SwarmUIImageProvider

    config["settings"]["strict"] = True
    config["providers"]["images"] = {
        "provider": "swarmui", "base_url": "https://image.example.org",
    }
    mocker.patch(
        "slide_stream.providers.images.requests.post",
        side_effect=RuntimeError("server down"),
    )

    out = tmp_path / "img.png"
    with pytest.raises(StrictModeError):
        SwarmUIImageProvider(config).generate_image("query", str(out))
    assert not out.exists()


def test_gemini_strict_raises_instead_of_text_fallback(config, tmp_path):
    from slide_stream.providers.images import GeminiImageProvider

    config["settings"]["strict"] = True
    config["api_keys"] = {"gemini": "g-test"}
    out = tmp_path / "img.png"
    # google-genai is not installed in the test env: the ImportError path must
    # also refuse the text fallback under strict.
    with pytest.raises(StrictModeError):
        GeminiImageProvider(config).generate_image("query", str(out))
    assert not out.exists()


def test_factory_unusable_image_fallback_drops_to_text(config, monkeypatch):
    """A configured fallback that is itself unusable must not be returned;
    the factory drops to the always-available text provider instead."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    config["api_keys"] = {}
    config["providers"]["images"]["provider"] = "dalle3"
    config["providers"]["images"]["fallback"] = "pexels"

    provider = ProviderFactory.create_image_provider(config)
    assert isinstance(provider, TextImageProvider)


# --- gTTS fallback counter (read by cli.py to warn the user) ------------------


def test_fallback_count_starts_at_zero_on_all_tts_providers(config):
    for provider_class in ProviderFactory.TTS_PROVIDERS.values():
        assert provider_class(config).fallback_count == 0


def test_fallback_count_increments_on_each_gtts_fallback(config, tmp_path, mocker):
    config["api_keys"] = {"elevenlabs": "el-test"}
    fake_client = mocker.MagicMock()
    fake_client.text_to_speech.convert.side_effect = RuntimeError("api down")
    mocker.patch("elevenlabs.client.ElevenLabs", return_value=fake_client)

    def fake_save(filename):
        with open(filename, "wb") as f:
            f.write(b"fallback")

    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = fake_save
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    provider = ElevenLabsTTSProvider(config)
    assert provider.fallback_count == 0
    provider.synthesize("one", str(tmp_path / "1.mp3"))
    provider.synthesize("two", str(tmp_path / "2.mp3"))
    assert provider.fallback_count == 2


def test_fallback_count_not_incremented_in_strict_mode(config, tmp_path, monkeypatch):
    """Strict mode refuses the fallback, so nothing was synthesized with the
    wrong voice and the counter must stay at zero."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config["settings"]["strict"] = True

    provider = ElevenLabsTTSProvider(config)
    assert provider.synthesize("x", str(tmp_path / "a.mp3")) is None
    assert provider.fallback_count == 0


# --- Per-instance session state ------------------------------------------------


def test_tts_session_state_is_per_instance(config):
    """Session state must live on the instance (set in __init__), not on the
    class, so two runs in one process can never share uploads/profiles."""
    from slide_stream.providers.tts import ChatterboxTTSProvider, VoiceboxTTSProvider

    assert "_session_voice" in vars(ChatterboxTTSProvider(config))
    voicebox = VoiceboxTTSProvider(config)
    assert "_session_profile_id" in vars(voicebox)
    assert "_profile_ready" in vars(voicebox)
    assert "_engine" in vars(KokoroTTSProvider(config))


# --- Kokoro output format ------------------------------------------------------


def test_kokoro_forces_wav_format_despite_mp3_filename(
    config, tmp_path, mocker, monkeypatch
):
    """soundfile picks the encoder from the extension; the .mp3 pipeline name
    must not make it try MP3 encoding (needs lame-enabled libsndfile). The
    write is forced to WAV, at the caller's own path."""
    import sys
    from pathlib import Path

    written = {}
    fake_sf = mocker.MagicMock()

    def fake_write(fname, samples, rate, format=None):
        written["fname"] = fname
        written["format"] = format
        Path(fname).write_bytes(b"RIFFdata")

    fake_sf.write.side_effect = fake_write
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", mocker.MagicMock())

    provider = KokoroTTSProvider(config)
    provider._engine = mocker.MagicMock()
    provider._engine.create.return_value = ([0.0], 24000)

    out = tmp_path / "audio.mp3"
    assert provider.synthesize("hi", str(out)) == str(out)
    assert written["fname"] == str(out)
    assert written["format"] == "WAV"


# --- Temp WAV cleanup on ffmpeg failure -----------------------------------------


def test_prepare_sample_as_wav_cleans_temp_on_ffmpeg_failure(tmp_path, mocker):
    import subprocess as sp
    import tempfile as tf
    from pathlib import Path

    from slide_stream.providers.tts import _prepare_sample_as_wav

    sample = tmp_path / "memo.m4a"
    sample.write_bytes(b"m4a")

    created = {}
    real_mkstemp = tf.mkstemp

    def recording_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created["path"] = path
        return fd, path

    mocker.patch(
        "slide_stream.providers.tts.tempfile.mkstemp",
        side_effect=recording_mkstemp,
    )
    mocker.patch(
        "slide_stream.providers.tts.shutil.which", return_value="/usr/bin/ffmpeg"
    )
    mocker.patch(
        "slide_stream.providers.tts.subprocess.run",
        side_effect=sp.CalledProcessError(1, "ffmpeg"),
    )

    with pytest.raises(sp.CalledProcessError):
        _prepare_sample_as_wav(sample)
    assert not Path(created["path"]).exists()


# --- OpenAI TTS uses the non-deprecated writer ----------------------------------


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
