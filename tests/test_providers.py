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
