"""Tests for the local voices (Kokoro, gTTS) and the voice floor (ADR-0012)."""

import pytest

from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.providers.base import StrictModeError
from slide_stream.providers.factory import ProviderFactory
from slide_stream.providers.tts import (
    ElevenLabsTTSProvider,
    GTTSProvider,
    KokoroTTSProvider,
)


@pytest.fixture
def config():
    """A minimal valid config (a fresh copy of the defaults)."""
    import copy

    return copy.deepcopy(DEFAULT_CONFIG)


# --- TextImageProvider (no network) ----------------------------------------


def _delenv_voicebox(monkeypatch):
    monkeypatch.delenv("VOICEBOX_BASE_URL", raising=False)
    monkeypatch.delenv("VOICEBOX_TOKEN", raising=False)


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


def _kokoro_installed(monkeypatch, installed: bool):
    monkeypatch.delenv("SLIDE_STREAM_VOICE_FLOOR", raising=False)
    monkeypatch.setattr(
        "slide_stream.providers.tts.KokoroTTSProvider.is_available",
        lambda self: installed,
    )


def test_voice_floor_is_kokoro_when_installed(config, monkeypatch):
    from slide_stream.providers.tts import floor_tts_provider, voice_floor_name

    _kokoro_installed(monkeypatch, True)
    assert voice_floor_name(config) == "kokoro"
    floor = floor_tts_provider(config)
    assert isinstance(floor, KokoroTTSProvider)
    assert floor.floor is True


def test_voice_floor_is_gtts_without_kokoro(config, monkeypatch):
    from slide_stream.providers.tts import floor_tts_provider

    _kokoro_installed(monkeypatch, False)
    assert isinstance(floor_tts_provider(config), GTTSProvider)


def test_voice_floor_can_be_forced_to_gtts(config, monkeypatch):
    from slide_stream.providers.tts import voice_floor_name

    _kokoro_installed(monkeypatch, True)
    config["settings"]["voice_floor"] = "gtts"
    assert voice_floor_name(config) == "gtts"


def test_unconfigured_default_uses_kokoro_floor(config, monkeypatch):
    """A bare install with Kokoro gets the offline voice, not gTTS."""
    _delenv_voicebox(monkeypatch)
    _kokoro_installed(monkeypatch, True)
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, KokoroTTSProvider)


def test_auto_tts_provider_is_the_floor(config, monkeypatch):
    config["providers"]["tts"]["provider"] = "auto"
    _kokoro_installed(monkeypatch, True)
    assert isinstance(ProviderFactory.create_tts_provider(config), KokoroTTSProvider)
    _kokoro_installed(monkeypatch, False)
    assert isinstance(ProviderFactory.create_tts_provider(config), GTTSProvider)


def test_broken_provider_falls_back_to_kokoro_floor(config, monkeypatch):
    config["providers"]["tts"]["provider"] = "elevenlabs"  # no key -> unusable
    _kokoro_installed(monkeypatch, True)
    assert isinstance(ProviderFactory.create_tts_provider(config), KokoroTTSProvider)


def test_per_slide_failure_uses_floor_once_per_run(config, monkeypatch, mocker):
    """A failing server voice falls to the floor; one floor engine is reused
    across slides so Kokoro loads its model once, not per slide."""
    from slide_stream.providers.tts import ElevenLabsTTSProvider

    _kokoro_installed(monkeypatch, True)
    synth = mocker.patch(
        "slide_stream.providers.tts.KokoroTTSProvider.synthesize",
        return_value="out.wav",
    )
    provider = ElevenLabsTTSProvider(config)
    assert provider._fallback_to_floor("one", "a.wav") == "out.wav"
    first_floor = provider._floor_provider
    assert provider._fallback_to_floor("two", "b.wav") == "out.wav"
    assert provider._floor_provider is first_floor
    assert provider.fallback_count == 2
    assert synth.call_count == 2


def test_per_slide_floor_respects_strict(config, monkeypatch):
    from slide_stream.providers.tts import ElevenLabsTTSProvider

    _kokoro_installed(monkeypatch, True)
    config["settings"]["strict"] = True
    provider = ElevenLabsTTSProvider(config)
    assert provider._fallback_to_floor("one", "a.wav") is None
    assert provider.fallback_count == 0


def test_kokoro_floor_ignores_another_providers_voice(config, mocker):
    """As the floor, Kokoro must not try an ElevenLabs/Voicebox voice name."""
    config["providers"]["tts"].update(provider="elevenlabs", voice="rachel")
    voices = _kokoro_voice_used(config, mocker)
    assert voices == ["af_sarah"]
    config["providers"]["tts"]["kokoro_voice"] = "bf_emma"
    assert _kokoro_voice_used(config, mocker) == ["bf_emma"]
    config["providers"]["tts"].update(provider="kokoro", voice="am_adam")
    assert _kokoro_voice_used(config, mocker) == ["am_adam"]


def _kokoro_voice_used(config, mocker):
    """Run Kokoro.synthesize with the engine faked; return the voices used."""
    import sys
    import types

    used = []

    class FakeKokoro:
        def __init__(self, *args, **kwargs):
            pass

        def create(self, text, voice, speed, lang):
            used.append(voice)
            return [0.0], 24000

    fake_module = types.SimpleNamespace(Kokoro=FakeKokoro)
    fake_sf = types.SimpleNamespace(write=lambda *a, **k: None)
    mocker.patch.dict(sys.modules, {"kokoro_onnx": fake_module, "soundfile": fake_sf})
    provider = KokoroTTSProvider(config, floor=True)
    mocker.patch.object(provider, "_model_files", return_value=("m", "v"))
    mocker.patch.object(provider, "_espeak_config", return_value=None)
    provider.synthesize("hello", "out.wav")
    return used


def test_kokoro_floor_still_reaches_gtts_in_strict(config, mocker):
    """Strict guards the chosen voice. The floor stands in for a voice that
    was never chosen, so Kokoro-as-floor may still fall to gTTS."""
    config["settings"]["strict"] = True
    gtts = mocker.patch(
        "slide_stream.providers.tts.GTTSProvider.synthesize", return_value="g.mp3"
    )
    floor = KokoroTTSProvider(config, floor=True)
    assert floor._fallback_to_gtts("hi", "g.mp3") == "g.mp3"
    gtts.assert_called_once()
    chosen = KokoroTTSProvider(config)  # Kokoro picked explicitly
    assert chosen._fallback_to_gtts("hi", "g.mp3") is None
