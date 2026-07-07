"""Tests for built-in character avatars, the static provider, and gTTS accents."""

import copy
import os

from typer.testing import CliRunner

from slide_stream.avatars import BUILTIN_AVATARS, avatar_names, resolve_avatar
from slide_stream.cli import app
from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.providers.factory import ProviderFactory


def _cfg():
    return copy.deepcopy(DEFAULT_CONFIG)


# --- registry ----------------------------------------------------------------


def test_builtin_avatar_files_exist():
    for name in avatar_names():
        path = resolve_avatar(name)
        assert path is not None and os.path.isfile(path), name
        assert path.endswith(".jpg")


def test_resolve_avatar_passthrough_and_case():
    assert resolve_avatar("/some/photo.png") == "/some/photo.png"
    assert resolve_avatar(None) is None
    # case-insensitive built-in name
    assert resolve_avatar("Teddy") == resolve_avatar("teddy")


def test_expected_characters_present():
    for name in ("teddy", "panda", "koala", "robot", "wizard", "owl"):
        assert name in BUILTIN_AVATARS


# --- static avatar provider --------------------------------------------------


def test_factory_registers_static():
    assert "static" in ProviderFactory.list_avatar_providers()


def test_static_provider_available_with_builtin(tmp_path):
    cfg = _cfg()
    cfg["providers"]["avatar"] = {"provider": "static", "source": "teddy"}
    provider = ProviderFactory.create_avatar_provider(cfg)
    assert provider.name == "static"
    assert provider.is_available() is True


def test_static_provider_unavailable_without_source():
    cfg = _cfg()
    cfg["providers"]["avatar"] = {"provider": "static"}
    from slide_stream.providers.avatar import StaticAvatarProvider

    assert StaticAvatarProvider(cfg).is_available() is False


def test_static_provider_renders_a_clip(tmp_path):
    """A real (tiny) encode: the built-in mascot becomes a corner-head clip."""
    cfg = _cfg()
    cfg["providers"]["avatar"] = {"provider": "static", "source": "teddy"}
    from slide_stream.providers.avatar import StaticAvatarProvider

    out = tmp_path / "head.mp4"
    result = StaticAvatarProvider(cfg).generate("unused.wav", str(out), 1)
    assert result == str(out)
    from moviepy import VideoFileClip

    with VideoFileClip(str(out)) as clip:
        assert clip.duration > 0


# --- gTTS accents ------------------------------------------------------------


def test_gtts_accent_maps_to_tld(mocker):
    from slide_stream.providers.tts import GTTSProvider

    cfg = _cfg()
    cfg["providers"]["tts"] = {"provider": "gtts", "accent": "australian"}

    def fake_save(filename):
        with open(filename, "wb") as f:
            f.write(b"a")

    fake = mocker.MagicMock()
    fake.save.side_effect = fake_save
    gtts = mocker.patch("gtts.gTTS", return_value=fake)

    GTTSProvider(cfg).synthesize("hello", "/tmp/out_accent.mp3")
    assert gtts.call_args.kwargs["tld"] == "com.au"


def test_gtts_default_tld_when_no_accent(mocker):
    from slide_stream.providers.tts import GTTSProvider

    cfg = _cfg()

    def fake_save(filename):
        with open(filename, "wb") as f:
            f.write(b"a")

    fake = mocker.MagicMock()
    fake.save.side_effect = fake_save
    gtts = mocker.patch("gtts.gTTS", return_value=fake)

    GTTSProvider(cfg).synthesize("hello", "/tmp/out_default.mp3")
    assert gtts.call_args.kwargs["tld"] == "com"


# --- CLI ---------------------------------------------------------------------


def test_avatars_command_lists_characters():
    result = CliRunner().invoke(app, ["avatars"])
    assert result.exit_code == 0
    assert "teddy" in result.output
    assert "Owl professor" in result.output


# --- puppet mouth-flap provider ----------------------------------------------


def _noise_wav(path, seconds=0.6):
    import random
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = bytearray()
        for _ in range(int(16000 * seconds)):
            frames += int(random.randint(-12000, 12000)).to_bytes(
                2, "little", signed=True
            )
        w.writeframes(bytes(frames))


def test_mouth_box_builtin_and_default():
    from slide_stream.avatars import DEFAULT_MOUTH_BOX, mouth_box

    assert mouth_box("teddy") != DEFAULT_MOUTH_BOX
    assert mouth_box("/some/custom.png") == DEFAULT_MOUTH_BOX
    assert mouth_box(None) == DEFAULT_MOUTH_BOX


def test_factory_registers_puppet():
    assert "puppet" in ProviderFactory.list_avatar_providers()


def test_puppet_available_with_builtin():
    cfg = _cfg()
    cfg["providers"]["avatar"] = {"provider": "puppet", "source": "teddy"}
    provider = ProviderFactory.create_avatar_provider(cfg)
    assert provider.name == "puppet"
    assert provider.is_available() is True


def test_puppet_renders_a_flap_clip(tmp_path):
    """Real render: the mascot + audio become a mouth-flap video."""
    from slide_stream.providers.avatar import PuppetAvatarProvider

    audio = tmp_path / "a.wav"
    _noise_wav(audio)
    cfg = _cfg()
    cfg["providers"]["avatar"] = {"provider": "puppet", "source": "teddy"}
    out = tmp_path / "head.mp4"
    result = PuppetAvatarProvider(cfg).generate(str(audio), str(out), 1)
    assert result == str(out)
    from moviepy import VideoFileClip

    with VideoFileClip(str(out)) as clip:
        assert clip.duration > 0


def test_puppet_custom_mouth_override(tmp_path):
    from slide_stream.providers.avatar import PuppetAvatarProvider

    cfg = _cfg()
    cfg["providers"]["avatar"] = {
        "provider": "puppet", "source": "teddy", "mouth": [0.4, 0.7, 0.2, 0.08],
    }
    box = PuppetAvatarProvider(cfg)._mouth_box("teddy")
    assert box == (0.4, 0.7, 0.2, 0.08)
