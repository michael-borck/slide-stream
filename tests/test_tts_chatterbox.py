"""Tests for the Chatterbox TTS provider and voice-sample handling."""

import pytest

from slide_stream.config_loader import DEFAULT_CONFIG


@pytest.fixture
def config():
    """A minimal valid config (a fresh copy of the defaults)."""
    import copy

    return copy.deepcopy(DEFAULT_CONFIG)


# --- TextImageProvider (no network) ----------------------------------------


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

    from slide_stream.ffmpeg_bin import ffmpeg_exe
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
        [str(ffmpeg_exe()), "-y", "-loglevel", "error", "-i", str(src_wav), str(sample)],
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
        "slide_stream.providers.tts.ffmpeg_exe", return_value="/usr/bin/ffmpeg"
    )
    mocker.patch(
        "slide_stream.providers.tts.subprocess.run",
        side_effect=sp.CalledProcessError(1, "ffmpeg"),
    )

    with pytest.raises(sp.CalledProcessError):
        _prepare_sample_as_wav(sample)
    assert not Path(created["path"]).exists()


# --- OpenAI TTS uses the non-deprecated writer ----------------------------------
