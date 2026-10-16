"""Tests for the Voicebox TTS provider: profiles, ephemeral clones, polling."""

import pytest

from slide_stream.config_loader import DEFAULT_CONFIG


@pytest.fixture
def config():
    """A minimal valid config (a fresh copy of the defaults)."""
    import copy

    return copy.deepcopy(DEFAULT_CONFIG)


# --- TextImageProvider (no network) ----------------------------------------


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
    generating.text = '{"status": "generating"}'
    done = mocker.MagicMock()
    done.text = '{"status": "completed"}'
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


def test_voicebox_status_body_parsing():
    from slide_stream.providers.tts import _parse_status_body

    # Plain JSON (older deployments).
    assert _parse_status_body('{"status": "completed"}') == {"status": "completed"}
    # SSE framing: single and multiple frames — last frame wins.
    sse_one = 'data: {"status": "failed", "error": "CUDA busy"}\n\n'
    assert _parse_status_body(sse_one) == {"status": "failed", "error": "CUDA busy"}
    sse_multi = (
        'data: {"status": "generating"}\n\n'
        'data: {"status": "completed", "duration": 1.5}\n\n'
    )
    assert _parse_status_body(sse_multi) == {"status": "completed", "duration": 1.5}
    # Unparseable junk (e.g. an SPA HTML catch-all) becomes an empty dict.
    assert _parse_status_body("<!doctype html><html></html>") == {}


def test_voicebox_polls_over_sse_status_stream(config, tmp_path, mocker):
    """Deployed Voicebox answers /status with text/event-stream frames."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config["providers"]["tts"] = {
        "provider": "voicebox", "base_url": "https://vb.org",
        "profile_id": "p1", "poll_interval": 0,
    }
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "generating"}
    mocker.patch("slide_stream.providers.tts.requests.post", return_value=gen)

    generating = mocker.MagicMock()
    generating.text = 'data: {"status": "generating"}\n\n'
    done = mocker.MagicMock()
    done.text = 'data: {"status": "completed"}\n\n'
    audio = mocker.MagicMock(content=b"OK")
    mocker.patch(
        "slide_stream.providers.tts.requests.get",
        side_effect=[generating, done, audio],
    )
    mocker.patch("slide_stream.providers.tts.time.sleep")

    out = tmp_path / "a.mp3"
    assert VoiceboxTTSProvider(config).synthesize("hi", str(out)) == str(out)
    assert out.read_bytes() == b"OK"


def test_voicebox_sse_failure_surfaces_error(config, tmp_path, mocker):
    """A terminal SSE 'failed' frame raises with the server's error detail."""
    from slide_stream.providers.tts import VoiceboxTTSProvider

    config["providers"]["tts"] = {
        "provider": "voicebox", "base_url": "https://vb.org", "profile_id": "p1",
    }
    config["settings"]["strict"] = True
    gen = mocker.MagicMock()
    gen.json.return_value = {"id": "g1", "status": "generating"}
    mocker.patch("slide_stream.providers.tts.requests.post", return_value=gen)

    failed = mocker.MagicMock()
    failed.text = 'data: {"status": "failed", "error": "CUDA error: busy"}\n\n'
    mocker.patch(
        "slide_stream.providers.tts.requests.get", return_value=failed
    )
    mocker.patch("slide_stream.providers.tts.time.sleep")

    result = VoiceboxTTSProvider(config).synthesize("hi", str(tmp_path / "a.mp3"))
    assert result is None  # falls back to gTTS after the RuntimeError


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
    queued.text = '{"status": "queued"}'
    done = mocker.MagicMock()
    done.text = '{"status": "completed"}'
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
