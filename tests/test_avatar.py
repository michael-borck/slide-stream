"""Tests for avatar providers and talking-head compositing (no GPU needed)."""

import copy
import wave

import pytest
from moviepy import ColorClip, VideoFileClip
from typer.testing import CliRunner

from slide_stream.cli import app
from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.media import create_video_fragment
from slide_stream.providers.avatar import NoneAvatarProvider, PrecomputedAvatarProvider
from slide_stream.providers.base import StrictModeError
from slide_stream.providers.factory import ProviderFactory
from slide_stream.providers.images import TextImageProvider


@pytest.fixture
def config():
    """A small/fast video config for encoding tests."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["settings"]["video"]["resolution"] = [320, 180]
    cfg["settings"]["video"]["fps"] = 8
    cfg["settings"]["video"]["default_slide_duration"] = 1.0
    return cfg


@pytest.fixture
def slide_image(config, tmp_path):
    out = tmp_path / "slide.png"
    TextImageProvider(config).generate_image(
        "q", str(out), slide={"title": "Avatar test", "content": ["point"]}
    )
    return out


def make_head_video(path, duration):
    """Write a tiny synthetic 'head' clip."""
    clip = ColorClip(size=(64, 48), color=(200, 30, 30), duration=duration)
    clip.write_videofile(str(path), fps=8, codec="libx264", logger=None)
    clip.close()
    return path


# --- Providers ---------------------------------------------------------------


def test_none_provider_is_available_but_generates_nothing(config):
    provider = NoneAvatarProvider(config)
    assert provider.name == "none"
    assert provider.is_available() is True
    assert provider.generate("a.mp3", "h.mp4", 1) is None


def test_precomputed_unavailable_without_assets_dir(config, tmp_path):
    provider = PrecomputedAvatarProvider(config)
    assert provider.is_available() is False

    config["providers"]["avatar"] = {"assets_dir": str(tmp_path / "missing")}
    assert PrecomputedAvatarProvider(config).is_available() is False

    config["providers"]["avatar"] = {"assets_dir": str(tmp_path)}
    assert PrecomputedAvatarProvider(config).is_available() is True


def test_precomputed_maps_slide_number_to_clip(config, tmp_path):
    (tmp_path / "head_2.mp4").write_bytes(b"fake")
    config["providers"]["avatar"] = {"assets_dir": str(tmp_path)}
    provider = PrecomputedAvatarProvider(config)

    assert provider.generate("a.mp3", "out.mp4", 2) == str(tmp_path / "head_2.mp4")
    # Slides without a matching clip render headless.
    assert provider.generate("a.mp3", "out.mp4", 3) is None


# --- Compositing (real encodes, small and fast) ------------------------------


def test_fragment_with_short_head_freezes_last_frame(config, slide_image, tmp_path):
    """Head shorter than the fragment: last frame is held to fill it."""
    head = make_head_video(tmp_path / "head.mp4", duration=0.5)
    out = tmp_path / "fragment.mp4"

    result = create_video_fragment(
        str(slide_image), None, str(out), config, head_video=str(head)
    )

    assert result == str(out)
    with VideoFileClip(str(out)) as clip:
        assert (clip.w, clip.h) == (320, 180)
        assert clip.duration == pytest.approx(1.0, abs=0.2)


def test_fragment_with_long_head_is_trimmed(config, slide_image, tmp_path):
    """Head longer than the fragment: trimmed to the fragment duration."""
    head = make_head_video(tmp_path / "head.mp4", duration=2.5)
    out = tmp_path / "fragment.mp4"

    result = create_video_fragment(
        str(slide_image), None, str(out), config, head_video=str(head)
    )

    assert result == str(out)
    with VideoFileClip(str(out)) as clip:
        assert clip.duration == pytest.approx(1.0, abs=0.2)


def test_fragment_head_overlay_changes_pixels(config, slide_image, tmp_path):
    """The composited head is actually visible in the configured corner."""
    head = make_head_video(tmp_path / "head.mp4", duration=1.0)
    plain = tmp_path / "plain.mp4"
    with_head = tmp_path / "with_head.mp4"

    create_video_fragment(str(slide_image), None, str(plain), config)
    create_video_fragment(
        str(slide_image), None, str(with_head), config, head_video=str(head)
    )

    with VideoFileClip(str(plain)) as a, VideoFileClip(str(with_head)) as b:
        frame_a = a.get_frame(0.4)
        frame_b = b.get_frame(0.4)
    assert frame_a is not None and frame_b is not None
    # Bottom-right corner (default position) must differ; top-left must not.
    h, w = frame_a.shape[:2]
    corner_a = frame_a[int(h * 0.75) :, int(w * 0.75) :]
    corner_b = frame_b[int(h * 0.75) :, int(w * 0.75) :]
    assert (corner_a != corner_b).any()
    assert (frame_a[: int(h * 0.25), : int(w * 0.25)] == frame_b[: int(h * 0.25), : int(w * 0.25)]).all()


def test_fragment_without_head_unchanged(config, slide_image, tmp_path):
    """Omitting head_video keeps the existing behaviour."""
    out = tmp_path / "fragment.mp4"
    result = create_video_fragment(str(slide_image), None, str(out), config)
    assert result == str(out)
    assert out.exists()


# --- Factory -----------------------------------------------------------------


def test_factory_default_avatar_is_none(config):
    provider = ProviderFactory.create_avatar_provider(config)
    assert isinstance(provider, NoneAvatarProvider)


def test_factory_selects_precomputed_when_assets_exist(config, tmp_path):
    config["providers"]["avatar"] = {"provider": "precomputed", "assets_dir": str(tmp_path)}
    provider = ProviderFactory.create_avatar_provider(config)
    assert isinstance(provider, PrecomputedAvatarProvider)


def test_factory_falls_back_to_none_when_assets_missing(config, tmp_path):
    config["providers"]["avatar"] = {
        "provider": "precomputed",
        "assets_dir": str(tmp_path / "missing"),
    }
    provider = ProviderFactory.create_avatar_provider(config)
    assert isinstance(provider, NoneAvatarProvider)


def test_factory_strict_raises_for_unusable_avatar(config, tmp_path):
    config["settings"]["strict"] = True
    config["providers"]["avatar"] = {
        "provider": "precomputed",
        "assets_dir": str(tmp_path / "missing"),
    }
    with pytest.raises(StrictModeError):
        ProviderFactory.create_avatar_provider(config)


def test_factory_strict_raises_for_unknown_avatar(config):
    config["settings"]["strict"] = True
    config["providers"]["avatar"] = {"provider": "hologram"}
    with pytest.raises(StrictModeError):
        ProviderFactory.create_avatar_provider(config)


def test_availability_report_includes_avatar(config):
    availability = ProviderFactory.check_provider_availability(config)
    assert availability["avatar"]["none"] is True
    assert availability["avatar"]["precomputed"] is False


# --- CLI end-to-end ----------------------------------------------------------


def write_silent_wav(filename, seconds=0.4):
    """A real, decodable audio file (WAV content behind any extension)."""
    with wave.open(str(filename), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * int(22050 * seconds))


@pytest.fixture
def fast_video_yaml():
    return (
        "settings:\n"
        "  video:\n"
        "    resolution: [320, 180]\n"
        "    fps: 8\n"
        "    default_slide_duration: 1.0\n"
        "    slide_duration_padding: 0.2\n"
    )


def test_cli_create_renders_video_with_precomputed_avatar(
    tmp_path, mocker, monkeypatch, fast_video_yaml
):
    """Full pipeline: markdown -> narrated video with head overlays."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    md = tmp_path / "deck.md"
    md.write_text("# One\n\n- a\n\n# Two\n\n- b\n")
    heads = tmp_path / "heads"
    heads.mkdir()
    make_head_video(heads / "head_1.mp4", 0.6)
    make_head_video(heads / "head_2.mp4", 0.6)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "providers:\n"
        "  avatar:\n"
        "    provider: precomputed\n"
        f"    assets_dir: {heads}\n" + fast_video_yaml
    )

    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = write_silent_wav
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app, ["create", str(md), str(out), "--config", str(cfg), "--strict"]
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    with VideoFileClip(str(out)) as clip:
        assert (clip.w, clip.h) == (320, 180)
        # Two slides, each ~0.4s audio + 0.2s padding.
        assert clip.duration == pytest.approx(1.2, abs=0.4)


def test_cli_no_avatar_flag_disables_configured_avatar(
    tmp_path, mocker, monkeypatch, fast_video_yaml
):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    md = tmp_path / "deck.md"
    md.write_text("# Only\n\n- a\n")
    heads = tmp_path / "heads"
    heads.mkdir()
    make_head_video(heads / "head_1.mp4", 0.6)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "providers:\n"
        "  avatar:\n"
        "    provider: precomputed\n"
        f"    assets_dir: {heads}\n" + fast_video_yaml
    )

    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = write_silent_wav
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    result = runner.invoke(
        app,
        ["create", str(md), str(tmp_path / "out.mp4"), "--config", str(cfg), "--no-avatar"],
    )

    assert result.exit_code == 0, result.output
    assert "Avatar Provider" not in result.output


def test_cli_avatar_flag_requires_configured_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    md = tmp_path / "deck.md"
    md.write_text("# Only\n\n- a\n")

    result = runner.invoke(app, ["create", str(md), "out.mp4", "--avatar"])

    assert result.exit_code == 1


# --- D-ID avatar provider (network mocked) -----------------------------------


def _did_config(config, **avatar):
    config["providers"]["avatar"] = {"provider": "d-id", **avatar}
    return config


def test_did_availability_needs_key_and_source(config, monkeypatch):
    from slide_stream.providers.avatar import DIDAvatarProvider

    monkeypatch.delenv("DID_API_KEY", raising=False)
    _did_config(config)
    assert DIDAvatarProvider(config).is_available() is False

    _did_config(config, api_key="k")  # key but no source
    assert DIDAvatarProvider(config).is_available() is False

    _did_config(config, api_key="k", source_image="/tmp/face.jpg")
    assert DIDAvatarProvider(config).is_available() is True


def test_did_generate_full_flow(config, tmp_path, mocker):
    """image upload -> audio upload -> create talk -> poll -> download."""
    from slide_stream.providers.avatar import DIDAvatarProvider

    face = tmp_path / "face.jpg"
    face.write_bytes(b"jpegbytes")
    audio = tmp_path / "slide_1.mp3"
    audio.write_bytes(b"audiobytes")
    _did_config(config, api_key="k", source_image=str(face), poll_interval=0)

    img_up = mocker.MagicMock()
    img_up.json.return_value = {"url": "https://d-id/img.jpg"}
    aud_up = mocker.MagicMock()
    aud_up.json.return_value = {"url": "https://d-id/aud.mp3"}
    create = mocker.MagicMock()
    create.json.return_value = {"id": "talk-1"}
    post = mocker.patch(
        "slide_stream.providers.avatar.requests.post",
        side_effect=[img_up, aud_up, create],
    )

    pending = mocker.MagicMock()
    pending.json.return_value = {"status": "started"}
    done = mocker.MagicMock()
    done.json.return_value = {"status": "done", "result_url": "https://d-id/out.mp4"}
    clip = mocker.MagicMock(content=b"MP4DATA")
    get = mocker.patch(
        "slide_stream.providers.avatar.requests.get",
        side_effect=[pending, done, clip],
    )
    mocker.patch("slide_stream.providers.avatar.time.sleep")

    out = tmp_path / "head_1.mp4"
    result = DIDAvatarProvider(config).generate(str(audio), str(out), 1)

    assert result == str(out)
    assert out.read_bytes() == b"MP4DATA"
    # Talk was created with the uploaded image + audio URLs.
    talk_body = post.call_args_list[2][1]["json"]
    assert talk_body["source_url"] == "https://d-id/img.jpg"
    assert talk_body["script"] == {"type": "audio", "audio_url": "https://d-id/aud.mp3"}
    assert get.call_args_list[-1][0][0] == "https://d-id/out.mp4"


def test_did_source_image_uploaded_once_per_run(config, tmp_path, mocker):
    from slide_stream.providers.avatar import DIDAvatarProvider

    face = tmp_path / "face.jpg"
    face.write_bytes(b"jpg")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"a")
    _did_config(config, api_key="k", source_image=str(face), poll_interval=0)

    def fake_post(url, **kwargs):
        r = mocker.MagicMock()
        if url.endswith("/images"):
            r.json.return_value = {"url": "https://d-id/img.jpg"}
        elif url.endswith("/audios"):
            r.json.return_value = {"url": "https://d-id/aud.mp3"}
        else:
            r.json.return_value = {"id": "t"}
        return r

    post = mocker.patch("slide_stream.providers.avatar.requests.post", side_effect=fake_post)
    done = mocker.MagicMock()
    done.json.return_value = {"status": "done", "result_url": "https://d-id/out.mp4"}
    mocker.patch("slide_stream.providers.avatar.requests.get",
                 return_value=mocker.MagicMock(content=b"x", **{"json.return_value": {"status": "done", "result_url": "u"}}))
    mocker.patch("slide_stream.providers.avatar.time.sleep")

    provider = DIDAvatarProvider(config)
    provider.generate(str(audio), str(tmp_path / "h1.mp4"), 1)
    provider.generate(str(audio), str(tmp_path / "h2.mp4"), 2)

    image_uploads = [c for c in post.call_args_list if c[0][0].endswith("/images")]
    assert len(image_uploads) == 1  # uploaded once, reused for slide 2


def test_did_error_returns_none(config, tmp_path, mocker):
    from slide_stream.providers.avatar import DIDAvatarProvider

    face = tmp_path / "face.jpg"
    face.write_bytes(b"jpg")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"a")
    _did_config(config, api_key="k", source_image=str(face))
    mocker.patch(
        "slide_stream.providers.avatar.requests.post",
        side_effect=RuntimeError("api down"),
    )
    assert DIDAvatarProvider(config).generate(str(audio), str(tmp_path / "o.mp4"), 1) is None


def test_factory_registers_did_and_strict(config, tmp_path):
    from slide_stream.providers.base import StrictModeError

    config["providers"]["avatar"] = {"provider": "d-id", "source_image": str(tmp_path / "x.jpg")}
    config["settings"]["strict"] = True  # no key -> unusable -> strict raises
    import pytest as _pytest
    with _pytest.raises(StrictModeError):
        ProviderFactory.create_avatar_provider(config)


# --- SadTalker avatar provider (ComfyUI, network mocked) ---------------------


def _sadtalker_config(config, **avatar):
    config["providers"]["avatar"] = {"provider": "sadtalker", **avatar}
    return config


def test_sadtalker_availability_needs_url_and_source(config, monkeypatch):
    from slide_stream.providers.avatar import SadTalkerAvatarProvider

    monkeypatch.delenv("COMFYUI_BASE_URL", raising=False)
    _sadtalker_config(config)
    assert SadTalkerAvatarProvider(config).is_available() is False

    _sadtalker_config(config, base_url="https://comfy.example.org")
    assert SadTalkerAvatarProvider(config).is_available() is False

    _sadtalker_config(config, base_url="https://comfy.example.org", source_image="/tmp/f.png")
    assert SadTalkerAvatarProvider(config).is_available() is True


def test_sadtalker_generate_flow(config, tmp_path, mocker, monkeypatch):
    from slide_stream.providers.avatar import SadTalkerAvatarProvider

    face = tmp_path / "face.png"
    face.write_bytes(b"png")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"wav")
    _sadtalker_config(
        config, base_url="https://comfy.example.org", source_image=str(face),
        preprocess="full", poll_interval=0,
    )
    # No ffmpeg in the test -> audio uploaded as-is.
    monkeypatch.setattr("slide_stream.providers.avatar.shutil.which", lambda _: None)

    img_up = mocker.MagicMock()
    img_up.json.return_value = {"name": "face.png"}
    aud_up = mocker.MagicMock()
    aud_up.json.return_value = {"name": "a.wav"}
    submit = mocker.MagicMock()
    submit.json.return_value = {"prompt_id": "pid-1"}
    post = mocker.patch(
        "slide_stream.providers.avatar.requests.post",
        side_effect=[img_up, aud_up, submit],
    )

    pending = mocker.MagicMock()
    pending.json.return_value = {"pid-1": {"status": {"status_str": "pending"}}}
    done = mocker.MagicMock()
    done.json.return_value = {
        "pid-1": {
            "status": {"status_str": "success"},
            "outputs": {"4": {"show_video_path": ["/basedir/output/out.mp4"]}},
        }
    }
    clip = mocker.MagicMock(content=b"MP4")
    get = mocker.patch(
        "slide_stream.providers.avatar.requests.get",
        side_effect=[pending, done, clip],
    )
    mocker.patch("slide_stream.providers.avatar.time.sleep")

    out = tmp_path / "head_1.mp4"
    result = SadTalkerAvatarProvider(config).generate(str(audio), str(out), 1)

    assert result == str(out)
    assert out.read_bytes() == b"MP4"
    # Workflow submitted with the uploaded filenames + configured preprocess.
    wf = post.call_args_list[2][1]["json"]["prompt"]
    assert wf["1"]["inputs"]["image"] == "face.png"
    assert wf["2"]["inputs"]["audio"] == "a.wav"
    assert wf["3"]["inputs"]["preprocess"] == "full"
    # Result downloaded via /view by basename.
    assert get.call_args_list[-1][1]["params"] == {"filename": "out.mp4", "type": "output"}


def test_sadtalker_workflow_error_returns_none(config, tmp_path, mocker, monkeypatch):
    from slide_stream.providers.avatar import SadTalkerAvatarProvider

    face = tmp_path / "f.png"
    face.write_bytes(b"p")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"w")
    _sadtalker_config(config, base_url="https://c.org", source_image=str(face), poll_interval=0)
    monkeypatch.setattr("slide_stream.providers.avatar.shutil.which", lambda _: None)

    img_up = mocker.MagicMock()
    img_up.json.return_value = {"name": "f.png"}
    aud_up = mocker.MagicMock()
    aud_up.json.return_value = {"name": "a.wav"}
    submit = mocker.MagicMock()
    submit.json.return_value = {"prompt_id": "p"}
    mocker.patch("slide_stream.providers.avatar.requests.post",
                 side_effect=[img_up, aud_up, submit])
    err = mocker.MagicMock()
    err.json.return_value = {"p": {"status": {"status_str": "error"}}}
    mocker.patch("slide_stream.providers.avatar.requests.get", return_value=err)
    mocker.patch("slide_stream.providers.avatar.time.sleep")

    result = SadTalkerAvatarProvider(config).generate(str(audio), str(tmp_path / "o.mp4"), 1)
    assert result is None


def test_find_output_path():
    from slide_stream.providers.avatar import _find_output_path

    # image kind -> ShowVideo node's show_video_path (no subfolder concept)
    assert _find_output_path(
        "image", {"4": {"show_video_path": ["/out/x.mp4"]}}
    ) == ("/out/x.mp4", "")
    assert _find_output_path("image", {"9": {"images": []}}) is None
    # video kind -> VHS_VideoCombine's gifs filename (+ subfolder when set)
    assert _find_output_path(
        "video", {"4": {"gifs": [{"filename": "w.mp4", "fullpath": "/out/w.mp4"}]}}
    ) == ("w.mp4", "")
    assert _find_output_path(
        "video",
        {"4": {"gifs": [{"filename": "w.mp4", "subfolder": "wav2lip",
                         "fullpath": "/out/wav2lip/w.mp4"}]}},
    ) == ("w.mp4", "wav2lip")


def test_factory_registers_sadtalker():
    assert "sadtalker" in ProviderFactory.list_avatar_providers()


# --- Wav2Lip (video) + ComfyUI auto-router -----------------------------------


def test_source_kind_detects_video_vs_image():
    from slide_stream.providers.avatar import _source_kind

    assert _source_kind("/x/me.mp4") == "video"
    assert _source_kind("/x/me.MOV") == "video"
    assert _source_kind("/x/me.png") == "image"
    assert _source_kind("/x/me.jpg") == "image"
    assert _source_kind("bareword") == "image"  # default


def test_wav2lip_uses_video_workflow(config, tmp_path, mocker, monkeypatch):
    from slide_stream.providers.avatar import Wav2LipAvatarProvider

    vid = tmp_path / "idle.mp4"
    vid.write_bytes(b"vid")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"w")
    config["providers"]["avatar"] = {
        "provider": "wav2lip", "base_url": "https://c.org",
        "source_video": str(vid), "poll_interval": 0,
    }
    monkeypatch.setattr("slide_stream.providers.avatar.shutil.which", lambda _: None)

    vid_up = mocker.MagicMock()
    vid_up.json.return_value = {"name": "idle.mp4"}
    aud_up = mocker.MagicMock()
    aud_up.json.return_value = {"name": "a.wav"}
    submit = mocker.MagicMock()
    submit.json.return_value = {"prompt_id": "p"}
    post = mocker.patch(
        "slide_stream.providers.avatar.requests.post",
        side_effect=[vid_up, aud_up, submit],
    )
    done = mocker.MagicMock()
    done.json.return_value = {
        "p": {
            "status": {"status_str": "success"},
            "outputs": {"4": {"gifs": [{"filename": "wav2lip_out.mp4",
                                        "fullpath": "/out/wav2lip_out.mp4"}]}},
        }
    }
    clip = mocker.MagicMock(content=b"MP4V")
    get = mocker.patch(
        "slide_stream.providers.avatar.requests.get", side_effect=[done, clip]
    )
    mocker.patch("slide_stream.providers.avatar.time.sleep")

    out = tmp_path / "head.mp4"
    result = Wav2LipAvatarProvider(config).generate(str(audio), str(out), 1)

    assert result == str(out)
    assert out.read_bytes() == b"MP4V"
    # It used the Wav2Lip (VHS) workflow, not SadTalker.
    wf = post.call_args_list[2][1]["json"]["prompt"]
    assert wf["1"]["class_type"] == "VHS_LoadVideo"
    assert wf["1"]["inputs"]["video"] == "idle.mp4"
    assert wf["3"]["class_type"] == "Wav2Lip"
    # Downloaded the gifs output by filename.
    assert get.call_args_list[-1][1]["params"]["filename"] == "wav2lip_out.mp4"


def test_comfyui_view_download_passes_subfolder(config, tmp_path, mocker, monkeypatch):
    """When the history entry places the output in a subfolder, the /view
    download must pass it through, or ComfyUI 404s the bare filename."""
    from slide_stream.providers.avatar import Wav2LipAvatarProvider

    vid = tmp_path / "idle.mp4"
    vid.write_bytes(b"vid")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"w")
    config["providers"]["avatar"] = {
        "provider": "wav2lip", "base_url": "https://c.org",
        "source_video": str(vid), "poll_interval": 0,
    }
    monkeypatch.setattr("slide_stream.providers.avatar.shutil.which", lambda _: None)

    vid_up = mocker.MagicMock()
    vid_up.json.return_value = {"name": "idle.mp4"}
    aud_up = mocker.MagicMock()
    aud_up.json.return_value = {"name": "a.wav"}
    submit = mocker.MagicMock()
    submit.json.return_value = {"prompt_id": "p"}
    mocker.patch(
        "slide_stream.providers.avatar.requests.post",
        side_effect=[vid_up, aud_up, submit],
    )
    done = mocker.MagicMock()
    done.json.return_value = {
        "p": {
            "status": {"status_str": "success"},
            "outputs": {"4": {"gifs": [{"filename": "out.mp4",
                                        "subfolder": "wav2lip",
                                        "fullpath": "/out/wav2lip/out.mp4"}]}},
        }
    }
    clip = mocker.MagicMock(content=b"MP4V")
    get = mocker.patch(
        "slide_stream.providers.avatar.requests.get", side_effect=[done, clip]
    )
    mocker.patch("slide_stream.providers.avatar.time.sleep")

    out = tmp_path / "head.mp4"
    assert Wav2LipAvatarProvider(config).generate(str(audio), str(out), 1) == str(out)
    assert get.call_args_list[-1][1]["params"] == {
        "filename": "out.mp4", "type": "output", "subfolder": "wav2lip",
    }


def test_comfyui_router_picks_engine_by_source(config, tmp_path):
    from slide_stream.providers.avatar import ComfyUIAvatarProvider

    # A photo source -> image kind (SadTalker).
    config["providers"]["avatar"] = {
        "provider": "comfyui", "base_url": "https://c.org", "source": str(tmp_path / "me.png"),
    }
    assert ComfyUIAvatarProvider(config)._kind() == "image"
    # A video source -> video kind (Wav2Lip).
    config["providers"]["avatar"]["source"] = str(tmp_path / "me.mp4")
    assert ComfyUIAvatarProvider(config)._kind() == "video"


def test_comfyui_router_builds_matching_workflow(config, tmp_path):
    from slide_stream.providers.avatar import ComfyUIAvatarProvider

    config["providers"]["avatar"] = {"provider": "comfyui", "base_url": "https://c.org"}
    p = ComfyUIAvatarProvider(config)
    img_wf = p._build_workflow("image", "me.png", "a.wav")
    assert img_wf["3"]["class_type"] == "SadTalker"
    vid_wf = p._build_workflow("video", "me.mp4", "a.wav")
    assert vid_wf["3"]["class_type"] == "Wav2Lip"
    assert vid_wf["4"]["inputs"]["pingpong"] is True


def test_factory_registers_wav2lip_and_comfyui():
    names = ProviderFactory.list_avatar_providers()
    assert "wav2lip" in names
    assert "comfyui" in names
