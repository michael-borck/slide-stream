"""Tests for the presenter overlay: providers without a server, compositing, factory, CLI options."""

import copy
import shutil
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


def test_fragment_with_short_head_loops_to_fill(config, slide_image, tmp_path):
    """Head much shorter than the fragment: looped so it keeps animating."""
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
        # Two slides, each with narration + padding (segment gaps add ~0.3s
        # per slide under ADR-0018).
        assert 1.0 <= clip.duration <= 5.0


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


def test_resolve_avatar_slides_variants():
    from slide_stream.cli import _resolve_avatar_slides as r

    assert r("all", 4) == {1, 2, 3, 4}
    assert r(None, 3) == {1, 2, 3}
    assert r("none", 4) == set()
    assert r("first", 4) == {1}
    assert r("last", 4) == {4}
    assert r("first,last", 4) == {1, 4}
    assert r("1,3,5", 4) == {1, 3}          # out-of-range dropped
    assert r("every:2", 5) == {1, 3, 5}
    assert r("every 3", 7) == {1, 4, 7}
    assert r([2, 4], 5) == {2, 4}
    assert r("all", 0) == set()             # no slides


def test_concatenate_with_transition_shortens_total():
    from slide_stream.media import concatenate_with_transition

    clips = [ColorClip((32, 24), color=(i * 40, 0, 0), duration=1.0) for i in range(3)]
    hard = concatenate_with_transition(clips, 0)
    assert abs(hard.duration - 3.0) < 0.05        # hard cut = sum of durations
    hard.close()

    clips2 = [ColorClip((32, 24), color=(i * 40, 0, 0), duration=1.0) for i in range(3)]
    faded = concatenate_with_transition(clips2, 0.4)
    # Two 0.4s overlaps remove ~0.8s from the 3.0s total.
    assert faded.duration < 2.9
    faded.close()


def test_cli_avatar_slides_limits_which_slides_get_a_head(
    tmp_path, mocker, monkeypatch, fast_video_yaml
):
    """--avatar-slides first,last renders a head only on those slides."""
    monkeypatch.chdir(tmp_path)
    md = tmp_path / "deck.md"
    md.write_text("# A\n\n- a\n\n# B\n\n- b\n\n# C\n\n- c\n")
    head = tmp_path / "head.mp4"
    make_head_video(head, 0.6)

    calls: list[int] = []
    fake_av = mocker.MagicMock()
    fake_av.name = "precomputed"

    def gen(audio, out, slide_num):
        calls.append(slide_num)
        shutil.copyfile(head, out)
        return out

    fake_av.generate.side_effect = gen
    mocker.patch(
        "slide_stream.cli.ProviderFactory.create_avatar_provider", return_value=fake_av
    )
    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = write_silent_wav
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"providers:\n  avatar:\n    provider: precomputed\n    assets_dir: {tmp_path}\n"
        + fast_video_yaml
    )
    out = tmp_path / "o.mp4"
    result = CliRunner().invoke(
        app, ["create", str(md), str(out), "--config", str(cfg),
              "--avatar-slides", "first,last"]
    )
    assert result.exit_code == 0, result.output
    assert sorted(calls) == [1, 3]  # slide 2 got narration but no head


def test_cli_reuse_avatar_renders_clip_once(
    tmp_path, mocker, monkeypatch, fast_video_yaml
):
    """--reuse-avatar renders one clip and reuses it across every slide."""
    monkeypatch.chdir(tmp_path)
    md = tmp_path / "deck.md"
    md.write_text("# A\n\n- a\n\n# B\n\n- b\n\n# C\n\n- c\n")
    head = tmp_path / "head.mp4"
    make_head_video(head, 0.6)

    calls: list[int] = []
    fake_av = mocker.MagicMock()
    fake_av.name = "wan-s2v"

    def gen(audio, out, slide_num):
        calls.append(slide_num)
        shutil.copyfile(head, out)
        return out

    fake_av.generate.side_effect = gen
    mocker.patch(
        "slide_stream.cli.ProviderFactory.create_avatar_provider", return_value=fake_av
    )
    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = write_silent_wav
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"providers:\n  avatar:\n    provider: precomputed\n    assets_dir: {tmp_path}\n"
        + fast_video_yaml
    )
    out = tmp_path / "o.mp4"
    result = CliRunner().invoke(
        app, ["create", str(md), str(out), "--config", str(cfg), "--reuse-avatar"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [1]  # rendered once, reused on slides 2 and 3
