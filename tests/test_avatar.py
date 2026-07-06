"""Tests for avatar providers and talking-head compositing (no GPU needed)."""

import copy

import pytest
from moviepy import ColorClip, VideoFileClip

from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.media import create_video_fragment
from slide_stream.providers.avatar import NoneAvatarProvider, PrecomputedAvatarProvider
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
