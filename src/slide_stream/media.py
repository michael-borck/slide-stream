"""Media handling functionality for Slide Stream."""

import os
import textwrap

import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx import Loop
from PIL import Image, ImageDraw
from rich.console import Console

from .providers.images import load_font

# Note: Configuration now comes from config parameter

err_console = Console(stderr=True, style="bold red")


def compose_slide_with_bullets(
    image_path: str,
    title: str,
    bullets: list[str],
    output_path: str,
    config: dict,
) -> str:
    """Render a video slide with the educator's bullet points beside the
    generated artwork.

    Bullets are the core of the slide (mirroring the enhanced-deck layout):
    the educator's text on the left, artwork on the right. Used when an image
    provider (SwarmUI, DALL-E, Imagen, …) produced the slide art — such art
    contains no text at all, so without this the video would show only
    pictures while the narration speaks bullets nobody can see. The text
    provider already draws bullets into its own cards and needs no composing.
    """
    from PIL import ImageDraw

    video_settings = config["settings"]["video"]
    image_settings = config["settings"].get("image", {})
    width, height = video_settings["resolution"]
    bg_color = image_settings.get("bg_color", "black")
    font_color = image_settings.get("font_color", "white")

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    title_size = int(height * 0.062)
    body_size = int(height * 0.042)
    title_font = load_font(title_size)
    body_font = load_font(body_size)
    margin = int(width * 0.055)
    wrap_cols = 34

    y = int(height * 0.075)
    if title:
        for line in textwrap.wrap(title, width=wrap_cols) or [title]:
            draw.text((margin, y), line, font=title_font, fill=font_color)
            y += int(title_size * 1.25)
        y += int(height * 0.04)

    for item in bullets:
        for line in textwrap.wrap(f"• {item}", width=wrap_cols) or [f"• {item}"]:
            if y > height * 0.84:  # keep clear of the presenter circle below
                break
            draw.text((margin, y), line, font=body_font, fill=font_color)
            y += int(body_size * 1.45)
        y += int(height * 0.022)

    # Artwork on the right: scaled to fit its region, centered vertically.
    art = Image.open(image_path).convert("RGB")
    region_w, region_h = int(width * 0.42), int(height * 0.72)
    scale = min(region_w / art.width, region_h / art.height, 1.0)
    if scale < 1.0:
        art = art.resize((max(1, int(art.width * scale)), max(1, int(art.height * scale))))
    art_x = width - int(width * 0.045) - art.width
    art_y = (height - art.height) // 2
    img.paste(art, (art_x, art_y))

    img.save(output_path)
    return output_path


def concatenate_with_transition(clips: list, transition_seconds: float):
    """Concatenate slide fragments, optionally crossfading between them.

    ``transition_seconds <= 0`` (or a single clip) is a plain hard-cut
    concatenation — the default, unchanged behaviour. Otherwise each slide
    after the first fades in over the previous one's tail, so slides dissolve
    into each other instead of cutting."""
    if transition_seconds <= 0 or len(clips) < 2:
        return concatenate_videoclips(clips)
    from moviepy.video.fx import CrossFadeIn

    # A transition can't be longer than (most of) the shortest clip it overlaps.
    shortest = min(c.duration for c in clips)
    d = min(transition_seconds, shortest * 0.9)
    faded = [clips[0]] + [c.with_effects([CrossFadeIn(d)]) for c in clips[1:]]
    return concatenate_videoclips(
        faded, method="compose", padding=-d  # type: ignore[arg-type]
    )


def _build_head_overlay(
    head_video: str,
    duration: float,
    frame_size: tuple[int, int],
    avatar_settings: dict,
):
    """Build a positioned, circle-masked talking-head clip for one fragment.

    The head clip's own audio is always dropped so the fragment's TTS track
    stays the single audio source (sync issues are impossible at composite
    time). The clip is trimmed to the fragment duration; a much shorter clip
    (a short talking loop, e.g. wan-s2v) is looped to fill it, while one that
    only falls short by the padding tail simply holds its last frame.

    Returns ``(overlay, source)``: the positioned overlay clip plus the
    underlying ``VideoFileClip``, which the caller must close — closing a
    derived or concatenated clip does not release the source's ffmpeg
    reader. If building the overlay fails, the source is closed here before
    the exception propagates.
    """
    width, height = frame_size
    size_frac = float(avatar_settings.get("size", 0.28))
    margin = int(avatar_settings.get("margin", 24))
    position = avatar_settings.get("position", "bottom-right")

    source = VideoFileClip(head_video)
    try:
        head = source.without_audio()

        # Center-crop to a square, then scale to the circle diameter.
        side = min(head.w, head.h)
        head = head.cropped(
            x_center=head.w / 2, y_center=head.h / 2, width=side, height=side
        )
        diameter = max(1, int(height * size_frac))
        head = head.resized((diameter, diameter))

        if head.duration > duration:
            head = head.subclipped(0, duration)
        elif head.duration < duration:
            # A clip much shorter than the slide is a short talking loop (e.g.
            # wan-s2v's default ~4s clip): loop it to fill, so the head keeps
            # animating rather than freezing partway through. A clip that only
            # falls short by the padding tail just holds its last frame — no
            # jarring snap back to the start pose.
            if head.duration < duration * 0.75:
                head = head.with_effects([Loop(duration=duration)])
            else:
                frame_time = max(head.duration - 1.0 / 30.0, 0)
                pad = head.to_ImageClip(t=frame_time).with_duration(
                    duration - head.duration
                )
                head = concatenate_videoclips([head, pad])

        # Circular alpha mask drawn with PIL.
        mask_image = Image.new("L", (diameter, diameter), 0)
        ImageDraw.Draw(mask_image).ellipse((0, 0, diameter - 1, diameter - 1), fill=255)
        mask_clip = ImageClip(
            np.array(mask_image) / 255.0, is_mask=True
        ).with_duration(duration)
        head = head.with_mask(mask_clip)

        positions = {
            "bottom-right": (width - diameter - margin, height - diameter - margin),
            "bottom-left": (margin, height - diameter - margin),
            "top-right": (width - diameter - margin, margin),
            "top-left": (margin, margin),
        }
        overlay = head.with_position(
            positions.get(position, positions["bottom-right"])
        )
        return overlay, source
    except Exception:
        # Never leak the ffmpeg reader when a later build step fails.
        source.close()
        raise


def create_video_fragment(
    image_path: str,
    audio_path: str | None,
    output_path: str,
    config: dict,
    head_video: str | None = None,
) -> str | None:
    """Create video fragment from image and audio, optionally with a
    talking-head overlay."""
    try:
        # Get settings from config
        video_settings = config["settings"]["video"]

        audio_clip = None
        image_clip = None
        final_clip = None
        head_clip = None
        head_source = None
        try:
            # Load audio if it exists
            if audio_path and os.path.exists(audio_path):
                audio_clip = AudioFileClip(audio_path)

            # Determine duration
            duration = (
                audio_clip.duration + video_settings["slide_duration_padding"]
                if audio_clip
                else video_settings["default_slide_duration"]
            )

            # Create image clip
            image_clip = ImageClip(image_path, duration=duration)  # type: ignore[attr-defined]

            # Normalize every fragment to the exact target resolution so that
            # concatenate_videoclips() receives uniformly-sized clips. We scale the
            # image to fit inside the frame (preserving aspect ratio) and center it
            # on a background canvas of the configured resolution.
            width, height = video_settings["resolution"]
            scale = min(width / image_clip.w, height / image_clip.h)
            # Avoid upscaling tiny images beyond their native size.
            scale = min(scale, 1.0)
            if scale != 1.0:
                image_clip = image_clip.resized(scale)

            background = ColorClip(
                size=(width, height),
                color=(0, 0, 0),
                duration=duration,
            )

            if head_video and os.path.exists(head_video):
                head_clip, head_source = _build_head_overlay(
                    head_video,
                    duration,
                    (width, height),
                    config["settings"].get("avatar", {}),
                )

            layers = [background, image_clip.with_position("center")]  # type: ignore[attr-defined]
            if head_clip is not None:
                layers.append(head_clip)
            image_clip = CompositeVideoClip(
                layers,
                size=(width, height),
            ).with_duration(duration)

            # Combine with audio (MoviePy 2.x: .with_audio())
            final_clip = image_clip.with_audio(audio_clip) if audio_clip else image_clip

            # Write video file. Pass the audio codec (default aac) so fragments
            # are playable everywhere — MoviePy otherwise defaults to mp3, which
            # QuickTime won't play inside an mp4 container.
            write_kwargs: dict = {
                "fps": video_settings["fps"],
                "codec": video_settings["codec"],
                "logger": None,
            }
            if audio_clip is not None:
                write_kwargs["audio_codec"] = video_settings.get("audio_codec", "aac")
            final_clip.write_videofile(output_path, **write_kwargs)

            return output_path
        finally:
            # Always release moviepy clips and ffmpeg handles, even if encoding
            # raised; otherwise a failed fragment leaks file descriptors.
            if audio_clip is not None:
                audio_clip.close()
            if image_clip is not None:
                image_clip.close()
            if head_clip is not None:
                head_clip.close()
            if head_source is not None:
                # Closing the derived overlay does not release the source
                # VideoFileClip's ffmpeg reader; close it explicitly.
                head_source.close()
            if final_clip is not None:
                final_clip.close()

    except Exception as e:
        err_console.print(f"  - Video fragment creation error: {e}")
        return None
