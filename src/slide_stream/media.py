"""Media handling functionality for Slide Stream."""

import os

import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw
from rich.console import Console

# Note: Configuration now comes from config parameter

err_console = Console(stderr=True, style="bold red")


def _build_head_overlay(
    head_video: str,
    duration: float,
    frame_size: tuple[int, int],
    avatar_settings: dict,
):
    """Build a positioned, circle-masked talking-head clip for one fragment.

    The head clip's own audio is always dropped so the fragment's TTS track
    stays the single audio source (sync issues are impossible at composite
    time). The clip is trimmed to the fragment duration, or its last frame is
    frozen to fill it (e.g. through the slide's padding second).
    """
    width, height = frame_size
    size_frac = float(avatar_settings.get("size", 0.28))
    margin = int(avatar_settings.get("margin", 24))
    position = avatar_settings.get("position", "bottom-right")

    head = VideoFileClip(head_video).without_audio()

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
    return head.with_position(positions.get(position, positions["bottom-right"]))


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
                head_clip = _build_head_overlay(
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

            # Write video file
            final_clip.write_videofile(
                output_path,
                fps=video_settings["fps"],
                codec=video_settings["codec"],
                logger=None,
            )

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
            if final_clip is not None:
                final_clip.close()

    except Exception as e:
        err_console.print(f"  - Video fragment creation error: {e}")
        return None
