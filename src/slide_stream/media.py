"""Media handling functionality for Slide Stream."""

import os

from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip
from rich.console import Console

# Note: Configuration now comes from config parameter

err_console = Console(stderr=True, style="bold red")


# Image generation functions moved to providers/images.py


# TTS functionality moved to providers/tts.py


def create_video_fragment(
    image_path: str, audio_path: str | None, output_path: str, config: dict
) -> str | None:
    """Create video fragment from image and audio."""
    try:
        # Get settings from config
        video_settings = config["settings"]["video"]
        
        # Load audio if it exists
        audio_clip = None
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
        image_clip = CompositeVideoClip(
            [background, image_clip.with_position("center")],
            size=(width, height),
        ).with_duration(duration)

        # Combine with audio (MoviePy 2.x: .with_audio())
        if audio_clip:
            final_clip = image_clip.with_audio(audio_clip)
        else:
            final_clip = image_clip

        # Write video file
        final_clip.write_videofile(
            output_path, 
            fps=video_settings["fps"], 
            codec=video_settings["codec"], 
            logger=None
        )

        # Clean up clips
        if audio_clip:
            audio_clip.close()
        image_clip.close()
        final_clip.close()

        return output_path

    except Exception as e:
        err_console.print(f"  - Video fragment creation error: {e}")
        return None
