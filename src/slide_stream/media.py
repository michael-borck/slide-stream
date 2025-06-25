"""Media handling functionality for Slide Stream."""

import os

from moviepy import AudioFileClip, ImageClip
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

        # Resize image to fit resolution if needed
        resolution = video_settings["resolution"]
        if image_clip.h > resolution[1]:
            image_clip = image_clip.resize(height=resolution[1])
        if image_clip.w > resolution[0]:
            image_clip = image_clip.resize(width=resolution[0])

        # Combine with audio - use with_audio for newer MoviePy versions
        if audio_clip:
            try:
                final_clip = image_clip.with_audio(audio_clip)
            except AttributeError:
                # Fallback for older MoviePy versions
                final_clip = image_clip.set_audio(audio_clip)
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
