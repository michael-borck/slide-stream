"""Media handling functionality for Slide Stream."""

import os
import textwrap

import requests
from gtts import gTTS
from moviepy import AudioFileClip, ImageClip
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

from .config import (
    BG_COLOR,
    DEFAULT_CONTENT_FONT_SIZE,
    DEFAULT_SLIDE_DURATION,
    DEFAULT_TITLE_FONT_SIZE,
    FONT_COLOR,
    IMAGE_DOWNLOAD_TIMEOUT,
    MAX_LINE_WIDTH,
    SLIDE_DURATION_PADDING,
    VIDEO_CODEC,
    VIDEO_FPS,
    VIDEO_RESOLUTION,
)

err_console = Console(stderr=True, style="bold red")


def search_and_download_image(query: str, filename: str) -> str:
    """Download image from Unsplash based on search query."""
    try:
        # Use picsum.photos as fallback since source.unsplash.com is deprecated
        url = f"https://picsum.photos/{VIDEO_RESOLUTION[0]}/{VIDEO_RESOLUTION[1]}"
        response = requests.get(
            url, timeout=IMAGE_DOWNLOAD_TIMEOUT, allow_redirects=True
        )
        response.raise_for_status()

        with open(filename, "wb") as f:
            f.write(response.content)

        return filename

    except requests.exceptions.RequestException as e:
        err_console.print(f"  - Image download error: {e}. Using a placeholder.")
        return create_text_image("Image not found", [f"Query: {query}"], filename)


def create_text_image(title: str, content_items: list, filename: str) -> str:
    """Create a text-based image for slides."""
    img = Image.new("RGB", VIDEO_RESOLUTION, color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Try to load custom fonts, fall back to default
    try:
        title_font = ImageFont.truetype("arial.ttf", DEFAULT_TITLE_FONT_SIZE)
        content_font = ImageFont.truetype("arial.ttf", DEFAULT_CONTENT_FONT_SIZE)
    except OSError:
        try:
            # Try alternative common font names
            title_font = ImageFont.truetype("DejaVuSans.ttf", DEFAULT_TITLE_FONT_SIZE)
            content_font = ImageFont.truetype(
                "DejaVuSans.ttf", DEFAULT_CONTENT_FONT_SIZE
            )
        except OSError:
            title_font = ImageFont.load_default()
            content_font = ImageFont.load_default()

    # Draw title
    draw.text(
        (VIDEO_RESOLUTION[0] * 0.1, VIDEO_RESOLUTION[1] * 0.1),
        title,
        font=title_font,
        fill=FONT_COLOR,
    )

    # Draw content items
    y_pos = VIDEO_RESOLUTION[1] * 0.3
    for item in content_items:
        wrapped_lines = textwrap.wrap(f"• {item}", width=MAX_LINE_WIDTH)
        for line in wrapped_lines:
            draw.text(
                (VIDEO_RESOLUTION[0] * 0.1, y_pos),
                line,
                font=content_font,
                fill=FONT_COLOR,
            )
            y_pos += 70
        y_pos += 30

    img.save(filename)
    return filename


def text_to_speech(text: str, filename: str) -> str | None:
    """Convert text to speech using gTTS."""
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        return filename
    except Exception as e:
        err_console.print(f"  - Audio generation error: {e}")
        return None


def create_video_fragment(
    image_path: str, audio_path: str | None, output_path: str
) -> str | None:
    """Create video fragment from image and audio."""
    try:
        # Load audio if it exists
        audio_clip = None
        if audio_path and os.path.exists(audio_path):
            audio_clip = AudioFileClip(audio_path)

        # Determine duration
        duration = (
            audio_clip.duration + SLIDE_DURATION_PADDING
            if audio_clip
            else DEFAULT_SLIDE_DURATION
        )

        # Create image clip
        image_clip = ImageClip(image_path, duration=duration)  # type: ignore[attr-defined]

        # Resize image to fit resolution if needed
        if image_clip.h > VIDEO_RESOLUTION[1]:
            image_clip = image_clip.resize(height=VIDEO_RESOLUTION[1])
        if image_clip.w > VIDEO_RESOLUTION[0]:
            image_clip = image_clip.resize(width=VIDEO_RESOLUTION[0])

        # Combine with audio
        final_clip = image_clip.set_audio(audio_clip) if audio_clip else image_clip

        # Write video file
        final_clip.write_videofile(
            output_path, fps=VIDEO_FPS, codec=VIDEO_CODEC, logger=None
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
