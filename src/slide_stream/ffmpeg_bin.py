"""Locate an ffmpeg executable.

moviepy depends on imageio-ffmpeg, which ships a static ffmpeg build for
macOS, Windows and Linux, and moviepy encodes video with it. So SlideStream
never needs a system ffmpeg: prefer one on the PATH (users may want a
specific build), else use the bundled binary.
"""

import shutil


def ffmpeg_exe() -> str | None:
    """Path to an ffmpeg executable, or None if neither exists."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # not installed, or no binary for this platform
        return None
