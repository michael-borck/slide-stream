"""Structural assertions for live e2e tests.

These check *integration health*, not aesthetics — a live render can't be
judged for quality by a machine, but it can be checked for the things that
break when a provider's API drifts: a valid media file, an audio track of
plausible length, a non-flat image, a head clip that actually moves.
"""

import numpy as np


def probe_video(path) -> dict:
    """Return {w, h, duration, has_audio} for a video file."""
    from moviepy import VideoFileClip

    with VideoFileClip(str(path)) as c:
        return {
            "w": c.w,
            "h": c.h,
            "duration": float(c.duration),
            "has_audio": c.audio is not None,
        }


def audio_duration(path) -> float:
    """Duration in seconds of an audio (or A/V) file."""
    from moviepy import AudioFileClip

    with AudioFileClip(str(path)) as a:
        return float(a.duration)


def region_motion(path, box=(0.0, 1.0, 0.0, 1.0), t0=0.4, t1=None) -> float:
    """Mean absolute pixel difference between two frames within a fractional
    ``box`` = (y0, y1, x0, x1). A proxy for "is this region animating".
    """
    from moviepy import VideoFileClip

    with VideoFileClip(str(path)) as c:
        end = c.duration - 1e-3
        t1 = t1 if t1 is not None else max(c.duration - 0.2, t0 + 0.1)
        f0 = c.get_frame(min(t0, end)).astype("int16")  # type: ignore[union-attr]
        f1 = c.get_frame(min(t1, end)).astype("int16")  # type: ignore[union-attr]
    h, w, _ = f0.shape
    y0, y1, x0, x1 = box
    a = f0[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    b = f1[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    return float(np.abs(b - a).mean())
