"""Tests for locating ffmpeg (system first, moviepy's bundled copy second)."""

import sys

from slide_stream.ffmpeg_bin import ffmpeg_exe


def test_prefers_system_ffmpeg(monkeypatch):
    monkeypatch.setattr("slide_stream.ffmpeg_bin.shutil.which", lambda _: "/opt/ffmpeg")
    assert ffmpeg_exe() == "/opt/ffmpeg"


def test_falls_back_to_bundled_ffmpeg(monkeypatch):
    monkeypatch.setattr("slide_stream.ffmpeg_bin.shutil.which", lambda _: None)
    exe = ffmpeg_exe()
    assert exe is not None
    assert "imageio_ffmpeg" in exe


def test_none_when_nothing_available(monkeypatch):
    monkeypatch.setattr("slide_stream.ffmpeg_bin.shutil.which", lambda _: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    assert ffmpeg_exe() is None

