"""Narration timing in segments, captions, transcripts and chapters
(ADR-0018).

A slide's narration is synthesised one segment (sentence) at a time and the
segments are joined into the slide's audio track, so every caption, chapter
and later bullet reveal is timed to the voice that was actually heard.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# A pause between segments, seconds.
SEGMENT_GAP = 0.3

# Caption cue targets: a cue is readable in one glance.
_MAX_CUE_CHARS = 84
_MAX_CUE_SECONDS = 6.0
_MIN_CUE_CHARS = 12

# Sentence enders, including CJK; a sentence ends at punctuation followed by
# whitespace + a capital/digit/quote, or at a paragraph break. A period after
# a single letter or a short numeral (initials, "1." list markers, "e.g.")
# does not end a sentence.
_SENTENCE_RE = re.compile(
    r"(?<=[.!?\u3002\uff01\uff1f])\s+(?=[A-Z0-9\u201c'\"])"
)
_PARA_RE = re.compile(r"\n\s*\n")
_SHORT_PUNCT_RE = re.compile(r"(?:^|[\s(])([A-Za-z]|\d{1,2})\.\s*$")

# An FFMETADATA header must not contain these control characters.
_FFMETA_SANITIZE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f=;#\n\\]")


class _Speaking(Protocol):
    """The part of a TTS provider segment synthesis needs."""

    name: str

    def synthesize(self, text: str, filename: str) -> str | None: ...  # pragma: no cover


@dataclass
class Cue:
    """A span of narration: ``start``/``end`` are seconds within the slide."""

    text: str
    start: float
    end: float


def split_segments(text: str) -> list[str]:
    """Narration text as speakable segments: sentences, merged so no segment
    is trivially short (a stray "Great." becomes part of its neighbour)."""
    paragraphs = [p.strip() for p in _PARA_RE.split(text.strip()) if p.strip()]
    raw: list[str] = []
    for para in paragraphs:
        pieces: list[str] = []
        for piece in _SENTENCE_RE.split(para):
            # CJK sentences end without whitespace; split those too.
            pieces.extend(
                p.strip() for p in re.split(r"(?<=[。！？])(?=.)", piece) if p.strip()
            )
        if not pieces:
            continue
        merged = [pieces[0]]
        for piece in pieces[1:]:
            tail = merged[-1]
            # CJK sentences end without whitespace; never merge across one.
            cjk_boundary = bool(tail) and tail[-1] in "\u3002\uff01\uff1f"
            tiny = len(piece) < _MIN_CUE_CHARS
            dangling = bool(_SHORT_PUNCT_RE.search(tail))
            if not cjk_boundary and (tiny or dangling):
                merged[-1] = f"{tail} {piece}"
            else:
                merged.append(piece)
        raw.extend(merged)
    return raw


def _silence(duration: float, fps: int, channels: int):
    import numpy as np
    from moviepy.audio.AudioClip import AudioArrayClip

    samples = np.zeros((max(1, int(duration * fps)), channels), dtype=np.float32)
    return AudioArrayClip(samples, fps=fps)


def synthesize_segments(
    provider: _Speaking, text: str, work_dir: Path, stem: str
) -> tuple[Path | None, list[Cue]]:
    """Speak ``text`` one segment at a time; return the joined audio path and
    each segment's ``(start, end)`` within it.

    Uses the provider's normal ``synthesize`` for every segment, so fallback
    and strict-mode behaviour are unchanged. Returns ``(None, [])`` when the
    first segment produces nothing (the caller treats that as the usual
    failed-audio case). Pause directives (``[pause 2s]``) become real silence
    between segments (ADR-0019).
    """
    from moviepy import AudioFileClip, concatenate_audioclips

    from .narration import split_pauses

    # Expand sentences into speak/pause items.
    items: list[tuple[str, float]] = []
    for segment in split_segments(text):
        pairs = split_pauses(segment)
        for j, (speak, pause) in enumerate(pairs):
            if speak:
                items.append((speak, pause if j < len(pairs) - 1 else 0.0))
            elif pause:
                if items:
                    prev_text, prev_pause = items[-1]
                    items[-1] = (prev_text, prev_pause + pause)
    if not items:
        return None, []

    paths: list[Path] = []
    clips: list[Any] = []
    cues: list[Cue] = []
    spoke_any = False
    t = 0.0
    try:
        for i, (speak, pause) in enumerate(items):
            path = work_dir / f"{stem}_seg{i + 1:02d}.mp3"
            result = provider.synthesize(speak, str(path))
            if result and Path(result).exists():
                spoke_any = True
                seg_path = Path(result)
                if seg_path != path:
                    seg_path.replace(path)
                paths.append(path)
                clip = AudioFileClip(str(path))
                duration = float(clip.duration)
                clips.append(clip)
                cues.append(Cue(speak, t, t + duration))
            else:
                # A failed segment keeps a silent gap so later captions
                # don't shift; strict mode has already returned None above.
                clips.append(_silence(0.8, 24000, 1))
                duration = 0.8
            t += duration
            gap = pause if pause > 0 else (
                SEGMENT_GAP if i < len(items) - 1 else 0.0
            )
            if gap > 0:
                clips.append(_silence(gap, 24000, 1))
                t += gap
        if not spoke_any:
            # Every segment failed (e.g. strict mode): that is the usual
            # failed-audio case, not a silent video.
            return None, []
        joined = concatenate_audioclips(clips)
        audio_path = work_dir / f"{stem}.mp3"
        joined.write_audiofile(str(audio_path), logger=None)
        return audio_path, cues
    finally:
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass


def caption_cues(segments: list[Cue]) -> list[Cue]:
    """Segments re-chunked for reading: short cues, long ones split at word
    boundaries with the time shared by character count."""
    cues: list[Cue] = []
    for seg in segments:
        text = " ".join(seg.text.split())
        if seg.end - seg.start <= _MAX_CUE_SECONDS or len(text) <= _MAX_CUE_CHARS:
            cues.append(Cue(text, seg.start, seg.end))
            continue
        words, chunks, current = text.split(), [], ""
        for word in words:
            if current and len(current) + 1 + len(word) > _MAX_CUE_CHARS:
                chunks.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            chunks.append(current)
        total = sum(len(c) for c in chunks)
        t = seg.start
        for chunk in chunks:
            dur = (seg.end - seg.start) * len(chunk) / total
            cues.append(Cue(chunk, t, t + dur))
            t += dur
    return cues


def slide_starts(durations: list[float], transition: float) -> list[float]:
    """Each slide's start in the final video, applying the crossfade overlap
    exactly as the assembler does."""
    starts = [0.0]
    overlap = 0.0
    if len(durations) >= 2 and transition > 0:
        overlap = min(transition, min(durations) * 0.9)
    for duration in durations[:-1]:
        starts.append(starts[-1] + duration - overlap)
    return starts


def _timestamp(seconds: float, comma: bool) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{ms:03d}"


def write_srt(cues: list[Cue], path: str | Path) -> Path:
    lines = []
    for i, cue in enumerate(cues, 1):
        lines += [
            str(i),
            f"{_timestamp(cue.start, True)} --> {_timestamp(cue.end, True)}",
            cue.text,
            "",
        ]
    out = Path(path)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_vtt(cues: list[Cue], path: str | Path) -> Path:
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines += [
            f"{_timestamp(cue.start, False)} --> {_timestamp(cue.end, False)}",
            cue.text,
            "",
        ]
    out = Path(path)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_transcript(
    slides: list[dict[str, Any]], slide_cues: list[list[Cue]], path: str | Path
) -> Path:
    """A readable Markdown transcript: slide titles as headings, narration
    under each."""
    lines = ["# Transcript", ""]
    for i, (slide, cues) in enumerate(zip(slides, slide_cues, strict=False), 1):
        title = str(slide.get("title", "")).strip() or f"Slide {i}"
        lines += [f"## {i}. {title}", ""]
        lines += [" ".join(cue.text for cue in cues).strip() or "(no narration)", ""]
    out = Path(path)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _chapter_title(slide: dict[str, Any], index: int) -> str:
    return str(slide.get("title", "")).strip() or f"Slide {index + 1}"


def _hhmm(seconds: float) -> str:
    """A YouTube chapter stamp: m:ss, or h:mm:ss past an hour."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def write_chapter_list(
    slides: list[dict[str, Any]], starts: list[float], path: str | Path
) -> Path:
    """A YouTube-style chapter list (``0:00 Title``)."""
    lines = [
        f"{_hhmm(start)} {_chapter_title(slide, i)}"
        for i, (slide, start) in enumerate(zip(slides, starts, strict=False))
    ]
    out = Path(path)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_ffmetadata(
    slides: list[dict[str, Any]], starts: list[float], total: float,
    path: str | Path,
) -> Path:
    """FFMETADATA chapters for embedding into the MP4 with ffmpeg."""
    lines = [";FFMETADATA1"]
    for i, (slide, start) in enumerate(zip(slides, starts, strict=False)):
        end = (
            starts[i + 1] if i + 1 < len(starts) else max(total, start + 0.001)
        )
        title = _FFMETA_SANITIZE.sub(" ", _chapter_title(slide, i)).strip() or f"Slide {i + 1}"
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={int(start * 1000)}",
            f"END={int(end * 1000)}",
            f"title={title}",
        ]
    out = Path(path)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def embed_chapters(video: str | Path, metadata: str | Path) -> None:
    """Write chapters into an MP4 in place (ffmpeg stream copy)."""
    import subprocess
    import tempfile

    from .ffmpeg_bin import ffmpeg_exe

    ffmpeg = ffmpeg_exe()
    if not ffmpeg:
        return
    video_path = Path(video)
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", dir=str(video_path.parent), delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(video_path),
             "-i", str(metadata), "-map", "0", "-map_metadata", "1",
             "-codec", "copy", str(tmp_path)],
            check=True,
        )
        tmp_path.replace(video_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
