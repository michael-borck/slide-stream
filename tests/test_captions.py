"""Tests for narration timing, captions, transcripts and chapters
(ADR-0018)."""

import wave
from pathlib import Path

from slide_stream.captions import (
    Cue,
    caption_cues,
    slide_starts,
    split_segments,
    synthesize_segments,
    write_chapter_list,
    write_ffmetadata,
    write_srt,
    write_transcript,
    write_vtt,
)


def _write_wav(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


class _FakeVoice:
    """Speaks every segment as a 1-second WAV, except configured failures."""

    name = "fake"

    def __init__(self, fail: set[int] | None = None):
        self.fail = fail or set()
        self.calls: list[str] = []

    def synthesize(self, text: str, filename: str):
        self.calls.append(text)
        n = len(self.calls)
        if n in self.fail:
            return None
        return str(_write_wav(Path(filename), 1.0))


def test_split_segments_sentences_and_abbreviations():
    text = "Hash tables are fast! E.g. a dict lookup. They use buckets.\n\nA new paragraph starts here."
    assert split_segments(text) == [
        "Hash tables are fast!",
        "E.g. a dict lookup.",
        "They use buckets.",
        "A new paragraph starts here.",
    ]


def test_split_segments_merges_tiny_fragments_and_list_markers():
    text = "1. First item\n2. Second item. Great. More detail follows here."
    segments = split_segments(text)
    assert segments[0].startswith("1. First item")
    assert not any(s.strip() == "Great." for s in segments)


def test_split_segments_cjk():
    assert split_segments("これは文です。次の文です！") == ["これは文です。", "次の文です！"]


def test_caption_cues_split_long_sentences_by_character_share():
    cues = caption_cues([Cue("word " * 60, 0.0, 12.0)])
    assert len(cues) > 1
    assert all(len(c.text) <= 84 for c in cues)
    assert abs(cues[-1].end - 12.0) < 0.01
    assert cues[0].start == 0.0
    assert " ".join(c.text for c in cues).split() == ["word"] * 60


def test_slide_starts_applies_crossfade_overlap():
    assert slide_starts([5.0, 5.0, 5.0], 0) == [0.0, 5.0, 10.0]
    assert slide_starts([5.0, 5.0, 5.0], 0.4) == [0.0, 4.6, 9.2]


def test_srt_and_vtt_formatting(tmp_path):
    cues = [Cue("Hello there.", 0.0, 1.5), Cue("Second line", 61.0, 62.25)]
    srt = write_srt(cues, tmp_path / "a.srt").read_text()
    assert "1\n00:00:00,000 --> 00:00:01,500\nHello there." in srt
    assert "00:01:01,000 --> 00:01:02,250" in srt
    vtt = write_vtt(cues, tmp_path / "a.vtt").read_text()
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt


def test_transcript_and_chapter_list(tmp_path):
    slides = [{"title": "Intro"}, {"title": ""}]
    write_transcript(slides, [[Cue("Welcome.", 0, 1)], []], tmp_path / "t.md")
    text = (tmp_path / "t.md").read_text()
    assert "## 1. Intro" in text and "Welcome." in text
    assert "## 2. Slide 2" in text and "(no narration)" in text
    write_chapter_list(slides, [0.0, 65.0], tmp_path / "c.txt")
    assert (tmp_path / "c.txt").read_text() == "0:00 Intro\n1:05 Slide 2\n"


def test_ffmetadata_sanitises_titles(tmp_path):
    write_ffmetadata([{"title": "A=B; C\n#D"}], [0.0], 10.0, tmp_path / "m")
    lines = (tmp_path / "m").read_text().splitlines()
    title = next(ln for ln in lines if ln.startswith("title="))
    assert "=" not in title[len("title="):]


def test_synthesize_segments_times_the_spoken_audio(tmp_path):
    voice = _FakeVoice()
    audio, cues = synthesize_segments(voice, "One sentence here. Two sentences now.", tmp_path, "s")
    assert audio is not None and audio.exists()
    # One synthesize call per segment, each timed; gaps sit between them.
    assert voice.calls == ["One sentence here.", "Two sentences now."]
    assert [c.text for c in cues] == voice.calls
    assert cues[0].start == 0.0
    assert abs(cues[1].start - (1.0 + 0.3)) < 0.2


def test_synthesize_segments_survives_a_later_failure(tmp_path):
    """A failed segment leaves a silent gap; later cues shift but keep their
    text, and the failed segment never becomes a caption."""
    voice = _FakeVoice(fail={2})
    audio, cues = synthesize_segments(
        voice, "First works. Second fails. Third appears.", tmp_path, "s"
    )
    assert audio is not None
    assert [c.text for c in cues] == ["First works.", "Third appears."]
    assert cues[0].end + 0.5 <= cues[1].start  # the gap was kept


def test_synthesize_segments_first_failure_means_no_audio(tmp_path):
    voice = _FakeVoice(fail={1})
    audio, cues = synthesize_segments(voice, "Nothing works here. Really.", tmp_path, "s")
    assert audio is None and cues == []
