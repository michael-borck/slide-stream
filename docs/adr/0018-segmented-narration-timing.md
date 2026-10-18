# 0018 — Narration is timed in segments

- Status: Accepted (2026-10)
- Drivers: captions, transcripts, chapters and (Phase 14) bullets that appear
  as they are spoken all need to know when each part of the narration is
  spoken. Today a slide's narration is synthesised in one call, so nothing
  knows when individual sentences happen.
- Related: 0008 (one narration path), 0012 (voice fallback levels).

## Decision

**A slide's narration is produced as segments, and the segments are the
audio.** A segment is a sentence (or, for scripts without sentence
punctuation, a line). For each slide:

1. The final narration text is split into segments.
2. The voice synthesises each segment (same provider instance, so Kokoro
   loads its model once per run).
3. The segment audio is joined with a short pause (0.3 s) into the slide's
   narration track. There is no second, whole-slide synthesis: what you hear
   is what was timed, so captions can never drift from the voice.

Each segment records its offset and duration within the slide. A slide's
start time in the video is the sum of the previous fragments' durations,
minus the crossfade overlap between fragments, which the assembler computes
from the same values it uses to join them.

**Caption cues** are the segments, re-chunked for reading: a cue is at most
about two lines (~84 characters) and a few seconds; a long sentence is split
at word boundaries with its time shared by character count.

**Consumers, all from the same timings:** `.srt` and `.vtt` captions beside
the video (default on), burned-in captions (`settings.accessibility.burn`,
drawn as translucent bands on the slide's own frames, no extra dependency),
a Markdown transcript per slide (default on), chapter markers in the MP4
via an FFMETADATA pass-through plus a YouTube-style `chapters.txt` (default
on), and later, bullet reveal in Phase 14.

**Settings** live in `settings.accessibility`: `captions` (`both`, `srt`,
`vtt`, `none`), `burn` (false), `transcript` (true), `chapters` (true).
Turning captions, transcript and chapters all off also turns segmenting off,
restoring the single-call synthesis.

**Alt text** is not part of the timing model: pictures are described by a
vision model on request (`--alt-text`), written into the `.pptx` and the
enriched deck. `doctor` reports contrast, text size and whether captions
are enabled.

## Consequences

- Narration is synthesised once per segment instead of once per slide: more
  calls, the same characters, so cloud TTS cost is unchanged. Slightly more
  synthesis time per slide.
- Prosody can differ from whole-slide synthesis; short segments generally
  sound fine, and the pause between segments is natural.
- A segment that falls back (voice server down mid-slide) can change voice
  within a slide; the existing fallback counter and `--strict` semantics
  apply per segment.
- Caption and chapter timings are exact for SlideStream's own slides. With
  designed slides (`--template`, `--preserve-style`, `--slides-pdf`) the
  same timings apply, since timing comes from the narration, not the frames.
- Tests that mock a voice see one `synthesize` call per segment.
