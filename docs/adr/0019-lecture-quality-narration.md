# 0019 — Lecture-quality narration: reveal, pauses, pronunciation, clips

- Status: Proposed (2026-10)
- Drivers: a narrated video that shows every bullet from second one feels
  like a slideshow with a voice-over. Code symbols ("percent folds it into
  the range") and acronyms are read awkwardly. Lecturers want short screen
  recordings inside a slide. All three build on ADR-0018's segment timings.
- Related: 0012 (fallback), 0017 (none), 0018 (segment timing).

## Decision

### 1. Bullets appear as they are spoken (video only)

When SlideStream can attribute narration to bullets **by construction**, it
synthesises one segment per bullet (after a title segment) and the video
frame switches to "bullets 1..k visible" as each segment starts. The final
state is the full slide; nothing is removed from the deck.

- **Attribution:** only the built-from-slide-text narration path attributes
  by construction (segments *are* title + bullets). LLM-written narration
  and `--script` files cannot be attributed reliably in v1: those slides
  stay static, and `doctor` says so. `--verbatim-notes` is likewise static.
- **Scope:** SlideStream's own slides (text cards and bullets-beside-
  artwork). Designed slides (template, Presenton, PDF) stay static — their
  frames are pictures.
- **Settings:** `settings.reveal: auto` (`auto` = on where attribution
  exists, `off` = never). The `.pptx` output is unaffected (PowerPoint
  animations are out of scope for python-pptx).

### 2. Pause markers are honoured, not deleted

`[pause]` and `[pause 2s]` (a duration in seconds, whole or decimal) become
real silence between segments — including in `--script` and
`--verbatim-notes` narration, which the user authored deliberately. Every
other bracketed or parenthetical direction is still stripped before voicing
(as today), and prose cues ("pause here") are still never touched — only
flagged by `doctor`. Pauses appear in captions as gaps between cues, never
as text.

### 3. A pronunciation dictionary

`settings.pronunciations` maps written form to spoken form and is applied to
the narration text after direction handling and before segmentation:

```yaml
settings:
  pronunciations:
    GPU: "G P U"
   /hash table/: "hash table"
    "->": " returns"
```

Keys are literal text (regex on word boundaries for word-like keys). A
small built-in map covers the common code symbols read from slide text
(`%` → " percent", `->` → " returns", `_` → space). Applied everywhere
narration is voiced, so captions and audio always agree.

### 4. Video clips on slides

A slide that references a video file (`![](demo.mp4)` in Markdown, or a
clip matched by `enrich`) plays the clip in the slide's picture area instead
of a still picture: the slide lasts at least as long as the clip (narration
keeps playing), the clip is scaled to fit without cropping, and audio from
the clip is dropped (the narration is the single audio source, as with
avatars). The `.pptx` output embeds the clip's first frame as a still.
`doctor` flags missing clip files and clips longer than their narration.

## Consequences

- Reveal only changes SlideStream's own slides; a designed deck keeps the
  template's static design (consistent with ADR-0014).
- Segment counts per slide grow (one per bullet); TTS cost is unchanged
  (same characters) and Kokoro still loads once per run.
- Honouring `[pause 2s]` means old decks that used it as a note now render
  actual silence; `doctor` reports the total pause time.
- The pronunciation map is a blunt string replacement; a key that appears
  inside another word is protected by word boundaries only for word-like
  keys (letters/digits). Users can see the applied result in the transcript.
- Clips are decoded per fragment; long clips increase render time and size.
