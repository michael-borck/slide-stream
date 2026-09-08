# 0008 — Dual output modes: `.mp4` video and editable `.pptx` deck

- Status: Accepted (2026-09)
- Drivers: teachers want a video to share AND a deck to present live; the
  artwork/narration work is identical for both

## Decision

`create` selects its pipeline tail by output extension
(`src/slide_stream/export_pptx.py`, cli.py):

- `.mp4` (default): the full video pipeline — TTS audio, avatar, fragments,
  ffmpeg assembly.
- `.pptx`: same per-slide pipeline up through artwork + final narration
  text, then stops. `export_pptx` writes native slides (title, bullets,
  artwork fitted right, bullets left) with the narration in the speaker
  notes — no TTS, no avatar, no ffmpeg.

PPTX stays input-only for `draft`/`parse`; this ADR covers output only.

## Consequences

- One narration code path (LLM / `--script` / `--verbatim-notes`) feeds both
  outputs, so wording never diverges between the video and the deck.
- Strict-mode narration failures abort a pptx export the same as a video.
- The exporter is deliberately dumb: layout is fixed (artwork right, bullets
  left) and theme comes from PowerPoint defaults; fancy deck design is out of
  scope — the pptx is a starting point the teacher edits.
