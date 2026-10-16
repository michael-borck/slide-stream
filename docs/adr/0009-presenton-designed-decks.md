# 0009 — Designed decks via an external Presenton, rasterised for video

- Status: Accepted (2026-10)
- Drivers: deck design (templates, layouts, charts) is a whole product of its
  own; SlideStream's fixed-layout export (ADR-0008) is a starting point, and a
  PowerPoint the user already styled loses its style in the video because
  `create` re-renders frames from the extracted text.

## Context

Presenton (https://github.com/presenton/presenton, Apache-2.0,
self-hosted, BYOK) already builds designed, editable PPTX/PDF from
per-slide Markdown. Its contribution policy accepts only small, focused
PRs (currently `electron/`-only), so upstreaming SlideStream's narration,
TTS, avatar, and video pipeline into Presenton is neither welcome nor
sensible. Copying the Presenton templates/layouts into SlideStream would
mean building and maintaining a template engine to render them.

## Decision

Workflow: **source → designed PPTX/PDF → slide pictures → narrated video**.
SlideStream integrates at arm's length and credits Presenton; nothing is
vendored or bundled.

- `slide_stream.presenton` is a thin client over Presenton's
  `POST /api/v1/ppt/presentation/generate` (`slides_markdown`, one entry per
  source slide, no added title/TOC slides) plus its `/{id}/export` endpoint
  to get the PDF of the same generation. Every request carries
  faithfulness `instructions` (keep code verbatim, invent nothing, keep
  titles), overridable as `providers.presenton.instructions`. The user runs their own Presenton;
  connection settings layer as flags > config (`providers.presenton`) >
  `PRESENTON_*` env > `localhost:5001`.
- `slide-stream presenton <deck> <out.pptx|.pdf>` produces the designed deck
  only, with narration written into its speaker notes.
- `create --template NAME` generates the designed pair, rasterises the PDF,
  uses the slides as full-frame frames, and saves the PPTX (narration in
  notes) beside the video.
- `slide_stream.slide_render` renders PDF pages to PNGs with pypdfium2
  (Apache-2.0/BSD-3; optional `slide-stream[presenton]` extra). PyMuPDF was
  rejected because it is AGPL-3.0, which would bind anyone embedding or
  redistributing SlideStream. `--preserve-style` renders an
  existing `.pptx` via headless LibreOffice so the user's own design
  survives; `--slides-pdf` accepts a PDF exported by hand. A page/slide
  count mismatch aborts the run rather than misaligning narration.

## Consequences

- Presenton owns design; SlideStream owns narration/TTS/avatar/video. Small
  generic bugs found while integrating go upstream as normal Presenton PRs.
- Presenton may rephrase slide text to fit layouts; narration stays faithful
  to the source, so the spoken words can differ slightly from the slide text.
- Video fidelity for designed slides depends on PDF rendering (LibreOffice
  for local PPTX), not on PowerPoint itself; the editable PPTX remains the
  design source of truth.
