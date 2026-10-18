# Changelog

## 2.18.0 — 2026-10-18

Slides in your own PowerPoint template (roadmap Phase 2c; ADR-0014):

- `create --pptx-template FILE` (or `settings.pptx_template`): new slides
  are built in the template's layouts, so its fonts, colours, backgrounds
  and logos style both the `.pptx` and the video. Works with a
  Quarto/pandoc `reference.pptx`; when a standard layout name is missing, a
  similar layout stands in and the run says so.
- Code keeps its syntax colouring and tables become native, editable
  PowerPoint tables, in templated decks and in the plain `.pptx` export.
- `slide-stream template init FILE` writes a starter template;
  `slide-stream template check FILE` reports what a template offers.
- `doctor` checks the template (and LibreOffice, for video) during
  `--dry-run`.


## 2.17.0 — 2026-10-17

Accessible outputs (roadmap Phase 12; ADR-0018):

- Every video now gets **captions** (`.srt` and `.vtt`), a Markdown
  **transcript** and **chapter markers** in the MP4 plus a
  `chapters.txt` list, by default. `settings.accessibility` turns any of
  them off or burns the captions into the picture.
- Narration is synthesised per sentence so the timings are exact; the
  joined sentences are the audio you hear. (Turning captions, transcript
  and chapters all off restores one synthesis call per slide.)
- `create deck.md deck.pptx --alt-text` and `enrich --alt-text` describe
  pictures with a vision LLM and write the text as the picture's alt text.
- `doctor` reports text contrast, text size and whether captions are on.


Notable changes, newest first. Full notes for each release are on
[GitHub Releases](https://github.com/michael-borck/slide-stream/releases).
Versions follow [semantic versioning](https://semver.org). The desktop app
(`desktop-v*` tags) installs the newest engine from PyPI, so it is not listed
here.

## 2.16.2 — unreleased

- **Fix:** `--llm-provider` on `create` and `draft` no longer reuses the
  configured provider's model, server URL, token or `think` setting. For
  example, `--llm-provider claude` over an Ollama config sent Claude the
  Ollama model name and failed.
- Live tests for Presenton and the Kokoro voice.

## 2.16.1 — 2026-10-17

Documentation, website and test-suite housekeeping (roadmap Phase 11). No
behaviour changes.

- Shorter README (it is the PyPI page) with links that work on PyPI.
- New provider reference, changelog and user stories; user guide rewritten
  as a reference; obsolete planning notes archived.
- Docs and website no longer claim FFmpeg must be installed.
- Website moved to `slidestream.borck.education`; accessibility fixes.
- Test files split by topic; tests for PDF and Word import; CI checks the
  bundled ffmpeg and a coverage floor.

## 2.16.0 — 2026-10-16

- **Fix:** a `# comment` inside a fenced code block no longer starts a new
  slide; fenced code and tables are parsed properly. `enrich` keeps them.
- **Fix:** video renders without a system ffmpeg, using the copy bundled with
  moviepy.
- **Fix:** `doctor` no longer counts LLM calls for scripted slides.
- Kokoro is the local voice floor: configured voice, then Kokoro (offline),
  then gTTS (ADR-0012). New TTS provider `auto` and `settings.voice_floor`.
- Code on slides is drawn with syntax colouring; tables keep their rows.
- Experimental, CLI-only Presenton designed decks: `slide-stream presenton`,
  `create --template`, `--preserve-style`, `--slides-pdf`, with a fidelity
  report and rebuilt code slides (ADR-0009, ADR-0013).

## 2.15.25 — 2026-09-15

- Security: host validation on every local-mode request, registered mascot
  names only, authenticated uploads before reading bodies, streaming size
  limits, explicit trusted proxies (`SLIDESTREAM_TRUSTED_PROXIES`); Docker
  binds to loopback by default.
- Rendering: enriched images reused only while the deck is unchanged;
  existing speaker notes kept by default; `.pptx` export and dry runs skip
  FFmpeg, TTS and avatar setup.
- Editable PowerPoint output from `create` (narration in speaker notes).

## 2.15.0 – 2.15.24 — 2026-08-26 to 2026-08-31

Mostly web UI and hosted demo work:

- Three output modes in the web UI; demo trimmed.
- Live render progress and a render trace panel; version in the footer.
- One-page teaser mode for the hosted demo, with idea-to-deck drafting.
- Ollama bearer auth and a `think` control for reasoning models.
- Editable enhanced slides, images without text, image style option;
  image and notes generated concurrently per slide.
- Bullets beside generated artwork in video slides (2.15.24).
- Hermetic tests; avatar polling resilience.

## 2.14.1 — 2026-07-25

- Presenter and transition controls in the web UI's Configure step.

## 2.14.0 — 2026-07-25

- Web UI becomes a Source → Deck → Configure → Render workflow.
- `create` uses images a deck already references, so `enrich` → `create`
  works.
- `--avatar-slides`, `--reuse-avatar`, `--transition`.
- ComfyUI renders retry once on a GPU out-of-memory error.

## 2.13.0 — 2026-07-21

- `slide-stream draft`: turn a text, Markdown, PDF, Word or PowerPoint file
  into a Markdown deck with an LLM.
- Clean errors instead of tracebacks when an LLM cannot start.

## 2.12.0 — 2026-07-21

- `[all]` install extra.
- `doctor` reports missing provider packages with the install command.
- Stage directions such as `[pause]` are removed before narration is voiced.
- `init` writes a minimal starter; `init --full` the complete reference.
- Documentation site published.

## 2.11.0 — 2026-07-19

- `enrich --notes fill|all`: AI presenter notes in the PowerPoint.
- `doctor` and `create --dry-run`: preflight estimates and warnings.
- Web UI preflight and PowerPoint output.
- Opt-in live end-to-end tests.

## 2.10.0 — 2026-07-18

- `wan-s2v` avatar provider: animates mascots and photos from narration
  audio, with no face detector.

## 2.0.0 — 2025-06-25

- Layered YAML configuration and the provider architecture (LLM, images,
  TTS). AI image generation and premium voices.

## 1.x

- 1.1: PowerPoint input with speaker notes.
- 1.0: first release, Markdown input.
