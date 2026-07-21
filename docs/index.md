# SlideStream documentation

SlideStream turns a Markdown or PowerPoint deck into a **narrated video
presentation** — spoken narration (not read-aloud bullets), optionally in a
clone of your own voice, illustrated with AI images, and optionally fronted by a
talking-head avatar. It runs on your machine or self-hosted; your decks and your
voice stay yours.

## Get started

```bash
pip install "slide-stream[all]"      # everything: AI providers + offline TTS + web UI
slide-stream init                     # a minimal, works-with-no-keys config
slide-stream create deck.md out.mp4   # render a narrated video
```

New here? Start with the **[User Guide](USER_GUIDE.md)**. Prefer recipes? The
**[Use Cases](USE_CASES.md)** guide walks through common tasks end to end.

## Find your way

- **[User Guide](USER_GUIDE.md)** — installation, configuration, providers, and the full workflow.
- **[Use Cases](USE_CASES.md)** — task-based recipes (narrate in your voice, add a presenter, export notes, …).
- **Providers & integrations** — self-hosted API notes for [Voicebox](voicebox-api.md) (TTS) and the talking-head backends ([options overview](talking-head-options.md)).
- **[Development workflow](DEVELOPMENT_WORKFLOW.md)** — for contributors.

!!! tip "Check your setup"
    `slide-stream doctor <deck>` reports any configured provider whose package
    or API key is missing — with the exact `pip install` line to fix it.

The full project, source, and issue tracker live on
[GitHub](https://github.com/michael-borck/slide-stream).
