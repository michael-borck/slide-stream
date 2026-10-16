# SlideStream documentation

SlideStream turns a Markdown file, a PowerPoint deck or a document into a
**narrated video presentation**, an editable PowerPoint, or both. Narration
is what a presenter would say, not the bullets read aloud; the voice runs
offline or can be a clone of your own; pictures and a presenter in the
corner are optional. It runs on your machine or your own servers.

## Get started

```bash
pip install "slide-stream[all]"      # everything: AI providers + offline TTS + web UI
slide-stream init                     # a minimal, works-with-no-keys config
slide-stream create deck.md out.mp4   # render a narrated video
slide-stream create deck.md out.pptx  # or an editable PowerPoint
```

New here? The **[use cases](USE_CASES.md)** walk through common jobs; the
**[user guide](USER_GUIDE.md)** is the reference.

## Find your way

- **[Use cases](USE_CASES.md)**: recipes for common jobs.
- **[User guide](USER_GUIDE.md)**: install, configuration, deck format, every command.
- **[Providers](PROVIDERS.md)**: every voice, picture, LLM, presenter and design option.
- **[Changelog](CHANGELOG.md)**: what changed in each release.
- **[Services](SERVICES.md)**: the self-hosted engines SlideStream can call, their licences and checks.
- **Project**: [roadmap](ROADMAP.md), [user stories](USER_STORIES.md), [decisions (ADRs)](adr/README.md), [development workflow](DEVELOPMENT_WORKFLOW.md).

!!! tip "Check your setup"
    `slide-stream doctor <deck>` reports any configured provider whose package
    or API key is missing — with the exact `pip install` line to fix it.

The full project, source, and issue tracker live on
[GitHub](https://github.com/michael-borck/slide-stream).
