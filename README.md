# SlideStream

Turn a Markdown file, a PowerPoint deck or a document into a narrated video
presentation, an editable PowerPoint, or both.

[![PyPI version](https://img.shields.io/pypi/v/slide-stream.svg)](https://pypi.org/project/slide-stream/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/michael-borck/slide-stream/blob/main/LICENSE)
[![CI](https://github.com/michael-borck/slide-stream/actions/workflows/ci.yml/badge.svg)](https://github.com/michael-borck/slide-stream/actions/workflows/ci.yml)

**[Website](https://slidestream.borck.education)** ·
**[Documentation](https://slidestream.borck.education/docs/)** ·
**[Desktop app](https://github.com/michael-borck/slide-stream/releases)** ·
**[Changelog](https://slidestream.borck.education/docs/CHANGELOG/)**

SlideStream writes narration that a presenter would say (not the bullets read
aloud), voices it, optionally in your own cloned voice, adds pictures and a
presenter in the corner, and assembles the video. It works with no account
and no API keys: the voice runs offline, and every AI service is optional.

## What it does

- **Sources:** Markdown decks (code blocks and tables included), PowerPoint
  files with speaker notes, or a text, Word or PDF document turned into a
  deck with `draft` (the web UI also starts from a one-line idea).
- **Narration:** written by an LLM of your choice (local Ollama or a cloud
  model), taken from speaker notes, or from your own script.
- **Voice:** offline Kokoro by default; your own cloned voice through a
  self-hosted Voicebox or Chatterbox server; or ElevenLabs and OpenAI.
- **Pictures:** text cards, your own folder, stock photos, or AI images from
  a local SwarmUI server or a cloud service.
- **Presenter:** a mascot or photo in the corner, from a still image to full
  lip-sync on your own GPU.
- **Outputs:** an MP4 video, or an editable `.pptx` with the narration in the
  speaker notes.
- **Design (experimental):** designed slides through a self-hosted
  [Presenton](https://github.com/presenton/presenton), with a check that the
  design kept your content; or keep your own PowerPoint's look.
- **Check first:** `slide-stream doctor` estimates length, cost and render
  time and warns about problems before you render.

Interfaces: the `slide-stream` command, a web UI (`slide-stream serve`), a
desktop app for macOS, Windows and Linux, and a Docker image for hosting.

## Install

```bash
pip install "slide-stream[all]"
```

`[all]` includes every AI provider, the offline voice, the web UI and
document import. Python 3.10 or newer. FFmpeg comes with the package; a
system install is optional. Smaller installs and every extra are listed in
the [user guide](https://slidestream.borck.education/docs/USER_GUIDE/#install).

Prefer no terminal? Download the
[desktop app](https://github.com/michael-borck/slide-stream/releases).

## Quick start

Write a deck. Each `#` heading starts a slide:

````markdown
# Hash Tables

- Find a value without searching
- Week 6, Data Structures

# A Hash Function

```python
def bucket_for(key: str, n: int) -> int:
    return hash(key) % n
```
````

Render it:

```bash
slide-stream create deck.md talk.mp4      # narrated video
slide-stream create deck.md deck.pptx     # editable PowerPoint, narration in the notes
slide-stream create slides.pptx talk.mp4  # narrate an existing PowerPoint
```

That works as is. To go further, write a config file and switch providers on:

```bash
slide-stream init         # a starter slidestream.yaml that works with no keys
slide-stream doctor deck.md
```

```yaml
# slidestream.yaml
providers:
  llm:
    provider: ollama        # writes the narration; or claude, openai, gemini, ...
    model: qwen3:4b
  images:
    provider: swarmui       # or text, local, pexels, dalle3, gemini, ...
    base_url: http://localhost:7801
  tts:
    provider: voicebox      # your cloned voice; falls back to Kokoro offline
    base_url: http://localhost:17493
    voice_sample: ./my_voice.wav
```

API keys come from the environment, never the file.

## Commands

| Command | Does |
|---|---|
| `create IN OUT` | Render a video (`.mp4`) or a deck (`.pptx`) |
| `doctor IN` | Preflight: estimates and warnings, no rendering |
| `draft DOC OUT.md` | Turn a text, Word, PDF or PowerPoint file into a Markdown deck |
| `enrich IN DIR` | Add a picture (and optionally AI speaker notes) to every slide |
| `presenton IN OUT` | Designed deck through Presenton (experimental) |
| `scan DIR` | Name a folder of pictures so slides can find them |
| `serve` | Web UI in the browser |
| `init` | Write a config file |
| `providers`, `voices`, `avatars` | List what is available |

`slide-stream COMMAND --help` lists every option.

## Documentation

- [Use cases](https://slidestream.borck.education/docs/USE_CASES/): recipes
  for common jobs (your own voice, a presenter, PowerPoint with notes,
  designed decks, self-hosting)
- [User guide](https://slidestream.borck.education/docs/USER_GUIDE/):
  install, configuration, deck format, every command
- [Providers](https://slidestream.borck.education/docs/PROVIDERS/): every
  voice, picture, LLM, presenter and design option
- [Services](https://slidestream.borck.education/docs/SERVICES/): the
  self-hosted engines, their licences and how they are checked
- [Roadmap](https://slidestream.borck.education/docs/ROADMAP/) and
  [user stories](https://slidestream.borck.education/docs/USER_STORIES/)
- [Hosting with Docker](https://github.com/michael-borck/slide-stream/blob/main/deploy/README.md)

## Privacy

Everything can run on your own machine or servers. A voice sample or photo
you upload is used for that render only and then deleted, on your server and
in the web UI. The only network calls are to the services you configure.

## Contributing

Development uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/michael-borck/slide-stream.git
cd slide-stream
uv sync --all-extras --group dev
uv run pytest && uv run ruff check . && uv run basedpyright
```

See the
[development workflow](https://slidestream.borck.education/docs/DEVELOPMENT_WORKFLOW/)
and the [architecture decisions](https://slidestream.borck.education/docs/adr/).

## Acknowledgements

SlideStream builds on [MoviePy](https://zulko.github.io/moviepy/),
[python-pptx](https://python-pptx.readthedocs.io/),
[Kokoro](https://github.com/thewh1teagle/kokoro-onnx),
[Pygments](https://pygments.org/), [pypdfium2](https://github.com/pypdfium2-team/pypdfium2),
[Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/),
and can call [Presenton](https://github.com/presenton/presenton),
[Voicebox](https://slidestream.borck.education/docs/voicebox-api/),
[ComfyUI](https://github.com/comfyanonymous/ComfyUI),
[SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) and
[Ollama](https://ollama.com). Credits and licences for every engine are in
[SERVICES](https://slidestream.borck.education/docs/SERVICES/).

## License

MIT. See [LICENSE](https://github.com/michael-borck/slide-stream/blob/main/LICENSE).
