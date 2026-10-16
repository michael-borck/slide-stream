# Development workflow

How to work on SlideStream: setup, layout, rules, tests and releases.
Design decisions are in the [ADRs](adr/README.md); planned work is in the
[roadmap](ROADMAP.md).

## Setup

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/). FFmpeg is
bundled; LibreOffice is only needed to try `--preserve-style` by hand (the
tests mock it).

```bash
git clone https://github.com/michael-borck/slide-stream.git
cd slide-stream
uv sync --all-extras --group dev
uv run pytest && uv run ruff check . && uv run basedpyright
```

A plain `uv sync` installs only the core and removes extras from `.venv`;
use `--all-extras --group dev` for development.

On an exFAT drive macOS writes `._*` files beside every file. They are
git-ignored; code that globs files should skip them.

## Layout

```text
src/slide_stream/
  cli.py            Typer commands: create, doctor, draft, enrich, scan,
                    presenton, serve, init, providers, voices, avatars
  config_loader.py  Layered YAML config, starter and full reference templates
  parser.py         Markdown decks: content, code blocks, tables, Markdown body
  powerpoint.py     .pptx input, speaker notes
  narration.py      Narration source choice, scripts, stage directions
  llm.py            LLM clients
  media.py          Slide frames, presenter overlay, video assembly
  code_render.py    Pygments syntax colouring
  export_pptx.py    Plain .pptx output (ADR-0008)
  doctor.py         Preflight report
  draft.py          Document to Markdown deck
  enrich.py         Pictures and AI notes into a new deck
  scan.py           Name picture folders
  presenton.py      Presenton client and design review (ADR-0009, ADR-0013)
  fidelity.py       Designed deck vs source check (ADR-0013)
  code_slides.py    SlideStream-built code slides in designed decks
  slide_render.py   PDF and .pptx to frames (pypdfium2, LibreOffice)
  ffmpeg_bin.py     ffmpeg on PATH, else moviepy's bundled copy
  serve.py          FastAPI web UI (demo, local and hosted modes)
  web_security.py   Host checks and request limits for serve
  avatars.py        Built-in mascots
  providers/        base.py, factory.py, tts.py, images.py, avatar.py
tests/              Mocked suite; tests/live/ for real services
e2e/                Render matrix (run.sh) and the Presenton trial
deploy/             Docker image and compose files
desktop/            Tauri desktop shell
landing/            Website front page
docs/               This site (MkDocs)
```

## Rules

- **Zero basedpyright errors, ruff clean.** Use `# type: ignore[...]` only
  for missing stubs, with the specific code.
- **Tests first** for new behaviour; every bug fix gets a regression test.
- **Hermetic tests** ([ADR-0007](adr/0007-hermetic-tests-env-scrubbing.md)):
  no network, a fake home folder, provider variables scrubbed, the voice
  floor pinned to gTTS (`tests/conftest.py`). Mock HTTP at the session or
  `requests` boundary.
- **New features are a pipeline stage or a provider/adapter**
  ([ADR-0010](adr/0010-orchestrator-over-external-services.md),
  [ADR-0011](adr/0011-project-folder-pipeline-state.md)). Add a roadmap entry
  and, for decisions, an ADR.
- **Every role keeps a fallback**
  ([ADR-0012](adr/0012-three-level-fallback-kokoro-floor.md)); `--strict`
  must fail rather than fall back for a provider the user chose.
- **Dependencies are permissively licensed** (MIT, BSD, Apache); no AGPL.
  Check the licence before adding one, and record new external engines in
  [SERVICES.md](SERVICES.md).
- **Docs follow code.** Provider changes update [PROVIDERS.md](PROVIDERS.md)
  (a test checks every registered provider is listed); config changes go in
  `config_loader.py`'s template, then regenerate the example file (below).

### Adding a provider

1. Subclass the role's base class in `providers/base.py` (`TTSProvider`,
   `ImageProvider`, `AvatarProvider`) and implement `name`,
   `is_available()` and the role method.
2. Register it in `providers/factory.py` (the class map and the
   `list_*_providers` description).
3. Route per-call failures to the fallback (for TTS,
   `self._fallback_to_floor`), respecting strict mode.
4. Add mocked tests, a row in PROVIDERS.md, a commented example in the
   config template, and for a server a row in SERVICES.md.

## Tests

```bash
uv run pytest                          # mocked suite
uv run pytest tests/test_parser.py -q  # one file
uv run pytest --cov=slide_stream       # with coverage
uv run pytest --run-live tests/live    # real services from ~/.slidestream.yaml
e2e/run.sh fast                        # render matrix against your stack
```

Live tests and `e2e/` use your real config and servers and are never run in
CI. `e2e/presenton/run.sh` repeats the Presenton trial.

CI (`.github/workflows/ci.yml`) runs ruff, basedpyright and the suite with
coverage on Python 3.10 to 3.13.

## Regenerating generated files

```bash
# The full config reference (a test checks they match):
uv run python -c "from slide_stream.config_loader import create_example_config; open('slidestream.example.yaml','w').write(create_example_config())"

# Preview the docs site:
uvx --with mkdocs-material mkdocs serve
```

## Releases

Use `twine`, not `uv publish`.

1. Add the release to [CHANGELOG.md](CHANGELOG.md).
2. Bump the version in `pyproject.toml` and `src/slide_stream/__init__.py`;
   run `uv lock`.
3. `uv run pytest && uv run ruff check . && uv run basedpyright`
4. `uv build`, then `twine check dist/slide_stream-X.Y.Z*`.
5. `twine upload dist/slide_stream-X.Y.Z*`
6. Commit (`chore: release X.Y.Z (…)`), tag `vX.Y.Z` (annotated), push both.
7. `gh release create vX.Y.Z` with notes from the changelog.

Pushing a version change rebuilds the Docker image from PyPI. The desktop
app picks up the new engine on its own; bump `desktop/` only to raise its
minimum version or change the shell.
