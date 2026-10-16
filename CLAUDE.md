# Claude Development Notes

This file contains notes and instructions for Claude (AI assistant) when working on this project.

## Project Overview

SlideStream is an AI-powered tool to create video presentations from Markdown and PowerPoint files. The project uses modern Python tooling and has zero type checker errors.

## Development Environment

- **Package Manager**: uv (for dependency management, virtual environment, building)
- **Type Checker**: basedpyright (configured for zero errors)
- **Linter/Formatter**: ruff (configured for modern Python standards)
- **Testing**: pytest with coverage reporting
- **CI/CD**: GitHub Actions (if configured)

## PyPI Publishing Workflow

**Important**: Use `twine` for PyPI uploads, not `uv publish`. The user has a `.pypirc` file configured that `uv` cannot use.

### Publishing Steps:

0. **Update `docs/CHANGELOG.md`** with the release's changes.

1. **Update Version**: 
   ```bash
   # Update version in both files:
   # - pyproject.toml: version = "x.y.z"
   # - src/slide_stream/__init__.py: __version__ = "x.y.z"
   ```

2. **Run Tests**:
   ```bash
   python -m pytest
   ```

3. **Build Package**:
   ```bash
   uv build
   ```

4. **Upload with Twine** (not uv):
   ```bash
   twine upload dist/slide_stream-x.y.z*
   ```

5. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Release version x.y.z"
   git push
   ```

## Key Architecture Notes

### File Structure
```
src/slide_stream/
├── __init__.py          # Version and package info
├── __main__.py          # python -m slide_stream entry point
├── cli.py               # CLI (Typer): create, doctor, draft, enrich, scan, presenton, serve, init, providers, voices, avatars
├── config_loader.py     # Layered YAML config (defaults → ~/.slidestream.yaml → project → CLI)
├── llm.py               # LLM integration (OpenAI, Gemini, Claude, Groq, openai-compatible)
├── media.py             # Image/audio/video assembly (moviepy)
├── parser.py            # Markdown parsing (heading-style and ---separator decks; fenced code + tables; per-slide `content` text and `markdown` body)
├── powerpoint.py        # PowerPoint (.pptx) parsing incl. speaker notes
├── narration.py         # Narration source choice, --script parsing, stage directions
├── doctor.py            # Preflight report (`doctor`, `create --dry-run`)
├── draft.py             # `draft`: document -> Markdown deck
├── export_pptx.py       # Plain .pptx output (ADR-0008)
├── web_security.py      # Host checks and request limits for serve
├── presenton.py         # Client for an external Presenton (designed decks; ADR-0009)
├── slide_render.py      # Designed slides -> pictures (pypdfium2; LibreOffice for .pptx)
├── fidelity.py          # Designed deck vs source: code/title/text/table kept, possibly invented (ADR-0013)
├── code_render.py       # Pygments syntax highlighting for frames and .pptx runs
├── code_slides.py       # SlideStream-built code slides in designed decks: theme sampling, frame, .pptx rewrite
├── ffmpeg_bin.py        # ffmpeg on PATH, else moviepy's bundled binary
├── enrich.py            # `enrich` command: add AI images to a deck
├── scan.py              # `scan` command: AI-rename image folders
├── serve.py             # FastAPI web UI (try-before-install demo + desktop backend)
├── avatars.py           # Built-in avatar mascot registry
├── avatar_images/       # Bundled avatar JPGs (shipped in the wheel)
└── providers/
    ├── base.py          # Provider protocols, strict mode (StrictModeError)
    ├── factory.py       # Provider selection/registry
    ├── tts.py           # TTS: gtts, kokoro, chatterbox, voicebox, elevenlabs, openai(-compat)
    ├── images.py        # Images: text, local, pexels, unsplash, dalle3, gemini, swarmui, ...
    └── avatar.py        # Avatars: static, puppet, precomputed, D-ID, ComfyUI (wan-s2v, SadTalker, Wav2Lip)
```
Also: `deploy/` (Docker), `desktop/` (Tauri app), `landing/` (website), `contrib/` (Voicebox and Chatterbox server-side cleanup scripts), `e2e/`, `docs/`.

### Type Safety
- Project maintains zero basedpyright errors
- Uses type guards for dynamic content (BeautifulSoup, etc.)
- Specific `# type: ignore[...]` only for missing type stubs
- Rules: docs/DEVELOPMENT_WORKFLOW.md

### Testing
- Comprehensive test coverage for all modules
- CLI tests use `typer.testing.CliRunner`
- PowerPoint tests create temporary .pptx files
- Tests avoid network calls (ADR-0007): fake home, provider env vars scrubbed, voice floor pinned to gTTS (tests/conftest.py)
- tests/test_docs.py checks every registered provider is in docs/PROVIDERS.md and README links are absolute

### CLI Design
- Subcommands: `create` (main render), `doctor`, `draft`, `enrich`, `scan`, `init`, `providers`, `voices`, `serve`, `avatars`, `presenton` (designed deck via an external Presenton; `create --template`/`--preserve-style`/`--slides-pdf` put designed slides into the video — ADR-0009)
- Supports both `.md` and `.pptx` input files (detection by extension)
- Behavior driven by layered YAML config (`config_loader.py`); CLI flags override config
- `--strict` must fail (not silently fall back) when a configured provider errors
- Rich progress bars and error formatting

## Documentation Structure

- **README.md**: also the PyPI page; keep it short and use absolute URLs only
- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**: Reference (install, config, deck format, commands)
- **[docs/USE_CASES.md](docs/USE_CASES.md)**: Task recipes
- **[docs/PROVIDERS.md](docs/PROVIDERS.md)**: Every provider per role; update when adding one
- **[docs/CHANGELOG.md](docs/CHANGELOG.md)**: Update with every release
- **[docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md)**: Development workflow and release process
- **[docs/adr/](docs/adr/)**: Architecture Decision Records (backfilled; read before changing provider/config/test design)
- **[docs/USER_STORIES.md](docs/USER_STORIES.md)**: Personas and stories with status; every story belongs to a roadmap phase or future work
- **[docs/ROADMAP.md](docs/ROADMAP.md)**: Phased plan. New features must be a pipeline stage (ADR-0011) or a service adapter (ADR-0010)
- **[docs/SERVICES.md](docs/SERVICES.md)**: Every external engine, its probe, tested version and licence; update it when adding an adapter
- **docs/archive/**: superseded plans and notes, not maintained
- **[e2e/](e2e/)**: End-to-end render matrix against the configured stack (`e2e/run.sh`)
- **[tests/fixtures/](tests/fixtures/)**: Test data files

## Version History (milestones)

- **1.x**: Markdown support, then PowerPoint (.pptx) with speaker notes
- **2.x**: Layered YAML config, provider architecture (TTS/images/LLM/avatar), `enrich`/`scan`/`serve` commands, avatars, web UI + Docker deploy, desktop app
- Current version lives in `pyproject.toml` / `src/slide_stream/__init__.py`; see git tags for the full history

## Common Tasks

### Adding New Features
1. Write tests first (TDD approach)
2. Implement feature with type safety
3. Update CLI help text if needed
4. Update USER_GUIDE/USE_CASES/PROVIDERS (README only for headline features)
5. Run full test suite
6. Check type coverage: `basedpyright`

### Debugging
- Set `providers.images.provider: text` (config) to avoid network calls during testing
- Run coverage explicitly when you need it (`uv run pytest --cov`); don't trust a stale `coverage.xml`
- Use Rich console for better error formatting

### Dependencies
- Core: `typer`, `rich`, `moviepy`, `numpy`, `pillow`, `markdown`, `beautifulsoup4`, `python-pptx`, `gtts`, `pyyaml`, `requests`, `pygments`
- Optional extras: `[openai]`, `[gemini]`, `[claude]`, `[groq]`, `[elevenlabs]`, `[local-tts]`, `[serve]`, `[import]`, `[presenton]` (pypdfium2 slide rendering; avoid AGPL deps such as PyMuPDF), `[all-ai]`, `[all]` (everything)
- Development (uv dependency group `dev`): `pytest`, `ruff`, `basedpyright`, `twine`

## Important Notes

- **Never use `uv publish`** - always use `twine upload`
- PowerPoint speaker notes are used for enhanced AI narration
- All tests must pass before release
- Type checker must show zero errors
- Follow semantic versioning for releases