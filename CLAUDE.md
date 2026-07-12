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
├── cli.py               # CLI (Typer): create, enrich, scan, init, voices, serve, avatars
├── config_loader.py     # Layered YAML config (defaults → ~/.slidestream.yaml → project → CLI)
├── llm.py               # LLM integration (OpenAI, Gemini, Claude, Groq, openai-compatible)
├── media.py             # Image/audio/video assembly (moviepy)
├── parser.py            # Markdown parsing (heading-style and ---separator decks)
├── powerpoint.py        # PowerPoint (.pptx) parsing incl. speaker notes
├── narration.py         # Narration script (--script) parsing
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
    └── avatar.py        # Avatars: static/puppet, D-ID, ComfyUI/SadTalker
```
Also: `deploy/` (Docker), `desktop/` (Tauri app), `landing/` (website), `contrib/voicebox/` (server-side sweep script), `docs/`.

### Type Safety
- Project maintains zero basedpyright errors
- Uses type guards for dynamic content (BeautifulSoup, etc.)
- Strategic `# type: ignore` for missing type stubs
- See `TYPE_SAFETY.md` for detailed documentation

### Testing
- Comprehensive test coverage for all modules
- CLI tests use `typer.testing.CliRunner`
- PowerPoint tests create temporary .pptx files
- Tests avoid network calls (default config uses the `text` image provider; TTS is mocked at the provider/gTTS boundary)

### CLI Design
- Subcommands: `create` (main render), `enrich`, `scan`, `init`, `voices`, `serve`, `avatars`
- Supports both `.md` and `.pptx` input files (detection by extension)
- Behavior driven by layered YAML config (`config_loader.py`); CLI flags override config
- `--strict` must fail (not silently fall back) when a configured provider errors
- Rich progress bars and error formatting

## Documentation Structure

- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**: Comprehensive user guide with examples
- **[docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md)**: Development workflow and release process
- **[docs/TYPE_SAFETY.md](docs/TYPE_SAFETY.md)**: Type safety documentation
- **[docs/TYPING_IMPROVEMENTS.md](docs/TYPING_IMPROVEMENTS.md)**: Type improvement roadmap
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
4. Update README.md with examples
5. Run full test suite
6. Check type coverage: `basedpyright`

### Debugging
- Set `providers.images.provider: text` (config) to avoid network calls during testing
- Run coverage explicitly when you need it (`uv run pytest --cov`); don't trust a stale `coverage.xml`
- Use Rich console for better error formatting

### Dependencies
- Core: `typer`, `rich`, `moviepy`, `numpy`, `pillow`, `beautifulsoup4`, `python-pptx`, `gtts`, `pyyaml`, `requests`
- Optional extras: `[openai]`, `[gemini]`, `[claude]`, `[groq]`, `[elevenlabs]`, `[local-tts]`, `[serve]`, `[all-ai]`
- Development (uv dependency group `dev`): `pytest`, `ruff`, `basedpyright`, `twine`

## Important Notes

- **Never use `uv publish`** - always use `twine upload`
- PowerPoint speaker notes are used for enhanced AI narration
- All tests must pass before release
- Type checker must show zero errors
- Follow semantic versioning for releases