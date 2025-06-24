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
├── cli.py              # Main CLI interface (Typer)
├── config.py           # Configuration constants
├── llm.py              # LLM integration (OpenAI, Gemini, Claude, etc.)
├── media.py            # Image/audio/video processing
├── parser.py           # Markdown parsing
└── powerpoint.py       # PowerPoint (.pptx) parsing
```

### Type Safety
- Project maintains zero basedpyright errors
- Uses type guards for dynamic content (BeautifulSoup, etc.)
- Strategic `# type: ignore` for missing type stubs
- See `TYPE_SAFETY.md` for detailed documentation

### Testing
- Comprehensive test coverage for all modules
- CLI tests use `typer.testing.CliRunner`
- PowerPoint tests create temporary .pptx files
- Tests avoid network calls (use `--image-source text`)

### CLI Design
- Single main command (no subcommands)
- Supports both `.md` and `.pptx` input files
- File type detection by extension
- Rich progress bars and error formatting

## Documentation Structure

- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**: Comprehensive user guide with examples
- **[docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md)**: Development workflow and release process
- **[docs/TYPE_SAFETY.md](docs/TYPE_SAFETY.md)**: Type safety documentation
- **[docs/TYPING_IMPROVEMENTS.md](docs/TYPING_IMPROVEMENTS.md)**: Type improvement roadmap
- **[tests/fixtures/](tests/fixtures/)**: Test data files

## Version History

- **1.0.0**: Initial release with Markdown support
- **1.1.0**: Added PowerPoint (.pptx) support with speaker notes

## Common Tasks

### Adding New Features
1. Write tests first (TDD approach)
2. Implement feature with type safety
3. Update CLI help text if needed
4. Update README.md with examples
5. Run full test suite
6. Check type coverage: `basedpyright`

### Debugging
- Use `--image-source text` to avoid network calls during testing
- Check `coverage.xml` for test coverage gaps
- Use Rich console for better error formatting

### Dependencies
- Core: `typer`, `rich`, `moviepy`, `pillow`, `beautifulsoup4`, `python-pptx`
- Optional AI providers via extras: `[openai]`, `[gemini]`, `[claude]`, `[groq]`, `[all-ai]`
- Development: `pytest`, `ruff`, `basedpyright`, `twine`

## Important Notes

- **Never use `uv publish`** - always use `twine upload`
- PowerPoint speaker notes are used for enhanced AI narration
- All tests must pass before release
- Type checker must show zero errors
- Follow semantic versioning for releases