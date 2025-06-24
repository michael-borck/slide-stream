# SlideStream Development Workflow

This document outlines the development workflow for maintaining and extending SlideStream.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Standards](#code-standards)
4. [Testing Workflow](#testing-workflow)
5. [Release Process](#release-process)
6. [Adding New Features](#adding-new-features)
7. [Debugging Guide](#debugging-guide)
8. [Contributing Guidelines](#contributing-guidelines)

## Project Structure

```
slide-stream/
├── src/slide_stream/           # Main package
│   ├── __init__.py            # Version and package metadata
│   ├── cli.py                 # CLI interface (Typer-based)
│   ├── config.py              # Configuration constants
│   ├── llm.py                 # AI/LLM integration
│   ├── media.py               # Image/audio/video processing
│   ├── parser.py              # Markdown parsing
│   └── powerpoint.py          # PowerPoint (.pptx) parsing
├── tests/                     # Test suite
│   ├── fixtures/              # Test data files
│   ├── test_cli.py           # CLI integration tests
│   ├── test_parser.py        # Markdown parser tests
│   └── test_powerpoint.py    # PowerPoint parser tests
├── docs/                      # Documentation
│   ├── USER_GUIDE.md         # Comprehensive user guide
│   ├── TYPE_SAFETY.md        # Type safety documentation
│   └── TYPING_IMPROVEMENTS.md # Type improvement roadmap
├── dist/                      # Build artifacts (auto-generated)
├── htmlcov/                   # Coverage reports (auto-generated)
├── pyproject.toml            # Project configuration
├── uv.lock                   # Dependency lock file
├── CLAUDE.md                 # AI assistant notes
└── README.md                 # Project overview
```

## Development Environment Setup

### Prerequisites

- **Python 3.10+**: Required for modern type hints
- **uv**: Package manager and build tool
- **FFmpeg**: Required for video processing

### Initial Setup

1. **Clone and enter project**:
```bash
git clone https://github.com/michael-borck/slide-stream.git
cd slide-stream
```

2. **Install dependencies**:
```bash
uv sync
```

3. **Activate virtual environment**:
```bash
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

4. **Verify installation**:
```bash
python -m pytest
basedpyright
ruff check
```

### IDE Configuration

#### VS Code
Recommended extensions:
- Python
- Pylance (Microsoft)
- Ruff

Settings (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "ruff",
  "python.typeChecking": "basic"
}
```

## Code Standards

### Type Safety

- **Zero basedpyright errors** is required
- Use type hints for all function parameters and return values
- Use type guards for dynamic content (BeautifulSoup, JSON, etc.)
- Strategic `# type: ignore` only for missing type stubs

Example:
```python
from typing import TypeGuard

def is_tag_with_text(element: Any) -> TypeGuard[Tag]:
    """Type guard to check if element is a BeautifulSoup Tag with text."""
    return hasattr(element, 'get_text') and callable(element.get_text)
```

### Code Style

- **Ruff** for linting and formatting
- **Line length**: 88 characters (Black-compatible)
- **Import sorting**: Automatic with ruff
- **Docstrings**: Required for all public functions

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Use Rich console for user-facing errors

Example:
```python
try:
    slides = parse_powerpoint(input_path)
except ValueError as e:
    err_console.print(f"Error parsing PowerPoint: {e}")
    raise typer.Exit(code=1)
```

## Testing Workflow

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src/slide_stream

# Run specific test file
python -m pytest tests/test_cli.py

# Run specific test
python -m pytest tests/test_cli.py::test_cli_version -v
```

### Test Structure

#### Unit Tests
- Test individual functions and classes
- Mock external dependencies (APIs, file system)
- Focus on business logic

#### Integration Tests
- Test CLI commands end-to-end
- Use fixtures for test data
- Avoid network calls in CI

#### Test Guidelines

1. **Use fixtures for test data**:
```python
@pytest.fixture
def sample_markdown():
    return """# Test Slide
- Point 1
- Point 2
"""
```

2. **Mock external services**:
```python
@pytest.fixture
def mock_llm_response():
    with patch('slide_stream.llm.query_llm') as mock:
        mock.return_value = "Enhanced content"
        yield mock
```

3. **Use `--image-source text` for tests**:
```python
result = runner.invoke(app, [
    "--input", "test.md",
    "--output", "test.mp4",
    "--image-source", "text"  # Avoid network calls
])
```

### Coverage Requirements

- **Minimum**: 80% line coverage
- **Target**: 90%+ line coverage
- **Focus**: Critical paths and error handling

## Release Process

### Version Management

Versions follow [semantic versioning](https://semver.org/):
- **Major**: Breaking changes (e.g., 1.0.0 → 2.0.0)
- **Minor**: New features (e.g., 1.0.0 → 1.1.0)
- **Patch**: Bug fixes (e.g., 1.0.0 → 1.0.1)

### Release Steps

1. **Update version** in two places:
```bash
# pyproject.toml
version = "1.2.0"

# src/slide_stream/__init__.py
__version__ = "1.2.0"
```

2. **Update version in tests**:
```python
# tests/test_cli.py
assert "1.2.0" in result.stdout
```

3. **Run full test suite**:
```bash
python -m pytest
basedpyright
ruff check
```

4. **Build package**:
```bash
uv build
```

5. **Commit and tag**:
```bash
git add .
git commit -m "Release version 1.2.0"
git tag v1.2.0
git push origin main --tags
```

6. **Upload to PyPI** (use twine, not uv):
```bash
twine upload dist/slide_stream-1.2.0*
```

### Release Checklist

- [ ] Version updated in `pyproject.toml` and `__init__.py`
- [ ] Tests updated for new version
- [ ] All tests passing
- [ ] Zero type checker errors
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if exists)
- [ ] Built with `uv build`
- [ ] Uploaded with `twine upload`
- [ ] Git tagged and pushed

## Adding New Features

### Feature Development Workflow

1. **Create feature branch**:
```bash
git checkout -b feature/new-feature-name
```

2. **Write tests first** (TDD approach):
```python
# tests/test_new_feature.py
def test_new_feature():
    result = new_feature("input")
    assert result == "expected_output"
```

3. **Implement feature**:
```python
# src/slide_stream/new_module.py
def new_feature(input_data: str) -> str:
    """Implement new feature with proper types."""
    return process_input(input_data)
```

4. **Update CLI if needed**:
```python
# src/slide_stream/cli.py
@app.command()
def new_command(
    option: Annotated[str, typer.Option(help="Description")]
):
    """Add new CLI command."""
    pass
```

5. **Add documentation**:
```markdown
<!-- docs/USER_GUIDE.md -->
### New Feature

Description of the new feature...
```

6. **Test thoroughly**:
```bash
python -m pytest tests/test_new_feature.py -v
python -m pytest  # All tests
basedpyright      # Type checking
```

### Adding New AI Providers

Example: Adding a new AI provider

1. **Update `llm.py`**:
```python
def get_llm_client(provider: str) -> Any:
    if provider == "new_provider":
        try:
            import new_provider_sdk
            return new_provider_sdk.Client()
        except ImportError:
            raise ImportError("pip install slide-stream[new_provider]")
    # ... existing providers
```

2. **Add to optional dependencies**:
```toml
# pyproject.toml
[project.optional-dependencies]
new_provider = ["new-provider-sdk>=1.0.0"]
all-ai = [
    "openai>=1.0.0",
    "anthropic>=0.7.0",
    "groq>=0.4.0",
    "google-generativeai>=0.3.0",
    "new-provider-sdk>=1.0.0",
]
```

3. **Update documentation**:
```markdown
<!-- docs/USER_GUIDE.md -->
#### New Provider
```bash
export NEW_PROVIDER_API_KEY="your-key"
pip install slide-stream[new_provider]
slide-stream -i slides.md -o video.mp4 --llm-provider new_provider
```

4. **Add tests**:
```python
def test_new_provider_integration():
    # Test the new provider
    pass
```

## Debugging Guide

### Common Issues

#### 1. Type Checker Errors
```bash
# Run basedpyright to see type issues
basedpyright

# Common fixes:
# - Add type hints
# - Use type guards for dynamic content
# - Add # type: ignore for external libraries
```

#### 2. Test Failures
```bash
# Run specific failing test with verbose output
python -m pytest tests/test_file.py::test_function -v -s

# Debug with pdb
python -m pytest tests/test_file.py::test_function --pdb
```

#### 3. Import Errors
```bash
# Verify package structure
python -c "import slide_stream; print(slide_stream.__version__)"

# Check uv environment
uv run python -c "import slide_stream"
```

#### 4. CLI Issues
```bash
# Test CLI directly
python -m slide_stream.cli --help

# Debug with rich tracebacks
export PYTHONPATH=src
python -c "
from slide_stream.cli import app
import typer
app()
"
```

### Debugging Tools

#### 1. Rich Console Debug
```python
from rich.console import Console
console = Console()
console.print("Debug info", style="red")
console.print_exception()  # Pretty tracebacks
```

#### 2. Type Checking Debug
```bash
# Verbose type checking
basedpyright --verbose

# Check specific file
basedpyright src/slide_stream/cli.py
```

#### 3. Test Debug
```python
# Add debug prints in tests
def test_something():
    result = function_under_test()
    print(f"Debug: {result}")  # Will show with -s flag
    assert result == expected
```

## Contributing Guidelines

### Pull Request Process

1. **Fork and clone**
2. **Create feature branch**
3. **Make changes with tests**
4. **Ensure all checks pass**:
   - `python -m pytest`
   - `basedpyright`
   - `ruff check`
5. **Submit pull request**

### Code Review Checklist

- [ ] Tests added/updated
- [ ] Type hints added
- [ ] Documentation updated
- [ ] No new basedpyright errors
- [ ] Follows existing code style
- [ ] CLI help text updated (if applicable)

### Commit Message Format

```
Add PowerPoint speaker notes support

- Extract speaker notes from PPTX files
- Use notes for enhanced AI narration
- Add comprehensive test coverage
- Update documentation with examples

Fixes #123
```

### Issue Guidelines

When reporting issues:

1. **Environment**: Python version, OS, SlideStream version
2. **Reproduction**: Minimal example to reproduce
3. **Expected vs Actual**: What you expected vs what happened
4. **Logs**: Any error messages or stack traces

Example issue:
```markdown
## Bug Report

**Environment:**
- SlideStream: 1.1.0
- Python: 3.11
- OS: macOS 14

**Steps to Reproduce:**
1. Create file `test.md` with content...
2. Run `slide-stream -i test.md -o video.mp4`
3. Error occurs

**Expected:** Video should be generated
**Actual:** Error: "No slides found"

**Error Log:**
```
[error messages here]
```
```

---

This development workflow ensures consistent, high-quality code and smooth collaboration. For questions, see [CLAUDE.md](../CLAUDE.md) for AI assistant guidelines.