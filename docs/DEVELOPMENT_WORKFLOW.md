# SlideStream 2.0 Development Workflow

This document outlines the development workflow for maintaining and extending SlideStream's modern provider architecture.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Standards](#code-standards)
4. [Testing Workflow](#testing-workflow)
5. [Release Process](#release-process)
6. [Adding New Features](#adding-new-features)
7. [Provider Development](#provider-development)
8. [Debugging Guide](#debugging-guide)
9. [Contributing Guidelines](#contributing-guidelines)

## Project Structure

```
slide-stream/
├── src/slide_stream/              # Main package
│   ├── __init__.py               # Version and package metadata
│   ├── cli.py                    # Modern CLI interface (Typer-based)
│   ├── config_loader.py          # YAML configuration system
│   ├── llm.py                    # LLM integration (legacy)
│   ├── media.py                  # Video processing utilities
│   ├── parser.py                 # Markdown parsing
│   ├── powerpoint.py             # PowerPoint (.pptx) parsing
│   └── providers/                # Provider system
│       ├── __init__.py
│       ├── base.py               # Abstract provider interfaces
│       ├── factory.py            # Provider factory and management
│       ├── images.py             # Image provider implementations
│       └── tts.py                # TTS provider implementations
├── tests/                        # Test suite
│   ├── fixtures/                 # Test data files
│   ├── test_cli.py              # CLI integration tests
│   ├── test_parser.py           # Markdown parser tests
│   └── test_powerpoint.py       # PowerPoint parser tests
├── docs/                         # Documentation
│   ├── USER_GUIDE.md            # Comprehensive user guide
│   ├── TYPE_SAFETY.md           # Type safety documentation
│   └── TYPING_IMPROVEMENTS.md   # Type improvement roadmap
├── dist/                         # Build artifacts (auto-generated)
├── htmlcov/                      # Coverage reports (auto-generated)
├── pyproject.toml               # Project configuration
├── uv.lock                      # Dependency lock file
├── CLAUDE.md                    # AI assistant development notes
└── README.md                    # Project overview
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
uv sync --dev
```

3. **Activate virtual environment**:
```bash
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

4. **Verify installation**:
```bash
uv run pytest
uv run basedpyright
uv run ruff check
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
  "python.typeChecking": "basic",
  "python.analysis.typeCheckingMode": "basic"
}
```

## Code Standards

### Architecture Principles

SlideStream 2.0 follows these principles:

1. **Configuration-First**: All behavior controlled via YAML config
2. **Provider Pattern**: Pluggable implementations for images, TTS, LLM
3. **Graceful Degradation**: Fallbacks when premium services unavailable
4. **Type Safety**: Zero basedpyright errors required

### Type Safety

- **Zero basedpyright errors** is required
- Use type hints for all function parameters and return values
- Use type guards for dynamic content (BeautifulSoup, JSON, etc.)
- Strategic `# type: ignore` only for missing type stubs

Example:
```python
from typing import TypeGuard, Dict, Any

def is_valid_config(data: Any) -> TypeGuard[Dict[str, Any]]:
    """Type guard to validate configuration structure."""
    return isinstance(data, dict) and "providers" in data
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
from rich.console import Console

err_console = Console(stderr=True, style="bold red")

try:
    config = load_config(config_file)
except ConfigurationError as e:
    err_console.print(f"Configuration Error: {e}")
    raise typer.Exit(code=1)
```

## Testing Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/slide_stream

# Run specific test file
uv run pytest tests/test_cli.py

# Run specific test
uv run pytest tests/test_cli.py::test_cli_version -v
```

### Test Structure

#### Unit Tests
- Test individual functions and classes
- Mock external dependencies (APIs, file system)
- Focus on business logic

#### Integration Tests
- Test CLI commands end-to-end
- Use fixtures for test data
- Test configuration loading and provider selection

#### Provider Tests
- Test each provider implementation
- Mock API calls to avoid network dependencies
- Test fallback behavior

### Test Guidelines

1. **Use fixtures for test data**:
```python
@pytest.fixture
def sample_config():
    return {
        "providers": {
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "images": {"provider": "text", "fallback": "text"},
            "tts": {"provider": "gtts"}
        }
    }
```

2. **Mock external services**:
```python
@pytest.fixture
def mock_dalle_response():
    with patch('slide_stream.providers.images.OpenAI') as mock:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(url="http://example.com/image.jpg")]
        mock_client.images.generate.return_value = mock_response
        mock.return_value = mock_client
        yield mock
```

3. **Test CLI with new format**:
```python
def test_cli_create_basic(runner, sample_markdown):
    result = runner.invoke(app, [
        "create", "test.md", "output.mp4"
    ])
    assert result.exit_code == 0
```

### Coverage Requirements

- **Minimum**: 80% line coverage
- **Target**: 90%+ line coverage
- **Focus**: Critical paths and error handling

## Release Process

### Version Management

Versions follow [semantic versioning](https://semver.org/):
- **Major**: Breaking changes (e.g., 1.x.x → 2.0.0)
- **Minor**: New features (e.g., 2.0.0 → 2.1.0)
- **Patch**: Bug fixes (e.g., 2.0.0 → 2.0.1)

### Release Steps

1. **Update version** in two places:
```bash
# pyproject.toml
version = "2.1.0"

# src/slide_stream/__init__.py
__version__ = "2.1.0"
```

2. **Update version in tests**:
```python
# tests/test_cli.py
assert "2.1.0" in result.stdout
```

3. **Run full test suite**:
```bash
uv run pytest
uv run basedpyright
uv run ruff check
```

4. **Build package**:
```bash
uv build
```

5. **Upload to PyPI** (use twine, not uv):
```bash
uv run twine upload dist/slide_stream-2.1.0*
```

6. **Commit and tag**:
```bash
git add .
git commit -m "Release version 2.1.0"
git tag v2.1.0
git push origin main --tags
```

### Release Checklist

- [ ] Version updated in `pyproject.toml` and `__init__.py`
- [ ] Tests updated for new version
- [ ] All tests passing
- [ ] Zero type checker errors
- [ ] Documentation updated
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
uv run pytest tests/test_new_feature.py -v
uv run pytest  # All tests
uv run basedpyright      # Type checking
```

## Provider Development

### Adding New Providers

SlideStream 2.0's provider system makes it easy to add new services.

#### Adding an Image Provider

1. **Create provider class**:
```python
# src/slide_stream/providers/images.py
class NewImageProvider(ImageProvider):
    @property
    def name(self) -> str:
        return "new_provider"
    
    def is_available(self) -> bool:
        api_keys = self.config.get("api_keys", {})
        return bool(api_keys.get("new_provider"))
    
    def generate_image(self, query: str, filename: str) -> str:
        # Implementation here
        pass
```

2. **Register in factory**:
```python
# src/slide_stream/providers/factory.py
IMAGE_PROVIDERS: Dict[str, Type[ImageProvider]] = {
    "text": TextImageProvider,
    "dalle3": DalleImageProvider,
    "new_provider": NewImageProvider,  # Add here
}
```

3. **Add to configuration schema**:
```python
# src/slide_stream/config_loader.py
# Update documentation and validation
```

4. **Add optional dependency**:
```toml
# pyproject.toml
[project.optional-dependencies]
new_provider = ["new-provider-sdk>=1.0.0"]
all-ai = [
    # existing providers...
    "new-provider-sdk>=1.0.0",
]
```

#### Adding a TTS Provider

1. **Create provider class**:
```python
# src/slide_stream/providers/tts.py
class NewTTSProvider(TTSProvider):
    @property
    def name(self) -> str:
        return "new_tts"
    
    def is_available(self) -> bool:
        # Check API key availability
        pass
    
    def synthesize(self, text: str, filename: str) -> Optional[str]:
        # Implementation here
        pass
```

2. **Register in factory**:
```python
# src/slide_stream/providers/factory.py
TTS_PROVIDERS: Dict[str, Type[TTSProvider]] = {
    "gtts": GTTSProvider,
    "elevenlabs": ElevenLabsTTSProvider,
    "new_tts": NewTTSProvider,  # Add here
}
```

### Provider Testing

Create comprehensive tests for new providers:

```python
# tests/test_providers.py
def test_new_provider_available():
    config = {"api_keys": {"new_provider": "test-key"}}
    provider = NewImageProvider(config)
    assert provider.is_available()

def test_new_provider_unavailable():
    config = {"api_keys": {}}
    provider = NewImageProvider(config)
    assert not provider.is_available()

@patch('requests.get')
def test_new_provider_generate_image(mock_get):
    # Mock API response
    mock_get.return_value.content = b"fake_image_data"
    
    config = {"api_keys": {"new_provider": "test-key"}}
    provider = NewImageProvider(config)
    
    result = provider.generate_image("test query", "/tmp/test.jpg")
    assert result == "/tmp/test.jpg"
```

### Configuration Updates

When adding providers, update the example configuration:

```python
# src/slide_stream/config_loader.py
def create_example_config() -> str:
    return """# SlideStream Configuration File

providers:
  images:
    provider: new_provider    # Add new option
    fallback: text
    
api_keys:
  new_provider: "${NEW_PROVIDER_API_KEY}"  # Add API key

# Document new provider options...
"""
```

## Debugging Guide

### Common Issues

#### 1. Configuration Errors
```bash
# Test configuration loading
uv run python -c "
from slide_stream.config_loader import load_config
config = load_config('test.yaml')
print(config)
"
```

#### 2. Provider Issues
```bash
# Check provider availability
uv run slide-stream providers
```

#### 3. Type Checker Errors
```bash
# Run basedpyright to see type issues
uv run basedpyright

# Common fixes:
# - Add type hints
# - Use type guards for dynamic content
# - Add # type: ignore for external libraries
```

#### 4. CLI Issues
```bash
# Test new CLI format
uv run slide-stream --help
uv run slide-stream create --help
uv run slide-stream providers
```

### Debugging Tools

#### 1. Configuration Debug
```python
from slide_stream.config_loader import load_config
from rich.console import Console

console = Console()
try:
    config = load_config()
    console.print("Config loaded:", config)
except Exception as e:
    console.print_exception()
```

#### 2. Provider Debug
```python
from slide_stream.providers.factory import ProviderFactory

config = load_config()
availability = ProviderFactory.check_provider_availability(config)
print("Provider availability:", availability)
```

#### 3. Rich Console Debug
```python
from rich.console import Console
console = Console()
console.print("Debug info", style="red")
console.print_exception()  # Pretty tracebacks
```

## Contributing Guidelines

### Pull Request Process

1. **Fork and clone**
2. **Create feature branch**
3. **Make changes with tests**
4. **Ensure all checks pass**:
   - `uv run pytest`
   - `uv run basedpyright`
   - `uv run ruff check`
5. **Submit pull request**

### Code Review Checklist

- [ ] Tests added/updated
- [ ] Type hints added
- [ ] Documentation updated
- [ ] No new basedpyright errors
- [ ] Follows existing code style
- [ ] Provider pattern followed (if applicable)
- [ ] Configuration updated (if applicable)

### Commit Message Format

```
Add DALL-E 3 image generation provider

- Implement DalleImageProvider with OpenAI integration
- Add fallback to text images when API unavailable
- Update configuration system to support DALL-E options
- Add comprehensive tests with API mocking
- Update documentation with setup instructions

Closes #42
```

### Issue Guidelines

When reporting issues:

1. **Environment**: Python version, OS, SlideStream version
2. **Configuration**: Share your `slidestream.yaml` (remove API keys)
3. **Reproduction**: Minimal example to reproduce
4. **Expected vs Actual**: What you expected vs what happened
5. **Logs**: Any error messages or stack traces

Example issue:
```markdown
## Bug Report

**Environment:**
- SlideStream: 2.0.0
- Python: 3.11
- OS: macOS 14

**Configuration:**
```yaml
providers:
  images:
    provider: dalle3
    fallback: text
```

**Steps to Reproduce:**
1. Create file `test.md` with content...
2. Run `slide-stream create test.md video.mp4`
3. Error occurs

**Expected:** Video should be generated with DALL-E images
**Actual:** Error: "DALL-E provider not available"

**Error Log:**
```
[error messages here]
```
```

### Adding Documentation

When adding features, update:

1. **README.md**: Overview and quick start
2. **USER_GUIDE.md**: Detailed usage instructions
3. **Configuration examples**: In `config_loader.py`
4. **CLI help text**: In command definitions

---

This development workflow ensures consistent, high-quality code and smooth collaboration in the modern SlideStream 2.0 architecture. For AI assistant guidelines, see [CLAUDE.md](../CLAUDE.md).