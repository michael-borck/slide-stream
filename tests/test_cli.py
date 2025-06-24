"""Tests for the CLI interface."""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slide_stream.cli import app


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_markdown():
    """Create sample markdown content."""
    return """# Test Slide

- First point
- Second point

# Another Slide

- More content
- Final point
"""


def test_cli_help(runner):
    """Test CLI help command."""
    result = runner.invoke(app, ["create", "--help"])
    assert result.exit_code == 0
    assert "Create a video from a Markdown file" in result.stdout


def test_cli_version(runner):
    """Test CLI version command."""
    result = runner.invoke(app, ["create", "--version"])
    assert result.exit_code == 0
    assert "SlideStream" in result.stdout
    assert "1.0.0" in result.stdout


def test_cli_create_missing_input(runner):
    """Test CLI with missing input file."""
    result = runner.invoke(app, ["create"])
    assert result.exit_code != 0


def test_cli_create_basic(runner, sample_markdown):
    """Test basic CLI create command."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_markdown)
        f.flush()
        
        # Test with text image source to avoid network calls
        result = runner.invoke(app, [
            "create",
            "--input", f.name,
            "--output", "test_output.mp4",
            "--image-source", "text"
        ])
        
        # Should not crash during parsing phase
        assert "Parsing Markdown" in result.stdout or result.exit_code == 0
        
        # Clean up
        Path(f.name).unlink(missing_ok=True)
        Path("test_output.mp4").unlink(missing_ok=True)