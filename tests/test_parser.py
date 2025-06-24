"""Tests for the markdown parser."""

import pytest

from slide_stream.parser import parse_markdown


def test_parse_markdown_basic():
    """Test basic markdown parsing."""
    markdown_text = """# First Slide

- Point 1
- Point 2

# Second Slide

- Another point
- Final point
"""
    
    slides = parse_markdown(markdown_text)
    
    assert len(slides) == 2
    assert slides[0]["title"] == "First Slide"
    assert slides[0]["content"] == ["Point 1", "Point 2"]
    assert slides[1]["title"] == "Second Slide"
    assert slides[1]["content"] == ["Another point", "Final point"]


def test_parse_markdown_empty():
    """Test parsing empty markdown."""
    slides = parse_markdown("")
    assert slides == []


def test_parse_markdown_no_lists():
    """Test parsing markdown with headers but no lists."""
    markdown_text = """# Title Only

Some paragraph text.

# Another Title

More text here.
"""
    
    slides = parse_markdown(markdown_text)
    
    assert len(slides) == 2
    assert slides[0]["title"] == "Title Only"
    assert slides[0]["content"] == ["Some paragraph text."]
    assert slides[1]["title"] == "Another Title"
    assert slides[1]["content"] == ["More text here."]


def test_parse_markdown_mixed_content():
    """Test parsing markdown with mixed content types."""
    markdown_text = """# First Slide

This is a paragraph.

- List item 1
- List item 2

# Second Slide

Another paragraph.
"""
    
    slides = parse_markdown(markdown_text)
    
    assert len(slides) == 2
    assert slides[0]["title"] == "First Slide"
    # The parser prioritizes lists over paragraphs when both are present
    assert "List item 1" in slides[0]["content"]
    assert "List item 2" in slides[0]["content"]
    # Paragraph might not be captured if a list is found
    assert slides[1]["title"] == "Second Slide"
    assert "Another paragraph." in slides[1]["content"]