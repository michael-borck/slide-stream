"""Tests for the markdown parser."""


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
    # All content under the heading is preserved: paragraph and list alike.
    assert "This is a paragraph." in slides[0]["content"]
    assert "List item 1" in slides[0]["content"]
    assert "List item 2" in slides[0]["content"]
    assert slides[1]["title"] == "Second Slide"
    assert "Another paragraph." in slides[1]["content"]


def test_parse_markdown_keeps_multiple_lists_and_subheadings():
    """No content is silently dropped under a slide: multiple lists, paragraphs,
    and h2/h3 sub-headings are all captured in document order."""
    markdown_text = """# Main Topic

Intro paragraph.

- First list item
- Second list item

## Subsection

- Third list item
- Fourth list item

Closing paragraph.
"""
    slides = parse_markdown(markdown_text)

    assert len(slides) == 1
    assert slides[0]["title"] == "Main Topic"
    content = slides[0]["content"]

    # Everything is preserved.
    assert "Intro paragraph." in content
    assert "First list item" in content
    assert "Second list item" in content
    assert "Subsection" in content  # h2 kept as content, not dropped
    assert "Third list item" in content  # second list captured (old code stopped at first)
    assert "Fourth list item" in content
    assert "Closing paragraph." in content

    # Document order is maintained.
    assert content.index("Intro paragraph.") < content.index("First list item")
    assert content.index("Second list item") < content.index("Subsection")
    assert content.index("Subsection") < content.index("Third list item")
    assert content.index("Fourth list item") < content.index("Closing paragraph.")
