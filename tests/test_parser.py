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


# --- separator-style and front-matter (ported from slide-vision) ------------


def test_parse_separator_style_deck():
    from slide_stream.parser import parse_markdown

    md = (
        "# First Slide\n"
        "- point a\n"
        "- point b\n"
        "---\n"
        "# Second Slide\n"
        "Some prose here.\n"
    )
    slides = parse_markdown(md)
    assert [s["title"] for s in slides] == ["First Slide", "Second Slide"]
    assert slides[0]["content"] == ["point a", "point b"]
    assert slides[1]["content"] == ["Some prose here."]


def test_front_matter_is_stripped():
    from slide_stream.parser import parse_markdown

    md = (
        "---\n"
        "title: My Deck\n"
        "author: Me\n"
        "---\n"
        "# Real Slide\n\n"
        "- a\n"
    )
    slides = parse_markdown(md)
    # Front matter did not become a slide.
    assert len(slides) == 1
    assert slides[0]["title"] == "Real Slide"


def test_front_matter_not_confused_with_separator_deck():
    """A deck opening with a --- separator (heading inside) is NOT front matter."""
    from slide_stream.parser import parse_markdown

    md = "---\n# One\n- a\n---\n# Two\n- b\n"
    slides = parse_markdown(md)
    assert [s["title"] for s in slides] == ["One", "Two"]


def test_heading_style_unchanged_without_separators():
    from slide_stream.parser import parse_markdown

    md = "# Only\n\n- x\n\n## Sub\n\n- y\n"
    slides = parse_markdown(md)
    assert len(slides) == 1
    assert slides[0]["title"] == "Only"
    assert "x" in slides[0]["content"]


def test_stray_thematic_break_keeps_heading_style():
    """One `---` in a multi-H1 deck must not flip the whole deck to
    separator mode (which would collapse it into two mashed slides)."""
    md = (
        "# One\n\n- a\n\n"
        "# Two\n\n- b\n\n"
        "---\n\n"
        "# Three\n\n- c\n"
    )
    slides = parse_markdown(md)
    assert [s["title"] for s in slides] == ["One", "Two", "Three"]
    assert slides[0]["content"] == ["a"]
    assert slides[1]["content"] == ["b"]
    assert slides[2]["content"] == ["c"]


def test_marp_deck_with_front_matter_parses_as_separator_style():
    """Front matter + `---` separators + one heading per slide (Marp style)."""
    md = (
        "---\n"
        "marp: true\n"
        "theme: default\n"
        "---\n"
        "# Intro\n\n- welcome\n\n"
        "---\n\n"
        "# Body\n\nSome prose.\n"
    )
    slides = parse_markdown(md)
    assert [s["title"] for s in slides] == ["Intro", "Body"]
    assert slides[0]["content"] == ["welcome"]
    assert slides[1]["content"] == ["Some prose."]


def test_separator_style_skips_image_markdown_lines():
    """Enrich-style markdown round-trips without image syntax in narration."""
    md = (
        "# One\n\n![One](images/slide_1.png)\n\n- a\n\n"
        "---\n\n"
        "# Two\n\n![Two](images/slide_2.png)\n\n- b\n"
    )
    slides = parse_markdown(md)
    assert [s["title"] for s in slides] == ["One", "Two"]
    for slide in slides:
        assert all("![" not in item for item in slide["content"])
    assert slides[0]["content"] == ["a"]
    assert slides[1]["content"] == ["b"]


def test_leading_prose_block_is_not_stripped_as_front_matter():
    """A separator deck whose first block is bullets/prose is a real slide,
    not YAML front matter."""
    md = "---\n- opening bullet\n- another\n---\n# Two\n- b\n"
    slides = parse_markdown(md)
    assert len(slides) == 2
    assert slides[0]["title"] == "opening bullet"
    assert slides[0]["content"] == ["another"]
    assert slides[1]["title"] == "Two"


def test_bullet_strip_preserves_bold_markers():
    """Stripping the leading bullet must not mangle `**bold**` emphasis."""
    md = "# One\n- **Bold** point\n---\n# Two\n- b\n"
    slides = parse_markdown(md)
    assert slides[0]["content"] == ["**Bold** point"]


def test_heading_style_captures_blockquote_and_code_block():
    md = (
        "# Only\n\n"
        "> A quoted insight.\n\n"
        "    result = compute()\n\n"
        "Closing text.\n"
    )
    slides = parse_markdown(md)
    assert len(slides) == 1
    content = " ".join(slides[0]["content"])
    assert "A quoted insight." in content
    assert "result = compute()" in content
    assert "Closing text." in content


def test_markdown_slides_flag_real_titles():
    heading_slides = parse_markdown("# One\n\n- a\n")
    assert heading_slides[0]["has_real_title"] is True

    separator_slides = parse_markdown("# One\n- a\n---\n# Two\n- b\n")
    assert all(s["has_real_title"] is True for s in separator_slides)
