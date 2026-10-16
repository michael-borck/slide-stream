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


# --- Pre-made per-slide images (enriched decks) ------------------------------


def test_heading_style_captures_slide_image():
    md = "# One\n\n![](images/slide_1.png)\n\n- a bullet\n\n# Two\n\n- b\n"
    slides = parse_markdown(md)
    assert slides[0]["image_path"] == "images/slide_1.png"
    assert "a bullet" in slides[0]["content"]  # bullets still captured
    assert "image_path" not in slides[1]  # no image on slide two


def test_separator_style_captures_slide_image():
    md = "# One\n\n![alt](images/slide_1.png)\n\n- a\n\n---\n\n# Two\n\n- b\n"
    slides = parse_markdown(md)
    assert len(slides) == 2
    assert slides[0]["image_path"] == "images/slide_1.png"
    # The image line is never narrated content.
    assert not any("slide_1.png" in c for c in slides[0]["content"])


def test_image_not_captured_when_absent():
    slides = parse_markdown("# Only\n\n- a\n")
    assert "image_path" not in slides[0]


# --- Code blocks, tables and the per-slide Markdown body --------------------

CODE_DECK = (
    "# Code\n\n"
    "```python\n"
    "# a comment, not a slide\n"
    "x = 1\n"
    "\n"
    "---\n"
    "y = 2\n"
    "```\n\n"
    "- after the code\n\n"
    "# Next\n\n"
    "- b\n"
)


def test_hash_comment_inside_fenced_code_does_not_start_a_slide():
    """Before fenced_code was enabled, `# a comment` became its own slide and
    shifted every later narration block out of alignment."""
    slides = parse_markdown(CODE_DECK)
    assert [s["title"] for s in slides] == ["Code", "Next"]
    code = slides[0]["content"][0]
    assert code.startswith("# a comment, not a slide\nx = 1")
    assert "y = 2" in code
    assert slides[0]["content"][1] == "after the code"


def test_fenced_code_markdown_body_is_verbatim():
    body = parse_markdown(CODE_DECK)[0]["markdown"]
    assert body.startswith("```python\n# a comment, not a slide\nx = 1\n")
    assert "\n---\ny = 2\n```" in body  # a --- inside code is kept
    assert body.endswith("- after the code")


def test_table_becomes_readable_rows_and_keeps_its_markdown():
    md = (
        "# Costs\n\n"
        "| Operation | List | Hash |\n"
        "|---|---|---|\n"
        "| Look up | O(n) | O(1) |\n"
    )
    (slide,) = parse_markdown(md)
    assert slide["content"] == ["Operation | List | Hash\nLook up | O(n) | O(1)"]
    assert slide["markdown"] == (
        "| Operation | List | Hash |\n|---|---|---|\n| Look up | O(n) | O(1) |"
    )


def test_markdown_body_keeps_subheadings_and_inline_formatting():
    md = "# Topic\n\n## Part A\n\n- **bold** and `code`\n\n# Two\n\n- b\n"
    slides = parse_markdown(md)
    assert slides[0]["content"] == ["Part A", "bold and code"]
    assert slides[0]["markdown"] == "## Part A\n\n- **bold** and `code`"
    assert slides[1]["markdown"] == "- b"


def test_markdown_body_drops_images_and_bare_rules():
    md = "# One\n\n![](images/slide_1.png)\n\n- a\n\n---\n\n# Two\n\n- b\n\n# Three\n"
    slides = parse_markdown(md)
    assert slides[0]["markdown"] == "- a"
    assert slides[2]["markdown"] == ""


def test_setext_headings_parse_without_a_markdown_body():
    """The line split only knows `# ` headings; when it disagrees with
    Python-Markdown, slides carry no body and callers fall back to content."""
    slides = parse_markdown("One\n===\n\n- a\n\nTwo\n===\n\n- b\n")
    assert [s["title"] for s in slides] == ["One", "Two"]
    assert all("markdown" not in s for s in slides)


def test_separator_style_keeps_code_block_as_one_item():
    md = (
        "# One\n\n```\n- not a bullet\n# not a title\n```\n\n- a\n"
        "---\n"
        "# Two\n- b\n"
    )
    slides = parse_markdown(md)
    assert [s["title"] for s in slides] == ["One", "Two"]
    assert slides[0]["content"] == ["- not a bullet\n# not a title", "a"]
    assert slides[0]["markdown"] == "```\n- not a bullet\n# not a title\n```\n\n- a"


def test_separator_inside_code_does_not_split_a_separator_deck():
    md = "# One\n```\nx\n---\ny\n```\n---\n# Two\n- b\n"
    slides = parse_markdown(md)
    assert [s["title"] for s in slides] == ["One", "Two"]
    assert slides[0]["content"] == ["x\n---\ny"]


def test_separator_style_promoted_title_is_not_repeated_in_body():
    slides = parse_markdown("---\n- opening bullet\n- another\n---\n# Two\n- b\n")
    assert slides[0]["title"] == "opening bullet"
    assert slides[0]["markdown"] == "- another"


def test_separator_style_table_matches_heading_style():
    table = "| a | b |\n|:--|--:|\n| 1 | 2 |\n"
    heading = parse_markdown("# T\n\n" + table)[0]["content"]
    separator = parse_markdown("# T\n" + table + "---\n# U\n- x\n")[0]["content"]
    assert heading == separator == ["a | b\n1 | 2"]


# --- Structured code blocks and tables (ADR-0013) ----------------------------


def test_code_blocks_keep_language_in_both_styles():
    heading = parse_markdown("# A\n\n```python\nx = 1\n```\n\n# B\n\n- b\n")
    assert heading[0]["code_blocks"] == [{"lang": "python", "code": "x = 1"}]
    assert "code_blocks" not in heading[1]
    separator = parse_markdown("# A\n```js\nx()\n```\n---\n# B\n- b\n")
    assert separator[0]["code_blocks"] == [{"lang": "js", "code": "x()"}]


def test_code_block_without_language_and_indented_code():
    (slide,) = parse_markdown("# A\n\n```\nplain\n```\n")
    assert slide["code_blocks"] == [{"lang": None, "code": "plain"}]
    (slide,) = parse_markdown("# I\n\n    first()\n        nested()\n")
    assert slide["code_blocks"][0]["code"] == "first()\n    nested()"


def test_tables_kept_as_cells_in_both_styles():
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    heading = parse_markdown("# T\n\n" + table)[0]
    separator = parse_markdown("# T\n" + table + "---\n# U\n- x\n")[0]
    assert heading["tables"] == separator["tables"] == [[["a", "b"], ["1", "2"]]]
