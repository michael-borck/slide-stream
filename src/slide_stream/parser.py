"""Markdown parsing functionality for Slide Stream."""

import re
from typing import Any

import markdown
from bs4 import BeautifulSoup, Tag

# A markdown image line, e.g. ``![Title](images/slide_1.png)`` — slide
# artwork (as written by ``enrich``), never spoken content.
_IMAGE_LINE_RE = re.compile(r"^!\[.*\]\(.*\)\s*$")
# The source path inside a markdown image, used to carry a pre-made per-slide
# image (e.g. an enriched deck's ``images/slide_1.png``) through to the renderer.
_IMAGE_SRC_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
# An ATX H1 heading line (``# Title``).
_H1_RE = re.compile(r"^#\s+\S")
# A YAML front-matter mapping line (``key: value``).
_YAML_KEY_RE = re.compile(r"^[A-Za-z_][\w.-]*\s*:(\s|$)")
# A single leading bullet marker (``- `` / ``* ``); unlike ``lstrip("-*")``
# this leaves emphasis such as ``**bold**`` intact.
_BULLET_RE = re.compile(r"^[-*]\s+")


def _strip_front_matter(text: str) -> str:
    """Drop a leading YAML front-matter block (``---`` ... ``---``).

    Only stripped when the opening ``---`` is the first line and the block
    actually looks like YAML front matter: no markdown headings, and every
    non-blank line is a ``key: value`` mapping (or an indented continuation
    line). A deck that merely opens with a ``---`` slide separator — bullets,
    prose, or headings in the first block — is left untouched. Ported from
    slide-vision.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            block = [line for line in lines[1:i] if line.strip()]
            if not block:
                return text
            for line in block:
                if line.lstrip().startswith("#"):
                    return text  # a heading: this is a slide, not front matter
                if not _YAML_KEY_RE.match(line) and not line[0].isspace():
                    return text  # bullets/prose: a real first slide
            return "\n".join(lines[i + 1:])
    return text


def _split_blocks(text: str) -> list[list[str]]:
    """Split lines into blocks delimited by ``---`` separator lines."""
    blocks: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.strip() == "---":
            blocks.append([])
        else:
            blocks[-1].append(line)
    return blocks


def _use_separator_style(text: str) -> bool:
    """True when ``---`` lines clearly delimit slides.

    Requires at least two non-empty blocks between separators, none of which
    contains more than one ``#`` H1 heading. A Marp-style deck (at most one
    heading per ``---``-separated block) parses by separator, while a deck
    structured around multiple H1s with a stray thematic break stays
    heading-style instead of collapsing into two mashed slides.
    """
    blocks = [b for b in _split_blocks(text) if any(line.strip() for line in b)]
    if len(blocks) < 2:
        return False
    for block in blocks:
        h1_count = sum(1 for line in block if _H1_RE.match(line.strip()))
        if h1_count > 1:
            return False
    return True


def _parse_separator_style(text: str) -> list[dict[str, Any]]:
    """Parse a ``---``-separated deck (Marp/reveal-style) into slides.

    Each block between ``---`` lines is one slide: the first heading line (or
    the first line) is the title, remaining non-empty lines become content
    (leading bullet markers stripped, image lines skipped). Ported from
    slide-vision.
    """
    slides: list[dict[str, Any]] = []
    for block_lines in _split_blocks(text):
        block = "\n".join(block_lines).strip()
        if not block:
            continue
        title = ""
        content: list[str] = []
        image_path = ""
        for line in block.splitlines():
            stripped = line.strip()
            if _IMAGE_LINE_RE.match(stripped):
                if not image_path:  # capture the first slide image; never spoken
                    m = _IMAGE_SRC_RE.search(stripped)
                    if m:
                        image_path = m.group(1).strip()
                continue
            if not title and stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
            elif stripped:
                content.append(_BULLET_RE.sub("", stripped))
        if not title and content:
            title, content = content[0], content[1:]
        slide: dict[str, Any] = {
            "title": title, "content": content, "has_real_title": bool(title)
        }
        if image_path:
            slide["image_path"] = image_path
        slides.append(slide)
    return slides


def parse_markdown(markdown_text: str) -> list[dict[str, Any]]:
    """Parse markdown text into slide data.

    A leading YAML front-matter block is stripped. Decks that clearly use
    ``---`` lines as slide separators are parsed by separator; otherwise each
    top-level (``#``) heading starts a new slide (a single stray thematic
    break in a multi-H1 deck does not flip the whole deck to separator mode).

    In heading style, everything between an ``h1`` and the next ``h1`` is
    collected as slide content so nothing is silently dropped: multiple lists,
    paragraphs, blockquotes, code blocks, tables, and ``h2``/``h3`` (or
    deeper) sub-headings are all preserved in document order. Sub-headings are
    kept as content lines rather than starting separate slides.
    """
    markdown_text = _strip_front_matter(
        markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    )
    if _use_separator_style(markdown_text):
        return _parse_separator_style(markdown_text)

    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    slides = []

    for header in soup.find_all("h1"):
        slide_title = header.get_text()
        content_items: list[str] = []
        slide_image = ""

        next_sibling = header.find_next_sibling()
        while next_sibling:
            # Type guard: only process Tag elements, skip NavigableString
            if isinstance(next_sibling, Tag):
                name = next_sibling.name
                if name == "h1":
                    # The next slide begins here; stop collecting content.
                    break
                elif name in ("ul", "ol"):
                    # Capture every list (not just the first) in document order.
                    content_items.extend(
                        item.get_text() for item in next_sibling.find_all("li")
                    )
                elif name == "p":
                    # A pre-made slide image (``![](images/slide_1.png)``) becomes
                    # the slide's artwork, not narration; capture the first one.
                    img = next_sibling.find("img")
                    if isinstance(img, Tag) and not slide_image:
                        src = img.get("src")
                        if isinstance(src, str) and src.strip():
                            slide_image = src.strip()
                    text = next_sibling.get_text()
                    if text:
                        content_items.append(text)
                elif name in ("h2", "h3", "h4", "h5", "h6"):
                    # Preserve sub-headings as content rather than dropping them.
                    text = next_sibling.get_text()
                    if text:
                        content_items.append(text)
                elif name in ("blockquote", "pre", "table"):
                    # Quotes, code blocks, and tables count as narration
                    # content too — do not silently drop their text.
                    text = next_sibling.get_text().strip()
                    if text:
                        content_items.append(text)
            next_sibling = next_sibling.find_next_sibling()

        slide: dict[str, Any] = {
            "title": slide_title, "content": content_items, "has_real_title": True
        }
        if slide_image:
            slide["image_path"] = slide_image
        slides.append(slide)

    return slides
