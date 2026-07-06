"""Markdown parsing functionality for Slide Stream."""

from typing import Any

import markdown
from bs4 import BeautifulSoup, Tag


def _strip_front_matter(text: str) -> str:
    """Drop a leading YAML front-matter block (``---`` ... ``---``).

    Only stripped when the opening ``---`` is the first line and the block
    contains no markdown heading, so a deck that merely opens with a ``---``
    slide separator is left untouched. Ported from slide-vision.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            block = lines[1:i]
            if any(line.lstrip().startswith("#") for line in block):
                return text
            return "\n".join(lines[i + 1:])
    return text


def _has_separators(text: str) -> bool:
    return any(line.strip() == "---" for line in text.splitlines())


def _parse_separator_style(text: str) -> list[dict[str, Any]]:
    """Parse a ``---``-separated deck (Marp/reveal-style) into slides.

    Each block between ``---`` lines is one slide: the first heading line (or
    the first line) is the title, remaining non-empty lines become content
    (leading bullet markers stripped). Ported from slide-vision.
    """
    blocks: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.strip() == "---":
            blocks.append([])
        else:
            blocks[-1].append(line)

    slides: list[dict[str, Any]] = []
    for block_lines in blocks:
        block = "\n".join(block_lines).strip()
        if not block:
            continue
        title = ""
        content: list[str] = []
        for line in block.splitlines():
            stripped = line.strip()
            if not title and stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
            elif stripped:
                content.append(stripped.lstrip("-*").strip())
        if not title and content:
            title, content = content[0], content[1:]
        slides.append({"title": title, "content": content})
    return slides


def parse_markdown(markdown_text: str) -> list[dict[str, Any]]:
    """Parse markdown text into slide data.

    A leading YAML front-matter block is stripped. Decks that use ``---`` lines
    as slide separators are parsed by separator; otherwise each top-level
    (``#``) heading starts a new slide.

    In heading style, everything between an ``h1`` and the next ``h1`` is
    collected as slide content so nothing is silently dropped: multiple lists,
    paragraphs, and ``h2``/``h3`` (or deeper) sub-headings are all preserved in
    document order. Sub-headings are kept as content lines rather than starting
    separate slides.
    """
    markdown_text = _strip_front_matter(
        markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    )
    if _has_separators(markdown_text):
        return _parse_separator_style(markdown_text)

    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    slides = []

    for header in soup.find_all("h1"):
        slide_title = header.get_text()
        content_items: list[str] = []

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
                    text = next_sibling.get_text()
                    if text:
                        content_items.append(text)
                elif name in ("h2", "h3", "h4", "h5", "h6"):
                    # Preserve sub-headings as content rather than dropping them.
                    text = next_sibling.get_text()
                    if text:
                        content_items.append(text)
            next_sibling = next_sibling.find_next_sibling()

        slides.append({"title": slide_title, "content": content_items})

    return slides
