"""Markdown parsing functionality for Slide Stream."""

from typing import Any

import markdown
from bs4 import BeautifulSoup, Tag


def parse_markdown(markdown_text: str) -> list[dict[str, Any]]:
    """Parse markdown text into slide data.

    Each top-level (``#``) heading starts a new slide. Everything between an
    ``h1`` and the next ``h1`` is collected as slide content so nothing is
    silently dropped: multiple lists, paragraphs, and ``h2``/``h3`` (or deeper)
    sub-headings are all preserved in document order. Sub-headings are kept as
    content lines rather than starting separate slides.
    """
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
