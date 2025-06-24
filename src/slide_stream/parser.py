"""Markdown parsing functionality for Slide Stream."""

from typing import Any

import markdown
from bs4 import BeautifulSoup, Tag


def parse_markdown(markdown_text: str) -> list[dict[str, Any]]:
    """Parse markdown text into slide data."""
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    slides = []

    for header in soup.find_all("h1"):
        slide_title = header.get_text()

        # Find the next sibling that is a list (ul or ol)
        next_sibling = header.find_next_sibling()
        content_items = []

        while next_sibling:
            # Type guard: only process Tag elements, skip NavigableString
            if isinstance(next_sibling, Tag):
                if next_sibling.name in ["ul", "ol"]:
                    content_items = [
                        item.get_text() for item in next_sibling.find_all("li")
                    ]
                    break
                elif next_sibling.name == "p":
                    # If it's a paragraph, add it as content
                    content_items.append(next_sibling.get_text())
                elif next_sibling.name in ["h1", "h2", "h3"]:
                    # Stop if we hit another header
                    break
            next_sibling = next_sibling.find_next_sibling()

        slides.append({"title": slide_title, "content": content_items})

    return slides
