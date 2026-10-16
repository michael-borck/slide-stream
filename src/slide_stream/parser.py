"""Markdown parsing functionality for Slide Stream.

Each parsed slide carries two views of its body:

- ``content``: plain-text items (bullets, paragraphs, code, table rows) used
  for narration, text cards and the plain ``.pptx`` exporter.
- ``code_blocks`` / ``tables`` (only when present): structured copies of
  the slide's code (``{"lang", "code"}``) and table cells, for syntax
  highlighting and the designed-deck fidelity check.
- ``markdown``: the slide's own Markdown source below its title, with image
  lines and bare ``---`` rules removed. Tools that pass Markdown on (Presenton,
  ``enrich``) use it so code blocks, tables and sub-headings survive intact.
  It is absent when the source cannot be split reliably (for example setext
  ``Title\\n===`` headings); callers then fall back to ``content``.
"""

import re
from collections.abc import Iterator
from typing import Any

import markdown
from bs4 import BeautifulSoup, Tag

# Python-Markdown extensions: fenced ```code``` blocks and pipe tables. Without
# fenced_code, a ``# comment`` line inside a code block becomes an <h1> and
# splits the slide in two.
_MD_EXTENSIONS = ["fenced_code", "tables"]
# Opening/closing line of a fenced code block (``` or ~~~).
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

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


def _fence_state(lines: list[str]) -> Iterator[tuple[str, bool]]:
    """Yield ``(line, in_code)`` pairs; fence marker lines count as code.

    A fence closes on a line starting with the same marker character and at
    least as many of them, as in CommonMark.
    """
    marker = ""
    for line in lines:
        m = _FENCE_RE.match(line)
        if not marker:
            if m:
                marker = m.group(1)
                yield line, True
            else:
                yield line, False
        else:
            if m and m.group(1)[0] == marker[0] and len(m.group(1)) >= len(marker):
                marker = ""
            yield line, True


def _split_blocks(text: str) -> list[list[str]]:
    """Split lines into blocks delimited by ``---`` separator lines.

    A ``---`` line inside a fenced code block is code, not a separator.
    """
    blocks: list[list[str]] = [[]]
    for line, in_code in _fence_state(text.splitlines()):
        if not in_code and line.strip() == "---":
            blocks.append([])
        else:
            blocks[-1].append(line)
    return blocks


def _clean_body(lines: list[str]) -> str:
    """A slide body for re-use as Markdown: drop image lines and bare ``---``
    rules (outside code, where they would read as slide separators), and
    trim surrounding blank lines."""
    kept = [
        line for line, in_code in _fence_state(lines)
        if in_code or not (
            _IMAGE_LINE_RE.match(line.strip()) or line.strip() == "---"
        )
    ]
    return "\n".join(kept).strip("\n")


def _heading_bodies(text: str) -> list[str]:
    """Markdown body under each ATX ``# `` heading, ignoring code blocks."""
    bodies: list[list[str]] = []
    for line, in_code in _fence_state(text.splitlines()):
        if not in_code and _H1_RE.match(line.strip()) and not line.startswith("    "):
            bodies.append([])
        elif bodies:
            bodies[-1].append(line)
    return [_clean_body(b) for b in bodies]


_TABLE_RULE_RE = re.compile(r"^\|?[\s:|-]+\|?$")


def _pipe_table_rows(lines: list[str]) -> list[list[str]]:
    """Cells of a pipe table's source lines, header rule skipped."""
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if "-" in stripped and _TABLE_RULE_RE.match(stripped):
            continue  # the |---|---| header rule
        rows.append([c.strip() for c in stripped.strip("|").split("|")])
    return rows


def _rows_text(rows: list[list[str]]) -> str:
    """A table as plain text: one row per line, cells joined by `` | ``."""
    return "\n".join(" | ".join(cells) for cells in rows)


def _flush_table(
    table: list[str], content: list[str], tables: list[list[list[str]]]
) -> None:
    """Move collected pipe-table lines into ``content`` as one item, and
    their cells into ``tables``."""
    if table:
        rows = _pipe_table_rows(table)
        content.append(_rows_text(rows))
        tables.append(rows)
        table.clear()


def _table_rows(table: Tag) -> list[list[str]]:
    """Cells of an HTML table, row by row."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        if isinstance(tr, Tag):
            rows.append([c.get_text().strip() for c in tr.find_all(["th", "td"])])
    return rows


def _fence_lang(line: str) -> str | None:
    """Language from a fence's info string (```` ```python ```` -> python)."""
    m = _FENCE_RE.match(line)
    if not m:
        return None
    info = line.strip()[len(m.group(1)):].strip()
    if not info:
        return None
    return info.split()[0].lstrip("{.").rstrip("}") or None


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
        h1_count = sum(
            1 for line, in_code in _fence_state(block)
            if not in_code and _H1_RE.match(line.strip())
        )
        if h1_count > 1:
            return False
    return True


def _parse_separator_style(text: str) -> list[dict[str, Any]]:
    """Parse a ``---``-separated deck (Marp/reveal-style) into slides.

    Each block between ``---`` lines is one slide: the first heading line (or
    the first line) is the title, remaining non-empty lines become content
    (leading bullet markers stripped, image lines skipped). A fenced code
    block becomes one content item holding the code, and a pipe table one
    item with a row per line, as in heading style. Ported from slide-vision.
    """
    slides: list[dict[str, Any]] = []
    for block_lines in _split_blocks(text):
        lines = "\n".join(block_lines).strip().splitlines()
        if not lines:
            continue
        title = ""
        title_line: int | None = None
        content: list[str] = []
        image_path = ""
        code: list[str] | None = None  # lines of the code block being read
        code_lang: str | None = None
        code_blocks: list[dict[str, Any]] = []
        table: list[str] = []  # consecutive | lines of a pipe table
        tables: list[list[list[str]]] = []
        for idx, (line, in_code) in enumerate(_fence_state(lines)):
            stripped = line.strip()
            if not in_code and stripped.startswith("|"):
                table.append(stripped)
                continue
            _flush_table(table, content, tables)
            if in_code:
                if _FENCE_RE.match(line):
                    if code is None:
                        code, code_lang = [], _fence_lang(line)
                        continue
                    _add_code(code, code_lang, content, code_blocks)
                    code = None
                elif code is not None:
                    code.append(line)
                continue
            if _IMAGE_LINE_RE.match(stripped):
                if not image_path:  # capture the first slide image; never spoken
                    m = _IMAGE_SRC_RE.search(stripped)
                    if m:
                        image_path = m.group(1).strip()
                continue
            if not title and stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                title_line = idx
            elif stripped:
                content.append(_BULLET_RE.sub("", stripped))
        _flush_table(table, content, tables)
        if code:  # unclosed fence at the end of the block
            _add_code(code, code_lang, content, code_blocks)
        if not title and content:
            title, content = content[0], content[1:]
            # The promoted first line is the title, not part of the body.
            title_line = next(
                (i for i, line in enumerate(lines)
                 if line.strip() and not _IMAGE_LINE_RE.match(line.strip())),
                None,
            )
        body = [line for i, line in enumerate(lines) if i != title_line]
        slide: dict[str, Any] = {
            "title": title, "content": content, "has_real_title": bool(title),
            "markdown": _clean_body(body),
        }
        if image_path:
            slide["image_path"] = image_path
        if code_blocks:
            slide["code_blocks"] = code_blocks
        if tables:
            slide["tables"] = tables
        slides.append(slide)
    return slides


def _add_code(
    code: list[str], lang: str | None, content: list[str],
    code_blocks: list[dict[str, Any]],
) -> None:
    """Record a finished code block as one content item and a code block."""
    text = "\n".join(code).strip("\n")
    if text.strip():
        content.append(text)
        code_blocks.append({"lang": lang, "code": text})


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

    html = markdown.markdown(markdown_text, extensions=_MD_EXTENSIONS)
    soup = BeautifulSoup(html, "html.parser")
    slides = []
    headers = soup.find_all("h1")
    # Only trust the line-based split when it found the same slides as
    # Python-Markdown (setext headings or an H1 nested in a quote would not).
    bodies = _heading_bodies(markdown_text)
    if len(bodies) != len(headers):
        bodies = []

    for slide_no, header in enumerate(headers):
        slide_title = header.get_text()
        content_items: list[str] = []
        code_blocks: list[dict[str, Any]] = []
        tables: list[list[list[str]]] = []
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
                elif name == "table":
                    rows = _table_rows(next_sibling)
                    if rows:
                        content_items.append(_rows_text(rows))
                        tables.append(rows)
                elif name == "pre":
                    # Code counts as narration content too. Keep leading
                    # indentation: strip only the surrounding blank lines.
                    text = next_sibling.get_text().strip("\n")
                    if text.strip():
                        content_items.append(text)
                        code_blocks.append({"lang": _code_lang(next_sibling), "code": text})
                elif name == "blockquote":
                    text = next_sibling.get_text().strip()
                    if text:
                        content_items.append(text)
            next_sibling = next_sibling.find_next_sibling()

        slide: dict[str, Any] = {
            "title": slide_title, "content": content_items, "has_real_title": True
        }
        if bodies:
            slide["markdown"] = bodies[slide_no]
        if slide_image:
            slide["image_path"] = slide_image
        if code_blocks:
            slide["code_blocks"] = code_blocks
        if tables:
            slide["tables"] = tables
        slides.append(slide)

    return slides


def _code_lang(pre: Tag) -> str | None:
    """Language of a rendered code block (``class="language-python"``)."""
    code = pre.find("code")
    classes = code.get("class") if isinstance(code, Tag) else None
    for cls in classes or []:
        if isinstance(cls, str) and cls.startswith("language-"):
            return cls[len("language-"):] or None
    return None
