"""Draft a slide deck from a source document (the ``draft`` command).

A deliberate *preprocessing* step, separate from rendering: extract the text
from a ``.txt`` / ``.md`` / ``.pdf`` / ``.docx`` / ``.pptx`` source, ask an LLM
to turn it into a slide outline in the app's heading-style Markdown, and write
that ``.md``. The user then runs ``create`` (and optionally ``enrich``) on the
result — each step explicit, so the workflow stays understandable.
"""

from pathlib import Path
from typing import Any

from .parser import parse_markdown

# Guard against sending an enormous document to the LLM. Roughly ~15k tokens of
# source; enough for a report/chapter, and long inputs are summarised anyway.
_MAX_SOURCE_CHARS = 60_000

# Source types we can extract text from (for the CLI's error message / help).
SUPPORTED_SUFFIXES = (".txt", ".md", ".markdown", ".pdf", ".docx", ".pptx")


class DraftError(Exception):
    """Raised when a source cannot be read or the draft cannot be produced."""


def extract_source_text(path: Path) -> str:
    """Extract plain text from a supported source document."""
    ext = path.suffix.lower()
    if ext in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8").strip()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".pptx":
        return _extract_pptx(path)
    raise DraftError(
        f"Unsupported source type '{ext or path.name}'. "
        f"Use one of: {', '.join(SUPPORTED_SUFFIXES)}."
    )


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        raise DraftError(
            'PDF support needs an extra: pip install "slide-stream[import]"'
        )
    reader = PdfReader(str(path))
    text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return text.strip()


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document  # python-docx  # type: ignore[import-not-found]
    except ImportError:
        raise DraftError(
            'Word (.docx) support needs an extra: pip install "slide-stream[import]"'
        )
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()


def _extract_pptx(path: Path) -> str:
    """Flatten an existing deck's titles, bullets, and notes — the 'rewrite'
    case, where the LLM restructures a messy deck into a cleaner one."""
    from .powerpoint import parse_powerpoint

    parts: list[str] = []
    for slide in parse_powerpoint(path):
        title = str(slide.get("title", "")).strip()
        if title:
            parts.append(f"# {title}")
        parts.extend(str(c).strip() for c in slide.get("content", []) if str(c).strip())
        notes = str(slide.get("notes", "")).strip()
        if notes and notes != "Click to add notes":
            parts.append(f"(notes: {notes})")
    return "\n".join(parts).strip()


def clamp_source(text: str) -> tuple[str, bool]:
    """Return (text, truncated) clamped to the LLM source budget."""
    if len(text) <= _MAX_SOURCE_CHARS:
        return text, False
    return text[:_MAX_SOURCE_CHARS], True


def build_draft_prompt(source_text: str, slides: int | None) -> str:
    """Build the LLM prompt that turns a document into a deck outline."""
    if slides:
        count = f"Create exactly {slides} slides."
    else:
        count = (
            "Choose a sensible number of slides for the material — roughly one "
            "per key idea (about 5–15 for a short document)."
        )
    return (
        "You are turning a source document into a presentation outline. "
        f"{count}\n\n"
        "Output ONLY Markdown in exactly this format, and nothing else:\n\n"
        "# First slide title\n\n- A concise point\n- Another point\n\n"
        "# Second slide title\n\n- A concise point\n\n"
        "Rules:\n"
        "- Every slide starts with a single '# ' heading (its title).\n"
        "- 2–5 short bullet points per slide; no sub-headings, no images, no "
        "speaker notes.\n"
        "- Distil the document's key points into a logical flow; do not copy "
        "long sentences verbatim.\n"
        "- No preamble, no commentary, and do NOT wrap the output in code "
        "fences.\n\n"
        "SOURCE DOCUMENT:\n\n"
        f"{source_text}"
    )


def clean_llm_markdown(text: str) -> str:
    """Strip a ```/```markdown code fence the model may wrap the deck in."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]  # drop opening ``` / ```markdown
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing ```
        stripped = "\n".join(lines).strip()
    return stripped


def validate_deck_markdown(markdown_text: str) -> list[dict[str, Any]]:
    """Parse the generated Markdown as a deck; raise if it has no slides."""
    slides = parse_markdown(markdown_text)
    if not slides:
        raise DraftError(
            "The model did not return a usable slide deck (no '# ' headings "
            "found). Try again, or adjust --slides."
        )
    return slides
