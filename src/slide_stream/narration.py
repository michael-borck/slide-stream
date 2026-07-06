"""Narration planning: pick the best source per slide and build LLM prompts.

The goal is narration that *complements* the slides instead of reading them
aloud. Source priority per slide:

1. Speaker notes (pptx) — cleaned up and fitted to the target length.
2. Slide text content — used as a starting point for what a presenter would
   SAY about the slide, never read verbatim.
3. Slide image (pptx picture-only slides) — described via a vision-capable
   LLM and related to the slide title.
4. Title only — a brief spoken introduction of the topic.
"""

from pathlib import Path
from typing import Any

# Average speaking rate used to convert a duration target into a word target.
DEFAULT_WPM = 150


def parse_script_file(path: str | Path) -> list[str]:
    """Parse a narration script file into one spoken block per slide.

    Blocks are separated by a line containing only three (or more) dashes.
    Each block is used verbatim as the narration for the corresponding slide,
    in order. Blank blocks are preserved so slide alignment is never shifted.
    """
    text = Path(path).read_text(encoding="utf-8")
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 3 and set(stripped) == {"-"}:  # a --- separator line
            blocks.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)
    blocks.append("\n".join(current).strip())
    return blocks


def narration_source(slide: dict[str, Any]) -> str:
    """Return which source narration should be built from for this slide."""
    if str(slide.get("notes", "")).strip():
        return "notes"
    if slide.get("content"):
        return "content"
    if slide.get("images"):
        return "image"
    return "title-only"


def target_words(target_seconds: float | None, wpm: float = DEFAULT_WPM) -> int | None:
    """Approximate word count for a spoken duration, or None for no target."""
    if not target_seconds:
        return None
    return max(10, round(target_seconds / 60 * wpm))


def _length_instruction(words: int | None, wpm: float = DEFAULT_WPM) -> str:
    if words is None:
        return (
            "Keep it concise and conversational — roughly what a presenter "
            "would say in under a minute."
        )
    seconds = round(words / wpm * 60)
    return (
        f"The narration must be approximately {words} words (about {seconds} "
        "seconds when spoken). Summarise if the source material is longer; "
        "expand naturally if it is shorter."
    )


def _bullets(slide: dict[str, Any]) -> str:
    items = [str(item).strip() for item in slide.get("content", []) if str(item).strip()]
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


def build_narration_prompt(
    slide: dict[str, Any],
    source: str,
    words: int | None,
    wpm: float = DEFAULT_WPM,
) -> str:
    """Build the LLM prompt for one slide's voiceover script."""
    title = str(slide.get("title", "")).strip()
    length = _length_instruction(words, wpm)

    common = (
        "You are writing the voiceover script for one slide of a lecture "
        "video. Write natural, flowing spoken language: no headings, no "
        "bullet markers, no stage directions, no quotation of the slide "
        "text. The narration must complement the slide rather than read it "
        f"aloud. {length} Output only the spoken script, nothing else.\n\n"
    )

    if source == "notes":
        return common + (
            "PRIMARY SOURCE — the presenter's speaker notes below. Clean "
            "them up into speakable prose, keeping their key points and "
            "intent. The slide title and bullets are context only.\n\n"
            f"Slide title: {title}\n"
            f"Slide bullets (context):\n{_bullets(slide)}\n\n"
            f"Speaker notes (primary source):\n{slide.get('notes', '')}"
        )

    if source == "content":
        return common + (
            "There are no speaker notes. Using the slide's title and "
            "bullets as your starting point, write what an engaging "
            "presenter would SAY about this slide — connect the ideas, add "
            "brief explanation or motivation, and do not recite the bullets "
            "verbatim.\n\n"
            f"Slide title: {title}\n"
            f"Slide bullets:\n{_bullets(slide)}"
        )

    if source == "image":
        return common + (
            "This slide contains only a title and the attached image. "
            "Describe what the image shows as a presenter would, and relate "
            f"it to the slide title '{title or 'this slide'}'. Focus on why "
            "the image matters, not a literal inventory of it."
        )

    # title-only
    return common + (
        "This slide contains only a title. Write a brief narration "
        f"introducing the topic: {title or 'this section'}."
    )
