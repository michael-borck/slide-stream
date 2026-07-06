"""Deck enrichment: add an image to each slide and write a new deck.

This is the ``enrich`` output track (ported/adapted from slide-vision): the
same slide input, but instead of a narrated video the output is a new,
editable deck — a Markdown file plus an ``images/`` folder, and optionally a
PowerPoint. Run ``create`` on the result to narrate it, or use ``create``
directly (it enriches internally) for a one-pass video.
"""

import zipfile
from pathlib import Path
from typing import Any

from .providers.base import ImageProvider


def _slide_query(slide: dict[str, Any]) -> str:
    """A search/keyword query for a slide's image."""
    title = str(slide.get("title", "")).strip()
    if title:
        return title
    for item in slide.get("content", []):
        text = str(item).strip()
        if text:
            return text
    return "presentation slide"


def enrich_deck(
    slides: list[dict[str, Any]],
    image_provider: ImageProvider,
    output_dir: Path,
    input_stem: str,
    *,
    also_pptx: bool = False,
    also_zip: bool = False,
) -> Path:
    """Write an enriched Markdown deck (and optional PPTX) into output_dir.

    Returns the output directory. Each slide gets an image from
    ``image_provider``; slides the ``local`` provider could not match are
    listed in ``prompts.md`` with ready-to-paste AI-image prompts.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    enriched: list[dict[str, Any]] = []
    for i, slide in enumerate(slides, 1):
        img_path = images_dir / f"slide_{i}.png"
        image_provider.generate_image(_slide_query(slide), str(img_path), slide=slide)
        # The local provider reports whether it matched a real folder image;
        # other providers always produce an image (or their own text fallback).
        matched = getattr(image_provider, "matched_last", True)
        enriched.append(
            {
                "index": i,
                "title": str(slide.get("title", "")).strip(),
                "content": [str(c).strip() for c in slide.get("content", []) if str(c).strip()],
                "image": img_path.name,
                "matched": matched,
            }
        )

    md_path = output_dir / f"{input_stem}.md"
    md_path.write_text(_build_markdown(enriched), encoding="utf-8")

    missing = [s for s in enriched if not s["matched"]]
    if missing:
        (output_dir / "prompts.md").write_text(_build_prompts(missing), encoding="utf-8")

    if also_pptx:
        _write_pptx(enriched, images_dir, output_dir / f"{input_stem}.pptx")

    if also_zip:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in output_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(output_dir.parent))

    return output_dir


def _build_markdown(slides: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for slide in slides:
        lines = [f"# {slide['title']}" if slide["title"] else f"# Slide {slide['index']}"]
        lines.append("")
        lines.append(f"![{slide['title']}](images/{slide['image']})")
        if slide["content"]:
            lines.append("")
            for item in slide["content"]:
                lines.append(f"- {item}")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks) + "\n"


def _build_prompts(missing: list[dict[str, Any]]) -> str:
    lines = ["# Image Prompts", "",
             "Slides with no matching local image. Paste a prompt into an AI "
             "image tool (DALL-E, Midjourney, ...) and drop the result into "
             "the images/ folder.", ""]
    for slide in missing:
        preview = " ".join(slide["content"])[:300]
        lines += [
            f"## Slide {slide['index']}: {slide['title']}",
            "",
            f'A high-quality, professional illustration for a presentation slide '
            f'titled "{slide["title"]}".',
        ]
        if preview:
            lines.append(f"The slide covers: {preview}.")
        lines += ["Style: clean, modern, no text overlays.", "", "---", ""]
    return "\n".join(lines)


def _write_pptx(slides: list[dict[str, Any]], images_dir: Path, out_path: Path) -> None:
    """Build a PowerPoint deck with one image slide per entry."""
    from pptx import Presentation
    from pptx.util import Emu, Inches

    prs = Presentation()
    # Default deck is 10" x 7.5"; fall back to that if the stubs report None.
    width = Emu(prs.slide_width) if prs.slide_width else Inches(10)
    blank = prs.slide_layouts[6]
    content_width = Emu(width - Inches(1))
    image_width = Emu(width - Inches(2))
    for slide in slides:
        s = prs.slides.add_slide(blank)
        # Title textbox across the top.
        title_box = s.shapes.add_textbox(Inches(0.5), Inches(0.3), content_width, Inches(1))
        title_box.text_frame.text = slide["title"] or f"Slide {slide['index']}"
        # Image centered below the title, scaled to fit.
        img = images_dir / slide["image"]
        if img.is_file():
            s.shapes.add_picture(str(img), Inches(1), Inches(1.5), width=image_width)
    prs.save(str(out_path))
