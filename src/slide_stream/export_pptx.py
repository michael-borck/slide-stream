"""Export parsed slides to a PowerPoint deck (.pptx).

The video pipeline's sibling output: instead of rendering an MP4, write each
slide as a native PowerPoint slide — title, bullet points, the generated
artwork, and the narration script in the speaker notes. Teachers get an
editable deck they can present live (notes visible in presenter view) while
the video pipeline stays the "watch it anywhere" path.
"""

from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Pt

# PowerPoint's standard widescreen slide size in EMUs (13.333" x 7.5").
SLIDE_WIDTH_EMU = 12_192_000
SLIDE_HEIGHT_EMU = 6_858_000

# Layout constants (EMU). Emu is an int subclass, so plain int maths is fine.
_MARGIN = 457_200  # 0.5"
_TITLE_BAND = 914_400  # 1.0" below the top margin for the title
_INK = RGBColor(0x1A, 0x1A, 0x1A)
_TITLE_PT = 32
_BODY_PT = 18


def export_pptx(
    slides: list[dict[str, Any]], output_path: str | Path
) -> Path:
    """Write ``slides`` (title/content/image_path/notes dicts) to a .pptx file.

    Each slide gets its title, its bullet points beside the artwork (or across
    the full width when there is no image), and its narration in the speaker
    notes. Returns the output path.
    """
    prs = Presentation()
    prs.slide_width = Emu(SLIDE_WIDTH_EMU)
    prs.slide_height = Emu(SLIDE_HEIGHT_EMU)
    blank_layout = prs.slide_layouts[6]

    for entry in slides:
        slide = prs.slides.add_slide(blank_layout)
        title = str(entry.get("title") or "").strip() or "Slide"
        alt = str(entry.get("alt") or "").strip()
        bullets = [
            str(c).strip() for c in (entry.get("content") or []) if str(c).strip()
        ]
        notes = str(entry.get("notes") or "").strip()

        _add_title(slide, title)
        image_path = entry.get("image_path")
        has_image = bool(image_path) and Path(str(image_path)).is_file()
        code_blocks = entry.get("code_blocks") or []
        tables = entry.get("tables") or []
        # Code and tables get their own blocks in the upper content area;
        # remaining bullets sit underneath.
        block = bool(code_blocks or tables)
        if code_blocks:
            _add_code_box(slide, code_blocks)
        elif tables:
            _add_table(slide, tables[0])
        if has_image and not block:
            _add_picture_fit(slide, str(image_path), alt=alt or None)
        if bullets:
            _add_bullets(slide, bullets, full_width=not has_image or block,
                         bottom_half=block)

        if notes:
            notes_frame = slide.notes_slide.notes_text_frame
            if notes_frame is not None:
                notes_frame.text = notes

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


def _add_title(slide: Any, title: str) -> None:
    box = slide.shapes.add_textbox(
        Emu(_MARGIN),
        Emu(_MARGIN),
        Emu(SLIDE_WIDTH_EMU - 2 * _MARGIN),
        Emu(_TITLE_BAND),
    )
    frame = box.text_frame
    frame.word_wrap = True
    run = frame.paragraphs[0].add_run()
    run.text = title
    run.font.size = Pt(_TITLE_PT)
    run.font.bold = True
    run.font.color.rgb = _INK


def _add_bullets(slide: Any, bullets: list[str], full_width: bool,
                 bottom_half: bool = False) -> None:
    """Write the bullet points in the area left of (or under) the artwork,
    or across the lower half under a code block or table."""
    if bottom_half:
        box_top = (SLIDE_HEIGHT_EMU + _TITLE_BAND) // 2
        box = slide.shapes.add_textbox(
            Emu(_MARGIN), Emu(box_top),
            Emu(SLIDE_WIDTH_EMU - 2 * _MARGIN),
            Emu(SLIDE_HEIGHT_EMU - box_top - _MARGIN),
        )
        frame = box.text_frame
        frame.word_wrap = True
        for i, bullet in enumerate(bullets):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            run = para.add_run()
            run.text = f"• {bullet}"
            run.font.size = Pt(_BODY_PT - 2)
            run.font.color.rgb = _INK
        return
    box_top = _MARGIN + _TITLE_BAND
    box_height = SLIDE_HEIGHT_EMU - box_top - _MARGIN
    if full_width:
        box_left, box_width = _MARGIN, SLIDE_WIDTH_EMU - 2 * _MARGIN
    else:
        box_left = _MARGIN
        box_width = SLIDE_WIDTH_EMU // 2 - 2 * _MARGIN

    box = slide.shapes.add_textbox(
        Emu(box_left), Emu(box_top), Emu(box_width), Emu(box_height)
    )
    frame = box.text_frame
    frame.word_wrap = True
    for i, bullet in enumerate(bullets):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        run = paragraph.add_run()
        run.text = f"• {bullet}"
        run.font.size = Pt(_BODY_PT)
        run.font.color.rgb = _INK


def _add_code_box(slide: Any, blocks: list[dict[str, Any]]) -> None:
    """An editable, syntax-coloured code box in the upper content area."""
    from .pptx_template import add_code_box

    add_code_box(
        slide, blocks,
        _MARGIN, _MARGIN + _TITLE_BAND,
        SLIDE_WIDTH_EMU - 2 * _MARGIN,
        (SLIDE_HEIGHT_EMU - _TITLE_BAND - 2 * _MARGIN) // 2,
    )


def _add_table(slide: Any, rows: list[list[str]]) -> None:
    """A native, editable PowerPoint table in the upper content area."""
    frame = slide.shapes.add_table(
        max(1, len(rows)),
        max(1, len(rows[0]) if rows else 1),
        Emu(_MARGIN), Emu(_MARGIN + _TITLE_BAND),
        Emu(SLIDE_WIDTH_EMU - 2 * _MARGIN),
        Emu(min(len(rows), 12) * 320000),
    ).table
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            tf = frame.cell(r, min(c, len(frame.columns) - 1)).text_frame
            tf.text = cell
            for para in tf.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(14 if r else 16)
                    run.font.bold = r == 0


def _add_picture_fit(slide: Any, image_path: str, alt: str | None = None) -> None:
    """Place the artwork in the right half, scaled to fit and centred."""
    box_left = SLIDE_WIDTH_EMU // 2
    box_top = _MARGIN + _TITLE_BAND
    box_width = SLIDE_WIDTH_EMU - box_left - _MARGIN
    box_height = SLIDE_HEIGHT_EMU - box_top - _MARGIN

    with PILImage.open(image_path) as im:
        image_w, image_h = im.size
    scale = min(box_width / image_w, box_height / image_h)
    draw_w, draw_h = int(image_w * scale), int(image_h * scale)
    left = box_left + (box_width - draw_w) // 2
    top = box_top + (box_height - draw_h) // 2
    picture = slide.shapes.add_picture(
        image_path, Emu(left), Emu(top), Emu(draw_w), Emu(draw_h)
    )
    if alt:
        # Screen readers read this as the picture's description.
        picture._element.nvPicPr.cNvPr.set("descr", alt)  # type: ignore[attr-defined]
