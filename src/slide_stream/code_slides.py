"""SlideStream-built code slides inside designed decks (ADR-0013).

No Presenton template has a layout for code, so it drops or paraphrases
code blocks. For those slides SlideStream builds the slide itself, in both
outputs so they match:

- the video frame (``render_code_slide``), and
- the editable ``.pptx`` (``rewrite_pptx_code_slides``), which keeps the
  slide's position, background and notes but replaces its shapes with the
  title and an editable, syntax-coloured code box.

Colours come from the neighbouring designed frames (``sample_theme``) so the
code slide does not stand out; SlideStream's own look is the fallback.
"""

import colorsys
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .code_render import (
    CodeRun,
    Color,
    draw_code_lines,
    find_code_block,
    highlight_lines,
    is_dark,
    to_rgb,
    wrap_runs,
)

CODE_SLIDE_MODES = ("auto", "always", "never")
# Courier New is installed with every office suite (PowerPoint, Keynote,
# LibreOffice, Google Slides). Consolas looks better but LibreOffice and
# Keynote substitute a proportional font for it, which breaks alignment.
PPTX_CODE_FONT = "Courier New"
_EMU_PER_PT = 12700


@dataclass
class Theme:
    """Colours for a code slide."""

    bg: Color
    fg: Color
    accent: Color
    panel: Color  # background of the code box
    dark: bool  # whether the code box is dark (picks the Pygments style)


def _mix(a: Color, b: Color, t: float) -> Color:
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def theme_from(bg: Color, accent: Color | None = None) -> Theme:
    """A complete theme from a background (and accent) colour."""
    dark = is_dark(bg)
    fg: Color = (240, 240, 240) if dark else (34, 34, 34)
    if accent is None:
        accent = (96, 165, 250) if dark else (79, 70, 229)
    # The code box: a little lighter on dark slides, a dark editor-style box
    # on light slides, which reads well and suits the dark Pygments style.
    panel = _mix(bg, (255, 255, 255), 0.07) if dark else (40, 42, 54)
    return Theme(bg=bg, fg=fg, accent=accent, panel=panel, dark=is_dark(panel))


def plain_theme(config: dict[str, Any]) -> Theme:
    """SlideStream's own look, from ``settings.image``."""
    image = config.get("settings", {}).get("image", {})
    return theme_from(to_rgb(image.get("bg_color", "black")))


def _quantize(c: tuple[int, ...]) -> Color:
    return (c[0] // 8 * 8, c[1] // 8 * 8, c[2] // 8 * 8)


def _distance(a: Color, b: Color) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def sample_theme(frames: list[Path]) -> Theme | None:
    """Theme from designed slide frames: the most common edge colour is the
    background, the most frequent saturated colour the accent. Returns None
    if there is nothing usable to sample."""
    edge: Counter[Color] = Counter()
    edge_sum: dict[Color, list[int]] = {}  # exact colour totals per bucket
    vivid: Counter[Color] = Counter()
    for path in frames:
        try:
            with Image.open(path) as im:
                small = im.convert("RGB").resize((160, 90))
        except OSError:
            continue
        w, h = small.size
        pixels = small.load()
        assert pixels is not None
        border = [(x, y) for x in range(w) for y in (0, 1, h - 2, h - 1)]
        border += [(x, y) for y in range(h) for x in (0, 1, w - 2, w - 1)]
        for x, y in border:
            px: Any = pixels[x, y]
            key = _quantize(px)
            edge[key] += 1
            total = edge_sum.setdefault(key, [0, 0, 0])
            for c in range(3):
                total[c] += px[c]
        for p in small.getdata():
            r, g, b = p[0], p[1], p[2]
            _, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if s > 0.4 and v > 0.5:
                vivid[_quantize((r, g, b))] += 1
    if not edge:
        return None
    bucket, count = edge.most_common(1)[0]
    # The exact average of that bucket, so the slide matches its neighbours
    # rather than being a shade off.
    bg = (
        round(edge_sum[bucket][0] / count),
        round(edge_sum[bucket][1] / count),
        round(edge_sum[bucket][2] / count),
    )
    # Among common vivid colours, one that stands out from the background.
    accent = next(
        (c for c, n in vivid.most_common(12) if _distance(c, bg) > 140 and n >= 4),
        None,
    )
    return theme_from(bg, accent)


def neighbour_frames(index: int, frames: list[Path], replaced: set[int]) -> list[Path]:
    """The nearest designed frames before and after a slide that are not
    being replaced themselves."""
    found: list[Path] = []
    for step in (-1, 1):
        i = index + step
        while 0 <= i < len(frames):
            if i not in replaced:
                found.append(frames[i])
                break
            i += step
    return found


def choose_code_slides(
    slides: list[dict[str, Any]], code_missing: list[int], mode: str
) -> list[int]:
    """0-based slides to rebuild: ``auto`` where the fidelity check found the
    code missing, ``always`` every slide with code, ``never`` none."""
    if mode == "never":
        return []
    if mode == "always":
        return [i for i, s in enumerate(slides) if s.get("code_blocks")]
    return sorted(i for i in code_missing if slides[i].get("code_blocks"))


def _split_content(slide: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The slide's code blocks, and its other content items."""
    blocks = list(slide.get("code_blocks") or [])
    others = [
        str(c).strip() for c in slide.get("content", [])
        if str(c).strip() and find_code_block(str(c), blocks) is None
    ]
    return blocks, others


def _code_runs(blocks: list[dict[str, Any]], dark: bool, fg: Color) -> list[list[CodeRun]]:
    """All of a slide's code blocks as coloured lines, a blank line apart."""
    lines: list[list[CodeRun]] = []
    for n, block in enumerate(blocks):
        if n:
            lines.append([])
        lines += highlight_lines(str(block.get("code", "")), block.get("lang"),
                                 dark=dark, fg=fg)
    return lines


def render_code_slide(
    dest: str | Path, slide: dict[str, Any], theme: Theme, size: tuple[int, int]
) -> Path:
    """Draw a code slide frame: title, accent rule, code box, other text."""
    from .providers.images import load_bold_font, load_font, load_mono_font

    width, height = size
    img = Image.new("RGB", size, theme.bg)
    draw = ImageDraw.Draw(img)
    margin = int(width * 0.06)
    y = int(height * 0.08)

    title = str(slide.get("title", "")).strip()
    if title and slide.get("has_real_title", True):
        title_size = int(height * 0.072)
        title_font = load_bold_font(title_size)
        for line in textwrap.wrap(title, width=48) or [title]:
            draw.text((margin, y), line, font=title_font, fill=theme.fg)
            y += int(title_size * 1.2)
        y += int(height * 0.012)
        draw.rectangle(
            (margin, y, margin + int(width * 0.05), y + max(3, int(height * 0.005))),
            fill=theme.accent,
        )
        y += int(height * 0.04)

    blocks, others = _split_content(slide)
    body_size = int(height * 0.032)
    body_font = load_font(body_size)
    other_lines = [
        line for item in others
        for line in (textwrap.wrap(f"• {item}", width=90) or [f"• {item}"])
    ]
    others_height = int(len(other_lines) * body_size * 1.45 + (height * 0.03 if other_lines else 0))

    bottom = int(height * 0.92) - others_height
    pad = int(height * 0.025)
    box_w = width - 2 * margin
    raw = _code_runs(blocks, theme.dark, (240, 240, 240) if theme.dark else (34, 34, 34))
    # The largest code font, within limits, at which the code fits the box.
    code_size = int(height * 0.042)
    min_size = max(10, int(height * 0.016))
    while True:
        font = load_mono_font(code_size)
        cols = max(10, int((box_w - 2 * pad) / max(1.0, font.getlength("M"))))
        lines = wrap_runs(raw, cols)
        line_h = code_size * 1.35
        if y + 2 * pad + len(lines) * line_h <= bottom or code_size <= min_size:
            break
        code_size -= 1
    box_bottom = min(bottom, int(y + 2 * pad + len(lines) * line_h))
    draw.rounded_rectangle(
        (margin, y, margin + box_w, box_bottom),
        radius=int(height * 0.015), fill=theme.panel,
    )
    draw_code_lines(draw, margin + pad, y + pad, lines, font, line_h,
                    max_y=box_bottom - pad + 1)
    y = box_bottom + int(height * 0.03)

    for line in other_lines:
        draw.text((margin, y), line, font=body_font, fill=theme.fg)
        y += int(body_size * 1.45)

    out = Path(dest)
    img.save(out)
    return out


def _hex(c: Color) -> str:
    return f"{c[0]:02X}{c[1]:02X}{c[2]:02X}"


def _title_font(prs: Any, index: int) -> tuple[str | None, Any]:
    """Font name and size of the title on the nearest other slide: the
    largest text run, which is the title on Presenton slides."""
    best: tuple[float, str | None, Any] = (0.0, None, None)
    order = sorted(range(len(prs.slides)), key=lambda i: (abs(i - index), i))
    for i in order:
        if i == index:
            continue
        for shape in prs.slides[i].shapes:
            if not getattr(shape, "has_text_frame", False) or not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    size = run.font.size
                    if size is not None and run.text.strip() and size.pt > best[0]:
                        best = (size.pt, run.font.name, size)
        if best[2] is not None:
            break
    return best[1], best[2]


def rewrite_pptx_code_slides(
    pptx_path: str | Path, slides: list[dict[str, Any]], indices: list[int],
    theme: Theme,
) -> None:
    """Replace the chosen slides' shapes with a title and an editable,
    syntax-coloured code box, in place.

    Slide order, count and speaker notes are untouched.
    """
    if not indices:
        return
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Emu, Pt

    prs = Presentation(str(pptx_path))
    sw, sh = int(prs.slide_width or Emu(12192000)), int(prs.slide_height or Emu(6858000))
    for index in indices:
        if index >= len(prs.slides):
            continue
        slide, source = prs.slides[index], slides[index]
        font_name, font_size = _title_font(prs, index)
        for shape in list(slide.shapes):
            element = shape._element
            element.getparent().remove(element)
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor.from_string(_hex(theme.bg))

        margin = int(sw * 0.06)
        y = int(sh * 0.08)
        title = str(source.get("title", "")).strip()
        if title and source.get("has_real_title", True):
            box = slide.shapes.add_textbox(
                Emu(margin), Emu(y), Emu(sw - 2 * margin), Emu(int(sh * 0.12))
            )
            frame = box.text_frame
            frame.word_wrap = True
            run = frame.paragraphs[0].add_run()
            run.text = title
            run.font.size = font_size or Pt(32)
            run.font.color.rgb = RGBColor.from_string(_hex(theme.fg))
            if font_name:
                run.font.name = font_name
            y += int(sh * 0.13)
            rule = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Emu(margin), Emu(y),
                Emu(int(sw * 0.05)), Emu(max(1, int(sh * 0.006))),
            )
            rule.fill.solid()
            rule.fill.fore_color.rgb = RGBColor.from_string(_hex(theme.accent))
            rule.line.fill.background()
            y += int(sh * 0.04)

        blocks, others = _split_content(source)
        lines = _code_runs(blocks, theme.dark, (240, 240, 240) if theme.dark else (34, 34, 34))
        others_h = int(len(others) * sh * 0.06)
        pad_x, pad_y = int(sh * 0.025), int(sh * 0.02)
        avail_h = int(sh * 0.92) - y - others_h - 2 * pad_y
        avail_w = sw - 2 * margin - 2 * pad_x
        cols = max((len("".join(r.text for r in line)) for line in lines), default=1)
        # Largest size up to 20 pt at which the code fits; Courier New's
        # glyphs are 0.6 em wide and lines are set at 1.2 em.
        code_pt = min(
            20.0,
            avail_w / _EMU_PER_PT / (max(cols, 1) * 0.6),
            avail_h / _EMU_PER_PT / (max(len(lines), 1) * 1.2),
        )
        code_pt = max(8.0, round(code_pt, 1))
        box_h = int(max(len(lines), 1) * code_pt * 1.2 * _EMU_PER_PT) + 2 * pad_y
        panel = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Emu(margin), Emu(y),
            Emu(sw - 2 * margin), Emu(box_h),
        )
        panel.fill.solid()
        panel.fill.fore_color.rgb = RGBColor.from_string(_hex(theme.panel))
        panel.line.fill.background()
        panel.adjustments[0] = 0.03
        frame = panel.text_frame
        frame.word_wrap = False
        frame.margin_left = frame.margin_right = Emu(pad_x)
        frame.margin_top = frame.margin_bottom = Emu(pad_y)
        frame.vertical_anchor = MSO_ANCHOR.TOP
        for n, line in enumerate(lines or [[]]):
            para = frame.paragraphs[0] if n == 0 else frame.add_paragraph()
            para.alignment = PP_ALIGN.LEFT
            for piece in line or [CodeRun("", theme.fg)]:
                run = para.add_run()
                run.text = piece.text
                run.font.name = PPTX_CODE_FONT
                run.font.size = Pt(code_pt)
                run.font.bold = piece.bold
                run.font.color.rgb = RGBColor.from_string(_hex(piece.color))
        y += box_h + int(sh * 0.03)

        if others:
            box = slide.shapes.add_textbox(
                Emu(margin), Emu(y), Emu(sw - 2 * margin), Emu(max(1, others_h))
            )
            frame = box.text_frame
            frame.word_wrap = True
            for n, item in enumerate(others):
                para = frame.paragraphs[0] if n == 0 else frame.add_paragraph()
                run = para.add_run()
                run.text = f"• {item}"
                run.font.size = Pt(16)
                run.font.color.rgb = RGBColor.from_string(_hex(theme.fg))
    prs.save(str(pptx_path))
