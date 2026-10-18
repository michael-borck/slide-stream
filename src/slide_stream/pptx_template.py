"""Design new slides from the user's own PowerPoint template (ADR-0014).

The user's ``.pptx`` (or ``.potx``) is the template, used as it is:
SlideStream fills its slide masters and layouts, so the deck and the video
get the user's own fonts, colours, backgrounds and logos. Layouts are picked
by pandoc's standard names, so a Quarto or pandoc ``reference.pptx`` works
unchanged; a layout with a similar placeholder signature stands in when a
name is missing.

Only master and layout styling carries over — formatting applied by hand to
the template's example slides does not, because the example slides are
removed. ``check_template`` surfaces all of this before a render.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pptx.util import Emu, Pt

# Layout roles, matched against pandoc's standard reference-deck names.
PANDOC_NAMES = {
    "title_slide": ("title slide",),
    "section_header": ("section header",),
    "title_content": ("title and content",),
    "two_content": ("two content",),
    "title_only": ("title only",),
    "blank": ("blank",),
}
ROLE_NAMES = {role: names[0].title() for role, names in PANDOC_NAMES.items()}

# The body font is never shrunk below this share of the layout's size.
_MIN_FONT_SHARE = 0.6
_BODY_PT = 18.0
_TITLE_PT = 32.0
_CODE_PANEL_RGB = (40, 42, 54)


@dataclass
class TemplateReport:
    """What was found in the template and which layout served each slide."""

    layouts: dict[str, str]  # role -> actual layout name used
    missing: list[str] = field(default_factory=list)  # roles with a stand-in
    overflow: list[str] = field(default_factory=list)  # "slide 4: ..." notes
    template_kind: str = "pptx"  # pptx | potx


def load_template(path: str | Path):
    """Open a ``.pptx`` or ``.potx`` template as a Presentation.

    python-pptx refuses ``.potx`` files by content type; the presentation
    part's content type is rewritten in memory so the file opens unchanged.
    """
    from pptx import Presentation

    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"template not found: {src}")
    if src.suffix.lower() != ".potx":
        return Presentation(str(src)), "pptx"

    import tempfile
    import zipfile

    with zipfile.ZipFile(src) as zin:
        names = zin.namelist()
        if "[Content_Types].xml" not in names:
            raise ValueError(f"{src.name} is not a PowerPoint template")
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
                for name in names:
                    data = zin.read(name)
                    if name == "[Content_Types].xml":
                        data = data.replace(
                            b"presentationml.template.main+xml",
                            b"presentationml.presentation.main+xml",
                        )
                    zout.writestr(name, data)
    try:
        return Presentation(str(tmp_path)), "potx"
    finally:
        tmp_path.unlink(missing_ok=True)


def _placeholder_types(layout: Any) -> set[Any]:
    kinds: set[Any] = set()
    for ph in layout.placeholders:
        kinds.add(ph.placeholder_format.type)
    return kinds


def _has_body(kinds: set[Any]) -> bool:
    from pptx.enum.shapes import PP_PLACEHOLDER

    return bool(kinds & {PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT})


def _layout_name(layout: Any) -> str:
    return str(layout.name or "")


def resolve_layouts(prs: Any) -> tuple[dict[str, Any], list[str]]:
    """Pick the layout for each role: an exact pandoc name first, then the
    first layout with a similar placeholder signature. Returns the chosen
    layouts and the roles that needed a stand-in."""
    from pptx.enum.shapes import PP_PLACEHOLDER

    by_name: dict[str, Any] = {}
    for layout in prs.slide_layouts:
        by_name.setdefault(_layout_name(layout).strip().lower(), layout)

    chosen: dict[str, Any] = {}
    missing: list[str] = []
    used: set[int] = set()
    for role, names in PANDOC_NAMES.items():
        layout = next((by_name[n] for n in names if n in by_name), None)
        if layout is not None:
            chosen[role] = layout
            used.add(id(layout))
    if "blank" not in chosen and prs.slide_layouts:
        # A blank layout always exists in practice; find the emptiest.
        chosen["blank"] = min(
            prs.slide_layouts, key=lambda ly: len(ly.placeholders)
        )
        used.add(id(chosen["blank"]))

    signatures = {
        "title_slide": lambda k: (
            bool(k & {PP_PLACEHOLDER.CENTER_TITLE, PP_PLACEHOLDER.TITLE})
            and len(k) >= 2
        ),
        "section_header": _has_body,
        "title_content": lambda k: (
            bool(k & {PP_PLACEHOLDER.TITLE}) and _has_body(k)
        ),
        "two_content": lambda k: (
            bool(k & {PP_PLACEHOLDER.TITLE})
            and sum(1 for t in k if t in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT)) >= 2
        ),
        "title_only": lambda k: (
            bool(k & {PP_PLACEHOLDER.TITLE}) and not _has_body(k)
        ),
    }
    for role in PANDOC_NAMES:
        if role in chosen or role == "blank":
            continue
        missing.append(role)
        want = signatures.get(role)
        layout = next(
            (ly for ly in prs.slide_layouts
             if id(ly) not in used and want is not None and want(_placeholder_types(ly))),
            chosen.get("blank"),
        )
        if layout is not None:
            chosen[role] = layout
            used.add(id(layout))
    return chosen, missing


def _drop_example_slides(prs: Any) -> None:
    for slide in list(prs.slides):
        rid = slide.slide_id
        xml_slides = prs.slides._sldIdLst  # type: ignore[attr-defined]
        for entry in list(xml_slides):
            if entry.get("id") == str(rid):
                xml_slides.remove(entry)


def _find_placeholder(slide: Any, kinds: set[Any]) -> Any | None:
    for ph in slide.placeholders:
        if ph.placeholder_format.type in kinds:
            return ph
    return None


def _set_title(slide: Any, text: str, size: float | None = None) -> bool:
    from pptx.enum.shapes import PP_PLACEHOLDER

    ph = _find_placeholder(
        slide, {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE}
    )
    if ph is None:
        return False
    frame = ph.text_frame
    frame.word_wrap = True
    frame.text = text
    if size:
        for para in frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(size)
    return True


def _set_body(ph: Any, bullets: list[str], size: float) -> None:
    frame = ph.text_frame
    frame.word_wrap = True
    for i, bullet in enumerate(bullets):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.text = bullet
        for run in para.runs:
            run.font.size = Pt(size)


def _body_size(ph: Any, bullets: list[str]) -> tuple[float, bool]:
    """The body font size that fits the placeholder: the layout's size,
    shrunk in steps to a floor. Returns (size, fits)."""
    width = int(ph.width or 0)
    height = int(ph.height or 0)
    size = _BODY_PT
    for para in ph.text_frame.paragraphs:
        for run in para.runs:
            if run.font.size:
                size = run.font.size.pt
                break
    floor = max(9.0, size * _MIN_FONT_SHARE)
    if not width or not height or not bullets:
        return size, True
    while size > floor:
        chars_per_line = max(
            8.0, width / (size * 12700 * 0.55)
        )
        lines = sum(max(1, -(-len(b) // int(chars_per_line))) for b in bullets)
        line_height = size * 12700 * 1.25
        if lines * line_height <= height:
            return size, True
        size = max(floor, size - 1.5)
    return size, False


def _picture_into(slide: Any, layout: Any, image_path: str, right: bool) -> None:
    from pptx.enum.shapes import PP_PLACEHOLDER
    from pptx.util import Emu

    ph = _find_placeholder(slide, {PP_PLACEHOLDER.PICTURE})
    if ph is not None:
        ph.insert_picture(image_path)
        return
    # Place the picture in the right placeholder's area (Two Content), or the
    # right half of the slide when there is nothing to measure.
    area = None
    if right:
        bodies = [
            p for p in layout.placeholders
            if p.placeholder_format.type in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT)
        ]
        if bodies:
            area = max(bodies, key=lambda p: int(p.left or 0))
    if area is not None:
        x, y = int(area.left or 0), int(area.top or 0)
        w, h = int(area.width or 0), int(area.height or 0)
    else:
        # The slide's own size is authoritative (layouts may not carry one).
        x, y = 0, 0
        w, h = int(slide.width or 0) // 2, int(slide.height or 0)
    pic = slide.shapes.add_picture(image_path, Emu(x), Emu(y), Emu(w), Emu(h))
    # Keep the picture's aspect ratio inside the area, centred.
    from PIL import Image

    with Image.open(image_path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    if scale < 1.0:
        pic.width, pic.height = Emu(int(iw * scale)), Emu(int(ih * scale))
        pic.left, pic.top = Emu(x + (w - int(iw * scale)) // 2), Emu(y + (h - int(ih * scale)) // 2)


def _add_table(slide: Any, rows: list[list[str]], x: int, y: int, w: int, h: int,
               overflow: list[str], index: int) -> None:
    from pptx.util import Emu, Pt

    body_h = max(h, int(len(rows) * 240000))
    frame = slide.shapes.add_table(
        len(rows), max(1, len(rows[0]) if rows else 1), Emu(x), Emu(y), Emu(w), Emu(body_h)
    ).table
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            tf = frame.cell(r, min(c, len(frame.columns) - 1)).text_frame
            tf.text = cell
            for para in tf.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(14 if r else 16)
                    run.font.bold = r == 0
    if body_h > h:
        overflow.append(f"slide {index}: the table is taller than its area")


def add_code_box(slide: Any, blocks: list[dict[str, Any]], x: int, y: int,
                 w: int, h: int) -> None:
    """An editable, syntax-coloured code box on a slide (ADR-0013). Used by
    the template builder and the plain exporter."""
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Emu, Pt

    from .code_render import highlight_lines
    from .code_slides import PPTX_CODE_FONT

    panel = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h)
    )
    panel.fill.solid()
    panel.fill.fore_color.rgb = RGBColor.from_string("282A36")
    panel.line.fill.background()
    panel.adjustments[0] = 0.03
    frame = panel.text_frame
    frame.word_wrap = False
    pad = int(h * 0.03)
    frame.margin_left = frame.margin_right = Emu(pad)
    frame.margin_top = frame.margin_bottom = Emu(pad)

    lines: list[list[Any]] = []
    for n, block in enumerate(blocks):
        if n:
            lines.append([])
        lines += highlight_lines(
            str(block.get("code", "")), block.get("lang"), dark=True
        )
    cols = max((len("".join(r.text for r in line)) for line in lines), default=1)
    avail_w, avail_h = w - 2 * pad, h - 2 * pad
    code_pt = min(
        20.0,
        avail_w / 12700 / (max(cols, 1) * 0.6),
        avail_h / 12700 / (max(len(lines), 1) * 1.2),
    )
    code_pt = max(8.0, round(code_pt, 1))
    for n, line in enumerate(lines or [[]]):
        para = frame.paragraphs[0] if n == 0 else frame.add_paragraph()
        para.alignment = PP_ALIGN.LEFT
        for piece in line:
            run = para.add_run()
            run.text = piece.text
            run.font.name = PPTX_CODE_FONT
            run.font.size = Pt(code_pt)
            run.font.bold = piece.bold
            run.font.color.rgb = RGBColor.from_string(
                f"{piece.color[0]:02X}{piece.color[1]:02X}{piece.color[2]:02X}"
            )


def _classify(index: int, entry: dict[str, Any]) -> str:
    content = [str(c).strip() for c in entry.get("content", []) if str(c).strip()]
    if entry.get("code_blocks"):
        return "code"
    if entry.get("tables"):
        return "table"
    if index == 0 and len(content) <= 1 and sum(len(c) for c in content) <= 120:
        return "title_slide"
    if not content:
        return "section_header"
    if entry.get("image_path"):
        return "two_content"
    return "title_content"


def build_deck(
    entries: list[dict[str, Any]],
    template_path: str | Path,
    output_path: str | Path,
) -> TemplateReport:
    """Write ``entries`` (title/content/image_path/notes/code_blocks/tables)
    as slides in the template's own layouts, with narration in the speaker
    notes. Returns a report of layout choices and overflow."""
    prs, kind = load_template(template_path)
    _drop_example_slides(prs)
    layouts, missing = resolve_layouts(prs)
    report = TemplateReport(
        layouts={role: _layout_name(ly) for role, ly in layouts.items()},
        missing=missing, template_kind=kind,
    )
    sw, sh = int(prs.slide_width or 0), int(prs.slide_height or 0)
    margin = int(sw * 0.05)

    for i, entry in enumerate(entries, 1):
        role = _classify(i - 1, entry)
        slide = prs.slides.add_slide(layouts.get(role, layouts["blank"]))
        title = str(entry.get("title") or "").strip()
        if title:
            _set_title(slide, title)

        content = [str(c).strip() for c in entry.get("content", []) if str(c).strip()]
        code_blocks = entry.get("code_blocks") or []
        # Content items that are one of the code blocks are shown in the code
        # box, not as body text.
        code_text = {str(b.get("code", "")).strip() for b in code_blocks}
        content = [c for c in content if c not in code_text]

        if role == "code":
            y = int(sh * 0.18)
            add_code_box(
                slide, code_blocks or [{"lang": None, "code": "\n".join(content)}],
                margin, y, sw - 2 * margin, int(sh * 0.62),
            )
            extra = content
        elif role == "table":
            tables = entry.get("tables") or []
            rows = tables[0] if tables else []
            _add_table(slide, rows, margin, int(sh * 0.2),
                       sw - 2 * margin, int(sh * 0.6), report.overflow, i)
            # The parser also emits the table as one text item; shown as the
            # table already, it must not repeat underneath.
            as_text = "\n".join(" | ".join(cells) for cells in rows)
            extra = [c for c in content if c.strip() != as_text.strip()]
        elif role == "two_content":
            body = _find_body_placeholder(slide)
            size, fits = _body_size(body, content) if body is not None else (_BODY_PT, True)
            if body is not None:
                _set_body(body, content, size)
            else:
                fits = False
            if not fits:
                report.overflow.append(f"slide {i}: the text does not fit")
            _picture_into(slide, layouts[role], str(entry["image_path"]), right=True)
            extra = []
        elif role in ("title_slide", "section_header", "title_content"):
            body = _find_body_placeholder(slide)
            size, fits = _body_size(body, content) if body is not None else (_BODY_PT, True)
            if body is not None:
                _set_body(body, content, size)
            elif content and role != "title_slide":
                box = slide.shapes.add_textbox(
                    Emu(margin), Emu(int(sh * 0.25)),
                    Emu(sw - 2 * margin), Emu(int(sh * 0.6)),
                )
                _set_body(box, content, _BODY_PT)
            if not fits:
                report.overflow.append(f"slide {i}: the text does not fit")
            extra = []
        else:  # blank
            extra = content

        if role == "title_slide" and not title:
            _set_title(slide, f"Slide {i}")
        if extra:
            box = slide.shapes.add_textbox(
                Emu(margin), Emu(int(sh * 0.82)),
                Emu(sw - 2 * margin), Emu(int(sh * 0.14)),
            )
            _set_body(box, extra, _BODY_PT)

        notes = str(entry.get("notes") or "").strip()
        if notes:
            frame: Any = slide.notes_slide.notes_text_frame
            frame.text = notes

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return report


def _find_body_placeholder(slide: Any) -> Any | None:
    from pptx.enum.shapes import PP_PLACEHOLDER

    return _find_placeholder(slide, {PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT})


def check_template(path: str | Path) -> TemplateReport:
    """Load a template and report its layouts without building anything."""
    prs, kind = load_template(path)
    layouts, missing = resolve_layouts(prs)
    return TemplateReport(
        layouts={role: _layout_name(ly) for role, ly in layouts.items()},
        missing=missing, template_kind=kind,
    )
