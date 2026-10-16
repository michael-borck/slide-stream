"""Syntax-highlighted code for slide frames and decks (ADR-0013).

Pygments (BSD-2-Clause) splits code into coloured runs. Frames draw the runs
with Pillow in a monospace font; the designed-deck fix writes them as
coloured text runs, so code stays editable in PowerPoint.
"""

from dataclasses import dataclass
from typing import Any

from PIL import ImageColor

Color = tuple[int, int, int]

DARK_STYLE = "monokai"
LIGHT_STYLE = "friendly"
_LIGHT_FG: Color = (34, 34, 34)
_DARK_FG: Color = (240, 240, 240)


@dataclass
class CodeRun:
    """A piece of one code line drawn in one colour."""

    text: str
    color: Color
    bold: bool = False


def to_rgb(color: Any) -> Color:
    """Any Pillow colour (name, ``#hex``, tuple) as an RGB tuple."""
    if isinstance(color, tuple):
        return (int(color[0]), int(color[1]), int(color[2]))
    rgb = ImageColor.getrgb(str(color))
    return (rgb[0], rgb[1], rgb[2])


def is_dark(color: Any) -> bool:
    """Whether a background colour needs light text."""
    r, g, b = to_rgb(color)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 128


def _lexer(code: str, lang: str | None) -> Any:
    from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer
    from pygments.util import ClassNotFound

    options = {"stripnl": False, "ensurenl": False, "tabsize": 4}
    if lang:
        try:
            return get_lexer_by_name(lang, **options)
        except ClassNotFound:
            pass
    else:
        try:
            return guess_lexer(code, **options)
        except ClassNotFound:
            pass
    return TextLexer(**options)


def highlight_lines(
    code: str, lang: str | None, *, dark: bool, fg: Color | None = None
) -> list[list[CodeRun]]:
    """Code as lines of coloured runs.

    ``lang`` is the fence's language (``python``); without one Pygments
    guesses, and an unknown language falls back to plain text in ``fg``.
    """
    from pygments.styles import get_style_by_name
    from pygments.token import Token

    style = get_style_by_name(DARK_STYLE if dark else LIGHT_STYLE)
    default = fg or (_DARK_FG if dark else _LIGHT_FG)
    # Plain text keeps the slide's own text colour, not the style's.
    base = style.style_for_token(Token.Text).get("color")
    lines: list[list[CodeRun]] = [[]]
    for ttype, value in _lexer(code, lang).get_tokens(code.expandtabs(4)):
        spec = style.style_for_token(ttype)
        raw = spec.get("color")
        color = to_rgb("#" + raw) if raw and raw != base else default
        bold = bool(spec.get("bold"))
        parts = value.split("\n")
        for i, part in enumerate(parts):
            if i:
                lines.append([])
            if part:
                lines[-1].append(CodeRun(part, color, bold))
    while lines and not lines[-1]:
        lines.pop()  # trailing newline from the lexer
    return lines


def wrap_runs(lines: list[list[CodeRun]], width: int) -> list[list[CodeRun]]:
    """Wrap lines longer than ``width`` characters, keeping colours.

    Continuation lines repeat the line's indentation plus two spaces.
    """
    out: list[list[CodeRun]] = []
    for line in lines:
        text = "".join(r.text for r in line)
        if len(text) <= width:
            out.append(line)
            continue
        indent = " " * (len(text) - len(text.lstrip(" "))) + "  "
        chars = [(ch, r.color, r.bold) for r in line for ch in r.text]
        first, limit = True, width
        while chars:
            cut = limit
            if len(chars) > limit:
                # Prefer breaking after the last space in the second half.
                spaces = [i for i, (ch, _, _) in enumerate(chars[:limit]) if ch == " "]
                if spaces and spaces[-1] >= limit // 2:
                    cut = spaces[-1] + 1
            chunk, chars = chars[:cut], chars[cut:]
            runs: list[CodeRun] = [] if first else [CodeRun(indent, chunk[0][1])]
            for ch, color, bold in chunk:
                if runs and runs[-1].color == color and runs[-1].bold == bold:
                    runs[-1].text += ch
                else:
                    runs.append(CodeRun(ch, color, bold))
            out.append(runs)
            first, limit = False, max(1, width - len(indent))
    return out


def find_code_block(
    item: str, code_blocks: list[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """The slide code block a content item came from, if any. Compared
    without surrounding whitespace, since some callers strip items."""
    for block in code_blocks or []:
        if str(block.get("code", "")).strip() == item.strip():
            return block
    return None


def draw_code_lines(
    draw: Any, x: float, y: float, lines: list[list[CodeRun]],
    font: Any, line_height: float, max_y: float | None = None,
) -> float:
    """Draw coloured lines with Pillow; returns the y below the last line."""
    for line in lines:
        if max_y is not None and y + line_height > max_y:
            break
        cx = x
        for run in line:
            draw.text((cx, y), run.text, font=font, fill=run.color)
            cx += font.getlength(run.text)
        y += line_height
    return y
