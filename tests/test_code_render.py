"""Tests for syntax-highlighted code rendering (ADR-0013)."""

from PIL import Image, ImageDraw

from slide_stream.code_render import (
    draw_code_lines,
    find_code_block,
    highlight_lines,
    is_dark,
    wrap_runs,
)
from slide_stream.providers.images import load_mono_font

CODE = 'def f(x):\n    return x + 1  # add one\n\nprint("hi")'


def _text(lines):
    return ["".join(r.text for r in line) for line in lines]


def test_highlight_keeps_text_and_indentation_exactly():
    lines = highlight_lines(CODE, "python", dark=True)
    assert _text(lines) == CODE.split("\n")


def test_highlight_colours_differ_by_token_type():
    (line,) = highlight_lines("def f(): return 1", "python", dark=True)
    colours = {r.color for r in line if r.text.strip()}
    assert len(colours) >= 3


def test_light_and_dark_styles_differ():
    dark = highlight_lines("def f(): pass", "python", dark=True)[0][0].color
    light = highlight_lines("def f(): pass", "python", dark=False)[0][0].color
    assert dark != light


def test_unknown_language_falls_back_to_plain_text():
    lines = highlight_lines("some words", "no-such-language", dark=True, fg=(1, 2, 3))
    assert _text(lines) == ["some words"]
    assert {r.color for r in lines[0]} == {(1, 2, 3)}


def test_wrap_keeps_every_character_and_indents_continuations():
    lines = highlight_lines("    value = alpha + beta + gamma + delta", "python", dark=True)
    wrapped = wrap_runs(lines, 20)
    texts = _text(wrapped)
    assert all(len(t) <= 20 for t in texts)
    assert texts[1].startswith("      ")  # 4-space indent + 2
    rejoined = texts[0] + "".join(t[6:] for t in texts[1:])
    assert rejoined == "    value = alpha + beta + gamma + delta"


def test_wrap_breaks_at_a_space_when_it_can():
    wrapped = _text(wrap_runs(highlight_lines("aaaa bbbb cccc", None, dark=True), 11))
    assert wrapped[0] == "aaaa bbbb "


def test_find_code_block_ignores_surrounding_whitespace():
    blocks = [{"lang": "python", "code": "    x = 1"}]
    assert find_code_block("x = 1", blocks) == blocks[0]
    assert find_code_block("y = 2", blocks) is None
    assert find_code_block("x = 1", None) is None


def test_is_dark():
    assert is_dark("black") and is_dark((20, 30, 40))
    assert not is_dark("white") and not is_dark("#f4efe6")


def test_draw_code_lines_paints_coloured_pixels():
    img = Image.new("RGB", (400, 120), "black")
    lines = highlight_lines("def f(): return 1", "python", dark=True)
    y = draw_code_lines(ImageDraw.Draw(img), 10, 10, lines, load_mono_font(20), 26)
    assert y == 36
    colours = {p for p in img.getdata() if p != (0, 0, 0)}
    assert len(colours) > 10
