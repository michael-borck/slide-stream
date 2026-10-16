"""Tests for media composition helpers."""


from PIL import Image

from slide_stream.media import compose_slide_with_bullets


def _config(resolution=(1280, 720)):
    import copy

    from slide_stream.config_loader import DEFAULT_CONFIG

    # Deep copy: a shallow .copy() shares the nested settings dicts, so
    # setting the resolution here leaked into every later test in the run.
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["settings"]["video"]["resolution"] = list(resolution)
    return cfg


def test_compose_slide_with_bullets_draws_text_and_art(tmp_path):
    art = tmp_path / "art.png"
    Image.new("RGB", (1024, 576), "darkgreen").save(art)
    out = tmp_path / "composed.png"

    result = compose_slide_with_bullets(
        str(art), "Photosynthesis", ["Light reactions", "Calvin cycle"],
        str(out), _config(),
    )

    assert result == str(out)
    img = Image.open(out)
    assert img.size == (1280, 720)
    # Text was drawn on the left half (some non-background pixels there).
    left = img.convert("L").crop((60, 80, 560, 620))
    assert left.getextrema() != left.getextrema()[0:1] + left.getextrema()[0:1]
    # Artwork occupies part of the right half (green pixels present).
    right = img.convert("RGB").crop((700, 200, 1200, 600))
    greens = [p for p in right.getdata() if p[1] > 60 and p[1] > p[0]]
    assert greens, "expected the green artwork on the right side"


def test_compose_slide_with_bullets_handles_empty_slide(tmp_path):
    art = tmp_path / "art.png"
    Image.new("RGB", (640, 360), "navy").save(art)
    out = tmp_path / "composed.png"

    result = compose_slide_with_bullets(str(art), "", [], str(out), _config())
    assert result == str(out)
    assert Image.open(out).size == (1280, 720)


# --- Text layout for code blocks and tables ---------------------------------


def test_content_lines_wraps_single_line_items_as_bullets():
    from slide_stream.providers.images import content_lines

    lines, is_block = content_lines("a b c d e f g h", width=8)
    assert is_block is False
    assert lines[0].startswith("• ")
    assert len(lines) > 1


def test_content_lines_keeps_code_indentation_and_line_breaks():
    from slide_stream.providers.images import content_lines

    code = "def f(x):\n    return x\n\ny = f(1)"
    lines, is_block = content_lines(code, width=40)
    assert is_block is True
    assert lines == ["def f(x):", "    return x", "", "y = f(1)"]


def test_content_lines_wraps_long_block_lines_under_their_indent():
    from slide_stream.providers.images import content_lines

    lines, _ = content_lines("x = 1\n    " + "word " * 10, width=20)
    assert lines[0] == "x = 1"
    assert all(len(line) <= 20 for line in lines)
    assert all(line.startswith("    ") for line in lines[1:])


def test_text_card_renders_code_and_table_slides(tmp_path):
    import copy

    from PIL import Image

    from slide_stream.config_loader import DEFAULT_CONFIG
    from slide_stream.parser import parse_markdown
    from slide_stream.providers.images import TextImageProvider

    md = "# Code\n\n```\ndef f():\n    pass\n```\n\n# T\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    provider = TextImageProvider(copy.deepcopy(DEFAULT_CONFIG))
    for i, slide in enumerate(parse_markdown(md)):
        out = tmp_path / f"s{i}.png"
        provider.generate_image("", str(out), slide=slide)
        with Image.open(out) as im:
            assert im.size == (1920, 1080)
