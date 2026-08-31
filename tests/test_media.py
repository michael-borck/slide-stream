"""Tests for media composition helpers."""


from PIL import Image

from slide_stream.media import compose_slide_with_bullets


def _config(resolution=(1280, 720)):
    from slide_stream.config_loader import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG.copy()
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
