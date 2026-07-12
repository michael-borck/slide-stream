"""Tests for the local image provider, enrich command, and scan command."""

import copy

from PIL import Image
from typer.testing import CliRunner

from slide_stream.cli import app
from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.enrich import enrich_deck
from slide_stream.providers.factory import ProviderFactory
from slide_stream.providers.images import LocalImageProvider
from slide_stream.scan import apply_renames, slugify


def make_image(path, color=(120, 120, 200)):
    Image.new("RGB", (48, 32), color=color).save(path)


def config_with_folder(folder):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["providers"]["images"]["provider"] = "local"
    cfg["providers"]["images"]["folder"] = str(folder)
    return cfg


# --- LocalImageProvider -------------------------------------------------------


def test_local_provider_matches_by_filename_keywords(tmp_path):
    make_image(tmp_path / "quantum-computing-diagram.png")
    make_image(tmp_path / "unrelated-cat-photo.png")
    cfg = config_with_folder(tmp_path)
    provider = LocalImageProvider(cfg)
    assert provider.is_available()

    out = tmp_path / "out.png"
    result = provider.generate_image(
        "Quantum Computing", str(out),
        slide={"title": "Quantum Computing", "content": ["qubits"]},
    )
    assert result == str(out)
    assert provider.matched_last is True
    # It copied the keyword-matching image, not the cat.
    assert out.read_bytes() == (tmp_path / "quantum-computing-diagram.png").read_bytes()


def test_local_provider_each_image_used_once(tmp_path):
    make_image(tmp_path / "photosynthesis.png")
    cfg = config_with_folder(tmp_path)
    provider = LocalImageProvider(cfg)

    slide = {"title": "Photosynthesis", "content": []}
    provider.generate_image("Photosynthesis", str(tmp_path / "a.png"), slide=slide)
    assert provider.matched_last is True
    # Second slide with the same query: the only image is used up -> text fallback.
    provider.generate_image("Photosynthesis", str(tmp_path / "b.png"), slide=slide)
    assert provider.matched_last is False


def test_local_provider_no_match_falls_back_to_text(tmp_path):
    make_image(tmp_path / "banana.png")
    cfg = config_with_folder(tmp_path)
    provider = LocalImageProvider(cfg)
    out = tmp_path / "out.png"

    result = provider.generate_image(
        "Astrophysics", str(out), slide={"title": "Astrophysics", "content": []}
    )
    assert result == str(out)
    assert provider.matched_last is False
    assert out.exists()  # a text card was rendered


def test_local_provider_unavailable_without_folder():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    assert LocalImageProvider(cfg).is_available() is False


def test_factory_registers_local_provider(tmp_path):
    cfg = config_with_folder(tmp_path)
    provider = ProviderFactory.create_image_provider(cfg)
    assert isinstance(provider, LocalImageProvider)
    assert "local" in ProviderFactory.list_image_providers()


# --- enrich_deck --------------------------------------------------------------


def test_enrich_deck_writes_markdown_and_images(tmp_path):
    make_image(tmp_path / "neurons.png")
    cfg = config_with_folder(tmp_path)
    provider = LocalImageProvider(cfg)
    slides = [
        {"title": "Neurons", "content": ["Axons", "Dendrites"]},
        {"title": "Astrophysics", "content": ["Stars"]},  # no matching image
    ]
    out = tmp_path / "out"

    enrich_deck(slides, provider, out, "deck")

    md = (out / "deck.md").read_text()
    assert "# Neurons" in md
    assert "![Neurons](images/slide_1.png)" in md
    assert "- Axons" in md
    assert "\n---\n" in md  # slide separator
    assert (out / "images" / "slide_1.png").exists()
    assert (out / "images" / "slide_2.png").exists()
    # Slide 2 had no local match -> prompts.md lists it.
    prompts = (out / "prompts.md").read_text()
    assert "Astrophysics" in prompts


def test_enrich_output_round_trips_without_image_markdown(tmp_path):
    """Parsing an enriched deck must not leak `![...](...)` lines into slide
    content, so create-on-enriched-deck narration never reads image syntax."""
    from slide_stream.parser import parse_markdown

    make_image(tmp_path / "neurons.png")
    cfg = config_with_folder(tmp_path)
    provider = LocalImageProvider(cfg)
    slides = [
        {"title": "Neurons", "content": ["Axons", "Dendrites"]},
        {"title": "Astrophysics", "content": ["Stars"]},
    ]
    out = tmp_path / "out"

    enrich_deck(slides, provider, out, "deck")
    parsed = parse_markdown((out / "deck.md").read_text())

    assert [s["title"] for s in parsed] == ["Neurons", "Astrophysics"]
    for slide in parsed:
        assert all("![" not in item for item in slide["content"])
    assert parsed[0]["content"] == ["Axons", "Dendrites"]
    assert parsed[1]["content"] == ["Stars"]


def test_enrich_deck_writes_pptx(tmp_path):
    make_image(tmp_path / "topic.png")
    cfg = config_with_folder(tmp_path)
    provider = LocalImageProvider(cfg)
    out = tmp_path / "out"

    enrich_deck(
        [{"title": "Topic", "content": ["a"]}], provider, out, "deck", also_pptx=True
    )

    pptx_path = out / "deck.pptx"
    assert pptx_path.exists()
    from pptx import Presentation

    prs = Presentation(str(pptx_path))
    assert len(prs.slides) == 1


def test_enrich_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    (tmp_path / "images").mkdir()
    make_image(tmp_path / "images" / "welcome-slide.png")
    deck = tmp_path / "deck.md"
    deck.write_text("# Welcome Slide\n\n- Point one\n\n# Second\n\n- Point two\n")

    result = runner.invoke(
        app,
        [
            "enrich", str(deck), str(tmp_path / "out"),
            "--image-provider", "local",
            "--image-folder", str(tmp_path / "images"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "deck.md").exists()
    assert (tmp_path / "out" / "images" / "slide_1.png").exists()


# --- scan ---------------------------------------------------------------------


def test_slugify():
    assert slugify("A Golden Retriever, in the Park!") == "a-golden-retriever-in-the-park"
    assert slugify("") == "image"
    assert slugify("!!!") == "image"


def test_apply_renames_disambiguates_collisions(tmp_path):
    from slide_stream.scan import RenameRecord

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    make_image(a)
    make_image(b)
    # Both propose the same slug.
    records = [
        RenameRecord(original=a, new_name="dog.png"),
        RenameRecord(original=b, new_name="dog.png"),
    ]
    applied = apply_renames(records, dry_run=False)

    names = sorted(p.name for _, p in applied)
    assert names == ["dog-1.png", "dog.png"]
    assert (tmp_path / "dog.png").exists()
    assert (tmp_path / "dog-1.png").exists()


def test_apply_renames_dry_run_changes_nothing(tmp_path):
    from slide_stream.scan import RenameRecord

    a = tmp_path / "original.png"
    make_image(a)
    apply_renames([RenameRecord(original=a, new_name="renamed.png")], dry_run=True)
    assert a.exists()
    assert not (tmp_path / "renamed.png").exists()


def test_scan_cli_dry_run_no_key_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    make_image(tmp_path / "img.png")
    runner = CliRunner()

    result = runner.invoke(app, ["scan", str(tmp_path), "--provider", "claude"])
    assert result.exit_code == 1
