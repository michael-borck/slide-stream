"""Tests for the `draft` command: document -> slide-deck Markdown."""

import pytest
from typer.testing import CliRunner

from slide_stream.cli import app
from slide_stream.draft import (
    DraftError,
    build_draft_prompt,
    clamp_source,
    clean_llm_markdown,
    extract_source_text,
    validate_deck_markdown,
)

runner = CliRunner()

_DECK_MD = "# Intro\n\n- Point one\n- Point two\n\n# Details\n\n- More\n"


@pytest.fixture(autouse=True)
def _no_home_config(monkeypatch):
    """Isolate from a real ~/.slidestream.yaml on the dev machine."""
    monkeypatch.setattr("slide_stream.config_loader.find_home_config", lambda: None)


# --- Text extraction ----------------------------------------------------------


def test_extract_txt(tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("Hello world.\nSecond line.")
    assert "Hello world." in extract_source_text(src)


def test_extract_markdown(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("# Heading\n\nBody text.")
    assert "Body text." in extract_source_text(src)


def test_extract_unsupported_suffix(tmp_path):
    src = tmp_path / "sheet.xlsx"
    src.write_text("x")
    with pytest.raises(DraftError, match="Unsupported source type"):
        extract_source_text(src)


def test_extract_pptx_roundtrips_titles_and_notes(tmp_path):
    """A .pptx source is flattened to titles/bullets/notes for the LLM."""
    from pptx import Presentation

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])  # title + content
    slide.shapes.title.text = "Photosynthesis"  # type: ignore[union-attr]
    slide.placeholders[1].text = "Light reactions"  # type: ignore[attr-defined]
    slide.notes_slide.notes_text_frame.text = "Mention chlorophyll"  # type: ignore[union-attr]
    pptx = tmp_path / "in.pptx"
    prs.save(pptx)

    text = extract_source_text(pptx)
    assert "# Photosynthesis" in text
    assert "Light reactions" in text
    assert "Mention chlorophyll" in text


# --- Prompt / output helpers --------------------------------------------------


def test_prompt_pins_exact_slide_count():
    assert "exactly 7 slides" in build_draft_prompt("source", 7)


def test_prompt_lets_llm_choose_when_no_count():
    prompt = build_draft_prompt("source", None)
    assert "Choose a sensible number" in prompt


def test_clamp_source_truncates_long_input():
    text = "x" * 200_000
    clamped, truncated = clamp_source(text)
    assert truncated is True
    assert len(clamped) < len(text)


def test_clamp_source_leaves_short_input():
    clamped, truncated = clamp_source("short")
    assert (clamped, truncated) == ("short", False)


def test_clean_llm_markdown_strips_code_fence():
    fenced = "```markdown\n# Title\n\n- a\n```"
    assert clean_llm_markdown(fenced) == "# Title\n\n- a"


def test_validate_deck_rejects_empty():
    with pytest.raises(DraftError, match="no '# ' headings"):
        validate_deck_markdown("just prose, no headings")


def test_validate_deck_accepts_real_deck():
    slides = validate_deck_markdown(_DECK_MD)
    assert [s["title"] for s in slides] == ["Intro", "Details"]


# --- CLI ----------------------------------------------------------------------


def test_cli_draft_writes_markdown(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "report.txt"
    src.write_text("A long report about widgets and gadgets.")

    mocker.patch("slide_stream.cli.get_llm_client", return_value=mocker.MagicMock())
    mocker.patch("slide_stream.cli.query_llm", return_value=_DECK_MD)

    result = runner.invoke(app, ["draft", str(src), "--llm-provider", "openai"])
    assert result.exit_code == 0, result.output
    out = tmp_path / "report.md"
    assert out.exists()
    assert "# Intro" in out.read_text()


def test_cli_draft_honours_output_arg_and_slides(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "report.txt"
    src.write_text("content")

    mocker.patch("slide_stream.cli.get_llm_client", return_value=mocker.MagicMock())
    q = mocker.patch("slide_stream.cli.query_llm", return_value=_DECK_MD)

    result = runner.invoke(
        app, ["draft", str(src), "deck.md", "--slides", "5", "--llm-provider", "openai"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "deck.md").exists()
    # The exact-count instruction reached the prompt.
    assert "exactly 5 slides" in q.call_args[0][2]


def test_cli_draft_requires_llm_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "report.txt"
    src.write_text("content")

    # No --llm-provider and default provider is 'none' -> clean error.
    result = runner.invoke(app, ["draft", str(src)])
    assert result.exit_code == 1
    assert "draft needs an LLM" in " ".join(result.output.split())


def test_cli_draft_refuses_overwrite_without_force(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "report.txt"
    src.write_text("content")
    (tmp_path / "report.md").write_text("existing")

    mocker.patch("slide_stream.cli.get_llm_client", return_value=mocker.MagicMock())
    mocker.patch("slide_stream.cli.query_llm", return_value=_DECK_MD)

    result = runner.invoke(app, ["draft", str(src), "--llm-provider", "openai"])
    assert result.exit_code == 1
    # Normalise whitespace: Rich wraps the long path across lines in narrow (CI)
    # terminals, which would otherwise split the "already exists" phrase.
    assert "already exists" in " ".join(result.output.split())
    assert (tmp_path / "report.md").read_text() == "existing"


def test_cli_draft_missing_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["draft", "nope.txt", "--llm-provider", "openai"])
    assert result.exit_code == 1
    assert "not found" in " ".join(result.output.split())


# --- PDF and Word input (the `import` extra) ---------------------------------


def _write_pdf(path, pages):
    """A minimal valid PDF with one line of Helvetica text per page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    font_obj = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {4 + 2 * i} 0 R /Resources << /Font << /F1 {font_obj} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = "%PDF-1.4\n", []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out.encode()))
        out += f"{n} 0 obj\n{body}\nendobj\n"
    xref = len(out.encode())
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode())
    return path


def test_extract_pdf_reads_every_page(tmp_path):
    pytest.importorskip("pypdf")
    pdf = _write_pdf(tmp_path / "report.pdf", ["Hash tables are fast", "Collisions happen"])
    text = extract_source_text(pdf)
    assert "Hash tables are fast" in text
    assert "Collisions happen" in text
    assert text.index("Hash tables") < text.index("Collisions")


def test_extract_docx_keeps_paragraphs_and_skips_blank_ones(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_heading("Report title", level=1)
    doc.add_paragraph("First finding.")
    doc.add_paragraph("")
    doc.add_paragraph("Second finding.")
    path = tmp_path / "report.docx"
    doc.save(str(path))
    assert extract_source_text(path) == "Report title\nFirst finding.\nSecond finding."


def test_extract_pdf_without_the_extra_explains_how_to_install(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(DraftError, match=r"slide-stream\[import\]"):
        extract_source_text(tmp_path / "x.pdf")


def test_extract_docx_without_the_extra_explains_how_to_install(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "docx", None)
    with pytest.raises(DraftError, match=r"slide-stream\[import\]"):
        extract_source_text(tmp_path / "x.docx")
