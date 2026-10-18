"""Tests for the CLI interface."""

# python-pptx ships incomplete type stubs; the PowerPoint fixture below builds
# a .pptx file via the runtime API, so suppress the stub-gap diagnostics here.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

import tempfile
import wave
from pathlib import Path

import pytest
from pptx import Presentation
from typer.testing import CliRunner

from slide_stream.cli import app

# Small/fast video settings so the real-encode tests stay quick.
FAST_VIDEO_CONFIG = (
    "settings:\n"
    "  video:\n"
    "    resolution: [320, 180]\n"
    "    fps: 8\n"
    "    default_slide_duration: 1.0\n"
    "    slide_duration_padding: 0.2\n"
)


def write_silent_wav(filename, seconds=0.3):
    """A real, decodable audio file (WAV content behind any extension)."""
    with wave.open(str(filename), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * int(22050 * seconds))


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_markdown():
    """Create sample markdown content."""
    return """# Test Slide

- First point
- Second point

# Another Slide

- More content
- Final point
"""


@pytest.fixture
def sample_powerpoint():
    """Create sample PowerPoint file."""
    temp_file = Path(tempfile.mkdtemp()) / "test.pptx"

    # Create a presentation
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])  # Title and Content layout

    title = slide.shapes.title
    title.text = "Test PowerPoint Slide"

    content = slide.placeholders[1]
    content.text = "First bullet point\nSecond bullet point"

    # Add speaker notes
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = "These are test speaker notes."

    prs.save(temp_file)
    return temp_file


def test_cli_help(runner):
    """Test CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Create a video from a Markdown (.md) or PowerPoint (.pptx) file" in result.stdout


def test_cli_version(runner):
    """Test CLI version command."""
    from slide_stream import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "SlideStream" in result.stdout
    assert __version__ in result.stdout


def test_cli_no_args_shows_help(runner):
    """Bare invocation prints help and exits 0 instead of erroring."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_cli_init_refuses_overwrite_without_force(runner, tmp_path):
    """init must not clobber an existing file unless --force is given."""
    target = tmp_path / "slidestream.yaml"
    target.write_text("existing: content\n")

    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code != 0
    assert target.read_text() == "existing: content\n"

    # --force overwrites the existing file (with the minimal starter by default).
    result = runner.invoke(app, ["init", str(target), "--force"])
    assert result.exit_code == 0
    written = target.read_text()
    assert "providers:" in written
    assert "provider: text" in written  # starter's no-key image default


def test_cli_init_default_writes_minimal_starter(runner, tmp_path):
    """The default `init` is the short starter, not the full reference."""
    target = tmp_path / "slidestream.yaml"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0
    text = target.read_text()
    # Starter defaults (no keys needed) and a pointer to the full reference.
    assert "provider: none" in text
    assert "init --full" in text
    # It must NOT be the exhaustive reference.
    assert "SlideStream Configuration File" not in text


def test_cli_init_full_writes_reference(runner, tmp_path):
    """`init --full` writes the complete all-providers reference."""
    target = tmp_path / "slidestream.yaml"
    result = runner.invoke(app, ["init", str(target), "--full"])
    assert result.exit_code == 0
    text = target.read_text()
    assert "SlideStream Configuration File" in text
    assert "elevenlabs" in text  # a provider only the full reference lists


def test_cli_create_missing_input(runner):
    """Test CLI with missing input file."""
    result = runner.invoke(app, ["create"])
    assert result.exit_code != 0


def test_cli_create_basic(runner, sample_markdown, tmp_path, mocker, monkeypatch):
    """create renders a real video from Markdown, with TTS mocked out."""
    monkeypatch.chdir(tmp_path)
    md_file = tmp_path / "slides.md"
    md_file.write_text(sample_markdown)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(FAST_VIDEO_CONFIG)

    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = write_silent_wav
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    output = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        ["create", str(md_file), str(output), "--config", str(config_file)],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()


def test_cli_create_pptx_output_skips_video_pipeline(
    runner, sample_markdown, tmp_path, mocker, monkeypatch
):
    """A .pptx output filename exports an editable deck: no TTS, no ffmpeg."""
    monkeypatch.chdir(tmp_path)
    md_file = tmp_path / "slides.md"
    md_file.write_text(sample_markdown)
    config_file = tmp_path / "config.yaml"
    # Text images keep the test hermetic; llm: none keeps narration default.
    config_file.write_text(
        "providers:\n"
        "  llm:\n    provider: none\n"
        "  images:\n    provider: text\n"
        + FAST_VIDEO_CONFIG
    )

    monkeypatch.setattr("slide_stream.cli.ffmpeg_exe", lambda: None)
    tts = mocker.patch("slide_stream.cli.ProviderFactory.create_tts_provider",
                       side_effect=AssertionError("TTS must not initialize for .pptx output"))
    avatar = mocker.patch("slide_stream.cli.ProviderFactory.create_avatar_provider",
                          side_effect=AssertionError("Avatar must not initialize for .pptx output"))

    output = tmp_path / "out.pptx"
    result = runner.invoke(
        app,
        ["create", str(md_file), str(output), "--config", str(config_file), "--strict"],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    tts.assert_not_called()
    avatar.assert_not_called()

    from pptx import Presentation

    prs = Presentation(str(output))
    assert len(prs.slides) > 0
    # Narration lands in the speaker notes, not an audio track.
    first_notes = prs.slides[0].notes_slide.notes_text_frame.text.strip()
    assert first_notes


def test_cli_pptx_dry_run_ignores_video_dependencies(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("slide_stream.cli.ffmpeg_exe", lambda: None)
    (tmp_path / "deck.md").write_text("# Slide\n- Bullet\n")
    (tmp_path / "slidestream.yaml").write_text(
        "providers:\n  tts:\n    provider: kokoro\n    voice_sample: /missing/voice.wav\n"
        "  avatar:\n    provider: sadtalker\nsettings:\n  strict: true\n"
    )
    result = runner.invoke(app, ["create", "deck.md", "out.pptx", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Video length" not in result.output
    assert not (tmp_path / "out.pptx").exists()


def test_cli_create_powerpoint(runner, sample_powerpoint, tmp_path, mocker, monkeypatch):
    """create renders a real video from PowerPoint, with TTS mocked out."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(FAST_VIDEO_CONFIG)

    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = write_silent_wav
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    output = tmp_path / "out.mp4"
    try:
        result = runner.invoke(
            app,
            [
                "create",
                str(sample_powerpoint),
                str(output),
                "--config",
                str(config_file),
            ],
        )

        assert result.exit_code == 0, result.output
        assert output.exists()
    finally:
        sample_powerpoint.unlink(missing_ok=True)


def test_cli_unsupported_file_type(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # ignore any local slidestream.yml/.env
    """Test CLI with unsupported file type."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Some text content")
        f.flush()

        result = runner.invoke(app, [
            "create", f.name, "test_output.mp4"
        ])

        assert result.exit_code != 0
        assert "Unsupported file type" in result.stdout

        # Clean up
        Path(f.name).unlink(missing_ok=True)


def test_cli_nonexistent_file(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    """Test CLI with non-existent input file."""
    result = runner.invoke(app, [
        "create", "nonexistent.md", "test_output.mp4"
    ])

    assert result.exit_code != 0
    assert "Input file not found" in result.stdout


def test_cli_strict_aborts_on_unavailable_tts_provider(
    runner, sample_markdown, tmp_path, monkeypatch
):
    """--strict fails fast when the configured TTS provider is unusable,
    instead of silently rendering the whole video with gTTS."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    md_file = tmp_path / "slides.md"
    md_file.write_text(sample_markdown)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "providers:\n  tts:\n    provider: elevenlabs\n"
    )

    result = runner.invoke(
        app,
        [
            "create",
            str(md_file),
            str(tmp_path / "out.mp4"),
            "--config",
            str(config_file),
            "--strict",
        ],
    )
    assert result.exit_code == 1
    assert not (tmp_path / "out.mp4").exists()


def test_cli_strict_aborts_on_llm_narration_failure(
    runner, sample_markdown, tmp_path, mocker, monkeypatch
):
    """--strict fails fast when LLM narration returns nothing, instead of
    silently falling back to raw-bullet narration."""
    # An unconfigured voicebox (the default TTS) quietly maps to gTTS; make
    # sure a developer's env vars can't turn it into a strict-mode error
    # before the LLM branch under test is even reached.
    monkeypatch.delenv("VOICEBOX_BASE_URL", raising=False)
    monkeypatch.delenv("VOICEBOX_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    md_file = tmp_path / "slides.md"
    md_file.write_text(sample_markdown)

    mocker.patch(
        "slide_stream.cli.get_llm_client", return_value=mocker.MagicMock()
    )
    # query_llm swallows all backend errors into None.
    mocker.patch("slide_stream.cli.query_llm", return_value=None)

    output = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        [
            "create",
            str(md_file),
            str(output),
            "--llm-provider",
            "claude",
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert "LLM narration failed" in result.output
    assert not output.exists()


def test_cli_scan_skips_images_whose_description_fails(runner, tmp_path, mocker):
    """A failed vision call must not rename the file to a slug of its own
    name; the image is skipped and reported instead."""
    (tmp_path / "a.png").write_bytes(b"img")
    (tmp_path / "b.png").write_bytes(b"img")

    mocker.patch(
        "slide_stream.scan.get_llm_client", return_value=mocker.MagicMock()
    )
    mocker.patch(
        "slide_stream.scan.query_llm_with_image",
        side_effect=["golden retriever in park", None],
    )

    result = runner.invoke(app, ["scan", str(tmp_path), "--apply"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "golden-retriever-in-park.png").exists()
    assert (tmp_path / "b.png").exists()  # untouched, not renamed to "b.png" slug
    assert "b.png" in result.output


def test_cli_scan_exits_nonzero_when_all_descriptions_fail(
    runner, tmp_path, mocker
):
    (tmp_path / "a.png").write_bytes(b"img")

    mocker.patch(
        "slide_stream.scan.get_llm_client", return_value=mocker.MagicMock()
    )
    mocker.patch("slide_stream.scan.query_llm_with_image", return_value=None)

    result = runner.invoke(app, ["scan", str(tmp_path), "--apply"])

    assert result.exit_code == 1
    assert (tmp_path / "a.png").exists()  # nothing was renamed


def test_cli_scan_dry_run_matches_apply(runner, tmp_path, mocker):
    """Dry-run resolves collisions against the planned renames (names taken
    and vacated), so its preview matches what --apply actually does."""
    (tmp_path / "a.png").write_bytes(b"img")
    (tmp_path / "b.png").write_bytes(b"img")

    mocker.patch(
        "slide_stream.scan.get_llm_client", return_value=mocker.MagicMock()
    )
    # a.png -> zebra.png (vacating "a.png"), b.png -> a.png (the vacated
    # name). Two invocations (dry-run then apply) consume the side effects.
    mocker.patch(
        "slide_stream.scan.query_llm_with_image",
        side_effect=["zebra", "a", "zebra", "a"],
    )

    dry = runner.invoke(app, ["scan", str(tmp_path)])
    assert dry.exit_code == 0, dry.output
    # Without vacated-name tracking the preview would show "a-1.png".
    assert "a-1.png" not in dry.output

    applied = runner.invoke(app, ["scan", str(tmp_path), "--apply"])
    assert applied.exit_code == 0, applied.output
    assert (tmp_path / "zebra.png").exists()
    assert (tmp_path / "a.png").exists()  # b.png took the vacated name
    assert not (tmp_path / "b.png").exists()


# --- --llm-provider must not inherit another provider's settings ----------


def test_llm_override_drops_the_old_providers_model_and_server():
    """`--llm-provider claude` over an Ollama config used to send the Ollama
    model name to Claude (404), along with Ollama's URL and token."""
    from slide_stream.cli import _override_llm

    cfg = {"providers": {"llm": {
        "provider": "ollama", "model": "granite4.2:3b", "think": False,
        "base_url": "http://box:11434", "api_key": "ollama-token",
    }}}
    _override_llm(cfg, "claude", None)
    assert cfg["providers"]["llm"] == {"provider": "claude", "model": None}


def test_llm_override_keeps_settings_for_the_same_provider_and_explicit_model():
    from slide_stream.cli import _override_llm

    same = {"providers": {"llm": {"provider": "ollama", "model": "m", "base_url": "u"}}}
    _override_llm(same, "ollama", None)
    assert same["providers"]["llm"] == {"provider": "ollama", "model": "m", "base_url": "u"}

    other = {"providers": {"llm": {"provider": "ollama", "model": "m", "base_url": "u"}}}
    _override_llm(other, "claude", "claude-haiku-5-5")
    assert other["providers"]["llm"] == {"provider": "claude", "model": "claude-haiku-5-5"}

    model_only = {"providers": {"llm": {"provider": "ollama", "model": "m"}}}
    _override_llm(model_only, None, "qwen3:4b")
    assert model_only["providers"]["llm"] == {"provider": "ollama", "model": "qwen3:4b"}


def test_cli_draft_llm_provider_flag_does_not_reuse_config_model(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slidestream.yaml").write_text(
        "providers:\n  llm:\n    provider: ollama\n    model: granite4.2:3b\n"
    )
    (tmp_path / "src.txt").write_text("Hash tables find values without searching.")
    mocker.patch("slide_stream.cli.get_llm_client", return_value=object())
    query = mocker.patch("slide_stream.cli.query_llm", return_value="# A\n\n- b\n")
    result = CliRunner().invoke(app, ["draft", "src.txt", "out.md", "--llm-provider", "claude"])
    assert result.exit_code == 0, result.output
    args, kwargs = query.call_args
    assert "granite4.2:3b" not in [*args, *kwargs.values()]


# --- Accessible outputs (ADR-0018) -------------------------------------------


def _write_silent_wav(filename, seconds=0.3):
    import wave

    with wave.open(str(filename), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * int(22050 * seconds))


def _speaking_tts(mocker):
    """A TTS provider that writes a real 1s WAV per synthesize call."""
    import wave

    def speak(text, filename):
        with wave.open(str(filename), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)
        return filename

    tts = mocker.MagicMock()
    tts.name = "fake"
    tts.synthesize.side_effect = speak
    return tts


DECK = tmp_deck = (
    "# Intro\n\n- welcome to the show\n\n# Details\n\n- here is the detail\n"
)


def test_create_writes_captions_transcript_and_chapters(tmp_path, mocker, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deck.md").write_text(DECK)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(FAST_VIDEO_CONFIG)
    mocker.patch("slide_stream.cli.ProviderFactory.create_tts_provider",
                 return_value=_speaking_tts(mocker))

    output = tmp_path / "talk.mp4"
    result = CliRunner().invoke(app, [
        "create", "deck.md", str(output), "--config", str(config_file),
    ])
    assert result.exit_code == 0, result.output
    srt = (tmp_path / "talk.srt").read_text()
    assert "-->" in srt and "welcome to the show" in srt
    assert (tmp_path / "talk.vtt").exists()
    transcript = (tmp_path / "talk.transcript.md").read_text()
    assert "## 1. Intro" in transcript and "## 2. Details" in transcript
    chapters = (tmp_path / "talk.chapters.txt").read_text().splitlines()
    assert chapters[0].endswith("Intro") and len(chapters) == 2


def test_create_captions_off_reverts_to_single_call(tmp_path, mocker, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deck.md").write_text(DECK)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        FAST_VIDEO_CONFIG
        + "settings:\n  accessibility:\n    captions: none\n"
        "    transcript: false\n    chapters: false\n"
    )
    tts = _speaking_tts(mocker)
    mocker.patch("slide_stream.cli.ProviderFactory.create_tts_provider",
                 return_value=tts)
    result = CliRunner().invoke(app, [
        "create", "deck.md", "talk.mp4", "--config", str(config_file),
    ])
    assert result.exit_code == 0, result.output
    assert tts.synthesize.call_count == 2  # one per slide, no segments
    assert not (tmp_path / "talk.srt").exists()
    assert not (tmp_path / "talk.transcript.md").exists()


def test_create_alt_text_writes_descriptions(tmp_path, mocker, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deck.md").write_text(DECK)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "providers:\n  llm:\n    provider: claude\n  images:\n    provider: text\n"
        + FAST_VIDEO_CONFIG
    )
    mocker.patch("slide_stream.cli.get_llm_client", return_value=object())
    describe = mocker.patch(
        "slide_stream.llm.describe_image_alt", return_value="A teal rectangle."
    )
    result = CliRunner().invoke(app, [
        "create", "deck.md", "deck.pptx", "--config", str(config_file),
        "--alt-text",
    ])
    assert result.exit_code == 0, result.output
    assert describe.call_count == 2

    from pptx import Presentation

    prs = Presentation(str(tmp_path / "deck.pptx"))
    alts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            descr = getattr(
                getattr(getattr(shape, "_element", None), "nvPicPr", None),
                "cNvPr", None,
            )
            if descr is not None and descr.get("descr"):
                alts.append(descr.get("descr"))
    assert alts == ["A teal rectangle.", "A teal rectangle."]


def test_create_alt_text_needs_a_llm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deck.md").write_text(DECK)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(FAST_VIDEO_CONFIG)
    result = CliRunner().invoke(app, [
        "create", "deck.md", "deck.pptx", "--config", str(config_file),
        "--alt-text",
    ])
    assert result.exit_code == 1
    assert "needs an LLM provider with vision" in result.output


def test_create_burned_captions_render(tmp_path, mocker, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deck.md").write_text(DECK)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        FAST_VIDEO_CONFIG
        + "settings:\n  accessibility:\n    burn: true\n"
    )
    mocker.patch("slide_stream.cli.ProviderFactory.create_tts_provider",
                 return_value=_speaking_tts(mocker))
    output = tmp_path / "talk.mp4"
    result = CliRunner().invoke(app, [
        "create", "deck.md", str(output), "--config", str(config_file),
    ])
    assert result.exit_code == 0, result.output
    from moviepy import VideoFileClip

    with VideoFileClip(str(output)) as clip:
        assert clip.duration > 1.0
        frame = clip.get_frame(0.2)
        assert frame is not None
        # The caption band darkens the bottom of the frame.
        band = frame[int(clip.h * 0.94):, :].mean()
        assert band < 255


# --- --pptx-template: slides in the user's own style (ADR-0014) ---------------


def _template_file(tmp_path):
    from pptx import Presentation

    tpl = tmp_path / "uni.pptx"
    Presentation().save(str(tpl))
    return tpl


def test_cli_pptx_template_output_uses_the_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tpl = _template_file(tmp_path)
    (tmp_path / "deck.md").write_text(
        "# Intro\n\n- welcome\n\n# A Function\n\n```python\nx = hash(1)\n```\n"
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(FAST_VIDEO_CONFIG)
    output = tmp_path / "out.pptx"
    result = CliRunner().invoke(app, [
        "create", "deck.md", str(output), "--config", str(config_file),
        "--pptx-template", str(tpl),
    ])
    assert result.exit_code == 0, result.output
    from pptx import Presentation

    prs = Presentation(str(output))
    assert len(prs.slides) == 2
    assert prs.slides[0].slide_layout.name == "Title Slide"
    notes = prs.slides[0].notes_slide.notes_text_frame.text
    assert "Intro" in notes
    assert any("hash" in sh.text_frame.text
               for sh in prs.slides[1].shapes if sh.has_text_frame)


def test_cli_pptx_template_video_uses_rendered_frames(tmp_path, mocker, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tpl = _template_file(tmp_path)
    (tmp_path / "deck.md").write_text("# Intro\n\n- welcome\n")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(FAST_VIDEO_CONFIG)
    full = _pngs(tmp_path, 1)
    monkeypatch.setattr(
        "slide_stream.cli.pptx_to_pdf",
        lambda *a, **k: tmp_path / "template.pdf",
    )
    monkeypatch.setattr(
        "slide_stream.cli.pdf_to_slide_images", lambda *a, **k: full
    )
    fake_tts = mocker.MagicMock()
    fake_tts.save.side_effect = _write_silent_wav
    mocker.patch("gtts.gTTS", return_value=fake_tts)

    output = tmp_path / "talk.mp4"
    result = CliRunner().invoke(app, [
        "create", "deck.md", str(output), "--config", str(config_file),
        "--pptx-template", str(tpl),
    ])
    assert result.exit_code == 0, result.output
    assert output.exists()
    # The filled deck is saved beside the video, with narration in the notes.
    deck = tmp_path / "talk.pptx"
    assert deck.exists()
    from pptx import Presentation

    notes = Presentation(str(deck)).slides[0].notes_slide.notes_text_frame.text
    assert "welcome" in notes


def test_cli_pptx_template_rejects_other_design_modes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tpl = _template_file(tmp_path)
    (tmp_path / "deck.md").write_text("# A\n\n- b\n")
    result = CliRunner().invoke(app, [
        "create", "deck.md", "out.mp4",
        "--pptx-template", str(tpl), "--preserve-style",
    ])
    assert result.exit_code != 0
    assert "Choose one designed-slide source" in result.output


def test_cli_template_command_init_and_check(tmp_path):
    runner = CliRunner()
    tpl = tmp_path / "starter.pptx"
    result = runner.invoke(app, ["template", "init", str(tpl)])
    assert result.exit_code == 0, result.output
    assert tpl.exists()
    # Refuses to overwrite without --force.
    assert runner.invoke(app, ["template", "init", str(tpl)]).exit_code != 0
    result = runner.invoke(app, ["template", "check", str(tpl)])
    assert result.exit_code == 0, result.output
    assert "Title Slide" in result.output


def test_cli_dry_run_reports_the_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tpl = _template_file(tmp_path)
    (tmp_path / "deck.md").write_text("# A\n\n- b\n")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(FAST_VIDEO_CONFIG)
    result = CliRunner().invoke(app, [
        "create", "deck.md", "out.mp4", "--config", str(config_file),
        "--pptx-template", str(tpl), "--dry-run",
    ])
    assert result.exit_code == 0, result.output
    assert "template layouts found" in result.output


def _pngs(tmp_path, count):
    from PIL import Image

    paths = []
    for i in range(count):
        p = tmp_path / f"full_{i + 1}.png"
        Image.new("RGB", (320, 180), (10 * (i + 1), 0, 0)).save(p)
        paths.append(p)
    return paths
