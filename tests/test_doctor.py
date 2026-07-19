"""Tests for the deck preflight (doctor / create --dry-run)."""

import copy
import wave
from io import BytesIO

from PIL import Image
from typer.testing import CliRunner

from slide_stream.cli import app
from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.doctor import BLOCKER, WARN, run_doctor


def cfg():
    return copy.deepcopy(DEFAULT_CONFIG)


def opts(**kw):
    base = {
        "mode": "create", "input_ext": ".md", "verbatim_notes": False,
        "script_blocks": None, "avatar_enabled": False,
        "narration_seconds": None, "output_path": None,
    }
    base.update(kw)
    return base


def has(report, substr, severity=None):
    return any(
        substr in f.message
        for f in report.findings
        if severity is None or f.severity == severity
    )


def write_wav(path, seconds, rate=22050):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def png_bytes(w, h):
    buf = BytesIO()
    Image.new("RGB", (w, h), "red").save(buf, format="PNG")
    return buf.getvalue()


# --- deck ---------------------------------------------------------------------


def test_doctor_counts_slides_and_missing_notes():
    slides = [
        {"title": "A", "content": ["x"], "notes": "note one"},
        {"title": "B", "content": ["y"]},  # no notes
    ]
    report = run_doctor(slides, cfg(), opts(input_ext=".pptx"))
    assert has(report, "2 slide(s)")
    assert has(report, "missing speaker notes", WARN)


def test_doctor_flags_empty_slide_as_blocker():
    slides = [{"title": "", "content": [], "notes": "", "has_real_title": False}]
    report = run_doctor(slides, cfg(), opts(input_ext=".pptx"))
    assert has(report, "empty slide", BLOCKER)


# --- narration ----------------------------------------------------------------


def test_doctor_flags_stage_direction_in_notes():
    slides = [{"title": "A", "content": [], "notes": "Welcome. [pause] Now click to advance."}]
    report = run_doctor(slides, cfg(), opts(input_ext=".pptx", verbatim_notes=True))
    assert has(report, "stage direction", WARN)


def test_doctor_flags_overlong_note_vs_target():
    long_note = " ".join(["word"] * 400)
    slides = [{"title": "A", "content": ["x"], "notes": long_note}]
    report = run_doctor(slides, cfg(), opts(input_ext=".pptx", narration_seconds=30))
    assert has(report, "summarised to fit", WARN)


# --- voice --------------------------------------------------------------------


def test_doctor_voice_sample_too_short_is_blocker(tmp_path):
    sample = tmp_path / "v.wav"
    write_wav(sample, 0.5)
    c = cfg()
    c["providers"]["tts"] = {"provider": "chatterbox", "voice_sample": str(sample)}
    report = run_doctor([{"title": "A", "content": ["x"]}], c, opts())
    assert has(report, "under ~5s", BLOCKER)


def test_doctor_voice_sample_missing_is_blocker():
    c = cfg()
    c["providers"]["tts"] = {"provider": "chatterbox", "voice_sample": "/no/such.wav"}
    report = run_doctor([{"title": "A", "content": ["x"]}], c, opts())
    assert has(report, "not found", BLOCKER)


def test_doctor_voice_sample_good_length_ok(tmp_path):
    sample = tmp_path / "v.wav"
    write_wav(sample, 18)
    c = cfg()
    c["providers"]["tts"] = {"provider": "chatterbox", "voice_sample": str(sample)}
    report = run_doctor([{"title": "A", "content": ["x"]}], c, opts())
    assert not has(report, "under ~5s")
    assert has(report, "18.0s")


# --- images -------------------------------------------------------------------


def test_doctor_flags_small_embedded_image():
    slides = [{"title": "A", "content": [], "images": [{"data": png_bytes(320, 180)}]}]
    report = run_doctor(slides, cfg(), opts(input_ext=".pptx"))
    assert has(report, "smaller than", WARN)


# --- avatar -------------------------------------------------------------------


def test_doctor_mascot_on_human_only_engine_is_blocker():
    c = cfg()
    c["providers"]["avatar"] = {"provider": "sadtalker", "base_url": "https://c",
                                "source_image": "teddy"}
    report = run_doctor([{"title": "A", "content": ["x"]}], c,
                        opts(avatar_enabled=True))
    assert has(report, "won't animate", BLOCKER)


def test_doctor_wan_s2v_slow_render_warning():
    c = cfg()
    c["providers"]["avatar"] = {"provider": "wan-s2v", "base_url": "https://c", "source": "owl"}
    report = run_doctor([{"title": "A", "content": ["x"]}], c, opts(avatar_enabled=True))
    assert has(report, "min/slide", WARN)
    assert not has(report, "needs a source")  # source is set


# --- providers / env ----------------------------------------------------------


def test_doctor_ffmpeg_missing_is_blocker(monkeypatch):
    monkeypatch.setattr("slide_stream.doctor.shutil.which", lambda _: None)
    c = cfg()
    c["providers"]["avatar"] = {"provider": "wan-s2v", "base_url": "https://c", "source": "owl"}
    report = run_doctor([{"title": "A", "content": ["x"]}], c, opts(avatar_enabled=True))
    assert has(report, "ffmpeg not found", BLOCKER)


# --- estimates + exit code ----------------------------------------------------


def test_doctor_estimates_video_length():
    report = run_doctor([{"title": "A", "content": ["x"]}], cfg(), opts(narration_seconds=30))
    assert any("Video length" in e for e in report.estimates)


def test_exit_code_respects_fail_on_warn():
    slides = [{"title": "A", "content": ["x"], "notes": "Welcome. [pause] click."}]
    report = run_doctor(slides, cfg(), opts(input_ext=".pptx", verbatim_notes=True))
    assert report.warnings >= 1 and report.blockers == 0
    assert report.exit_code(fail_on_warn=False) == 0
    assert report.exit_code(fail_on_warn=True) == 1


# --- CLI ----------------------------------------------------------------------


def test_cli_create_dry_run_does_not_render(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    deck = tmp_path / "deck.md"
    deck.write_text("# One\n\n- a\n\n# Two\n\n- b\n")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "providers:\n  tts: {provider: gtts}\n  avatar: {provider: none}\n"
    )
    out = tmp_path / "out.mp4"
    result = CliRunner().invoke(
        app, ["create", str(deck), str(out), "--config", str(cfg_file), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert not out.exists()  # nothing rendered


def test_cli_doctor_command_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    deck = tmp_path / "deck.md"
    deck.write_text("# One\n\n- a\n")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("providers:\n  tts: {provider: gtts}\n  avatar: {provider: none}\n")
    result = CliRunner().invoke(app, ["doctor", str(deck), "--config", str(cfg_file)])
    assert result.exit_code == 0, result.output
    assert "Deck check" in result.output
    assert "1 slide(s)" in result.output
