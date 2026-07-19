"""Live full deck: a 3-slide render through the real CLI with the configured
stack (LLM narration + images + TTS + video assembly). Avatar is disabled here
(it has its own slow test) so this stays quick.

    uv run pytest tests/live/test_live_full_deck.py --run-live -q
"""

import copy

import yaml
from typer.testing import CliRunner

from slide_stream.cli import app

from ._live_helpers import probe_video

DECK = """# Welcome

- Slide stream live end to end test
- Three short slides

# Details

- Rendered with the configured provider stack
- Narration, images and audio

# Wrap up

- Thanks for watching
"""


def test_live_full_deck_renders(live_config, tmp_path):
    cfg = copy.deepcopy(live_config)
    # Avatar has its own (slow) test; keep the deck fast and cheap.
    cfg["providers"]["avatar"] = {"provider": "none"}
    cfg.setdefault("settings", {}).setdefault("video", {})
    cfg["settings"]["video"]["resolution"] = [854, 480]
    cfg["settings"]["video"]["fps"] = 24

    deck = tmp_path / "deck.md"
    deck.write_text(DECK)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out.mp4"

    result = CliRunner().invoke(
        app, ["create", str(deck), str(out), "--config", str(cfg_file)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists(), "no output video produced"

    info = probe_video(out)
    assert info["has_audio"], "final deck has no narration audio track"
    assert info["duration"] > 2.0, f"deck too short: {info}"
    print(f"\nFull deck rendered: {info}")
