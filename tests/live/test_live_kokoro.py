"""Live Kokoro: the offline voice floor really speaks (ADR-0012).

Loads the real model (about 340 MB, downloaded once to
~/.cache/slide-stream/kokoro/). Run with:
    uv run pytest tests/live/test_live_kokoro.py --run-live -q
"""

import copy
from pathlib import Path

import pytest

from slide_stream.providers.factory import ProviderFactory
from slide_stream.providers.tts import KokoroTTSProvider, voice_floor_name

from ._live_helpers import audio_duration

pytestmark = pytest.mark.skipif(
    not KokoroTTSProvider({}).is_available(),
    reason='Kokoro not installed (pip install "slide-stream[local-tts]")',
)


def test_live_kokoro_floor_speaks(live_config, tmp_path, monkeypatch):
    monkeypatch.delenv("SLIDE_STREAM_VOICE_FLOOR", raising=False)
    cfg = copy.deepcopy(live_config)
    cfg["providers"]["tts"] = {"provider": "auto"}
    cfg.setdefault("settings", {})["voice_floor"] = "auto"
    assert voice_floor_name(cfg) == "kokoro"

    provider = ProviderFactory.create_tts_provider(cfg)
    assert provider.name == "kokoro"
    out = tmp_path / "kokoro.mp3"  # WAV data behind an .mp3 name, as in renders
    result = provider.synthesize("The offline voice is working.", str(out))
    assert result, "Kokoro returned no audio"
    assert Path(result).stat().st_size > 10_000
    assert audio_duration(result) > 1.0
    assert provider.fallback_count == 0  # Kokoro itself spoke, not gTTS
