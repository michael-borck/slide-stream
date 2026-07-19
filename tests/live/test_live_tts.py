"""Live TTS: the configured voice provider produces real, playable audio.

Catches TTS API drift and silent server failures. Run with:
    uv run pytest tests/live/test_live_tts.py --run-live -q
"""

from pathlib import Path

from slide_stream.providers.factory import ProviderFactory

from ._live_helpers import audio_duration


def test_live_tts_synthesizes_real_audio(live_config, tmp_path):
    provider = ProviderFactory.create_tts_provider(live_config)
    out = tmp_path / "tts.mp3"
    result = provider.synthesize(
        "This is a live text to speech smoke test for slide stream.", str(out)
    )
    assert result, f"{provider.name}.synthesize returned None"
    p = Path(result)
    assert p.exists() and p.stat().st_size > 1000, "audio file missing or tiny"
    assert audio_duration(p) > 0.8, "synthesized audio suspiciously short"
    print(f"\nTTS provider exercised: {provider.name}")
