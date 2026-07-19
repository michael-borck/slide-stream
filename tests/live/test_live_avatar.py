"""Live avatar: the configured engine turns narration audio into a head clip
that actually moves. For wan-s2v this is a GPU render (minutes) — marked slow.

    uv run pytest tests/live/test_live_avatar.py --run-live -q
    # skip the slow render:  --run-live -m "not slow"
"""

from pathlib import Path

import pytest

from slide_stream.providers.factory import ProviderFactory

from ._live_helpers import region_motion


@pytest.mark.slow
def test_live_avatar_animates(live_config, tmp_path):
    av_cfg = live_config["providers"].setdefault("avatar", {})
    # Supply a mascot source + cheap render settings only if the config didn't
    # already (a detector-free engine like wan-s2v needs a source; a user's own
    # photo/settings win via setdefault).
    av_cfg.setdefault("source", "owl")
    av_cfg.setdefault("clip_seconds", 4)
    av_cfg.setdefault("steps", 10)
    av_cfg.setdefault("timeout", 1800)

    provider = ProviderFactory.create_avatar_provider(live_config)
    if provider.name == "none":
        pytest.skip("no avatar provider configured/available")

    # Real driving audio from the configured TTS (gtts fallback is fine here).
    tts = ProviderFactory.create_tts_provider(live_config)
    audio = tmp_path / "drive.mp3"
    assert tts.synthesize("Hello, this is a talking head smoke test.", str(audio)), (
        "could not synthesize driving audio"
    )

    out = tmp_path / "head.mp4"
    result = provider.generate(str(audio), str(out), 1)
    assert result and Path(result).exists(), f"{provider.name} produced no clip"

    motion = region_motion(result)
    assert motion > 1.0, f"{provider.name} clip looks static (motion={motion:.2f})"
    print(f"\nAvatar engine exercised: {provider.name} (motion={motion:.2f})")
