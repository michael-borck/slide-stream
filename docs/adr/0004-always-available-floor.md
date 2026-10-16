# 0004 — gTTS and text cards are the always-available floor

- Status: Superseded by [0012](0012-three-level-fallback-kokoro-floor.md) (2026-10). Accepted (backfilled 2026-09)
- Drivers: first run must produce a watchable video with zero keys, zero
  servers, and no GPU

## Context

The hosted demo rate-limits strangers; local first-timers have configured
nothing. Requiring a key before the first render kills adoption.

## Decision

Every layer ends in a floor that needs nothing:

- TTS: gTTS (free, network required, accent-selectable via Google TLDs)
- Images: `text` — the slide's real title and bullets rendered onto the frame
- Avatar: none

Unconfigured server TTS (the voicebox default with no `base_url`) quietly
becomes gTTS rather than erroring: "not configured yet" is not a failure.
Mixed-voice degradation is surfaced (fallback counts + `doctor` nudge), and
strict mode exists for when the exact voice matters.

## Consequences

- The demo/first-run path is bulletproof; per-image provider failures fall
  straight to a text card mid-run.
- The floor can mask a misconfiguration (e.g. a dead `base_url` renders a
  whole video in the wrong voice). Mitigations: the ⚠ fallback warning, and
  strict mode recommended for cloned-voice setups.
- gTTS needs outbound internet — fully-offline users must pick kokoro or an
  OpenAI-compatible server explicitly.
