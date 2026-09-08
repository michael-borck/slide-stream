# 0007 — Hermetic tests; provider env vars scrubbed

- Status: Accepted (2026-09, after it bit us — commit 573347a, then the
  conftest env-scrub fix)
- Drivers: the mocked suite must produce identical results on any machine

## Decision

`tests/conftest.py` isolates every non-`live` test from the host:

- `Path.home()` points at an empty temp dir — a real `~/.slidestream.yaml`
  can never change which providers load.
- Provider connection env vars (VOICEBOX_*/CHATTERBOX_*/SWARMUI_*/COMFYUI_*/
  OLLAMA_*/OPENAI_BASE_URL) are deleted per test.
- Live end-to-end checks live in `tests/live/` and only run with
  `--run-live`; they deliberately read the real home config.

## The incident that pinned this down

`uv run` auto-loads the repo's `.env`. Once real tokens appeared there, the
factory (correctly, per its "env-configured counts as configured" rule)
treated voicebox as a deliberate choice — so strict-mode tests that expected
the bare-install gTTS path started aborting. Only a fully clean environment
(`env -i`, `--no-env-file`) made the suite green again, hence the explicit
scrub rather than documentation.

## Consequences

- Tests cannot verify the env-var configuration path; that behaviour is
  covered by explicit monkeypatch tests instead (e.g. `_SERVER_TTS_ENV_VARS`
  handling in test_providers.py).
- `.env` is for runtimes, conftest is for tests; never "fix" a failing
  provider test by exporting a real token.
