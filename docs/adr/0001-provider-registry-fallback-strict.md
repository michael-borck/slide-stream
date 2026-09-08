# 0001 — Provider registry with fallback chains and strict mode

- Status: Accepted (backfilled 2026-09)
- Drivers: every AI layer (LLM / images / TTS / avatar) has cloud, stock-photo,
  and self-hosted implementations of very different reliability

## Context

A render touches four independent providers. Any of them can be unconfigured
(no key yet), unreachable (laptop asleep, GPU box busy), or failing at
runtime. Teachers need "some video, always"; lecturers with a cloned voice or
a paid image budget need "exactly the configured provider, or nothing".

## Decision

`providers/factory.py` is the single registry. Each layer resolves as:
configured provider → configured `fallback` (if usable) → a hard floor
(gTTS / text cards / no avatar). Strict mode (`settings.strict`, `--strict`)
converts every fallback into a `StrictModeError` abort, and runtime failures
honour it per-slide too (`providers/base.py`). Unavailability at startup
(missing key/base_url) and failure mid-run are reported distinctly, and
`slide-stream doctor` explains the why (see `doctor.py`).

## Consequences

- Adding a provider is one class + one registry line + one `list_*` blurb.
- Silent quality degradation is possible (e.g. elevenlabs → gTTS mixed voices
  in one video); the run summary warns about fallback counts and nudges
  toward `doctor` (cli.py `_doctor_nudge`).
- Strict mode must stay opt-in per run/config; the free floor is what makes
  the hosted demo and first-run experience work with zero keys.
