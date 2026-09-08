# 0002 — Layered YAML configuration with `${ENV}` expansion

- Status: Accepted (backfilled 2026-09)
- Drivers: one person runs the same pipeline from the repo, the desktop app,
  Docker, and per-deck folders, against personal servers and keys

## Context

Server URLs and tokens are personal and stable; deck choices (narration
length, strict mode, which provider for this deck) vary per project. Keys
should live in env vars, but servers on a LAN have no natural env var story
in Docker and desktop contexts.

## Decision

Config layers, later winning, deep-merged (`config_loader.py`):

1. built-in defaults (incl. `providers.tts.provider: voicebox`, avatar off)
2. `~/.slidestream.yaml` / `.yml` — personal: server URLs, keys, providers
3. `./slidestream.yaml` / `.yml` or `--config` — deck/project overrides
4. CLI flags

Any value may reference `${ENV_VAR}`; unset vars expand to `""` (backward
compatible), so a token missing from the environment degrades rather than
crashing the YAML load. The home file is deliberately the "set once" layer:
the desktop app and `serve` read it from any working directory.

## Consequences

- A deck repo can stay shareable while pointing at someone's private GPU box.
- Two files CAN drift (this bit us: an old `voice_sample` in the home file
  kept overriding a project profile after the sample was deleted). Keep
  providers in ONE layer; project files should carry settings only.
- `""` expansion is quiet by design, but strict mode still catches the
  resulting "configured but unusable" provider at startup.
