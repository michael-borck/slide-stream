# 0010 — SlideStream orchestrates external services through adapters

- Status: Proposed (2026-10)
- Drivers: local-first generation needs many specialist engines (voice,
  images, avatars, slide design, music, document extraction). Most of them are
  other people's projects. Users need one tool, one config, and a clear view of
  which engines are live before a render starts.
- Related: generalises 0001 (registry and strict mode), 0003 (OpenAI-compatible
  servers) and 0009 (Presenton). Borrows the shape of the lens family contract.

## Context

SlideStream already calls Voicebox, Chatterbox, ComfyUI, SwarmUI, Ollama and
OpenAI-compatible servers. ADR-0009 added Presenton. Each integration arrived
on its own terms:

- `providers/base.py` defines interfaces for images, TTS and avatars. It also
  defines `LLMProvider`, but nothing implements it. LLM calls go through
  `llm.py` instead.
- Presenton has its own client in `presenton.py`, outside `providers/`.
- `is_available()` checks configuration only, for example "a `base_url` and a
  profile are set" (`providers/tts.py:493`). Nothing asks whether the server is
  up, which version it runs, or what it offers. A dead `base_url` shows up as a
  fallback halfway through a render, which ADR-0004 lists as a known cost.

The lens family solves a similar problem with a shared contract: every engine
serves `GET /health`, `GET /manifest` and `POST /analyse`, implemented once in
`lens-contract`. That works because lens owns every engine. SlideStream does
not own Presenton, ComfyUI or Voicebox and cannot add endpoints to them.

Deployment is already handled elsewhere in the same ecosystem. Puente
(`locopuente` on PyPI) stands up ComfyUI, SwarmUI, Voicebox, Ollama and MusicGen
from one `puente.yml`, and has its own `doctor`.

## Decision

1. **SlideStream is an orchestrator.** It runs one job across several
   services. It does not bundle, fork or vendor them, and by default it does
   not install, start or stop them. That is the job of the user, Puente, or the
   service's own installer. A managed lifecycle for one specific service needs
   its own ADR.

2. **The contract lives in the adapter.** SlideStream reaches every service
   through an adapter class. One adapter serves one role. Each adapter
   implements:

   | Method | Does |
   |---|---|
   | `is_configured()` | Today's `is_available()`: required config is present. No network. |
   | `probe(timeout)` | Returns `ServiceStatus(reachable, version, capabilities, detail)` from the service's own health or info endpoint. Cheap and read-only. |
   | role method(s) | `generate_text`, `synthesize`, `generate_image`, `generate`, `design`, `compose`, `extract`, `check` |
   | `close()` | Per-run cleanup. Already exists on TTS (voice-clone deletion). |

   `capabilities` is role-specific: voices for TTS, models for LLM and images,
   templates for design, and so on. The UI builds its pickers from it.

3. **Roles.** `llm`, `tts`, `images`, `avatar`, `design`, `music`, `ingest`,
   `check`. LLM moves under the same interface by wrapping `llm.py`, not
   rewriting it. Presenton becomes the first `design` adapter.

4. **Every role falls back service → local → bare minimum** (ADR-0012).
   Voice: Kokoro, then gTTS. Pictures: a local folder, then text cards.
   Presenter: static or puppet, then none. Design: preserve, then `basic`.
   Music: a user track, then none. Ingest: built-in extraction. Check: none.

5. **One machine-readable status.** `slide-stream services [--json]` probes
   every configured adapter and prints the results. `doctor`, the web UI and
   the desktop Services panel read the same data. A UI control for a role
   appears only when its adapter probes reachable. Otherwise the UI says how to
   enable it, and never offers a picker that fails on use.

6. **Pinned and tested.** `docs/SERVICES.md` records the version each adapter
   was tested against. Each adapter gets a contract test under `tests/live/`,
   run with `--run-live` against that version. The mocked suite stays hermetic
   (ADR-0007).

7. **First-party services speak the lens shape.** Any service SlideStream
   ships itself (the `contrib/` scripts, a future music wrapper) serves
   `GET /health` and `GET /manifest` like a lens engine, so first-party
   services look the same across both families.

8. **Credit.** `docs/SERVICES.md` names each upstream project, its licence and
   its link. Copied code keeps its upstream licence and NOTICE, as in 0009.

## Consequences

- Adding an engine means one adapter, one `SERVICES.md` row and tests. It adds
  no new code paths to the pipeline.
- Preflight finds a dead server before rendering. Strict mode fails fast.
  Default mode warns and uses the floor.
- Probing costs one round-trip per configured service. Keep timeouts short
  (about 2 s) and cache results for the length of a run.
- `providers/base.py` grows. `llm.py` and `presenton.py` move behind adapters.
  Existing provider names and config keys stay as they are.
- Anything above the floor needs a running service. Puente is the recommended
  way to run a local stack. SlideStream may read Puente's connection details
  if Puente gains a machine-readable `connect --json`, but it must not depend
  on Puente.
- Option B of the Presenton analysis (SlideStream starts Presenton's Docker
  image per command) is deferred. Running Presenton as a Puente service meets
  the same need without SlideStream managing containers.
- Desktop users without a local stack still get the floor and any cloud
  providers they configure. Hosted or bundled engines for them are a separate
  decision.
