# Architecture Decision Records

Decisions that shape SlideStream, recorded so nobody has to reverse-engineer
them from code. Backfilled 2026-09 from module docstrings, config comments,
and the git history (the repo's visible history starts in the 2.15.x era).

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-provider-registry-fallback-strict.md) | Provider registry with fallback chains and strict mode | Accepted |
| [0002](0002-layered-yaml-config.md) | Layered YAML configuration with `${ENV}` expansion | Accepted |
| [0003](0003-openai-compatible-self-hosted.md) | OpenAI-compatible APIs as the default self-hosted shape | Accepted |
| [0004](0004-always-available-floor.md) | gTTS and text cards are the always-available floor | Superseded by 0012 |
| [0005](0005-privacy-first-voice-cloning.md) | Privacy-first voice cloning (ephemeral, deleted after the run) | Accepted |
| [0006](0006-avatar-render-economics.md) | Avatar render economics: one GPU clip per deck | Accepted |
| [0007](0007-hermetic-tests-env-scrubbing.md) | Hermetic tests; provider env vars scrubbed | Accepted |
| [0008](0008-dual-output-video-and-pptx.md) | Dual output modes: `.mp4` video and editable `.pptx` deck | Accepted |
| [0009](0009-presenton-designed-decks.md) | Designed decks via an external Presenton, rasterised for video | Accepted |
| [0010](0010-orchestrator-over-external-services.md) | SlideStream orchestrates external services through adapters | Proposed |
| [0011](0011-project-folder-pipeline-state.md) | A project folder holds the pipeline's state | Proposed |
| [0012](0012-three-level-fallback-kokoro-floor.md) | Three-level fallback; Kokoro is the voice floor | Accepted |
| [0013](0013-faithful-designed-decks.md) | Designed decks stay faithful: fidelity check and code slides | Accepted |
| [0014](0014-powerpoint-template-design.md) | Design new slides from the user's own PowerPoint template | Accepted (not yet implemented) |

## Format

Each ADR is short: Context (the forces), Decision (what we chose), and
Consequences (what that costs or enables). New ADRs get the next number and
are never edited after acceptance — supersede them with a new record instead.
