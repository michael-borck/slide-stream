# 0005 — Privacy-first voice cloning (ephemeral, deleted after the run)

- Status: Accepted (backfilled 2026-09)
- Drivers: a lecturer's voice sample uploaded to a shared self-hosted server
  is sensitive, recognisable biometric data

## Decision

Clone flows leave nothing behind by default (`providers/tts.py`):

- Chatterbox: the reference clip uploads once per run under a random UUID
  (`upload_reference`), never under the speaker's name;
  `contrib/chatterbox/cleanup_uuid_voices.sh` reaps UUID files server-side.
- Voicebox: a throwaway profile is created for the run and deleted via the
  API afterwards — a deep delete that removes samples and reference audio
  too. Rendered generations are deleted per-slide via `DELETE /history/{id}`
  as soon as their audio downloads (`delete_generations: true` default).
- Profiles deleted even on partial failure: the profile id is recorded
  before the sample upload so `close()` can reap a half-built clone.
- A persistent `profile_id` remains available for people who want zero
  per-run cloning cost; `voice_sample` wins over `profile_id` when both are
  set (an explicit upload is the more recent intent).

## Consequences

- Shared servers never hold a name→voice mapping; a cron (contrib scripts)
  or API deletes are the cleanup paths, not trust.
- Poll/timeout paths deliberately do NOT delete a generation still rendering
  server-side (a sweep script reaps those later) — deleting mid-render would
  orphan the run.
