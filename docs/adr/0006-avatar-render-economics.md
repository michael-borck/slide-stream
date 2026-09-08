# 0006 — Avatar render economics: one GPU clip per deck

- Status: Accepted (backfilled 2026-09)
- Drivers: wan-s2v/sadtalker take minutes per slide on consumer GPUs (an RTX
  3090 renders a short clip in ~10 min, cold model load dominating);
  decks have 10–30 slides

## Decision

`settings.avatar` treats GPU renders as scarce:

- `reuse_clip` (default false): render ONE head clip from slide 1's voice and
  loop it under every later slide — one GPU render per deck, approximate
  lip-sync. Intended for cartoon mascots, where "talking loop" beats "correct
  phonemes"; do not combine with `full_length: true` (clip then matches only
  slide 1's narration length).
- `slides` ("all" | N | [1, 5]): put heads only on chosen slides.
- `reuse_clip` interplay: heads wanted but no narration audio → counted as a
  headless slide, never a silent stall.
- ComfyUI hiccups between slides are survived and retried (commit 51b10a9);
  transient GPU OOM retries with backoff (`oom_retries`, `oom_backoff`), with
  a bounded poll `timeout` per clip.
- No-GPU alternatives ship for the common cases: `static` (still image),
  `puppet` (loudness mouth-flap), `precomputed` (ready-made clips), and
  hosted D-ID as the BYO-key escape hatch.

## Consequences

- A 20-slide mascot video costs the same GPU time as a 2-slide one when
  `reuse_clip` is on — the setting that makes mascots practical.
- Sync is visibly approximate on reused clips; humans/photo-real avatars
  should pay for per-slide renders instead.
