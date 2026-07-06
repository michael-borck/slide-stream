# Talking-Head Avatar — Options & Decision Notes

Reference for choosing how to generate the lip-synced talking head that
slide-stream composites over each slide. Written 2026 during the avatar work;
revisit when picking a local engine to prototype.

## The one distinction that matters: two families

Every talking-head tool falls into one of two families, and this is what
determines how "janky" the result looks.

### Family 1 — photo → full-head synthesis
One **still photo** + audio. The model synthesizes *everything*: head pose,
blinks, expressions, and lips. Convenient (only a photo needed) but jankier —
all the motion is invented, so it can look uncanny.

- **SadTalker** — 3D-pose based. Easiest to run (ComfyUI node), but known for
  "unnatural, upward-tilted head poses."
- **D-ID / Hedra** (cloud) — the polished, hosted version of this family. No
  jank, but bills per minute of video (~$1–2/min). **D-ID is already
  implemented in slide-stream** as the `d-id` avatar provider (BYOK).

### Family 2 — video → mouth-region swap
An existing **video** of the person + audio. The model replaces *only the
mouth* to match the audio; head motion, blinks, and body stay **real
footage** → far more natural. Needs base video, not just a photo.

- **Wav2Lip** (2020) — excellent sync but renders the mouth at 96×96 and
  upscales, so the mouth looks **blurry**. Dead simple. Feed a still image and
  it just holds a static frame.
- **MuseTalk** (2024) — better than Wav2Lip, near-realtime, but **degrades
  facial features** (e.g. a beard goes sparse) and is a **painful install with
  pinned models**. Not recommended — superseded.
- **LatentSync** (2024/2025) — diffusion-based, the **current quality leader**:
  "lip-sync accuracy significantly surpasses others," best visual quality. Has
  a ComfyUI node. **This is the one to pick for the video route.**

## Key insight for slide-stream specifically

slide-stream composites the head as a **small corner circle (~28% of frame
height)**, not full-screen. At that size, SadTalker's head-bob jank and
Wav2Lip's mouth blur are *much* less visible than in the full-frame demos.
**"Good enough" is easier to hit than it looks — test at the real corner size
before investing in a heavier engine.**

## The "does it loop?" question

Family 2 (mouth-swap) tools output video the **length of the audio**. If a
slide's narration is longer than the base clip, the base video is **looped** to
cover it. For no visible jump, record the base footage as a seamless
**ping-pong (boomerang) idle loop** (~15s of the lecturer idling/listening).
Family 1 (photo) sidesteps looping — it synthesizes motion for any duration —
at the cost of realism.

## The ladder (free-janky → paid-polished)

1. **SadTalker** (local, free, Family 1) — jankier, but you already have the
   ComfyUI node. **Prototype here first**, judged at corner-circle size.
2. **LatentSync** (local, free, Family 2) — best local quality; needs a
   ping-pong idle base video + GPU. The upgrade if SadTalker isn't good enough.
3. **D-ID** (cloud, paid, Family 1) — polished, no jank, ~$1–2/min. Already
   shipped as the `d-id` provider — the BYOK "just works" tier.

## Integration notes

- Any local clip can be dropped in today via the **`precomputed` avatar
  provider**: name clips `head_1.mp4`, `head_2.mp4`, … in a folder and set
  `providers.avatar.assets_dir`. This is how to test a SadTalker output at
  corner size with zero new code.
- **You already have the serving infra.** SwarmUI is ComfyUI-backed, and
  SadTalker / Wav2Lip / LatentSync all run as ComfyUI nodes. So instead of
  MuseTalk's separate FastAPI server (the original Phase-4 plan), a future
  **`comfyui` avatar provider** could drive a workflow on the existing box —
  POST image/video + audio → get the clip back — same shape as the `swarmui`
  image provider. This is the preferred path over standing up MuseTalk.

## Current status

- `none`, `precomputed`, and `d-id` avatar providers are implemented.
- MuseTalk (original Phase-4 target) is **deprioritized** in favour of
  LatentSync (better) and/or a ComfyUI-driven provider (reuses existing infra).
- Next action: prototype SadTalker locally, view via `precomputed` at corner
  size, then decide between "good enough", LatentSync, or D-ID.
