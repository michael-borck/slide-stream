# Talking-Head Avatar Feature — Implementation Plan for slide-stream

Written 2026-07-06 from the lecturer-clone session, after reviewing the actual
slide-stream 2.0 codebase. Intended to be picked up by the session working in
`../slide-stream`. This replaces the lecturer-clone project: slide-stream
becomes the single app, lecturer-clone gets archived.

## Target architecture: ONE app (confirmed 2026-07-06)

All three repos converge into slide-stream as composable stages over one
deck input (markdown/pptx). Each stage is optional and combinable:

- **images** (from slide-vision): enrich slides with images — already mostly
  in slide-stream's image providers; port slide-vision's unique bits (local
  folder provider, `scan` AI-rename, enrich-markdown output mode, better
  parser).
- **narration** (slide-stream core): per-slide TTS — already there.
- **avatar** (from lecturer-clone, this plan): lip-synced talking head
  overlay.

End state: `slide-stream enrich deck.md` (deck out), `slide-stream create
deck.md out.mp4` (narrated video), `--avatar` (talking head), any
combination via config. lecturer-clone and slide-vision are then archived —
neither has anything else to salvage (lecturer-clone: only this doc and
`_notes/mikebot.mp3`).

## Goal

`slide-stream create deck.md out.mp4` gains an optional **avatar mode**: a
lip-synced talking head of the lecturer, composited as a circle in a corner
over each slide, driven by the same per-slide TTS audio the pipeline already
produces.

Inputs the user supplies once:
- **base video**: 2–5 min real footage of the lecturer, head-and-shoulders,
  sitting/idling/gesturing naturally (this is what makes it not look janky —
  only the mouth region is synthesized)
- **cloned voice**: self-hosted Chatterbox (see TTS section — likely zero new
  code)

## Why slide-stream fits (verified against the code)

- Pipeline is already **per-slide**: each slide gets its own
  `slide_N.mp3` + `fragment_N.mp4` in `cli.py` (`create`, ~line 300), then
  fragments are concatenated. The avatar step slots between TTS and
  `create_video_fragment`.
- `media.py::create_video_fragment` already composites with MoviePy
  (`CompositeVideoClip`) — adding a positioned, circle-masked overlay clip is
  the same idiom.
- Provider architecture (`providers/base.py`, `factory.py`) is the right
  extension point: add an `AvatarProvider` alongside `ImageProvider` /
  `TTSProvider`, with the same strict-mode / fallback semantics.
- The `openai-compatible` TTS provider pattern (config `base_url` pointing at
  a local server) is exactly the right pattern for the GPU lip-sync service.

## Architecture decision (made with Michael 2026-07-06)

- **One app**: extend slide-stream. Do NOT bake GPU/torch deps into the pip
  package. The lip-sync engine runs as an external local HTTP service; the
  slide-stream provider is a thin client.
- **Engine: MuseTalk** (first target) — ~8GB VRAM, near-real-time, only
  replaces the mouth region of real footage. LatentSync is the quality
  upgrade later (there's a ComfyUI workflow JSON for it in
  `lecturer-clone/_notes/workflows/video/`). HeyGem was considered; worth a
  separate evaluation but not the first integration.

## 1. Voice clone (probably zero code)

Chatterbox community servers (e.g. `chatterbox-tts-api`) expose an
OpenAI-compatible `/v1/audio/speech` endpoint. slide-stream's existing
`OpenAICompatTTSProvider` (`providers/tts.py:237`) already supports that:

```yaml
providers:
  tts:
    provider: openai-compatible
    base_url: http://gpu-box:5005/v1
    voice: michael          # server-side voice name for the cloned sample
settings:
  strict: true              # never silently fall back to gTTS for a lecture
```

Optional nicety later: a dedicated `chatterbox` provider that uploads a
`voice_sample: path/to/mikebot.mp3` per run. Not needed for v1.
Note the existing strict-mode comment in the example config already mentions
"cloned lecturer voice" — set `strict: true` in avatar workflows.

## 2. Avatar provider (new)

### Config

```yaml
providers:
  avatar:
    provider: none            # none | musetalk (HTTP client) | precomputed
    base_url: http://gpu-box:7010
    base_video: ./assets/base_video.mp4

settings:
  avatar:
    position: bottom-right    # bottom-left | bottom-right | top-left | top-right
    size: 0.28                # circle diameter as fraction of frame height
    margin: 24                # px from edges
    shape: circle             # circle | rounded | rect (circle for v1)
```

Add `avatar` to `DEFAULT_CONFIG` in `config_loader.py` with
`provider: none` so existing configs/tests are untouched. Keep it OUT of
`required_providers` in `validate_config` (or add with a default) so old
YAML files still validate.

### Protocol (providers/avatar.py)

```python
class AvatarProvider(Protocol):
    name: str
    def is_available(self) -> bool: ...
    def generate(self, audio_path: str, output_path: str) -> str | None:
        """Return path to a lip-synced head video for this audio, or None."""
```

Implementations:
- `NoneAvatarProvider` — default, disables the feature.
- `HTTPAvatarProvider` (`musetalk`) — POSTs the fragment audio to
  `{base_url}/lipsync` (multipart: audio file; the base video is registered
  server-side once, or sent by path), receives an mp4. Honor strict mode: on
  failure, strict → abort; non-strict → fall back to audio-only fragment
  (current behavior) and count it in the run summary.
- `PrecomputedAvatarProvider` — maps `slide_N.mp3` → user-supplied
  `head_N.mp4` from a directory. This exists so the overlay compositing can be
  built and tested with NO GPU (record yourself once, or use any clip).

Factory: `ProviderFactory.create_avatar_provider(config)` mirroring the TTS
factory, plus listing in the `providers` CLI command.

### CLI wiring (cli.py `create`)

Per slide, after `tts_provider.synthesize(...)`:

```python
head_path = None
if avatar_provider.name != "none" and audio_file:
    head_path = avatar_provider.generate(str(audio_path), str(temp_dir / f"head_{slide_num}.mp4"))
fragment_file = create_video_fragment(img_path, audio_path, fragment_path, config, head_video=head_path)
```

Add a `--avatar/--no-avatar` flag to override config per run, mirroring
`--strict`.

### Compositing (media.py)

In `create_video_fragment`, when `head_video` is given:
- `VideoFileClip(head_video)`, center-crop to square, resize to
  `size * frame_height`.
- Circular alpha mask: render a white-circle-on-black image with PIL
  (PIL/numpy are already transitive deps via MoviePy) and attach via
  `.with_mask(ImageClip(mask, is_mask=True))`.
- Position from `position` + `margin`; add to the `CompositeVideoClip` list
  after the slide image.
- Duration: head clip should match audio duration (the lip-sync service
  output will, by construction). If head > slide duration (padding), freeze
  or trim the tail; if shorter, hold the last frame.
- The head clip's own audio must be dropped (`.without_audio()`) — the
  fragment keeps the TTS track as the single audio source, so sync issues
  are impossible at composite time.

Testable today on the Mac: synthetic ColorClip "head" video + real audio →
assert output resolution, duration, and that it encodes. No GPU needed.

## 3. MuseTalk service (runs on the GPU box, separate from the package)

A ~100-line FastAPI wrapper around MuseTalk's inference, shipped in
slide-stream's repo under `contrib/` or `docs/` (not in the wheel):
- startup: load models once, preprocess the base video (face detection /
  landmarks are cacheable per base video — MuseTalk supports a prepared
  "avatar" that makes per-request inference fast)
- `POST /lipsync` (audio file) → mp4 bytes
- `GET /health` → used by `is_available()`

2× RTX 2060 8GB plan: GPU0 = Chatterbox server, GPU1 = MuseTalk server via
`CUDA_VISIBLE_DEVICES`.

## 4. Later phases (explicitly out of v1)

- Merge slide-vision's unique code into slide-stream — do NOT delete
  slide-vision until this port is done; it has real, well-tested code
  (reviewed 2026-07-06):
  - `providers/local.py` — LocalProvider: keyword-scores local image
    filenames per slide, each image used once. slide-stream has no local
    image provider at all.
  - `scanner.py` — `scan` command: AI vision renames images to keyword
    slugs (dry-run, collision-safe, writes a report). Pairs with
    LocalProvider.
  - `parser.py` — strictly better parser: `---` separator-style decks AND
    heading-style, YAML front-matter stripping, no markdown/bs4 deps.
  - `providers/factory.py` + scanner plugin system — providers via entry
    points, `module:Attribute` CLI specs, or runtime registration; adopt
    for avatar/TTS/image providers instead of the hardcoded registry.
  - `writer.py` — enrich-markdown output mode + `prompts.md` for unmatched
    slides + zip; becomes the `enrich` subcommand.
  - Skip: its dalle/pexels/unsplash providers (slide-stream duplicates).
- LatentSync backend as a quality option behind the same HTTP contract.
- Full-screen presenter mode (no slides, head only) — trivial once the
  avatar fragment exists: composite onto a background instead of a corner.
- Archive lecturer-clone and slide-vision repos once merged.

## Suggested build order (each step independently testable)

1. `PrecomputedAvatarProvider` + circle overlay in `media.py` + tests
   (pure MoviePy, runs on the Mac, proves the look end-to-end with a
   hand-recorded head clip).
2. Config + factory + CLI flag + `providers` listing + strict-mode paths.
3. Chatterbox: config-only recipe in the example config; verify with the
   real voice (`lecturer-clone/_notes/mikebot.mp3` is the sample).
4. MuseTalk FastAPI wrapper on the GPU box + `HTTPAvatarProvider`.
5. Quality pass: mask feathering (soft edge), optional ring/border, position
   presets; evaluate LatentSync/HeyGem against MuseTalk output.
