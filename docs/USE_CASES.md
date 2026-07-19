# SlideStream by use case

Task-first recipes for the things people actually want to do. Each shows the
CLI, the config, and the web-UI equivalent. For the full reference see
[USER_GUIDE.md](USER_GUIDE.md).

Every recipe takes a **deck** — a Markdown file (`.md`) or a PowerPoint
(`.pptx`, whose speaker notes become the narration source).

- [1. Slides → narrated video (fastest)](#1-slides--narrated-video-fastest)
- [2. Narrate in your own voice](#2-narrate-in-your-own-voice)
- [3. Add a talking-head presenter](#3-add-a-talking-head-presenter)
- [4. Export a PowerPoint with AI speaker notes](#4-export-a-powerpoint-with-ai-speaker-notes)
- [5. Check a deck before you render](#5-check-a-deck-before-you-render)
- [6. Run the web UI (yourself or hosted)](#6-run-the-web-ui-yourself-or-hosted)

---

## 1. Slides → narrated video (fastest)

No account, no keys. Uses free gTTS for the voice and renders text-card images.

```bash
slide-stream create deck.md talk.mp4
```

- `--narration-seconds 30` — aim for ~30s of speech per slide.
- `--llm-provider gemini` (or `openai`/`claude`/`groq`) — write presenter-style
  narration instead of reading the bullets. Needs that provider's API key.
- `--image-provider pexels` — real stock photos instead of text cards
  (free Pexels key), or `gemini` for AI images (~$0.02 each).

**Web UI:** drop the deck in, click **Create video**.

---

## 2. Narrate in your own voice

Record **10–30 seconds** of clean speech, then clone it *for that render only*
via a self-hosted [Voicebox](voicebox-api.md) or Chatterbox server.

```bash
slide-stream create deck.md talk.mp4 --voice-sample my_voice.wav
```

Or in config (survives across runs), with `--strict` so it never silently
falls back to the wrong voice:

```yaml
providers:
  tts:
    provider: chatterbox            # or voicebox
    base_url: https://voice.example.org
    voice_sample: ./my_voice.wav    # ephemeral clone, deleted server-side after
    api_key: "${CHATTERBOX_TOKEN}"  # if the server is behind auth
settings:
  strict: true
```

Fully offline instead? `providers.tts.provider: kokoro` (needs
`pip install "slide-stream[local-tts]"`, ~340 MB one-time model download).

**Web UI:** open **Voice & presenter**, upload your sample. The browser
remembers it so you don't re-pick it each time; the server stores nothing.

---

## 3. Add a talking-head presenter

A presenter appears in a circle in the corner. Three levels, cheapest first.

### a. A still image (no GPU)
A mascot or your photo, held in the corner.

```yaml
providers:
  avatar: { provider: static, source: owl }   # built-in: teddy/panda/koala/robot/wizard/owl
```
List the built-ins with `slide-stream avatars`. Use any image path for your own.

### b. Animated — **any** character (Wan2.2-S2V)
`wan-s2v` has **no face detector**, so it lip-syncs the built-in mascots
*and* human head shots from the narration audio, via a self-hosted ComfyUI
server. This is the one that animates a teddy or an owl.

```yaml
providers:
  avatar:
    provider: wan-s2v
    base_url: https://comfyui.example.org
    api_key: "${COMFYUI_TOKEN}"       # if the server checks a Bearer token
    source: owl                       # a built-in name, or ./me.jpg
    # clip_seconds: 4                  # short clip looped under the narration (default)
    # full_length: true                # or render the whole narration (much slower)
```
```bash
slide-stream create deck.md talk.mp4 --avatar
```
> **Heads-up:** S2V is heavy — roughly a few minutes of GPU per slide. Use
> [`--dry-run`](#5-check-a-deck-before-you-render) first for a time estimate.

### c. Animated — human face only
For a real photo/video you can also use `sadtalker` (photo), `wav2lip`
(video), or hosted `d-id`. These use a **human** face detector, so they do
**not** animate the stylized mascots — use `wan-s2v` for those.

**Web UI:** pick a **Mascot presenter** or upload a photo, tick **Animate the
presenter**. If the server has `wan-s2v` (or `d-id`) configured, mascots
animate for real; otherwise they get a no-GPU cartoon mouth-flap.

---

## 4. Export a PowerPoint with AI speaker notes

Want editable slides, not a video. `enrich` adds an image to every slide and
writes a new deck (Markdown + `images/`, and a `.pptx` with `--pptx`).

```bash
slide-stream enrich deck.md out/ --pptx --image-provider pexels
```

Add AI **presenter notes** into the PowerPoint's notes pane:

```bash
slide-stream enrich deck.md out/ --pptx --notes all   # write for every slide
slide-stream enrich deck.md out/ --pptx --notes fill  # keep existing, fill the gaps
```

The notes are written as a spoken script, so they **round-trip**: run
`slide-stream create out/deck.pptx talk.mp4` and the narration comes straight
from those notes. `--notes` needs an LLM provider configured.

**Web UI:** switch **Output** to *PowerPoint deck* and pick a notes option; you
get a `.zip` (deck + images) to download.

---

## 5. Check a deck before you render

A preflight that reports estimates and warnings **without rendering** — handy
before an expensive avatar run, or as a CI gate.

```bash
slide-stream doctor deck.pptx
slide-stream create deck.pptx out.mp4 --dry-run   # same report, with create's exact flags
```

It reports: slide count, slides missing notes, **stage directions** in notes
that would be read aloud ("[pause]", "click to advance"), voice-sample length
(too short/long), image resolution vs the video frame, mascot-vs-engine
mismatches, missing `ffmpeg`/API keys, and **estimated duration, cost, and
render time**.

```bash
slide-stream doctor deck.pptx --fail-on-warn    # non-zero exit if any warnings (CI)
```

**Web UI:** click **Check deck first** before **Create**.

---

## 6. Run the web UI (yourself or hosted)

Upload a deck (+ optional voice/photo) in the browser, render, download.

```bash
pip install "slide-stream[serve]"
slide-stream serve --host 127.0.0.1 --port 8080
```

- **Desktop app** — a one-click bundle of the same UI (no terminal); see the
  [releases page](https://github.com/michael-borck/slide-stream/releases).
- **Self-host for others** — Docker Compose behind a reverse proxy; see
  [`deploy/README.md`](../deploy/README.md). Providers are chosen in a
  server-side `slidestream.yaml`; secrets come from the environment.

The web UI, the desktop app, and the Docker deploy are the **same** interface,
so everything above (voice, mascots, PowerPoint output, the preflight) is
available on all three.
