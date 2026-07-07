---
title: "Voicebox Voice-Clone & TTS API"
---

How to clone a voice and generate speech with it via Voicebox's HTTP API, so an
external tool can add Voicebox as a **TTS provider** (alongside Chatterbox).

> **Verified** 2026-07-07 against a live Voicebox (`puente up voicebox`, port
> 17493): all documented endpoints exist in its OpenAPI spec, and creating a
> cloned-voice profile via `POST /profiles` works. Voicebox exposes interactive
> FastAPI docs at `/docs` (and the raw schema at `/openapi.json`) — the
> authoritative reference if the API evolves.

---

## What Voicebox is (and how it relates to Chatterbox)

Voicebox is a self-hosted **voice studio** with a FastAPI backend. Importantly,
it is **multi-backend** — its `engine` parameter selects the TTS model:
`qwen`, `qwen_custom_voice`, `luxtts`, `chatterbox`, `chatterbox_turbo`,
`tada`, `kokoro`. So Voicebox can even drive Chatterbox internally.

It is **not** a thin UI over the standalone Chatterbox server — it's an
independent stack. As a provider it offers a clean "profile + generate" model.

**Endpoint:** `http://<box>:17493` (default port). Add an nginx-proxy-manager
host (e.g. `voice.locopuente.org`) for public/TLS access, like the other
services. FastAPI exposes interactive docs at `/docs` — use it to confirm
schemas live.

---

## The voice-clone model: profiles + samples

Voicebox clones a voice as a **profile** built from one or more **samples**
(reference audio + its transcript). You then generate speech against the
profile. Three steps:

### 1. Create a voice profile

```
POST /profiles
Content-Type: application/json

{
  "name": "Lecturer A",
  "description": "Course narration voice",
  "language": "en",
  "voice_type": "cloned"
}
```
→ returns a `VoiceProfileResponse` with an `id` (the `profile_id`).

Languages: `zh en ja ko de fr ru pt es it he ar da el fi hi ms nl no pl sv sw tr`.

### 2. Add a sample (this is the clone)

Upload reference audio + the exact text spoken in it (multipart form):

```
POST /profiles/{profile_id}/samples
Content-Type: multipart/form-data

file=@reference.wav
reference_text=The exact words spoken in the reference clip.
```
Accepted audio: `.wav .mp3 .m4a .ogg .flac .aac .webm .opus`.
→ returns a `ProfileSampleResponse`. Add multiple samples for a better clone.

### 3. Generate speech with the cloned voice

```
POST /generate
Content-Type: application/json

{
  "profile_id": "<id from step 1>",
  "text": "Text to speak in the cloned voice.",
  "language": "en",
  "engine": "chatterbox",
  "seed": null,
  "normalize": true
}
```
→ returns a `GenerationResponse` with an `id` and (when finished) `audio_path`,
`duration`, `status`.

`engine` options: `qwen` (default) | `qwen_custom_voice` | `luxtts` |
`chatterbox` | `chatterbox_turbo` | `tada` | `kokoro`. For long text, tune
`max_chunk_chars` (default 800) and `crossfade_ms` (default 50).

### 4. Fetch the generated audio

```
GET /audio/{generation_id}
```
Streams the audio file. (Also `GET /history/{generation_id}/export-audio`.)

**Async note:** `/generate` may return `status: "generating"`. Poll
`GET /generate/{generation_id}/status` until complete, or use the streaming
variant `POST /generate/stream`.

---

## Minimal provider flow (pseudocode)

```python
B = "http://box:17493"

# one-time per voice: create profile + add sample(s)
pid = post(f"{B}/profiles", json={"name": "Lecturer A", "language": "en",
                                   "voice_type": "cloned"})["id"]
post(f"{B}/profiles/{pid}/samples",
     files={"file": open("reference.wav", "rb")},
     data={"reference_text": "The exact words in the clip."})

# per generation:
gen = post(f"{B}/generate", json={"profile_id": pid, "text": "Hello world",
                                  "engine": "chatterbox"})
# poll if needed:
while get(f"{B}/generate/{gen['id']}/status")["status"] == "generating":
    sleep(1)
audio = get_bytes(f"{B}/audio/{gen['id']}")
```

Reuse the `profile_id` across many generations — cloning (steps 1–2) is one-time
per voice; step 3 is per utterance.

---

## Should you use this instead of Chatterbox?

Both do voice-clone TTS. For a **provider abstraction** in the calling tool:

- **Chatterbox (:8004)** — the currently-integrated, verified TTS. Simplest if
  you just need "text → cloned-voice audio."
- **Voicebox (:17493)** — richer: named profiles, multiple engines (incl.
  Chatterbox), effects, personality rewriting. More setup, still "under
  evaluation" in Puente.

Adding both as selectable TTS providers is reasonable — they don't conflict.
See [service topology](/service-topology/) for the full picture.
