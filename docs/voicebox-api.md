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

## Transcribing a clip

To clone from audio alone (no hand-typed transcript), transcribe it first:

```
POST /transcribe
Content-Type: multipart/form-data

file=@reference.wav
language=en          # optional
model=<whisper-size> # optional; server default otherwise
```
→ `{"text": "...", "duration": <seconds>}`, via server-side Whisper.

**First-call note:** if the Whisper model is not cached yet, this returns
**HTTP 202** with `{"downloading": true, ...}` and fetches the model in the
background. Wait and retry; it is not an error.

---

## Deleting what you created

Voicebox exposes three delete endpoints:

```
DELETE /profiles/{profile_id}          # profile + its samples + reference audio
DELETE /profiles/samples/{sample_id}   # one sample
DELETE /history/{generation_id}        # one generation + its rendered audio
```

`DELETE /profiles/{id}` is a **deep** delete (verified in
`backend/services/profiles.py::delete_profile`): it removes the sample rows,
the profile row, `rmtree`s the profile's directory under `get_profiles_dir()`
— which is where the uploaded reference audio, i.e. the clone itself, lives —
and clears the combined-audio cache. You do **not** need to delete samples
individually first.

> **It does not cascade to generations.** `Generation.profile_id` is a plain
> `ForeignKey` column with no SQLAlchemy `relationship(..., cascade=...)`, and
> SQLite does not enforce FKs unless `PRAGMA foreign_keys=ON`. Deleting a
> profile therefore leaves orphaned history rows whose rendered audio is still
> on disk and still fetchable via `GET /audio/{generation_id}`. To leave nothing
> behind, delete each generation as well.

---

## Listing generation history

To find a profile's generations (e.g. before deleting the profile):

```
GET /history?profile_id=<id>&limit=100&offset=0
```

Query params: `profile_id` (filter to one profile), `limit` and `offset`
(pagination). Response shape:

```json
{"items": [{"id": "...", "profile_id": "...", ...}, ...], "total": 42}
```

> **Note:** this is what `contrib/voicebox/sweep_ephemeral_profiles.py`
> assumes. Verify it against the live server's `/openapi.json` — in
> particular that `/history` honours the `profile_id` filter. The sweep
> script does not blindly trust it: it re-checks each item's own
> `profile_id` field client-side and never deletes items lacking that field,
> so a server that ignores the filter cannot cause it to delete the whole
> history.

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

### Ephemeral clone (what SlideStream's `voicebox` provider does)

When `voice_sample` is configured instead of `profile_id`, the provider clones
for the run and leaves nothing behind. Cloning needs a transcript of the clip;
if `reference_text` is not given, the clip is first transcribed via
`POST /transcribe` (see "Transcribing a clip" above), so only the audio is
required.

```python
reference_text = reference_text or transcribe(reference.wav)  # POST /transcribe
pid = create_profile(name=f"slide-stream-{uuid4()}")  # random: no speaker-recognisable label
add_sample(pid, reference.wav, reference_text)
try:
    for slide in deck:                       # per slide
        gen_id = generate(pid, slide.text)
        try:
            audio = get_bytes(f"{B}/audio/{gen_id}")
        finally:
            delete(f"{B}/history/{gen_id}")  # drop the rendered narration
finally:
    delete(f"{B}/profiles/{pid}")            # drop the clone, even on crash
```

Both deletes must sit in `finally` blocks: a run that dies mid-render would
otherwise strand exactly the clone you were trying not to store. Set
`delete_generations: false` to keep the rendered audio in Voicebox's history.

### Backstop: sweeping orphans

A `finally` cannot run if the process is `SIGKILL`ed, OOM-killed, or the box
loses power. `contrib/voicebox/sweep_ephemeral_profiles.py` reaps whatever such
a run left behind — it deletes each profile's generations, then the profile.
Stdlib only; copy it onto the Voicebox host.

```
*/15 * * * * /opt/slide-stream/sweep_ephemeral_profiles.py \
    --base-url http://localhost:17493 --max-age-minutes 60
```

The grace period is what stops it deleting the clone out from under a render
still in progress, so keep `--max-age-minutes` comfortably above your longest
deck. It only ever deletes profiles that are `voice_type: cloned` **and** named
exactly `slide-stream-<uuid4>`; a profile you made by hand cannot match. Use
`--dry-run` first, and `--keep-generations` to reap clones but retain history.
It exits non-zero if any deletion failed, so cron will mail you.

---

## Should you use this instead of Chatterbox?

Both do voice-clone TTS. For a **provider abstraction** in the calling tool:

- **Chatterbox (:8004)** — the currently-integrated, verified TTS. Simplest if
  you just need "text → cloned-voice audio." No delete API: uploaded references
  are named with a UUID and reaped by the cron in `contrib/chatterbox/`.
- **Voicebox (:17493)** — richer: named profiles, multiple engines (incl.
  Chatterbox), effects, personality rewriting. More setup, still "under
  evaluation" in Puente. **Can delete clones over the API**, so the ephemeral
  flow above needs no server-side cleanup job.

Adding both as selectable TTS providers is reasonable — they don't conflict.
See [service topology](/service-topology/) for the full picture.
