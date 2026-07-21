---
title: "Wav2Lip Video Lip-Sync API (via ComfyUI)"
---

How to re-sync a talking-head **video** to new **audio** (video → lip-synced
MP4) by calling the Wav2Lip node through ComfyUI's HTTP API. This is the
video-driven counterpart to [SadTalker](sadtalker-api.md) (which is photo-driven).

> **Verified working** 2026-07-07: a reference video + 3s audio produced a
> lip-synced h264+aac MP4 end-to-end via this exact API flow, on the 8 GB
> RTX 2060 Super.

---

## Why Wav2Lip (and not LatentSync/MuseTalk)

For the video → lip-sync family on **8 GB VRAM**, Wav2Lip is the pragmatic pick:

- **LatentSync 1.6** is the quality leader but is spec'd for ~20 GB VRAM — it
  will not fit the 2060 Super. Revisit only with a 16 GB+ card.
- **MuseTalk** has a painful pinned-2024 install and degrades facial features.
- **Wav2Lip** is light (~2 GB at inference), installs cleanly, and runs on 8 GB.
  Trade-off: it renders the mouth region at 96×96 and pastes it back, so the
  mouth is a bit soft with a faint seam. At small "corner-circle" avatar sizes
  this is barely visible.

---

## Endpoint

**Base URL:** `https://comfyui.locopuente.org` (ComfyUI native API, port 8188).
Same workflow-graph API as SadTalker: `POST /prompt` → poll `GET /history/{id}`
→ read the output path. (⚠️ still unauthenticated — add auth before public use.)

---

## Nodes involved

Installed by the comfyui `post_start` hook when `install_wav2lip: true`:

- **`Wav2Lip`** — the sync engine. Inputs: `images` (IMAGE batch = video frames),
  `mode` (`sequential` / `repetitive`), `face_detect_batch` (INT), `audio`
  (AUDIO). Outputs: `images` (synced frames), `audio`.
- **`VHS_LoadVideo`** — video file → IMAGE frames + audio (VideoHelperSuite).
- **`VHS_VideoCombine`** — frames + audio → MP4. Supports `pingpong` (boomerang
  loop) — useful to hide the seam when looping a short clip under longer audio.

Wav2Lip works on **frames**, so a real video flow needs VHS to load/encode.

---

## Workflow JSON (API format) — full video → video

The **verified** graph: load reference video → Wav2Lip with new audio → encode MP4.

```json
{
  "1": {"class_type": "VHS_LoadVideo", "inputs": {
        "video": "reference.mp4", "force_rate": 0, "custom_width": 0,
        "custom_height": 0, "frame_load_cap": 0, "skip_first_frames": 0,
        "select_every_nth": 1}},
  "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}},
  "3": {"class_type": "Wav2Lip", "inputs": {
        "images": ["1", 0], "mode": "sequential",
        "face_detect_batch": 8, "audio": ["2", 0]}},
  "4": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["3", 0], "audio": ["3", 1], "frame_rate": 25,
        "loop_count": 0, "filename_prefix": "wav2lip_out",
        "format": "video/h264-mp4", "pingpong": false, "save_output": true}}
}
```

Success response from `/history/{id}`:

```json
{"4": {"gifs": [{"filename": "wav2lip_out_00001-audio.mp4",
       "fullpath": "/basedir/output/wav2lip_out_00001-audio.mp4",
       "format": "video/h264-mp4"}]}}
```

The MP4 (h264 video + aac audio muxed) is at that `fullpath` on the shared
output volume (`~/.puente/comfyui-basedir/output/`), or via
`GET /view?filename=...&type=output`.

---

## Notes

- **Input files** (reference video, audio) must be in ComfyUI's `input/` dir
  before referencing them — upload via `POST /upload/image` (multipart, accepts
  video/audio too) or place on the shared volume `~/.puente/comfyui-basedir/input/`.
- **Output length = audio length.** If the reference video is shorter than the
  audio, `mode: "repetitive"` loops the source frames to cover it. For a
  seamless loop, set `pingpong: true` on VHS_VideoCombine (boomerang).
- **Reference video** should be a real face roughly facing camera. ~10–15s of
  slight natural idle motion looks best; a 1s clip looped many times shows a
  visible repeat.
- **Static image instead of video:** feed a single image as `images` (via
  LoadImage) — Wav2Lip will animate the mouth on a held frame (stiffer, but works).
- **Speed:** ~real-time-ish on the 2060 for short clips; face detection is the
  main cost (tune `face_detect_batch`).

---

## Reproducing on a new machine

`install_wav2lip: true` under `comfyui` in `puente.yml`, then `puente up comfyui`
installs both nodes, VHS requirements, and the wav2lip weight automatically
(idempotent). The S3FD face-detector weight auto-downloads on first inference.
