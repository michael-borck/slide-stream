# LivePortrait Reenactment API (via ComfyUI)

How to animate a portrait from a **driving video** (expressive face reenactment)
via the LivePortrait node in ComfyUI. This is the highest-quality avatar option —
full-face reenactment with no mouth-box seam, above SadTalker and Wav2Lip.

> **Verified working** 2026-07-07 on the 8 GB RTX 2060 Super: portrait + driving
> video → clean full-face reenactment MP4 (768×768 h264). LivePortrait is
> VRAM-light, so it needed no special hardware even then — and it runs on GPU 0's
> 24 GB RTX 3090 with room to spare. (LatentSync's ~20 GB, once disqualifying,
> now fits too — see `gpu-swap-3090.md`.)

---

## What it does (vs the other avatar tools)

- **SadTalker** — photo → *synthesized* motion (convenient, a bit stiff).
- **Wav2Lip** — video → mouth-only swap (soft mouth, faint seam).
- **LivePortrait** — photo → motion *copied from a real driving video*: blinks,
  brows, head turns, expression. The most natural and expressive, because it
  transfers a real performance onto the portrait.

Input is a **driving video** (a real face performance), not audio. If you only
have audio, use SadTalker or Wav2Lip; LivePortrait's audio-driven mode (JoyVASA)
lives in the standalone `faster_liveportrait` image, not this ComfyUI node.

> **Licensing:** the working cropper uses Insightface's `buffalo_l`, which is
> **non-commercial**. Fine for a proof-of-concept / evaluation. For commercial
> deployment, switch to the MediaPipe cropper (MIT/Apache) — note the shipped
> mediapipe currently has a numpy-2 import issue to resolve first.

---

## Endpoint

`https://comfyui.locopuente.org` — same ComfyUI workflow API as SadTalker/Wav2Lip
(`POST /prompt` → poll `GET /history/{id}` → read output path).

Nodes (installed by the comfyui `post_start` hook when `install_liveportrait: true`):
`DownloadAndLoadLivePortraitModels`, `LivePortraitLoadCropper` (Insightface) /
`LivePortraitLoadMediaPipeCropper`, `LivePortraitCropper`, `LivePortraitProcess`,
`LivePortraitComposite`. Uses `VHS_LoadVideo` / `VHS_VideoCombine` for video I/O.

---

## Workflow JSON (API format) — verified

```json
{
  "1": {"class_type": "DownloadAndLoadLivePortraitModels", "inputs": {}},
  "2": {"class_type": "LivePortraitLoadCropper",
        "inputs": {"onnx_device": "CUDA", "keep_model_loaded": true}},
  "3": {"class_type": "LoadImage", "inputs": {"image": "portrait.png"}},
  "4": {"class_type": "VHS_LoadVideo",
        "inputs": {"video": "driving.mp4", "force_rate": 0, "custom_width": 0,
                   "custom_height": 0, "frame_load_cap": 0,
                   "skip_first_frames": 0, "select_every_nth": 1}},
  "5": {"class_type": "LivePortraitCropper",
        "inputs": {"pipeline": ["1",0], "cropper": ["2",0], "source_image": ["3",0],
                   "dsize": 512, "scale": 2.3, "vx_ratio": 0.0, "vy_ratio": -0.125,
                   "face_index": 0, "face_index_order": "large-small", "rotate": false}},
  "6": {"class_type": "LivePortraitProcess",
        "inputs": {"pipeline": ["1",0], "crop_info": ["5",1], "source_image": ["3",0],
                   "driving_images": ["4",0], "lip_zero": false, "lip_zero_threshold": 0.03,
                   "stitching": true, "delta_multiplier": 1.0, "mismatch_method": "constant",
                   "relative_motion_mode": "relative",
                   "driving_smooth_observation_variance": 3e-7}},
  "8": {"class_type": "LivePortraitComposite",
        "inputs": {"source_image": ["3",0], "cropped_image": ["6",0],
                   "liveportrait_out": ["6",1]}},
  "7": {"class_type": "VHS_VideoCombine",
        "inputs": {"images": ["8",0], "frame_rate": 25, "loop_count": 0,
                   "filename_prefix": "liveportrait", "format": "video/h264-mp4",
                   "pingpong": false, "save_output": true}}
}
```

Success → `/history/{id}` returns the MP4 under the VHS node's `gifs[0].fullpath`
(`/basedir/output/...`), fetchable via `GET /view?...&type=output`.

**Wiring gotcha:** `LivePortraitProcess` output 1 is `LP_OUT` (not an image) — it
MUST go through `LivePortraitComposite` (with source + cropped_image) to produce
displayable frames. Feeding it straight to VHS fails type validation.

---

## Parameter notes

- `scale` / `vy_ratio` — face crop framing; the values above are the common defaults.
- `stitching: true` — blends the animated face back into the full frame.
- `relative_motion_mode: "relative"` — transfers motion *relative* to the source
  pose (most natural); `off` copies absolute pose.
- `driving_smooth_observation_variance` — motion smoothing; lower = more faithful.

Reproduce on a new machine: `install_liveportrait: true` under `comfyui` in
`puente.yml`, then `puente up comfyui` (installs node + ~716MB models + insightface).
