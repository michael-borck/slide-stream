---
title: "SadTalker Talking-Head API (via ComfyUI)"
---

How to generate talking-head videos (photo + audio → lip-synced MP4) by calling
the SadTalker node through ComfyUI's HTTP API. This is the contract for a remote
tool (e.g. the slide-stream avatar provider) driving the box.

> **Verified working** 2026-07-06: a generated portrait + 3s audio produced a
> 768×768 h264 talking-head clip end-to-end via this exact API flow.

---

## Endpoint

**Base URL:** `https://comfyui.locopuente.org` (ComfyUI native API, port 8188
behind nginx-proxy-manager → `192.168.20.120:8188`).

SadTalker runs as a ComfyUI custom node, so there is no dedicated
"SadTalker endpoint" — you POST a **workflow graph** to ComfyUI's `/prompt`.

> ⚠️ **No auth.** ComfyUI's API is currently unauthenticated — anyone who
> reaches the host can run any workflow. Add HTTP basic-auth / an access rule in
> nginx-proxy-manager before relying on this publicly.

---

## The 3-step flow

1. **Submit** — `POST /prompt` with `{"prompt": <workflow>}` → returns
   `{"prompt_id": "..."}`.
2. **Poll** — `GET /history/{prompt_id}` until the entry appears with
   `status.status_str == "success"`.
3. **Read the result** — the output MP4 path is in the response under the
   ShowVideo node's `show_video_path`.

Inputs (the face image and the audio) must exist in ComfyUI's `input/` directory
before you reference them by filename. Upload via `POST /upload/image`
(multipart) — it accepts audio files too despite the name — or place them on the
shared volume `~/.puente/comfyui-basedir/input/`.

---

## Workflow JSON (API format)

This is the **proven** graph. `LoadImage` + `LoadAudio` feed `SadTalker`; the
`ShowVideo` node is the required terminal/output node that surfaces the result
path.

```json
{
  "1": {"class_type": "LoadImage", "inputs": {"image": "portrait.png"}},
  "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}},
  "3": {"class_type": "SadTalker", "inputs": {
        "image": ["1", 0],
        "audio": ["2", 0],
        "poseStyle": 0,
        "faceModelResolution": "256",
        "preprocess": "full",
        "stillMode": true,
        "batchSizeInGeneration": 2,
        "gfpganAsFaceEnhancer": false,
        "useIdleMode": false,
        "idleModeTime": 5,
        "useRefVideo": false,
        "refInfo": "pose"
  }},
  "4": {"class_type": "ShowVideo", "inputs": {"show_video_path": ["3", 1]}}
}
```

Success response from `/history/{id}` contains:

```json
{"4": {"show_video_path": ["/basedir/output/20260706133111.mp4"]}}
```

Fetch that file from the shared output volume
(`~/.puente/comfyui-basedir/output/`), or via `GET /view?filename=...&type=output`.

---

## Parameter guide

| Input | Values | Notes |
|-------|--------|-------|
| `image` | filename in `input/` | **Must be a real front-facing face.** If no face is detected SadTalker raises "can not detect the landmark from source image". |
| `audio` | filename in `input/` | Output video length = audio length. |
| `poseStyle` | `0`–`45` | Head-pose style seed. |
| `faceModelResolution` | `"256"` / `"512"` | 256 faster, 512 sharper. |
| `preprocess` | `crop` / `resize` / `full` / `extcrop` / `extfull` | **Use `full`** for robustness — `crop` is stricter and fails on many images. |
| `stillMode` | `true` / `false` | `true` = minimal head motion. **Recommended** for the corner-circle avatar use case (much less jank). |
| `gfpganAsFaceEnhancer` | `true` / `false` | Face upscaling/enhancement. |
| `useRefVideo` + `refVideo` + `refInfo` | bool / VIDEOSTRING / `pose\|blink\|pose+blink\|all` | Drive pose/blink from a reference video instead of synthesizing. |

Outputs: `video_path` (index 0), `show_video_path` (index 1) — both STRING paths.

---

## Known-good example (reproduce the verification)

```python
import json, urllib.request, time
BASE = "https://comfyui.locopuente.org"

wf = { ... }  # the workflow JSON above, with your image/audio filenames

req = urllib.request.Request(BASE + "/prompt",
        data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"})
pid = json.load(urllib.request.urlopen(req))["prompt_id"]

while True:
    h = json.load(urllib.request.urlopen(f"{BASE}/history/{pid}"))
    if pid in h and h[pid]["status"]["status_str"] == "success":
        path = h[pid]["outputs"]["4"]["show_video_path"][0]
        print("video:", path)
        break
    time.sleep(5)
```

SadTalker on the 2060 Super takes ~30s for a few-second clip. Longer audio → longer.
```
