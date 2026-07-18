# Non-human talking avatars — remote API (LivePortrait animal + Wan2.2-S2V)

How a **remote app** drives puente's two non-human avatar paths over HTTP. Both run
as nodes inside the shared ComfyUI instance and are called through ComfyUI's native
workflow API — there is no separate microservice.

- **Wan2.2-S2V** — still image + audio → lip-synced video. **No face detector at
  all**, so it works on teddy bears, cartoon animals, robots. This is the
  recommended path for non-human subjects. See also [wan-s2v-api.md](wan-s2v-api.md).
- **LivePortrait (animal mode)** — portrait + driving *video* → reenactment, using
  animal-trained generator weights. Higher quality *when it works*, but the cropper
  is still human-only, so it needs a manual crop for non-human subjects. See also
  [liveportrait-api.md](liveportrait-api.md).

> Why two: every other avatar tool (SadTalker, Wav2Lip, LivePortrait, LatentSync)
> starts by detecting a **human** face and cropping to it, so they fail on non-human
> characters. Wan2.2-S2V has no detector anywhere in its path; LivePortrait animal
> mode swaps the *generator* weights but not the detector, so you must feed it a
> pre-cropped head. See memory `nonhuman-avatar-face-detection`.

---

## Endpoint & auth

**Base URL:** `https://comfyui.locopuente.org` (ComfyUI native API, internal port 8188).

⚠️ **Currently unauthenticated.** Add auth (Caddy basic_auth / bearer) before any
public app uses this. Treat the base URL as a secret until then.

Everything below is standard ComfyUI HTTP API — the same four calls for both models:

| Step | Call |
|---|---|
| 1. Upload inputs | `POST /upload/image` (multipart; accepts images, audio, video despite the name) |
| 2. Queue the graph | `POST /prompt` with `{"prompt": <graph>, "client_id": <uuid>}` → returns `{"prompt_id": ...}` |
| 3. Poll for completion | `GET /history/{prompt_id}` — empty until done, then holds the outputs |
| 4. Fetch the result | `GET /view?filename=...&subfolder=...&type=output` |

Optional but recommended between runs (see VRAM gotcha for S2V):
`POST /free` with `{"unload_models": true, "free_memory": true}`.

### The generic call flow (language-agnostic)

```
uuid        = new client id (any uuid v4)
POST /upload/image   (character image)        -> {"name": "char.png",  "subfolder": "", "type": "input"}
POST /upload/image   (audio.wav OR driving.mp4)-> {"name": "voice.wav", ...}
# build the graph JSON (below), referencing the returned "name" values
POST /prompt {"prompt": graph, "client_id": uuid} -> {"prompt_id": PID}
loop: GET /history/{PID} until the body is non-empty
        -> outputs[node_id]["gifs"][0] has {filename, subfolder, fullpath, format}
GET /view?filename=<f>&subfolder=<s>&type=output  -> the MP4 bytes
```

`fullpath` in the history response is the path **on the server's shared volume**
(`~/.puente/comfyui-basedir/output/`). A remote app that isn't on the box should use
`/view` (step 4) to pull the bytes, not `fullpath`.

Uploaded inputs land in `~/.puente/comfyui-basedir/input/`; you may also drop files
there directly if the app runs on the host, skipping step 1.

---

## Path A — Wan2.2-S2V (recommended for non-human)

**Input:** one still image + one audio clip. **No driving video, no crop, no face
input.** The reference image goes straight into the VAE as pixels.

### Graph (API format)

```json
{
  "1": {"class_type": "UNETLoader",
        "inputs": {"unet_name": "wan2.2_s2v_14B_fp8_scaled.safetensors",
                   "weight_dtype": "default"}},
  "2": {"class_type": "CLIPLoader",
        "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
  "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
  "4": {"class_type": "AudioEncoderLoader",
        "inputs": {"audio_encoder_name": "wav2vec2_large_english_fp16.safetensors"}},
  "5": {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}},
  "6": {"class_type": "AudioEncoderEncode",
        "inputs": {"audio_encoder": ["4", 0], "audio": ["5", 0]}},
  "7": {"class_type": "LoadImage", "inputs": {"image": "character.png"}},
  "8": {"class_type": "CLIPTextEncode",
        "inputs": {"clip": ["2", 0],
                   "text": "a plush teddy bear talking, muzzle opening and closing to form words, warm lighting"}},
  "9": {"class_type": "CLIPTextEncode",
        "inputs": {"clip": ["2", 0], "text": "blurry, distorted, static, still"}},
  "10": {"class_type": "WanSoundImageToVideo",
         "inputs": {"positive": ["8", 0], "negative": ["9", 0], "vae": ["3", 0],
                    "audio_encoder_output": ["6", 0], "ref_image": ["7", 0],
                    "width": 448, "height": 608, "length": 49}},
  "11": {"class_type": "KSampler",
         "inputs": {"model": ["1", 0], "positive": ["10", 0], "negative": ["10", 1],
                    "latent_image": ["10", 2], "seed": 42, "steps": 20, "cfg": 6.0,
                    "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0}},
  "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
  "13": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": 16, "audio": ["5", 0]}},
  "14": {"class_type": "SaveVideo",
         "inputs": {"video": ["13", 0], "filename_prefix": "s2v", "format": "mp4", "codec": "h264"}}
}
```

Node/socket names track ComfyUI core (`comfy_extras/nodes_wan.py`); if a socket name
drifts, `GET /object_info/WanSoundImageToVideo` returns the live input schema.

### Rules that will bite you

- **`length` must be `4n+1`.** Wan runs at **16 fps**: 3s ≈ 49, 5s ≈ 81.
- **`POST /free` before every run.** The 16.4 GB unet stays resident, so the *second*
  run OOMs in `CLIPTextEncode` (not in diffusion). Free models between generations.
  Peak is ~21 GB of the 3090's 24 GB — little headroom.
- **Prompt the mouth anatomy explicitly** — there's no detector, so the text prompt is
  how the model learns where the mouth is. An owl has a beak, not lips.
- **Steps:** 10 is fast but can drift identity / blur frame 1; raise to ~20 to tighten.
- **Do NOT use `WanAnimateToVideo`** — it's a different pose/face-driven model that
  reintroduces face detection.
- Cost: ~4 min for 3s @ 10 steps, 448×608, on the 3090.

---

## Path B — LivePortrait animal mode (portrait + driving video)

Same graph as [liveportrait-api.md](liveportrait-api.md), with **two** changes:

1. **Select animal weights** — set `mode: "animal"` on the model loader node:

   ```json
   "1": {"class_type": "DownloadAndLoadLivePortraitModels",
         "inputs": {"mode": "animal"}}
   ```

2. **Bypass the human cropper.** Kijai's wrapper ships **no animal face detector**, so
   `LivePortraitCropper` logs "No face detected" on a non-human subject and produces
   nothing. Do **not** wire the cropper. Instead pre-crop the head yourself (square,
   head roughly centred) and feed that image straight into `LivePortraitProcess` in
   place of the cropper's output:

   ```json
   "6": {"class_type": "LivePortraitProcess",
         "inputs": {"pipeline": ["1", 0],
                    "crop_info": null,
                    "source_image": ["3", 0],
                    "driving_images": ["4", 0],
                    "lip_zero": false, "stitching": true,
                    "delta_multiplier": 1.0,
                    "relative_motion_mode": "relative",
                    "driving_smooth_observation_variance": 3e-7}}
   ```

   `["3",0]` is your **manually cropped** `LoadImage`. Output 1 of
   `LivePortraitProcess` is `LP_OUT` — route it through `LivePortraitComposite` before
   VHS, exactly as in the human graph, or type validation fails.

**Reality check:** animal mode only swaps the generator/motion models — it is a
partial answer. If the character isn't a real animal head (a plush toy, a stylised
cartoon), reenactment quality is unreliable. For anything genuinely non-human,
**prefer Path A (Wan2.2-S2V)**, which has no detector to fail.

---

## Enabling on a fresh box

In `puente.yml` under `comfyui:` then `puente up comfyui` (idempotent, downloads only):

```yaml
comfyui:
  install_wan_s2v: true          # Path A — ~22GB weights, installs NO code
  install_liveportrait: true     # Path B — node + ~716MB models + insightface
  liveportrait_animal: true      # Path B animal weights — ~520MB (sets PUENTE_LP_ANIMAL=1)
```

`install_wan_s2v` fetches only weights (the nodes are ComfyUI core), so it can't
disturb the shared numpy-2 / torch-2.6 venv — that safety is why S2V beat LatentSync.

---

## Minimal client (reference)

```python
import requests, uuid, time

BASE = "https://comfyui.locopuente.org"

def upload(path):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/upload/image", files={"image": f})
    r.raise_for_status()
    return r.json()["name"]          # use this name in the graph

def free():
    requests.post(f"{BASE}/free", json={"unload_models": True, "free_memory": True})

def run(graph, out_node):
    cid = str(uuid.uuid4())
    pid = requests.post(f"{BASE}/prompt",
                        json={"prompt": graph, "client_id": cid}).json()["prompt_id"]
    while True:
        h = requests.get(f"{BASE}/history/{pid}").json()
        if h.get(pid):
            break
        time.sleep(2)
    g = h[pid]["outputs"][out_node]["gifs"][0]
    vid = requests.get(f"{BASE}/view",
                       params={"filename": g["filename"],
                               "subfolder": g.get("subfolder", ""),
                               "type": "output"})
    return vid.content               # MP4 bytes

# Wan2.2-S2V:
free()                               # MUST free before queueing
img = upload("teddy.png"); wav = upload("voice.wav")
# ...patch graph["7"]["inputs"]["image"]=img, graph["5"]["inputs"]["audio"]=wav...
mp4 = run(graph, "14")
```

For LivePortrait the output node is the `VHS_VideoCombine` id, and the result is under
`gifs[0]` there too.
