# Wan2.2-S2V — talking characters that aren't human

Audio-driven video generation: **still image + audio → lip-synced video**. Unlike every
other avatar tool in puente, this one works on **non-human characters** — teddy bears,
cartoon animals, robots.

## Why this exists

SadTalker, Wav2Lip, LivePortrait and LatentSync all begin their pipeline by **detecting a
human face** and cropping to it (S3FD, 3DMM, InsightFace buffalo_l, mediapipe). A teddy
bear has no human landmark geometry, so detection returns nothing and the tool fails —
often *silently*, warning "No face detected" while still reporting success.

Wan2.2-S2V has **no detector, no landmarker and no crop anywhere in its path**. Its
conditioning is `wav2vec2(audio)` + `vae.encode(ref_image)` → latents (see ComfyUI's
`comfy_extras/nodes_wan.py::wan_sound_to_video`). The reference image goes straight into
the VAE as pixels. There is no face-detection stage for a non-human subject to fail.

Verified 2026-07-13 on a plush teddy bear: real, varied, phoneme-like muzzle articulation.

## Install

`install_wan_s2v: true` under `comfyui:` in `puente.yml`, then `puente up comfyui`.

**This installs no code.** The nodes (`WanSoundImageToVideo`, `AudioEncoderLoader`,
`AudioEncoderEncode`) ship in ComfyUI **core**. The hook only downloads weights, so —
unlike LatentSync — it cannot disturb the shared numpy-2 / torch-2.6 venv that SwarmUI and
the other avatar nodes depend on. That safety is the main reason S2V was chosen.

Weights (~22GB, from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`):

| File | Size | → `/basedir/models/` |
|---|---|---|
| `wan2.2_s2v_14B_fp8_scaled.safetensors` | 16.4GB | `diffusion_models/` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.7GB | `text_encoders/` |
| `wav2vec2_large_english_fp16.safetensors` | 0.6GB | `audio_encoders/` |
| `wan_2.1_vae.safetensors` | 0.25GB | `vae/` |

Use the **fp8** diffusion model, not the bf16 build (32.6GB — will not fit a 24GB card).
If fp8 is still tight, `QuantStack/Wan2.2-S2V-14B-GGUF` goes down to ~13.9GB (Q4_K_M).

## Gotchas

**VRAM: free between runs.** The 16.4GB unet stays resident after a generation, leaving no
room for the 6.7GB text encoder — so the *second* run OOMs in `CLIPTextEncode`, not in the
diffusion model. Before queueing, `POST /free {"unload_models": true, "free_memory": true}`.
Peak usage is ~21GB of the 3090's 24GB, so there is very little headroom.

**`length` must be 4n+1.** Wan runs at **16fps**, so 3s ≈ 49 frames, 5s ≈ 81.

**Do NOT wire `WanAnimateToVideo`.** `nodes_wan.py` also contains `face_video` code around
L1127-1209 — that belongs to `WanAnimateToVideo`, a *different* pose/face-driven model. It
is not on the S2V path and it would reintroduce the face-detection problem.

**Prompt the mouth explicitly.** With no detector, the text prompt is the main lever for
telling the model where the mouth is and how it moves. Describe the actual anatomy — an owl
has a beak, not lips. Low step counts (10) can drift the character's identity from the
reference and produce a blurry first frame; raise steps to tighten it.

## Workflow (ComfyUI HTTP API)

```
UNETLoader(wan2.2_s2v_14B_fp8_scaled)  ─────────────┐
CLIPLoader(umt5_xxl_fp8, type="wan")  ─> CLIPTextEncode(+/-) ─┐
VAELoader(wan_2.1_vae) ─────────────────────────────┐         │
AudioEncoderLoader(wav2vec2_large_english_fp16)     │         │
   └─> AudioEncoderEncode(audio=LoadAudio) ──┐      │         │
LoadImage(character.png) ───────ref_image──┐ │      │         │
                                           v v      v         v
                              WanSoundImageToVideo(width,height,length=4n+1)
                                    │ positive, negative, latent
                                    v
                              KSampler(uni_pc/simple) -> VAEDecode
                                    -> CreateVideo(fps=16, audio) -> SaveVideo(mp4/h264)
```

`ref_image` is a plain `IMAGE`. There is no cropper node and no face input — that is the
whole point.

Working script: `s2v_test.py` (scratchpad). Typical cost: ~4 min for 3s @ 10 steps,
448x608, on the 3090.

See also `docs/sadtalker-api.md`, `docs/wav2lip-api.md`, `docs/liveportrait-api.md` — use
those for **human** subjects, where they are cheaper and faster than S2V.
