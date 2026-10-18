# Providers

Every engine SlideStream can use for each job, how to switch it on, and what
happens when it isn't available. Set providers in `slidestream.yaml` (see
[configuration](USER_GUIDE.md#configuration)) or per run with a flag such as
`--tts-provider` or `--llm-provider` (`enrich` also takes `--image-provider`). For the servers behind the
self-hosted providers (ports, health checks, licences) see
[SERVICES.md](SERVICES.md).

**Fallbacks.** Each job falls back from the configured service to a local
engine to a bare minimum, so a render always finishes
([ADR-0012](adr/0012-three-level-fallback-kokoro-floor.md)). With
`settings.strict: true` (or `--strict`) a provider you chose explicitly is
never replaced; the run fails instead. `slide-stream doctor <deck>` reports
any configured provider whose package or key is missing.

API keys come from the environment. Reference them in config as
`"${VAR}"`, or let the provider read its standard variable.

## Voice (text to speech)

`providers.tts.provider`. Default: `voicebox`, which with no server set
quietly uses the voice floor below.

| Provider | What it is | Needs |
|---|---|---|
| `auto` | The voice floor: Kokoro if installed, else gTTS. Written by `slide-stream init`. | nothing |
| `kokoro` | Kokoro-82M, fully offline; voices include `af_sarah`, `af_bella`, `am_adam`, `am_michael` | `slide-stream[local-tts]` (in `[all]`); about 340 MB downloaded once |
| `gtts` | Google Text-to-Speech; English accents via `accent:` (`australian`, `british`, `american`, `canadian`, `indian`, `irish`, `south-african`) | internet; sends text to Google |
| `voicebox` | Self-hosted [Voicebox](voicebox-api.md) studio, several engines (`engine: kokoro`, `chatterbox`, `qwen`, …); a stored `profile_id`, or `voice_sample` for a clone deleted after the run | `base_url` |
| `chatterbox` | Self-hosted [Chatterbox TTS Server](https://github.com/devnen/Chatterbox-TTS-Server) voice cloning; `voice_sample` uploaded under a random name per run | `base_url` |
| `elevenlabs` | ElevenLabs voices (`voice: rachel`, …) | `ELEVENLABS_API_KEY`, `slide-stream[elevenlabs]` |
| `openai` | OpenAI TTS (`voice: alloy`, `echo`, `fable`, `nova`, `onyx`, `shimmer`) | `OPENAI_API_KEY`, `slide-stream[openai]` |
| `openai-compatible` | Any OpenAI-style `/v1/audio/speech` server (LocalAI, Kokoro-FastAPI, …) | `base_url` |

**Voice floor.** When no voice is configured, or the chosen one fails, the
floor speaks: Kokoro when installed (offline, text stays on the machine),
else gTTS. Force one with `settings.voice_floor: kokoro` or `gtts`. As the
floor, Kokoro uses `providers.tts.kokoro_voice` (default `af_sarah`).

**Your own voice.** Record 10 to 30 seconds of clean speech and set
`voice_sample: ./me.wav` (or pass `--voice-sample`) with `voicebox` or
`chatterbox`. The sample is uploaded for that run only and deleted
afterwards ([ADR-0005](adr/0005-privacy-first-voice-cloning.md)). Turn on
`strict` so a server problem never swaps in another voice. Non-WAV samples
are converted with ffmpeg (bundled).

```yaml
providers:
  tts:
    provider: voicebox
    base_url: https://voice.example.org
    voice_sample: ./my_voice.wav
    api_key: "${VOICEBOX_TOKEN}"
settings:
  strict: true
```

## Pictures

`providers.images.provider`. Default: `text`. `fallback:` names the
provider to use if the main one can't start; a single picture that fails
mid-render becomes a text card.

| Provider | What it is | Needs |
|---|---|---|
| `text` | The slide's title and content drawn on the frame, code syntax-coloured | nothing |
| `local` | Pictures from a folder, matched by file name (`folder:`); `slide-stream scan` can name them for you | a folder |
| `swarmui` | Self-hosted [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) (`model:`, `steps:`, `width:`, `height:`) | `base_url` |
| `openai-compatible` | Any OpenAI-style `/v1/images` server | `base_url` |
| `gemini` | Google Imagen, about $0.02 a picture (`model:` picks the tier) | `GEMINI_API_KEY`, `slide-stream[gemini]` |
| `dalle3` | OpenAI image generation | `OPENAI_API_KEY`, `slide-stream[openai]` |
| `pexels` | Pexels stock photos | `PEXELS_API_KEY` |
| `unsplash` | Unsplash stock photos | `UNSPLASH_ACCESS_KEY` |

With a picture provider other than `text`, video slides show the bullets on
the left and the picture on the right (`settings.video.bullets_overlay`).

## Narration writer (LLM)

`providers.llm.provider`. Default: `none`, which speaks the slide text,
speaker notes or `--script` as written. With an LLM, SlideStream writes what
a presenter would *say*, fitted to `settings.narration.target_seconds`, and
describes picture-only slides with a vision model. `draft`, `enrich --notes`
and `scan` need one.

| Provider | Needs |
|---|---|
| `ollama` | a local [Ollama](https://ollama.com); `base_url` for a remote one, `think: false` for reasoning models |
| `openai-compatible` | `base_url` (LocalAI, vLLM, LM Studio, …) |
| `claude` | `ANTHROPIC_API_KEY`, `slide-stream[claude]`; default model `claude-haiku-4-5` |
| `openai` | `OPENAI_API_KEY`, `slide-stream[openai]` |
| `gemini` | `GEMINI_API_KEY`, `slide-stream[gemini]` |
| `groq` | `GROQ_API_KEY`, `slide-stream[groq]` |

Set `model:` (or `--llm-model`) to choose the model; built-in defaults for
some providers are older models.

## Presenter (talking head)

`providers.avatar.provider`, used when you pass `--avatar`. Default: `none`.
`slide-stream avatars` lists the built-in mascots (teddy, panda, koala,
robot, wizard, owl); `source:` takes a mascot name or an image path.

| Provider | What it does | Needs | Mascots? |
|---|---|---|---|
| `static` | Holds the image in a corner circle | nothing | yes |
| `puppet` | Cartoon mouth that opens with the voice's loudness; `mouth: [cx, cy, w, h]` for your own image | nothing | yes |
| `precomputed` | Uses ready-made `head_1.mp4`, `head_2.mp4`, … from `assets_dir` | clips | yes |
| `wan-s2v` | Wan2.2-S2V lip-sync from a still image and the narration, no face detector ([API](wan-s2v-api.md)); `clip_seconds` or `full_length` | ComfyUI `base_url`, a GPU (minutes per slide) | yes |
| `sadtalker` | Talking head from a photo ([API](sadtalker-api.md)) | ComfyUI `base_url`, `source_image` | no, human faces only |
| `wav2lip` | Lip-sync on a short video ([API](wav2lip-api.md)) | ComfyUI `base_url`, `source_video` | no |
| `comfyui` | Picks SadTalker for a photo, Wav2Lip for a video | ComfyUI `base_url`, `source` | no |
| `d-id` | Hosted D-ID, about $1–2 per minute of video | `DID_API_KEY`, `source_image` | yes |

Show the head on some slides only with `--avatar-slides first,last` (also
`all`, `none`, `1,3,5`, `every:3`), or render one clip and loop it with
`--reuse-avatar` ([ADR-0006](adr/0006-avatar-render-economics.md)).
Position and size: `settings.avatar` (`position`, `size`, `margin`).
Background on choosing an engine: [talking-head options](talking-head-options.md).

## Slide design

How slides look. Plain SlideStream slides need nothing; the others are
optional.

| Option | What it does | Needs |
|---|---|---|
| (default) | SlideStream's own slides: plain layout, code and tables drawn properly | nothing |
| `--pptx-template FILE` | New slides in your own PowerPoint template ([ADR-0014](adr/0014-powerpoint-template-design.md)); `template init` / `template check` help | python-pptx (core); LibreOffice for video |
| `--preserve-style` | Narrates a styled `.pptx` and keeps its look in the video | LibreOffice, `slide-stream[presenton]` |
| `--slides-pdf FILE` | Uses a PDF you exported as the video frames | `slide-stream[presenton]` |
| `--template NAME` | Designed slides from [Presenton](https://github.com/presenton/presenton) (experimental); see [USE_CASES](USE_CASES.md#designed-slides-instead-presenton) | a Presenton server, `slide-stream[presenton]` |
| `--pptx-template FILE` | New slides in your own PowerPoint template ([ADR-0014](adr/0014-powerpoint-template-design.md)) | planned |

Presenton settings live under `providers.presenton`: `base_url`, `api_key`,
`template`, `instructions`, `code_slides` (`auto`, `always`, `never`),
`fidelity` (`warn`, `strict`, `off`), `fidelity_min_kept`
([ADR-0013](adr/0013-faithful-designed-decks.md)).
