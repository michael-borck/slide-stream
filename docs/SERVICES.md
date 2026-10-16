# Services SlideStream can use

SlideStream orchestrates other engines; it does not bundle them
([ADR-0010](adr/0010-orchestrator-over-external-services.md)). This page lists
every engine it can call, what each one does for SlideStream, how SlideStream
checks that it is up, and whose project it is. It is also our
acknowledgements page: these projects do the heavy lifting.

Everything here is optional. With nothing configured, SlideStream still
renders a video, with Kokoro (or gTTS if Kokoro isn't installed) for the
voice and text cards for the pictures. Every role falls back service →
local → bare minimum
([ADR-0012](adr/0012-three-level-fallback-kokoro-floor.md)).

**Running a local stack.** [Puente](https://github.com/michael-borck/loco-puente)
(`pip install locopuente`) can stand up most of the self-hosted services below
from one config file. SlideStream works with any instance at any URL; Puente
is a convenience, not a requirement.

## Reading the tables

- **Role** is the pipeline job the service does (ADR-0010).
- **Probe** is the cheap read-only request an adapter's `probe()` sends to
  check the service is up. Entries marked † are proposals. Confirm them
  during that adapter's live test and remove the mark.
- **Tested** is the last version checked against a live service. "Not
  recorded" means it works in practice but nobody wrote the version down. Fill
  these in as live tests land.

## Self-hosted services

| Role | Service | Adapter(s) | Default port | Probe | In Puente | Tested | Licence |
|---|---|---|---|---|---|---|---|
| llm | [Ollama](https://github.com/ollama/ollama) | `ollama` | 11434 | `GET /api/version`, `GET /api/tags` for models † | yes (native) | not recorded | MIT |
| llm, tts, images | Any OpenAI-compatible server (LocalAI, vLLM, Kokoro-FastAPI, openedai-speech, ...) | `openai-compatible` | varies | `GET /v1/models` † | n/a | not recorded | varies |
| tts | Voicebox (see [voicebox-api.md](voicebox-api.md)) | `voicebox` | 17493 | `GET /openapi.json` or `GET /profiles` † | yes | endpoints verified 2026-07-07; version not recorded | to confirm |
| tts | [Chatterbox](https://github.com/resemble-ai/chatterbox) server | `chatterbox` | set `base_url` | to decide † | no | not recorded | MIT (model) |
| images | [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) | `swarmui` | 7801 | `POST /API/GetNewSession` (already the first call of a render) | yes | not recorded | MIT |
| avatar | [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running Wan2.2-S2V, SadTalker or Wav2Lip workflows | `wan-s2v`, `sadtalker`, `wav2lip`, `comfyui` | 8188 | `GET /system_stats`; `GET /object_info/<node>` to confirm a workflow's nodes are installed † | yes | SadTalker verified 2026-07-06; others not recorded | GPL-3.0 (ComfyUI); models vary, see licence notes |
| design | [Presenton](https://github.com/presenton/presenton) | `presenton` (today `presenton.py`; becomes an adapter) | 5001 in SlideStream's config | `GET /api/v1/auth/status` (no login needed) | not yet (proposed) | `v0.9.11-beta`, live trial 16 Oct 2026 | Apache-2.0 |
| music | [ACE-Step 1.5 XL](https://github.com/ace-step/ACE-Step-1.5) (songs with lyrics, or instrumentals). **Default.** | planned | set `base_url` | to decide † | not yet (proposed) | not tested | code and weights MIT; output usable commercially |
| music | [YuE2](https://github.com/multimodal-art-projection/YuE) (songs with lyrics). Alternative. | planned | set `base_url`; also runs in ComfyUI | to decide † | not yet | not tested | code Apache-2.0; weights CC BY-NC 4.0 plus a creator permission, see licence notes |

## Built in or local libraries (no server)

| Role | Engine | How you get it | Notes |
|---|---|---|---|
| tts | Kokoro-82M via `kokoro-onnx` | `slide-stream[local-tts]` (in `[all]`) | The voice floor when installed. Fully offline. About 340 MB model download, once. |
| tts | gTTS | core dependency | The bare-minimum voice. Needs internet; sends the text to Google. |
| images | Text cards | core | The picture floor: the slide's title and bullets on the frame. |
| images | Local folder | core | `providers.images.provider: local` matches files by name. |
| avatar | Static image, puppet mouth-flap, precomputed clips | core | No GPU. |
| design | `basic` fixed layout | core | Today's renderer and `.pptx` exporter (ADR-0008). |
| design | User's PowerPoint template (`--pptx-template`, planned, ADR-0014) | core (python-pptx); LibreOffice for video | New slides in the template's own layouts and style. |
| design | Render a styled `.pptx` or user PDF to frames | `slide-stream[presenton]` ([pypdfium2](https://github.com/pypdfium2-team/pypdfium2), Apache-2.0/BSD-3) plus LibreOffice for `.pptx` | `--preserve-style`, `--slides-pdf` (ADR-0009). |
| ingest | Built-in text extraction (`draft.py`) | core; `slide-stream[import]` for PDF and Word | The ingest floor. |

## Cloud services (bring your own key)

| Role | Service | Adapter | Key |
|---|---|---|---|
| llm | OpenAI, Gemini, Claude, Groq | `openai`, `gemini`, `claude`, `groq` | `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY` |
| tts | ElevenLabs, OpenAI | `elevenlabs`, `openai` | `ELEVENLABS_API_KEY`, `OPENAI_API_KEY` |
| images | DALL-E 3, Gemini Imagen, Pexels, Unsplash | `dalle3`, `gemini`, `pexels`, `unsplash` | per provider |
| avatar | D-ID | `d-id` | `DID_API_KEY` |

For cloud services `probe()` checks that the key is present. A real
authenticated call is optional and must never spend credit.

## Lens family services (planned adapters)

These are first-party engines from the
[lens family](https://github.com/michael-borck/lens-analysers). They already
serve the lens contract, so their adapters can probe `GET /health` and
`GET /manifest` directly.

| Role | Service | Default port | Would do |
|---|---|---|---|
| ingest | `document-analyser` | 8000 | Better text extraction from PDF, DOCX and PPTX than the built-in floor. Needs Python 3.11+, so call it over HTTP rather than importing it. |
| check | `speech-analyser` | 8001 | Check the narration audio: speaking rate, long silences. |
| check | `video-analyser` | 8002 | Check the finished video: on-slide text vs narration, scene timing. |

## Music engines

ACE-Step 1.5 XL is the default because its licence puts no limits on what
users do with the music. YuE2 is the alternative for users who prefer its
songs and fall inside its creator permission.

| | ACE-Step 1.5 XL | YuE2 |
|---|---|---|
| Model size | 4B DiT decoder; base, sft and turbo (8-step) variants | 3B |
| Hardware | 12 GB VRAM with offload, 20 GB+ recommended; MLX on Apple Silicon | ComfyUI package `Comfy-Org/YuE2` |
| Weights | MIT | CC BY-NC 4.0, plus a creator permission (16 Sept 2026) |
| Output use | Commercial use allowed. The model card says the training data is licensed, royalty-free, public domain or synthetic. | Individuals may publish and sell outputs. Non-commercial for academic use. Companies need a commercial licence. |

Puente also runs MusicGen on port 7860. Its weights are CC-BY-NC-4.0, so it is
not a planned adapter. Nothing stops a user pointing SlideStream at it once
the music role exists.

## Ports

Defaults collide. `document-analyser` and the example OpenAI-compatible TTS
server in `slidestream.example.yaml` both use 8000. Gradio apps, including
Puente's MusicGen and the Gradio UIs of most music models, default to 7860.
SlideStream never assumes a port: every service is reached through its
configured `base_url`, and the port above is only the default to suggest.

## Licence notes that affect your videos

SlideStream is MIT. Calling a service over HTTP does not change that. Some
engines put limits on what you can do with their *output* or *weights*:

- **ACE-Step 1.5 XL** is MIT, weights included, and its model card allows
  commercial use of the output. No limits to warn about.
- **YuE2 weights** are CC BY-NC 4.0. A creator permission added on
  16 September 2026 lets individual creators, musicians and personal users
  publish and sell what they generate. Academic use is free but
  non-commercial. A company using it commercially, such as an institution
  selling a course with a generated soundtrack, needs a commercial licence
  from the YuE team.
- **MusicGen weights** are CC-BY-NC-4.0. Music generated with them may not be
  fine for commercial or monetised videos.
- **Wav2Lip** and the **Insightface** models used by some face pipelines are
  for non-commercial research use. See [liveportrait-api.md](archive/liveportrait-api.md)
  for the Insightface case.
- **ComfyUI** is GPL-3.0. SlideStream talks to it over HTTP as a separate
  program, which keeps the licences separate.

For any generated music, "trained on legally compliant data" is the model
author's claim, and the law on AI-generated output is still unsettled. The
licences above remove the obvious problems, not every possible one.

`doctor` should warn when a configured engine has a non-commercial licence.

### Python dependencies

Optional extras must stay on permissive licences (MIT, BSD, Apache), so
anyone can embed or redistribute SlideStream. That is why PDF pages are
rendered with pypdfium2 (Apache-2.0/BSD-3) and not PyMuPDF, which is AGPL-3.0.
Check the licence of any new dependency before adding it.

## Adding a service

1. Write an adapter for its role (ADR-0010): `is_configured()`, `probe()`,
   the role method, and `close()` if it leaves anything on the server.
2. Add mocked tests. Keep them hermetic (ADR-0007).
3. Add a live contract test under `tests/live/` and run it with `--run-live`.
4. Add a row here with the probe, the tested version and the licence.
5. Add a config example to `slidestream.example.yaml`.
6. If it is useful outside SlideStream and Puente can run it, propose it as a
   Puente service.
