# 0012 — Three-level fallback; Kokoro is the voice floor

- Status: Accepted (2026-10). Supersedes 0004.
- Drivers: SlideStream is local-first. The old voice floor, gTTS, sends the
  narration text to Google and needs internet. Kokoro runs offline and is
  already installed by `slide-stream[all]`, so the desktop app has it, but
  SlideStream only used it when chosen by name. When a configured service
  failed, SlideStream dropped straight to the cloud floor and skipped the
  local engine sitting on the machine.
- Related: 0001 (fallback and strict mode), 0010 (adapters and `probe()`).

## Decision

**Every pipeline role falls back through up to three levels, in order:**

1. **Service**: the engine the user configured (Voicebox, ComfyUI,
   Presenton, a cloud API). Best quality.
2. **Local**: a built-in engine that runs on the machine with no server.
3. **Bare minimum**: needs nothing, so a render always finishes.

| Role | Local | Bare minimum |
|---|---|---|
| Voice | Kokoro (when its packages are installed) | gTTS |
| Pictures | Images from a configured folder | Text cards |
| Presenter | Static image or puppet mascot | None |
| Design | Render a styled `.pptx` (LibreOffice) or user PDF | Fixed layout |
| Narration text | None built in; Ollama counts as a service | Slide text as written |
| Music (planned) | A user-supplied track | None |

Strict mode is unchanged. It refuses to leave level 1 for a voice or engine
the user chose explicitly. A level the user never chose may still fall
further: when Kokoro stands in as the floor and fails, gTTS takes over even
in strict mode.

**The voice floor is implemented now:**

- `floor_tts_provider()` returns Kokoro when `kokoro_onnx` and `soundfile`
  are installed, otherwise gTTS. `settings.voice_floor: auto|kokoro|gtts`
  (or `SLIDE_STREAM_VOICE_FLOOR`) can force one.
- The factory uses it for an unconfigured default, for a configured provider
  that is unusable, and for an unknown provider name. Server providers use it
  for per-slide failures, and keep one floor engine per run so Kokoro loads
  its model once.
- New TTS provider value `auto` asks for the floor directly. The
  `slide-stream init` starter config writes it.
- As the floor, Kokoro ignores `providers.tts.voice` (it names another
  provider's voice) and uses `providers.tts.kokoro_voice`, default `af_sarah`.
- An explicit `provider: gtts` still means gTTS, so its accents still work.

The other roles follow this rule as their adapters are built (0010, roadmap
Phase 3). Today pictures and presenters drop straight to the bare minimum
when a service fails.

## Consequences

- Bare installs and the desktop app narrate offline, and the text stays on
  the machine.
- The first Kokoro render downloads about 340 MB of model files, which is
  slow and needs internet once. An offline first run falls to gTTS, and
  fails only if there is no internet at all.
- Kokoro loads its model per run (a few seconds), which is slower to start
  than gTTS for a one-slide test.
- The voice changes for existing users who relied on the unconfigured
  default with Kokoro installed. They can pin `provider: gtts` or
  `settings.voice_floor: gtts`.
- The mocked test suite pins the floor to gTTS (`tests/conftest.py`) so no
  test loads or downloads the model. Floor tests unpin it.
