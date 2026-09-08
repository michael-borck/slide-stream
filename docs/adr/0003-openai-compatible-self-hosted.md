# 0003 — OpenAI-compatible APIs as the default self-hosted shape

- Status: Accepted (backfilled 2026-09)
- Drivers: self-hosters run Ollama, LocalAI, vLLM, speaches, Kokoro-FastAPI,
  SwarmUI, ComfyUI — each behind different APIs

## Decision

Where a service family speaks an OpenAI shape, SlideStream speaks it and lets
`base_url` select the backend — one code path, no per-vendor SDKs:

- LLM: `ollama` and `openai-compatible` clients (`llm.py`), incl. bearer
  tokens for authenticated proxies and `think: false` for reasoning models,
  with `<think>` blocks stripped from every provider's output.
- TTS: `openai-compatible` `/v1/audio/speech` (works with speaches, LocalAI,
  openedai-speech, Kokoro-FastAPI).
- Images: `openai-compatible` `/v1/images/generations`, handling both URL and
  inline `b64_json` responses.

Services that do NOT speak OpenAI shapes get a native provider instead:
SwarmUI's GetNewSession → GenerateText2Image flow (`providers/images.py`),
Voicebox's profiles/generate/status API and ComfyUI's workflow API
(`providers/avatar.py`) — including Voicebox's SSE-framed status responses
(`_parse_status_body`), which older code assumed were plain JSON.

## Consequences

- New self-hosted services usually need zero code (config only).
- Server-side API drift (like Voicebox's SSE status) surfaces as confusing
  client-side parse errors; the fix is tolerant parsing + tests pinned to the
  observed wire format.
- `openai-compatible` requires an explicit `base_url` — an OpenAI key alone
  must never silently route to the real OpenAI API.
