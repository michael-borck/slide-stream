# Self-hosting SlideStream (the web UI)

Run the `slide-stream serve` web UI on a VPS with Docker. Renders happen on the
box; users upload a deck (+ optional voice/photo) in the browser and download a
video.

## Quick start

```bash
cd deploy
cp .env.example .env                 # set SLIDESTREAM_TOKEN, keys, demo flag
cp slidestream.yaml.example slidestream.yaml   # choose your providers
docker compose up -d --build
```

The UI is now on `http://<host>:8080`. Put a reverse proxy (nginx / Caddy /
Nginx Proxy Manager) in front for TLS on `app.slidestream.eduserver.au`.

`restart: unless-stopped` keeps it running across crashes and reboots.

## How config works

- **`.env`** — the token, the demo banner flag, the port, and provider API
  keys (injected as environment variables).
- **`slidestream.yaml`** — mounted read-only as the server's layered config
  (`~/.slidestream.yaml`). It selects the providers and references secrets as
  `${VAR}`, so no plaintext keys live in the file. Each render job layers its
  own per-request bits (uploaded voice sample, photo) on top.

## Demo vs full

`SLIDESTREAM_DEMO=true` shows a banner in the UI: *"Hosted demo — install
locally for full control over voice, image and video generation."* Set it
`false` for a private/full instance where you've wired up your own
Chatterbox / SwarmUI / SadTalker servers.

## Notes

- The image bundles the cloud AI clients (`[all-ai,serve]`). For offline TTS
  (Kokoro) add `[local-tts]` to the Dockerfile — it pulls ~340MB of models.
- Voice samples and photos are **used per render and deleted** — no biometric
  data is stored on the server. The user's browser remembers them (IndexedDB)
  so they need not re-upload each job.
- Keep `--workers` low (default 1): renders are CPU/GPU heavy.
