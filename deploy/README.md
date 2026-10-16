# Self-hosting SlideStream (the web UI)

Run the `slide-stream serve` web UI on a VPS with Docker. Renders happen on the
box; users upload a deck (+ optional voice/photo) in the browser and download a
video.

## Quick start

The image is prebuilt and hosted on GitHub's registry, so you only need three
files — no repo clone, no local build:

```bash
mkdir slidestream-demo && cd slidestream-demo
base=https://raw.githubusercontent.com/michael-borck/slide-stream/main/deploy
curl -O "$base/docker-compose.yml"
curl -o .env             "$base/.env.example"          # set token, demo flag, keys
curl -o slidestream.yaml "$base/slidestream.yaml.example"   # choose providers

docker compose up -d          # pulls ghcr.io/michael-borck/slide-stream
```

The UI listens on `http://127.0.0.1:8080` on the Docker host. Put a reverse
proxy in front for HTTPS. For a containerized proxy, use a private Docker
network or restrict `BIND_HOST` to an interface reachable only by that proxy.
`restart: unless-stopped` keeps it running across crashes and reboots.

### Updating

```bash
docker compose pull && docker compose up -d
```

Pin a specific version with `SLIDESTREAM_IMAGE_TAG=2.16.0` in `.env` (image
tags match the PyPI versions). To build
locally instead of pulling, see the commented `build:` block in
`docker-compose.yml` (you'll also need the `Dockerfile`).

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
voice (Voicebox, Chatterbox), picture (SwarmUI) and presenter (ComfyUI
with wan-s2v or SadTalker) servers. See the
[providers reference](https://slidestream.borck.education/docs/PROVIDERS/).

## Reverse proxy and client IPs

The demo rate limit is 3 videos/hour per client IP. Forwarded headers are
ignored by default. Set `SLIDESTREAM_TRUSTED_PROXIES` to a comma-separated
list of proxy peer IPs or CIDRs as seen by the application, for example
`172.20.0.2/32`. Docker networking can change the address the app sees.
Only listed peers may supply the rightmost `X-Forwarded-For` entry. The proxy
must append the connecting client's address or overwrite the header.

The former `SLIDESTREAM_TRUSTED_PROXY=1` switch no longer enables trust.
Migrate it to the explicit address list. Keep the application port inaccessible
to public clients. When running through Uvicorn directly, pass
`--no-proxy-headers` so the app can validate the original socket peer;
`slide-stream serve` already does this.

The server bounds request bodies while receiving them, including chunked
uploads, before multipart parsing. The aggregate limit is the sum of the
deck, voice, and photo limits plus 1 MiB for form fields and multipart headers.
Set the reverse proxy's request-body limit to the same value, 76 MiB with the
defaults. Token-protected writes reject missing or invalid credentials before
reading their bodies.

## Notes

- The image bundles the cloud AI clients and the web UI (`[all-ai,serve]`).
  Without Kokoro the fallback voice is gTTS, which needs internet. For the
  offline voice add `local-tts` to the Dockerfile's extras and mount a
  volume at `/home/app/.cache/slide-stream` (the model is about 340 MB).
  Designed slides (`presenton`) are not in the image.
- Voice samples and photos are **used per render and deleted** — no biometric
  data is stored on the server. The deck and the per-job config are wiped when
  the render finishes; in demo mode the video is deleted right after download,
  and any leftovers are reaped after `SLIDESTREAM_JOB_TTL_MIN` (default 60)
  minutes from completion. Queued and running jobs keep their working files.
  The user's browser remembers voice/photo (IndexedDB) so they need
  not re-upload each job.
- Uploads are capped (30 MB deck, 30 MB voice, 15 MB photo) — override with
  `SLIDESTREAM_MAX_DECK_MB` / `SLIDESTREAM_MAX_VOICE_MB` /
  `SLIDESTREAM_MAX_PHOTO_MB`.
- Keep `--workers` low (default 1): renders are CPU/GPU heavy.
