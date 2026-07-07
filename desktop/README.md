# SlideStream Desktop

A native desktop app (macOS / Linux / Windows) for SlideStream: slides in,
narrated video out — no terminal required.

## How it works

A thin [Tauri](https://tauri.app) shell around the Python engine:

1. On launch it finds the [`uv`](https://docs.astral.sh/uv/) Python manager
   (offering a one-click install if missing).
2. It starts `slide-stream serve` in **local mode** on a free `127.0.0.1`
   port via `uvx` — the first run downloads the engine from PyPI (a few
   minutes); afterwards it starts in seconds from uv's cache.
3. The window then *is* the SlideStream web UI, plus a ⚙ **Settings** page
   (local mode only) that edits `~/.slidestream.yaml` — your providers,
   servers and keys, with a sane-defaults template.
4. Closing the window shuts the server down (`POST /api/quit`).

Everything runs on your machine. No account, no telemetry; the only network
calls are to the AI providers *you* configure (and PyPI for the engine).

Requires `slide-stream >= 2.9` (local mode). `ffmpeg` is recommended for
voice cloning and avatars (`brew install ffmpeg` / distro package / ffmpeg.org).

## Building

CI builds installers for all three platforms:
**Actions → "Build desktop apps" → Run workflow** (artifacts), or push a
`desktop-v*` tag to attach them to a GitHub release. Builds are unsigned for
now — on macOS use right-click → Open on first launch.

Locally:

```bash
cd desktop
npx --yes @tauri-apps/cli@^2 dev     # run in dev
npx --yes @tauri-apps/cli@^2 build   # produce the installer for this OS
```

On an exFAT working copy (external drives), point Cargo at an internal disk
first: `export CARGO_TARGET_DIR=~/.cache/slidestream-target` (AppleDouble
`._*` files inside `target/` break the Tauri build script otherwise).

## Layout

```
desktop/
  ui/index.html        launcher (bootstrap progress, uv install, log)
  src-tauri/           Tauri (Rust) shell: find uv, spawn serve, clean quit
  app-icon.png         source icon (npx tauri icon regenerates the sets)
```
