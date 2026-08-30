#!/usr/bin/env python3
"""Two-slide talking-mascot diagnostic: enrich -> create, full logs.

Run from the repo root (tokens must be in your shell):
    export OLLAMA_TOKEN=... CHATTERBOX_TOKEN=... COMFYUI_TOKEN=...
    uv run python scripts/mascot_video_test.py 2>&1 | tee mascot-test/run.log

Stage 1 enriches the deck (SwarmUI images + Ollama speaker notes).
Stage 2 creates a video from the ENRICHED deck with teddy lip-syncing
(wan-s2v via ComfyUI). Isolates: image gen / narration / TTS / avatar /
ffmpeg compositing and stitching — the log names each one.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DECK = (
    "# Coffee\n\n"
    "- It fuels developers worldwide\n"
    "- Espresso, filter, or cold brew\n\n"
    "# Tea\n\n"
    "- The quieter alternative\n"
    "- Matcha, oolong, earl grey\n"
)

CONFIG = """providers:
  llm:
    provider: ollama
    base_url: https://ollama.locollm.org
    api_key: "${OLLAMA_TOKEN}"
    model: gemma4:12b
    think: false
  tts:
    provider: voicebox
    base_url: https://voicebox.locoensayo.org
    api_key: "${CHATTERBOX_TOKEN}"
    voice: Emily.wav
  images:
    provider: swarmui
    base_url: https://swarmui.locopuente.org
    api_key: "${SWARMUI_TOKEN}"
    model: juggernautXL_v9
    timeout: 300            # first generation may load the model (cold start)
    fallback: text
  avatar:
    provider: wan-s2v
    base_url: https://comfyui.locopuente.org
    api_key: "${COMFYUI_TOKEN}"
    source: teddy          # bundled mascot — lip-syncs via wan-s2v
    clip_seconds: 2        # short looping clip = faster test
settings:
  strict: false
  narration:
    target_seconds: 20
"""


def run(label: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], text=True, env=env)
    if proc.returncode != 0:
        print(f"\n!! {label} FAILED (exit {proc.returncode}) — see output above")
        sys.exit(proc.returncode or 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="mascot-test")
    ap.add_argument("--skip-enrich", action="store_true",
                    help="reuse the previous run's enhanced slides")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not on PATH — stage 2 will fail.")
    for var in ("OLLAMA_TOKEN", "CHATTERBOX_TOKEN", "COMFYUI_TOKEN"):
        if not os.getenv(var):
            print(f"WARNING: {var} is not set in this shell.")

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    # Isolate the test from the user's global ~/.slidestream.yml (stale
    # voice_sample/provider settings there silently override the test config).
    child_home = work / "home"
    child_home.mkdir(exist_ok=True)
    child_env = dict(os.environ, HOME=str(child_home))
    deck = work / "test-deck.md"
    deck.write_text(DECK, encoding="utf-8")
    cfg = work / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")

    out_dir = work / "enhanced"
    if not args.skip_enrich:
        run("STAGE 1: enrich (SwarmUI images + Ollama speaker notes)", env=child_env, cmd=[
            sys.executable, "-m", "slide_stream", "enrich",
            deck, out_dir, "--config", cfg, "--pptx", "--notes", "all",
        ])
    enriched_md = out_dir / "test-deck.md"
    if not enriched_md.exists():
        print(f"!! {enriched_md} missing — enrich produced no deck")
        sys.exit(1)
    print(f"\nEnhanced deck: {enriched_md}")
    print(f"Slide images : {sorted(p.name for p in (out_dir / 'images').glob('*'))}")

    video = work / "mascot-video.mp4"
    run("STAGE 2: create (narrated video, teddy lip-syncing via wan-s2v)", env=child_env, cmd=[
        sys.executable, "-m", "slide_stream", "create",
        enriched_md, video, "--config", cfg,
    ])

    if video.exists():
        mb = video.stat().st_size / 1e6
        print(f"\n✅ Video written: {video} ({mb:.1f} MB)")
        print("Check: teddy talks bottom-left over the SwarmUI art, Emily's")
        print("voice narrates, and the two slides crossfade.")
    else:
        print("\n!! No video produced — see stage output above.")


if __name__ == "__main__":
    main()
