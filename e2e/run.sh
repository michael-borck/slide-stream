#!/usr/bin/env bash
# SlideStream end-to-end render matrix against the LAN AI stack.
#
# Each scenario exercises the full per-slide pipeline (parse -> narrate ->
# artwork -> output) with a different output mode, layered on top of the
# personal stack in ~/.slidestream.yml (LLM/TTS/images) and the tokens in
# .env (uv run loads it automatically from the repo root).
#
# Usage:  e2e/run.sh [narration-only|static-teddy|pptx|panda|fast|all]
#         fast = narration-only + static-teddy + pptx (default)
#         panda takes ~10 min (one wan-s2v render on the GPU box).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="e2e/out"
mkdir -p "$OUT"

run_one() {
  local name="$1" deck="$2" config="$3" output="$4"
  shift 4
  echo ""
  echo "━━━ e2e: $name ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if uv run slide-stream create \
      "e2e/decks/$deck" "$OUT/$output" \
      --config "e2e/scenarios/$config" "$@" 2>&1 | tee "$OUT/$name.log"; then
    echo "✅ $name -> $OUT/$output"
    return 0
  else
    echo "❌ $name failed (log: $OUT/$name.log)"
    return 1
  fi
}

failures=0

case "${1:-fast}" in
  narration-only)
    run_one narration-only single_slide.md narration_only.yaml narration_only.mp4 ;;
  static-teddy)
    run_one static-teddy single_slide.md static_teddy.yaml static_teddy.mp4 --avatar ;;
  pptx)
    run_one pptx two_slides.md pptx_export.yaml slides_and_notes.pptx ;;
  panda)
    run_one panda two_slides.md two_slide_panda.yaml two_slide_panda.mp4 --avatar ;;
  fast)
    run_one narration-only single_slide.md narration_only.yaml narration_only.mp4 || failures=$((failures+1))
    run_one static-teddy single_slide.md static_teddy.yaml static_teddy.mp4 --avatar || failures=$((failures+1))
    run_one pptx two_slides.md pptx_export.yaml slides_and_notes.pptx || failures=$((failures+1))
    ;;
  all)
    run_one narration-only single_slide.md narration_only.yaml narration_only.mp4 || failures=$((failures+1))
    run_one static-teddy single_slide.md static_teddy.yaml static_teddy.mp4 --avatar || failures=$((failures+1))
    run_one pptx two_slides.md pptx_export.yaml slides_and_notes.pptx || failures=$((failures+1))
    run_one panda two_slides.md two_slide_panda.yaml two_slide_panda.mp4 --avatar || failures=$((failures+1))
    ;;
  *)
    echo "Unknown scenario: $1 (use narration-only|static-teddy|pptx|panda|fast|all)"
    exit 2
    ;;
esac

echo ""
if [ "$failures" -eq 0 ]; then
  echo "🎉 e2e suite passed. Outputs in $OUT/"
else
  echo "⚠ $failures scenario(s) failed. See logs in $OUT/"
  exit 1
fi
