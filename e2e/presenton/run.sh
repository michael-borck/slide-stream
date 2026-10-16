#!/usr/bin/env bash
# Phase 1 Presenton trial (docs/ROADMAP.md). Needs a running Presenton
# v0.9.11-beta on localhost:5001 and PRESENTON_API_KEY exported; see
# e2e/presenton/README.md.
#
# Usage:  e2e/presenton/run.sh [lecture|creative|styled|baseline|all]   (default: all)
#         TEMPLATE_LECTURE=modern TEMPLATE_CREATIVE=editorial e2e/presenton/run.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
HERE="e2e/presenton"
OUT="e2e/out/presenton"
CONFIG="$HERE/config.yaml"
LECTURE_TEMPLATE="${TEMPLATE_LECTURE:-general}"
CREATIVE_TEMPLATE="${TEMPLATE_CREATIVE:-editorial}"
mkdir -p "$OUT"

failures=0

step() {
  local name="$1"
  shift
  echo ""
  echo "━━━ presenton trial: $name ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  local start=$SECONDS
  if "$@" 2>&1 | tee "$OUT/$name.log"; then
    echo "✅ $name ($((SECONDS - start)) s)" | tee -a "$OUT/timings.txt"
  else
    echo "❌ $name failed after $((SECONDS - start)) s (log: $OUT/$name.log)" \
      | tee -a "$OUT/timings.txt"
    failures=$((failures + 1))
  fi
}

need_presenton() {
  if [[ -z "${PRESENTON_API_KEY:-}" ]]; then
    echo "PRESENTON_API_KEY is not set. Create a key in Presenton's Admin panel."
    exit 2
  fi
  if ! curl -s -o /dev/null -m 5 http://localhost:5001/; then
    echo "Presenton is not answering on http://localhost:5001. Start it first."
    exit 2
  fi
}

lecture() {
  need_presenton
  step "lecture-deck" uv run slide-stream presenton "$HERE/lecture.md" \
    "$OUT/lecture.pptx" --template "$LECTURE_TEMPLATE" --config "$CONFIG" --force
  step "lecture-video" uv run slide-stream create "$HERE/lecture.md" \
    "$OUT/lecture.mp4" --template "$LECTURE_TEMPLATE" \
    --script "$HERE/lecture_script.txt" --config "$CONFIG"
}

creative() {
  need_presenton
  step "creative-video" uv run slide-stream create "$HERE/creative.md" \
    "$OUT/creative.mp4" --template "$CREATIVE_TEMPLATE" \
    --script "$HERE/creative_script.txt" --config "$CONFIG"
}

baseline() {
  # The same decks in SlideStream's own plain layout, for side-by-side
  # comparison with the designed versions. No Presenton needed.
  step "lecture-basic" uv run slide-stream create "$HERE/lecture.md" \
    "$OUT/lecture-basic.mp4" --script "$HERE/lecture_script.txt" --config "$CONFIG"
  step "creative-basic" uv run slide-stream create "$HERE/creative.md" \
    "$OUT/creative-basic.mp4" --script "$HERE/creative_script.txt" --config "$CONFIG"
}

styled() {
  # No Presenton involved: LibreOffice renders the deck's own design.
  if ! command -v soffice >/dev/null && ! command -v libreoffice >/dev/null; then
    echo "LibreOffice not found. Install it: brew install --cask libreoffice"
    failures=$((failures + 1))
    return
  fi
  [[ -f "$HERE/styled.pptx" ]] || uv run python "$HERE/make_styled_deck.py"
  step "styled-video" uv run slide-stream create "$HERE/styled.pptx" \
    "$OUT/styled.mp4" --preserve-style --verbatim-notes --config "$CONFIG"
}

echo "run started $(date '+%Y-%m-%d %H:%M')" >> "$OUT/timings.txt"
case "${1:-all}" in
  lecture) lecture ;;
  creative) creative ;;
  styled) styled ;;
  baseline) baseline ;;
  all) lecture; creative; styled; baseline ;;
  *) echo "Unknown deck: $1 (use lecture|creative|styled|baseline|all)"; exit 2 ;;
esac

echo ""
if [[ $failures -eq 0 ]]; then
  echo "All steps passed. Outputs in $OUT/. Fill in the record in $HERE/README.md."
else
  echo "$failures step(s) failed. Logs are in $OUT/."
fi
exit $((failures > 0))
