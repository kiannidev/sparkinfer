#!/usr/bin/env bash
# Basket accuracy gate: run the correctness harness for each SN74 basket model.
#
#   bench/scripts/accuracy_basket.sh [--download] [--text FILE]
#
# Today Qwen3-MoE runs end-to-end; Gemma 4 is skipped gracefully until
# gemma4_gguf_score is wired in the runtime (see dashboard "wiring" status).
#
# Env overrides: MODELS_DIR, ARCH, LLAMACPP_DIR (see _common.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

DOWNLOAD=0; EXTRA=()
while [ $# -gt 0 ]; do case "$1" in
  --download) DOWNLOAD=1 ;;
  --text)     EXTRA+=(--text); shift; EXTRA+=("$1") ;;
  -h|--help)
    sed -n '2,10p' "$0"
    exit 0
    ;;
  *) echo "!! unexpected arg: $1"; exit 1 ;;
esac; shift; done

ARCH="$(detect_arch)"
FAILED=0; SKIPPED=0; PASSED=0

run_preset() {
  local preset="$1"
  export MODEL_PRESET="$preset"
  apply_model_preset

  if [ ! -x "$ROOT/build/runtime/$SCORE_TOOL" ]; then
    resolve_runner "$ARCH" 2>/dev/null || true
  fi
  if [ ! -x "${SI_BIN:-$ROOT/build/runtime}/$SCORE_TOOL" ] && [ ! -x "$ROOT/build/runtime/$SCORE_TOOL" ]; then
    echo
    echo "=== SKIP $preset — $SCORE_TOOL not built (runtime wiring pending) ==="
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  echo
  echo "========================================"
  echo " basket: $preset  ($SCORE_TOOL)"
  echo "========================================"
  local args=()
  [ "$DOWNLOAD" = 1 ] && args+=(--download)
  args+=("${EXTRA[@]}")
  if "$HERE/accuracy.sh" --model "$preset" "${args[@]}"; then
    PASSED=$((PASSED + 1))
  else
    echo "!! FAIL basket accuracy: $preset"
    FAILED=$((FAILED + 1))
  fi
}

echo ">> basket accuracy gate (sm_$ARCH)"
run_preset qwen
run_preset gemma4

echo
echo "=== basket summary ==="
echo "passed : $PASSED"
echo "skipped: $SKIPPED"
echo "failed : $FAILED"

[ "$FAILED" -eq 0 ]
