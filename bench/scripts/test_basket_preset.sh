#!/usr/bin/env bash
# Smoke test for basket model presets in _common.sh (no GPU required).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_preset() {
  local preset="$1" score="$2" bench="$3"
  local got
  got="$(MODEL_PRESET="$preset" bash -c "source '$HERE/_common.sh' && printf '%s:%s' \"\$SCORE_TOOL\" \"\$BENCH_TOOL\"")"
  local got_score="${got%%:*}" got_bench="${got#*:}"
  [ "$got_score" = "$score" ] || { echo "!! $preset: expected SCORE_TOOL=$score got $got_score"; exit 1; }
  [ "$got_bench" = "$bench" ] || { echo "!! $preset: expected BENCH_TOOL=$bench got $got_bench"; exit 1; }
  echo "ok preset=$preset score=$got_score bench=$got_bench"
}

check_preset qwen   qwen3_gguf_score   qwen3_gguf_bench
check_preset gemma4 gemma4_gguf_score   gemma4_gguf_bench
echo ">> all basket preset checks passed"
