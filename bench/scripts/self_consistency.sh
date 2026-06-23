#!/usr/bin/env bash
# Self-consistency gate: compare two qwen3_gguf_score dumps on the same token sequence.
#
#   bench/scripts/self_consistency.sh [--download | <model.gguf>]
#
# Default mode runs the score tool twice on the same build:
#   SPARKINFER_MMVQ=0  (bf16 dequant-GEMV)
#   SPARKINFER_MMVQ=1  (int8 dp4a MMVQ)
# and asserts greedy-identical output (argmax agreement = 1.0, self-KL ≈ 0).
#
# Pass two existing score files to compare arbitrary builds:
#   bench/scripts/self_consistency.sh /tmp/baseline.txt /tmp/candidate.txt
#
# Env overrides: MODELS_DIR, MODEL_FILE, ARCH (see _common.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

SCORE_A=""; SCORE_B=""
while [ $# -gt 0 ]; do case "$1" in
  --download) GGUF="$MODELS_DIR/$MODEL_FILE" ;;
  -h|--help)
    sed -n '2,14p' "$0"
    exit 0
    ;;
  *)
    if [ -z "$SCORE_A" ]; then SCORE_A="$1"
    elif [ -z "$SCORE_B" ]; then SCORE_B="$1"
    else echo "!! unexpected arg: $1"; exit 1
    fi
    ;;
esac; shift; done

if [ -n "$SCORE_A" ] && [ -n "$SCORE_B" ]; then
  echo "=== self-consistency: score dump comparison ==="
  python3 "$HERE/self_consistency.py" "$SCORE_A" "$SCORE_B"
  exit $?
fi

GGUF="${GGUF:-$MODELS_DIR/$MODEL_FILE}"
ARCH="$(detect_arch)"
resolve_runner "$ARCH"
[ "$GGUF" = "$MODELS_DIR/$MODEL_FILE" ] && ensure_model
ensure_tokenizer
[ -f "$GGUF" ] || { echo "!! GGUF not found: $GGUF"; exit 1; }

IDS="$(python3 -c "from tokenizers import Tokenizer; print(' '.join(map(str, Tokenizer.from_file('$MODELS_DIR/tokenizer.json').encode(open('$HERE/eval_text.txt').read().strip()).ids)))")"
echo ">> eval tokens: $(echo "$IDS" | wc -w)"

run_score() {
  local out="$1" mmvq="$2"
  SPARKINFER_MMVQ="$mmvq" si_run qwen3_gguf_score "$GGUF" 20 $IDS > "$out" 2>/dev/null || true
  if ! grep -q "^PPL" "$out"; then
    fallback_build "$ARCH"
    SPARKINFER_MMVQ="$mmvq" si_run qwen3_gguf_score "$GGUF" 20 $IDS > "$out" 2>/dev/null
  fi
}

echo ">> sparkinfer score (SPARKINFER_MMVQ=0) ..."
run_score /tmp/spark_score_mmvq0.txt 0
echo ">> sparkinfer score (SPARKINFER_MMVQ=1) ..."
run_score /tmp/spark_score_mmvq1.txt 1

echo
echo "=== self-consistency: MMVQ=0 vs MMVQ=1 ==="
out="$(python3 "$HERE/self_consistency.py" /tmp/spark_score_mmvq0.txt /tmp/spark_score_mmvq1.txt)"
echo "$out"

AGREE="$(printf '%s\n' "$out" | sed -n 's/.*METRIC agree=\([0-9.][0-9.]*\).*/\1/p' | head -1)"
SELFKL="$(printf '%s\n' "$out" | sed -n 's/.*selfkl=\([0-9.][0-9eE+-]*\).*/\1/p' | head -1)"
AGREE="${AGREE:-0}"; SELFKL="${SELFKL:-99}"

python3 - "$AGREE" "$SELFKL" <<'PY'
import sys
agree, selfkl = float(sys.argv[1]), float(sys.argv[2])
if agree < 0.99 or selfkl > 0.01:
    print(f"!! FAIL self-consistency gate (agree={agree:.4f}, selfkl={selfkl:.6f})")
    sys.exit(1)
print(f">> PASS self-consistency gate (agree={agree:.4f}, selfkl={selfkl:.6f})")
PY
