#!/usr/bin/env bash
# Automatic evaluation of a sparkinfer build: build → correctness → speed → label.
# Runs ON a GPU box (the vast orchestrator clones the repo + invokes this). Emits a JSON
# verdict as the last stdout line:  RESULT_JSON {...}
#
#   bench/scripts/evaluate.sh [--ref GIT_REF] [--baseline-ref GIT_REF]
#                           [--frontier TPS] [--ceiling TPS] [--gguf PATH]
#
# correctness = token-match / KL vs llama.cpp (accuracy.sh)
#             + optional score-vs-baseline gate (--baseline-ref, ~100% top-1 + self-KL≈0)
# speed = median of 3 bench runs · label = significance gate + headroom bucket (label.py).
# Source-built (NO_PREBUILT) so the measured artifact is the submitted code.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/_common.sh"

REF=""; BASELINE_REF=""; FRONTIER=0; CEILING=0; GGUF=""
while [ $# -gt 0 ]; do case "$1" in
  --ref) shift; REF="$1" ;;
  --baseline-ref) shift; BASELINE_REF="$1" ;;
  --frontier) shift; FRONTIER="$1" ;;
  --ceiling) shift; CEILING="$1" ;;
  --gguf) shift; GGUF="$1" ;;
  *) ;;
esac; shift; done
[ -z "$GGUF" ] && GGUF="$MODELS_DIR/$MODEL_FILE"
export LLAMACPP_DIR="${LLAMACPP_DIR:-/workspace/.llamacpp}"   # persist llama.cpp across evals
ARCH="$(detect_arch)"
TEXT="$HERE/eval_text.txt"

score_token_ids() {
  python3 -c "from tokenizers import Tokenizer; print(' '.join(map(str, Tokenizer.from_file('$MODELS_DIR/tokenizer.json').encode(open('$TEXT').read().strip()).ids)))"
}

run_score_dump() {
  local out="$1"
  local ids; ids="$(score_token_ids)"
  si_run qwen3_gguf_score "$GGUF" 20 $ids > "$out" 2>/dev/null || true
  if ! grep -q "^PPL" "$out"; then
    fallback_build "$ARCH"
    si_run qwen3_gguf_score "$GGUF" 20 $ids > "$out" 2>/dev/null
  fi
}

BASELINE_AGREE=""; BASELINE_SELFKL=""
if [ -n "$BASELINE_REF" ]; then
  SUBMISSION_REF="$(git -C "$ROOT" rev-parse HEAD)"
  echo ">> [0/4] baseline score ($BASELINE_REF) from source (sm_$ARCH) ..." >&2
  git -C "$ROOT" fetch -q origin "$BASELINE_REF" 2>/dev/null || true
  git -C "$ROOT" checkout -q "$BASELINE_REF"
  BASELINE_COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
  rm -rf "$ROOT/build"
  NO_PREBUILT=1 ensure_sparkinfer "$ARCH"
  ensure_tokenizer
  run_score_dump /tmp/baseline_score.txt
  git -C "$ROOT" checkout -q "$SUBMISSION_REF"
  echo ">> baseline commit: $BASELINE_COMMIT" >&2
fi

if [ -n "$REF" ]; then git -C "$ROOT" fetch -q origin "$REF" 2>/dev/null || true; git -C "$ROOT" checkout -q "$REF"; fi
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"

echo ">> [1/4] build submission ($COMMIT) from source (sm_$ARCH) ..." >&2
rm -rf "$ROOT/build"; NO_PREBUILT=1 ensure_sparkinfer "$ARCH"

echo ">> [2/4] speed — median of 3 bench runs ..." >&2
ts=()
for _ in 1 2 3; do
  t=$(si_run qwen3_gguf_bench "$GGUF" 128 2>/dev/null | sed -n 's/.*decode tg *: *\([0-9.][0-9.]*\).*/\1/p')
  ts+=("${t:-0}")
done
TPS=$(printf '%s\n' "${ts[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')

echo ">> [3/4] correctness — token-match / KL vs llama.cpp ..." >&2
acc=$("$HERE/accuracy.sh" "$GGUF" 2>/dev/null || true)
# parse the unambiguous METRIC line (not the human-readable text, which contains "bar >= 0.90")
TOP1=$(printf '%s\n' "$acc" | sed -n 's/.*METRIC .*top1=\([0-9.][0-9.]*\).*/\1/p' | head -1)
KL=$(printf   '%s\n' "$acc" | sed -n 's/.*METRIC .*kl=\([0-9.][0-9.]*\).*/\1/p' | head -1)
TOP1="${TOP1:-0}"; KL="${KL:-99}"

if [ -n "$BASELINE_REF" ]; then
  echo ">> [4/4] baseline gate — score vs $BASELINE_REF ($BASELINE_COMMIT) ..." >&2
  cp /tmp/spark_score.txt /tmp/submission_score.txt 2>/dev/null || run_score_dump /tmp/submission_score.txt
  sc_out="$(python3 "$HERE/self_consistency.py" /tmp/baseline_score.txt /tmp/submission_score.txt)"
  echo "$sc_out" >&2
  BASELINE_AGREE=$(printf '%s\n' "$sc_out" | sed -n 's/.*METRIC agree=\([0-9.][0-9.]*\).*/\1/p' | head -1)
  BASELINE_SELFKL=$(printf '%s\n' "$sc_out" | sed -n 's/.*selfkl=\([0-9.][0-9eE+-]*\).*/\1/p' | head -1)
  BASELINE_AGREE="${BASELINE_AGREE:-0}"; BASELINE_SELFKL="${BASELINE_SELFKL:-99}"
else
  echo ">> [4/4] label (no --baseline-ref; skip score-vs-baseline gate) ..." >&2
fi

if [ -n "$BASELINE_AGREE" ]; then
  python3 "$HERE/label.py" "$TPS" "$FRONTIER" "$CEILING" "$TOP1" "$KL" "$COMMIT" \
    "$BASELINE_AGREE" "$BASELINE_SELFKL" "$BASELINE_COMMIT"
else
  python3 "$HERE/label.py" "$TPS" "$FRONTIER" "$CEILING" "$TOP1" "$KL" "$COMMIT"
fi
