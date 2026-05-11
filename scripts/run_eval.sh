#!/usr/bin/env bash
# Stage-1 final eval: SFT / GRPO / DPO three-way comparison on rlvr_demo_test.
# Also usable as the generic eval driver — pass extra --models / --retain_baseline
# via env vars when calling for stage 4 retain_pct.
#
# Env knobs:
#   EVAL_GPU=4
#   MODELS="ckpts/legalgpt-8b-sft/checkpoint-1068 ckpts/legalgpt-8b-grpo ckpts/legalgpt-8b-dpo"
#   EVAL_DATASET=data/processed/rlvr_demo_test
#   OUT=outputs/stage1_eval.json
#   RETAIN_BASELINE=""              # leave empty for stage 1; set for stage 4
#   SWANLAB_RUN_NAME=eval-stage1-compare
set -euo pipefail

LAWGPT_HOME=${LAWGPT_HOME:-/local_data/kzy/lawGPT}
source "$LAWGPT_HOME/scripts/env.sh"

EVAL_GPU=${EVAL_GPU:-4}
MODELS=${MODELS:-"/local_data/kzy/ckpt/ckpts/legalgpt-8b-sft/checkpoint-1068 /local_data/kzy/ckpt/ckpts/legalgpt-8b-grpo /local_data/kzy/ckpt/ckpts/legalgpt-8b-dpo"}
EVAL_DATASET=${EVAL_DATASET:-data/processed/rlvr_demo_test}
OUT=${OUT:-outputs/stage1_eval.json}
SWANLAB_RUN_NAME=${SWANLAB_RUN_NAME:-eval-stage1-compare}

cd "$LAWGPT_HOME"
mkdir -p outputs

retain_arg=()
if [[ -n "${RETAIN_BASELINE:-}" ]]; then
  retain_arg=(--retain_baseline "$RETAIN_BASELINE")
fi

echo "[run_eval] gpu=cuda:$EVAL_GPU"
echo "[run_eval] models: $MODELS"
echo "[run_eval] dataset: $EVAL_DATASET -> $OUT"

CUDA_VISIBLE_DEVICES="$EVAL_GPU" python stages/eval/run_eval.py \
  --models $MODELS \
  --eval_dataset "$EVAL_DATASET" \
  --output "$OUT" \
  "${retain_arg[@]}" \
  --swanlab_run_name "$SWANLAB_RUN_NAME"
