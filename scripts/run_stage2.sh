#!/usr/bin/env bash
# Stage-2 router behaviour analysis on Qwen3-30B-A3B (forward only, no training).
# 60 GB bf16 model on 1×80 GB GPU; ~30-60 min on the demo legal_eval_by_domain set.
#
# Env knobs:
#   MODEL_PATH=/local_data/kzy/models/Qwen3-30B-A3B
#   EVAL_JSONL=data/processed/legal_eval_by_domain.jsonl
#   OUTPUT_JSON=outputs/router_variance.json
#   STAGE2_GPU=0
set -euo pipefail

LAWGPT_HOME=${LAWGPT_HOME:-/local_data/kzy/lawGPT}
source "$LAWGPT_HOME/scripts/env.sh"

MODEL_PATH=${MODEL_PATH:-/local_data/kzy/models/Qwen3-30B-A3B}
EVAL_JSONL=${EVAL_JSONL:-data/processed/legal_eval_by_domain.jsonl}
OUTPUT_JSON=${OUTPUT_JSON:-outputs/router_variance.json}
STAGE2_GPU=${STAGE2_GPU:-0}

cd "$LAWGPT_HOME"

echo "[run_stage2] analyzing $MODEL_PATH on cuda:$STAGE2_GPU"
CUDA_VISIBLE_DEVICES="$STAGE2_GPU" python stages/stage2_moe_router/analyze_router.py \
  --model "$MODEL_PATH" \
  --eval_jsonl "$EVAL_JSONL" \
  --top_k 8 --num_experts 128 \
  --output_json "$OUTPUT_JSON" \
  --swanlab_run_name "${SWANLAB_RUN_NAME:-stage2-analyze-v1}"
