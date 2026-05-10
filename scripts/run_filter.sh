#!/usr/bin/env bash
# DAPO Dynamic Sampling — pre-filter rlvr_demo before GRPO.
# Runs vllm rollout (8B SFT model on cuda:0) over the rlvr_demo dataset,
# computes RLVR reward spread per prompt, drops prompts with spread < threshold.
#
# Usage: bash scripts/run_filter.sh
#
# Env knobs:
#   MODEL_PATH=/path/to/sft/ckpt
#   DATASET_PATH=data/processed/rlvr_demo
#   OUTPUT_PATH=data/processed/rlvr_demo_filtered
#   N_ROLLOUTS=8
#   MIN_SPREAD=0.1
#   FILTER_GPU=0    # rollout GPU index
set -euo pipefail

LAWGPT_HOME=${LAWGPT_HOME:-/local_data/kzy/lawGPT}
source "$LAWGPT_HOME/scripts/env.sh"

MODEL_PATH=${MODEL_PATH:-/local_data/kzy/ckpt/ckpts/legalgpt-8b-sft/checkpoint-1068}
DATASET_PATH=${DATASET_PATH:-data/processed/rlvr_demo}
OUTPUT_PATH=${OUTPUT_PATH:-data/processed/rlvr_demo_filtered}
N_ROLLOUTS=${N_ROLLOUTS:-8}
MIN_SPREAD=${MIN_SPREAD:-0.1}
FILTER_GPU=${FILTER_GPU:-0}

cd "$LAWGPT_HOME"
mkdir -p outputs

echo "[run_filter] model=$MODEL_PATH"
echo "[run_filter] dataset=$DATASET_PATH -> $OUTPUT_PATH"
echo "[run_filter] N=$N_ROLLOUTS, min_spread=$MIN_SPREAD, GPU=$FILTER_GPU"

CUDA_VISIBLE_DEVICES="$FILTER_GPU" python stages/stage1_sft_grpo/data/filter_dynamic_sampling.py \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --output_path "$OUTPUT_PATH" \
  --n_rollouts "$N_ROLLOUTS" \
  --min_spread "$MIN_SPREAD" \
  --report_path outputs/dyn_sampling_report.json
