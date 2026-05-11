#!/usr/bin/env bash
# Stage-1 DPO baseline (against the GRPO checkpoint).
# Two phases:
#   bash scripts/run_dpo.sh build     # vllm-rollout the SFT model, score with
#                                     # legal_reward_fn, emit (chosen, rejected) pairs
#   bash scripts/run_dpo.sh train     # accelerate launch DPOTrainer on those pairs
#   bash scripts/run_dpo.sh           # build then train (default)
#
# Env knobs (all optional):
#   SFT_PATH=/path/to/sft/ckpt
#   PREF_JSONL=data/processed/preferences.jsonl
#   OUTPUT_DIR=ckpts/legalgpt-8b-dpo
#   BUILD_GPU=1            # single GPU for vllm rollout in build phase
#   TRAIN_GPUS=1,2,3       # comma-separated GPUs for DPO training
#   N_PER_PROMPT=4         # rollouts per RLVR prompt
set -euo pipefail

LAWGPT_HOME=${LAWGPT_HOME:-/local_data/kzy/lawGPT}
source "$LAWGPT_HOME/scripts/env.sh"

SFT_PATH=${SFT_PATH:-/local_data/kzy/ckpt/ckpts/legalgpt-8b-sft/checkpoint-1068}
PREF_JSONL=${PREF_JSONL:-data/processed/preferences.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/local_data/kzy/ckpt/ckpts/legalgpt-8b-dpo}
BUILD_GPU=${BUILD_GPU:-1}
TRAIN_GPUS=${TRAIN_GPUS:-1,2,3}
N_PER_PROMPT=${N_PER_PROMPT:-4}
BETA=${BETA:-0.1}
LR=${LR:-5e-7}

cd "$LAWGPT_HOME"
phase=${1:-all}

run_build() {
  echo "[run_dpo] build preferences on cuda:$BUILD_GPU"
  CUDA_VISIBLE_DEVICES="$BUILD_GPU" python stages/stage1_sft_grpo/train_dpo.py \
    --build-preferences \
    --model_path "$SFT_PATH" \
    --rlvr_dataset data/processed/rlvr_demo \
    --preference_jsonl "$PREF_JSONL" \
    --n_per_prompt "$N_PER_PROMPT"
}

run_train() {
  NUM_TRAIN_PROCS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
  echo "[run_dpo] training on cuda:$TRAIN_GPUS ($NUM_TRAIN_PROCS procs)"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" accelerate launch \
    --num_processes="$NUM_TRAIN_PROCS" \
    --config_file stages/stage1_sft_grpo/configs/accelerate_zero3.yaml \
    stages/stage1_sft_grpo/train_dpo.py \
    --model_path "$SFT_PATH" \
    --preference_jsonl "$PREF_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --beta "$BETA" --learning_rate "$LR" \
    --swanlab_run_name "${SWANLAB_RUN_NAME:-stage1-dpo-baseline}"
}

case "$phase" in
  build) run_build ;;
  train) run_train ;;
  all)   run_build && run_train ;;
  *) echo "usage: $0 [build|train|all]" >&2; exit 2 ;;
esac
