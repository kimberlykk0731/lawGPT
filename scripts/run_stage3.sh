#!/usr/bin/env bash
# Stage 3 — Dense → MoE upcycle + smoke + domain fine-tune + smoke.
#
# Four phases, run together by default:
#   upcycle    mergekit-moe writes /local_data/kzy/ckpt/ckpts/qwen3-1.7b-moe-8e
#              (CPU, no GPU; needs ~30 GB free disk for the merged shards)
#   smoke      analyze the freshly upcycled router on 1 GPU (~1 min)
#   sft        domain fine-tune on the demo SFT corpus, ZeRO-3 on TRAIN_GPUS
#              (1.7B is small — 1 epoch on demo is fast)
#   smoke_after  router re-analysis on the fine-tuned model
#
# Usage:
#   bash scripts/run_stage3.sh                          # all four phases
#   bash scripts/run_stage3.sh upcycle smoke            # subset
#   bash scripts/run_stage3.sh sft -- --resume_from_checkpoint auto
#                                                       # forward extra flags
#                                                       # to train_sft.py via `--`
#
# Env knobs:
#   STAGE3_BASE=/local_data/kzy/models/Qwen3-1.7B
#   STAGE3_MOE=/local_data/kzy/ckpt/ckpts/qwen3-1.7b-moe-8e
#   STAGE3_MOE_SFT=/local_data/kzy/ckpt/ckpts/qwen3-1.7b-moe-8e-sft
#   SMOKE_GPU=5
#   TRAIN_GPUS=5,6,7
set -euo pipefail

LAWGPT_HOME=${LAWGPT_HOME:-/local_data/kzy/lawGPT}
source "$LAWGPT_HOME/scripts/env.sh"

STAGE3_BASE=${STAGE3_BASE:-/local_data/kzy/models/Qwen3-1.7B}
STAGE3_MOE=${STAGE3_MOE:-/local_data/kzy/ckpt/ckpts/qwen3-1.7b-moe-8e}
STAGE3_MOE_SFT=${STAGE3_MOE_SFT:-/local_data/kzy/ckpt/ckpts/qwen3-1.7b-moe-8e-sft}
SMOKE_GPU=${SMOKE_GPU:-5}
TRAIN_GPUS=${TRAIN_GPUS:-5,6,7}
UPCYCLE_YAML=${UPCYCLE_YAML:-stages/stage3_dense_to_moe/upcycle_qwen3_1_7b.yaml}

cd "$LAWGPT_HOME"

run_upcycle() {
  echo "[stage3] upcycle Qwen3-1.7B -> $STAGE3_MOE via mergekit-moe (CPU)"
  # Patch the YAML's `base_model` path to our local cache so mergekit doesn't
  # try to hit HuggingFace. We write to a temp file rather than the tracked
  # YAML so the repo file stays Qwen/Qwen3-1.7B-canonical.
  tmpyaml=$(mktemp)
  sed "s|Qwen/Qwen3-1\\.7B|$STAGE3_BASE|g" "$UPCYCLE_YAML" > "$tmpyaml"
  bash stages/stage3_dense_to_moe/run_upcycle.sh "$tmpyaml" "$STAGE3_MOE"
  rm -f "$tmpyaml"
}

run_smoke() {
  local model=$1 run=$2
  echo "[stage3] smoke $run on cuda:$SMOKE_GPU"
  CUDA_VISIBLE_DEVICES="$SMOKE_GPU" python stages/stage3_dense_to_moe/smoke_test.py \
    --model "$model" \
    --top_k 2 --num_experts 8 \
    --swanlab_run_name "$run"
}

run_sft() {
  local NUM_TRAIN_PROCS
  NUM_TRAIN_PROCS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
  echo "[stage3] domain SFT on cuda:$TRAIN_GPUS ($NUM_TRAIN_PROCS procs)"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" accelerate launch \
    --num_processes="$NUM_TRAIN_PROCS" \
    --config_file stages/stage1_sft_grpo/configs/accelerate_zero3.yaml \
    stages/stage1_sft_grpo/train_sft.py \
    --model_name_or_path "$STAGE3_MOE" \
    --dataset_path data/processed/sft_demo \
    --output_dir "$STAGE3_MOE_SFT" \
    --max_seq_length 2048 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --bf16 --gradient_checkpointing \
    --save_steps 500 \
    --swanlab_run_name "${SWANLAB_RUN_NAME:-stage3-moe-sft-v1}" \
    "${EXTRA_ARGS[@]}"
}

# Split positional args into phase names and `--`-suffixed extra args
# forwarded to the inner trainer (e.g. --resume_from_checkpoint auto).
phases=()
EXTRA_ARGS=()
seen_sep=0
for a in "$@"; do
  if [[ "$a" == "--" ]]; then seen_sep=1; continue; fi
  if [[ $seen_sep -eq 1 ]]; then EXTRA_ARGS+=("$a"); else phases+=("$a"); fi
done
if [[ ${#phases[@]} -eq 0 ]]; then
  phases=(upcycle smoke sft smoke_after)
fi
for phase in "${phases[@]}"; do
  case "$phase" in
    upcycle)     run_upcycle ;;
    smoke)       run_smoke "$STAGE3_MOE" stage3-smoke-test ;;
    sft)         run_sft ;;
    smoke_after) run_smoke "$STAGE3_MOE_SFT" stage3-smoke-test-after ;;
    *) echo "unknown phase: $phase" >&2; exit 2 ;;
  esac
done
