#!/usr/bin/env bash
# Stage-1 GRPO launcher (trl 0.16+ server-mode vLLM).
#
# Two processes, one tmux session:
#   1) `trl vllm-serve` on cuda:3 — exposes /generate and /health on $VLLM_PORT
#   2) `accelerate launch` GRPOTrainer on cuda:0,1,2 — connects to that server
#
# Usage:
#   bash scripts/run_grpo.sh [extra args forwarded to train_grpo.py]
#
# Env knobs (all optional; defaults match the README story):
#   MODEL_PATH=/path/to/sft/ckpt        # default: existing SFT checkpoint-1068
#   DATASET_PATH=data/processed/...     # default: rlvr_demo_filtered (DAPO-filtered)
#   OUTPUT_DIR=/path/to/output          # default: ckpts/legalgpt-8b-grpo
#   VLLM_PORT=8000                      # rollout server port
#   VLLM_GPU=cuda:3                     # rollout-server GPU (last of CVD)
#   TRAIN_GPUS=0,1,2                    # comma-separated training GPUs
#   NUM_GENERATIONS=8                   # rollouts per prompt (must divide eff_batch)
#   SMOKE=1                             # cap to --max_steps 2 for a quick test
#
# The script forwards any positional args to train_grpo.py — `--max_steps 2`,
# `--num_train_epochs`, etc. SwanLab: `export SWANLAB_API_KEY=...` before running,
# or `export SWANLAB_MODE=local` for offline.

set -euo pipefail

LAWGPT_HOME=${LAWGPT_HOME:-/local_data/kzy/lawGPT}
source "$LAWGPT_HOME/scripts/env.sh"

MODEL_PATH=${MODEL_PATH:-/local_data/kzy/ckpt/ckpts/legalgpt-8b-sft/checkpoint-1068}
DATASET_PATH=${DATASET_PATH:-data/processed/rlvr_demo_filtered}
OUTPUT_DIR=${OUTPUT_DIR:-/local_data/kzy/ckpt/ckpts/legalgpt-8b-grpo}
VLLM_PORT=${VLLM_PORT:-8000}
# 8×A800 layout: 7 train procs + 1 vLLM rollout server.
# Effective batch = num_train_procs × PER_DEVICE_BATCH × GRAD_ACCUM = 7 × 2 × 4 = 56,
# divisible by NUM_GENERATIONS=8 (trl 0.17+ requires this). Override via env vars.
VLLM_GPU=${VLLM_GPU:-cuda:7}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_COMPLETION_LEN=${MAX_COMPLETION_LEN:-1024}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.7}

# Visible GPUs = train + rollout. trl checks num_processes < device_count.
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS,${VLLM_GPU#cuda:}"
NUM_TRAIN_PROCS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)

if [[ ! -d "$DATASET_PATH" && ! -d "$LAWGPT_HOME/$DATASET_PATH" ]]; then
  echo "[run_grpo] dataset_path '$DATASET_PATH' not found." >&2
  echo "  Run filter_dynamic_sampling first, or set DATASET_PATH=data/processed/rlvr_demo." >&2
  exit 2
fi

# Smoke-test mode: cap to a couple of optimisation steps from CLI overrides.
SMOKE=${SMOKE:-0}
EXTRA_TRAIN_ARGS=("$@")
if [[ "$SMOKE" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--max_steps 2)
fi

# --- 1) start vllm-serve on $VLLM_GPU --------------------------------------------------
mkdir -p "$LAWGPT_HOME/logs"
SERVE_LOG="$LAWGPT_HOME/logs/vllm_serve.log"
echo "[run_grpo] starting trl vllm-serve on $VLLM_GPU :$VLLM_PORT (model=$MODEL_PATH)"

# Run vllm-serve in its own session via setsid so we can kill the whole
# process group (trl spawns multiprocessing workers that survive a plain
# `kill $parent`). `--fork` keeps it as our child for waitpid.
setsid bash -c '
  export CUDA_VISIBLE_DEVICES="'"${VLLM_GPU#cuda:}"'"
  exec trl vllm-serve \
    --model "'"$MODEL_PATH"'" \
    --host 127.0.0.1 \
    --port "'"$VLLM_PORT"'" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "'"$VLLM_GPU_MEM_UTIL"'" \
    --max_model_len "'"$VLLM_MAX_MODEL_LEN"'" \
    --dtype bfloat16
' > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
trap '
  echo "[run_grpo] killing vllm-serve pgid=$SERVE_PID (and descendants)"
  # Negative pid signals the whole process group started by setsid
  kill -9 -$SERVE_PID 2>/dev/null || true
  # Belt-and-suspenders: any orphaned multiprocessing.spawn workers (PPID=1)
  pkill -9 -f "from multiprocessing.spawn import spawn_main" 2>/dev/null || true
  pkill -9 -f "trl vllm-serve" 2>/dev/null || true
' EXIT

# Wait for /health to come up
echo -n "[run_grpo] waiting for vllm-serve /health ..."
for i in $(seq 1 120); do
  if curl -fs --max-time 2 "http://127.0.0.1:$VLLM_PORT/health/" > /dev/null 2>&1; then
    echo " up after ${i}s"; break
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo " FAILED — vllm-serve died early. tail:"
    tail -40 "$SERVE_LOG" >&2
    exit 1
  fi
  sleep 5
done
curl -fs --max-time 2 "http://127.0.0.1:$VLLM_PORT/health/" > /dev/null 2>&1 \
  || { echo " TIMED OUT after 600s. tail:"; tail -40 "$SERVE_LOG" >&2; exit 1; }

# --- 2) accelerate launch GRPOTrainer --------------------------------------------------
echo "[run_grpo] launching accelerate trainer ($NUM_TRAIN_PROCS train procs on $TRAIN_GPUS)"
cd "$LAWGPT_HOME"
accelerate launch --num_processes="$NUM_TRAIN_PROCS" \
  --config_file stages/stage1_sft_grpo/configs/accelerate_zero3.yaml \
  stages/stage1_sft_grpo/train_grpo.py \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --use_vllm \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "$VLLM_PORT" \
  --num_generations "$NUM_GENERATIONS" --beta 0.04 \
  --epsilon 0.2 --epsilon_high 0.28 \
  --max_prompt_length "$MAX_PROMPT_LEN" --max_completion_length "$MAX_COMPLETION_LEN" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
  --logging_steps 5 --save_steps 200 \
  --swanlab_run_name "${SWANLAB_RUN_NAME:-stage1-grpo-v1}" \
  "${EXTRA_TRAIN_ARGS[@]}"

echo "[run_grpo] training finished, shutting down vllm-serve"
