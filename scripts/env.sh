#!/usr/bin/env bash
# source this in every shell / tmux pane that runs lawGPT scripts.
# rationale: server root disk (/) is full — we redirect every dotfile cache
# to /local_data/kzy so transformers/torch/triton/swanlab/pip can write.

# repo + python env
export LAWGPT_HOME=/local_data/kzy/lawGPT
source /local_data/kzy/miniconda3/etc/profile.d/conda.sh
conda activate /local_data/kzy/conda/envs/lawgpt

# CUDA toolkit
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Force every cache off the full root partition
export TMPDIR=/local_data/kzy/tmp
export XDG_CACHE_HOME=/local_data/kzy/.cache
export HF_HOME=/local_data/kzy/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=/local_data/kzy/.cache/torch
export TRITON_CACHE_DIR=/local_data/kzy/.cache/triton
export PIP_CACHE_DIR=/local_data/kzy/.cache/pip
export SWANLAB_SAVE_DIR=/local_data/kzy/.swanlab
export NUMBA_CACHE_DIR=/local_data/kzy/.cache/numba
export MPLCONFIGDIR=/local_data/kzy/.cache/matplotlib

mkdir -p \
  $TMPDIR $XDG_CACHE_HOME \
  $HF_HOME $HF_DATASETS_CACHE \
  $TORCH_HOME $TRITON_CACHE_DIR $PIP_CACHE_DIR \
  $SWANLAB_SAVE_DIR $NUMBA_CACHE_DIR $MPLCONFIGDIR

# Belt & suspenders: symlink the most stubborn dotdirs from $HOME to /local_data/kzy
for d in .swanlab .cache .triton; do
  link=$HOME/$d
  target=/local_data/kzy/$d
  mkdir -p "$target"
  if [ ! -e "$link" ] || [ -L "$link" ]; then
    ln -sfn "$target" "$link"
  fi
done

# HF mirror (China network)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# vLLM telemetry tries to write /home/kzy/.config (root disk full) — disable.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

# ZeRO-3 + GRPO sees memory fragmentation late in training (PyTorch's own
# advice on the OOM trace). expandable_segments=True turns on a chunked
# allocator that releases gaps back to the pool.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# pip mirrors
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_DEFAULT_TIMEOUT=120

# Repo conventions
export PYTHONPATH=$LAWGPT_HOME:${PYTHONPATH:-}
cd $LAWGPT_HOME 2>/dev/null || true

# Optional secrets file (untracked — put SWANLAB_API_KEY etc. here so it
# survives across panes/sessions). Create with:
#   cat > $LAWGPT_HOME/scripts/secrets.sh <<EOF
#   export SWANLAB_API_KEY=...
#   export SWANLAB_PROJECT=legalgpt-2026
#   EOF
#   chmod 600 $LAWGPT_HOME/scripts/secrets.sh
if [ -f "$LAWGPT_HOME/scripts/secrets.sh" ]; then
  # shellcheck source=/dev/null
  source "$LAWGPT_HOME/scripts/secrets.sh"
fi
