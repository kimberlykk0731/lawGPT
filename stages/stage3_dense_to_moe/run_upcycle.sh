#!/usr/bin/env bash
# Dense→MoE upcycle via mergekit
set -euo pipefail

CONFIG="${1:-stages/stage3_dense_to_moe/upcycle_qwen3_1_7b.yaml}"
OUT="${2:-ckpts/qwen3-1.7b-moe-8e}"

if ! command -v mergekit-moe >/dev/null 2>&1; then
  echo "mergekit-moe not installed. pip install mergekit" >&2
  exit 1
fi

mergekit-moe "$CONFIG" "$OUT" \
  --copy-tokenizer \
  --allow-crimes \
  --out-shard-size 5B \
  --lazy-unpickle

echo "[upcycle] done -> $OUT"
echo "Next: run smoke_test.py to verify the model loads and routes properly."
