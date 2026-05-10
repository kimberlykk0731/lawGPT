# Stage 5 — 工程消融 + 部署 Bench

精简到两组「能直接出图、面试讲得清楚」的消融 + vLLM 吞吐 bench。
PT 消融已下线（结论沉淀到顶层 README 的"不做的事"，不再单独跑）。

## 1. vLLM 吞吐 bench（核心数：6×）

```bash
python stages/stage5_engineering/benchmark_throughput.py \
  --models ckpts/legalgpt-8b-grpo ckpts/legalgpt-1.7b-distilled \
  --prompts_jsonl data/processed/rlvr_demo/prompts.jsonl \
  --n_prompts 1000 \
  --batch_size 64 \
  --max_tokens 256
```

输出末尾打印 "legalgpt-1.7b-distilled: 6.0x"。
首选在单卡 A10/L20 上跑，多卡反而难比；纯前向延迟 + 吞吐比，不要混进训练用的 H100。

## 2. Ablation 5.1 — 是否扩词表（设计稿，未实跑）

> 结论：法律领域 OOV 不严重，预期差距 < 0.5pt → **不扩**（顶层 README §10 已下线）。
> 这一节保留为设计稿，便于面试讲清楚"为什么不扩"。

详细方法（BPE 挖词 + embedding resize + 前 1k step 高 LR）见 [`ablations/vocab_extension.md`](ablations/vocab_extension.md)。

## 3. Ablation 5.2 — GRPO rollout: vLLM vs HF generate

> 期望速度 3×。这条用 swanlab 看 wall-clock 即可，不用单独跑大数据集。

```bash
# Arm A: 不用 vLLM（HF generate 路径，3 卡训）
deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_grpo.py \
  --model_path    ckpts/legalgpt-8b-sft \
  --dataset_path  data/processed/rlvr_demo \
  --output_dir    ckpts/grpo-arm-a \
  --use_vllm false \
  --num_train_epochs 1 \
  --swanlab_run_name stage5-grpo-arm-a-hf

# Arm B: 用 vLLM
deepspeed --num_gpus=3 stages/stage1_sft_grpo/train_grpo.py \
  --model_path    ckpts/legalgpt-8b-sft \
  --dataset_path  data/processed/rlvr_demo \
  --output_dir    ckpts/grpo-arm-b \
  --use_vllm --vllm_device cuda:3 \
  --num_train_epochs 1 \
  --swanlab_run_name stage5-grpo-arm-b-vllm
```

对比两个 swanlab run 的 wall-clock + tokens/s。详见 [`ablations/grpo_vllm_speedup.md`](ablations/grpo_vllm_speedup.md)。

## 4. swanlab 命名约定（消融 run）

```python
from swanlab.integration.wandb import wandb   # shim
wandb.init(
    project="legalgpt-2026",
    name=f"stage5-{ablation}-{arm}",        # e.g. stage5-vocab-ext-arm-b
    tags=["stage5", "ablation", ablation, arm],
    config=vars(args),
)
```

最终汇总指标：

| 指标 | 来源 |
|---|---|
| `final/crime_f1`, `final/contract_f1` | 训练完成后跑一遍 holdout 写入 |
| `compare/delta_vs_baseline` | Arm B - Arm A，简历那条数 |
| `sys/tokens_per_s`, `sys/speedup_vs_teacher` | 从 benchmark_throughput.py 输出 |
