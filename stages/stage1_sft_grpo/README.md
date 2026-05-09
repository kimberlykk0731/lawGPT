# Stage 1 — Qwen3-8B SFT + GRPO 对齐

围绕"罪名预测可验证任务"的端到端 SFT→GRPO 链路。GRPO vs DPO 是项目目标里
**+Δpt** 那条数据。

## RL 算法选型

主线 **GRPO**（trl 工具链最稳，输出短结构化 JSON 不需要 GSPO 的长序列优化），
另外 cherry-pick **DAPO** 的两个补丁——这两个改动都不需要重写 trainer，对当前任务最划算：

| 补丁 | 目的 | 实现位置 |
|---|---|---|
| **Clip-Higher** (`epsilon_low=0.2, epsilon_high=0.28`) | 上下不对称 PPO clip，防止 policy 过快崩塌 | `train_grpo.py` 直接传 `--epsilon`/`--epsilon_high` 给 trl |
| **Dynamic Sampling**（离线版） | 过滤 reward spread < 阈值的 prompt（GRPO 上 advantage ≈ 0 浪费算力） | `data/filter_dynamic_sampling.py` 用 SFT model + vLLM 预跑 N 次 rollout 评估 |

不用 **GSPO**：sequence-level importance ratio 主要解决长 CoT (>1k token) 和 MoE
训练的不稳定性，本任务输出短（JSON 数组几十~几百 token）+ dense 8B 模型，优势打不出来。

不用全套 **DAPO**：Token-level loss / Overlong reward shaping 主要为长 reasoning 设计，
工程复杂度高于收益。

## 0. 环境准备

```bash
export SWANLAB_PROJECT=legalgpt-2026
export SWANLAB_API_KEY=<your-key>
export PYTHONPATH=$PWD:$PYTHONPATH
```

## 1. 数据准备

数据全套已经由顶层 `stages/data_prep.py` 一键产出。如果只想看老入口（手工方式）：

- `data/build_sft_350k.py` — 旧 SFT 入口，单独使用需要 `--cail-raw` + `--external-instruction-jsonl`
- `data/build_rlvr_dataset.py` — 旧 RLVR 入口，需要预先准备 `clean_cail.jsonl` + `contract_clauses.jsonl`

主链路推荐直接用 `data_prep.py`。

```bash
# Dynamic Sampling 离线过滤（在 SFT 完成后跑，预算 ~2-3 GPU·h）
python stages/stage1_sft_grpo/data/filter_dynamic_sampling.py \
  --model_path     ckpts/legalgpt-8b-sft \
  --dataset_path   data/processed/rlvr_demo \
  --output_path    data/processed/rlvr_demo_filtered \
  --n_rollouts     8 \
  --min_spread     0.1 \
  --report_path    outputs/dyn_sampling_report.json
```

## 2. 训练

```bash
# 2.1) 全参 SFT（4× H100, demo ~3-5h）
deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_sft.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_path  data/processed/sft_demo \
  --output_dir    ckpts/legalgpt-8b-sft \
  --max_seq_length 4096 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 --warmup_ratio 0.03 \
  --bf16 --gradient_checkpointing \
  --deepspeed stages/stage1_sft_grpo/configs/ds_zero3_bf16.json \
  --swanlab_run_name stage1-sft-v1

# 2.2) GRPO（3 卡训 + 1 卡 vLLM rollout, demo ~3-4h；含 DAPO Clip-Higher）
deepspeed --num_gpus=3 stages/stage1_sft_grpo/train_grpo.py \
  --model_path    ckpts/legalgpt-8b-sft \
  --dataset_path  data/processed/rlvr_demo_filtered \
  --output_dir    ckpts/legalgpt-8b-grpo \
  --use_vllm --vllm_device cuda:3 \
  --num_generations 8 --beta 0.04 \
  --epsilon 0.2 --epsilon_high 0.28 \
  --num_train_epochs 4 \
  --swanlab_run_name stage1-grpo-v1

# 2.3) DPO baseline（先生成偏好对，再训练）
python stages/stage1_sft_grpo/train_dpo.py --build-preferences \
  --model_path    ckpts/legalgpt-8b-sft \
  --rlvr_dataset  data/processed/rlvr_demo \
  --preference_jsonl data/processed/preferences.jsonl

deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_dpo.py \
  --model_path        ckpts/legalgpt-8b-sft \
  --preference_jsonl  data/processed/preferences.jsonl \
  --output_dir        ckpts/legalgpt-8b-dpo \
  --beta 0.1 --learning_rate 5e-7 \
  --swanlab_run_name    stage1-dpo-baseline
```

## 3. 评估（GRPO vs DPO 对比）

```bash
python stages/eval/run_eval.py \
  --models ckpts/legalgpt-8b-sft ckpts/legalgpt-8b-grpo ckpts/legalgpt-8b-dpo \
  --eval_dataset    data/processed/rlvr_demo_test \
  --output          outputs/stage1_eval.json \
  --swanlab_run_name eval-stage1-compare
```

终端会直接打印每模型 overall_mean、Δpt、retain_pct。简历那条 "+Δpt" 就在
`stage1_eval.json` 的 `comparison[].delta_pt`。

## 4. swanlab 监控

参见 [`../SWANLAB_METRICS.md`](../SWANLAB_METRICS.md) 的 Stage 1 节。GRPO run 关键指标：

| 指标 | 来源 | 看什么 |
|---|---|---|
| `actor/pg_loss`, `actor/kl_loss`, `actor/entropy_loss` | trl 默认 | 策略/KL/熵 三大件 |
| `critic/rewards/mean`, `critic/rewards/std` | trl 默认 | std 太小说明 group 退化（Dynamic Sampling 已经过滤过一轮） |
| `legal/crime_f1`, `legal/contract_f1` | callback | 任务级 F1（最终指标的训练时代理） |
| `legal/parse_failure_rate` | callback | 应快速降到 < 5% |
| `critic/format_reward/frac` | callback | 偏离 ~50% 警示 reward hacking |
| `response_length/mean`, `response_length/clip_ratio` | trl 默认 | clip > 30% 要扩 max_completion_length |
| `actor/clip_frac_high`, `actor/clip_frac_low` | trl 默认 | Clip-Higher 后 high 应明显高于 low（说明上界放开起作用了） |

## 5. 产物

| 路径 | 用途 |
|---|---|
| `ckpts/legalgpt-8b-sft` | Stage 4 蒸馏起点（学生 warmup 用同一份 SFT 数据训 1.7B） |
| `ckpts/legalgpt-8b-grpo` | **教师**：Stage 4 黑盒 / logits / on-policy 蒸馏全用它 |
| `ckpts/legalgpt-8b-dpo` | 仅用于 +Δpt 对照数 |
