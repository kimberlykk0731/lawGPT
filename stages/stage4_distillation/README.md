# Stage 4 — 蒸馏（Qwen3-8B → Qwen3-1.7B）

> 项目主菜：保留教师 90%+ 性能，vLLM 单 A10 吞吐 ×6。
> retain_pct 由 `stages/eval/run_eval.py --retain_baseline` 直接打印。

## 0. 输入产物

| 来自 | 产物 |
|---|---|
| Stage 1 SFT | `ckpts/legalgpt-8b-sft` |
| Stage 1 GRPO | **`ckpts/legalgpt-8b-grpo`**（教师 = 这个） |

## 1. Stage A — 黑盒 SFT 蒸馏（warmup）

教师 vLLM 离线生成 → 学生 SFT 打底。Demo 规模用 `distill_prompts.jsonl`（5k 条）。

```bash
# A.1 教师离线推理（A10 单卡 demo ~30min）
python stages/stage4_distillation/stage_a_blackbox_sft.py \
  --teacher_model ckpts/legalgpt-8b-grpo \
  --prompts_jsonl data/processed/distill_prompts.jsonl \
  --output_jsonl  data/distilled/teacher_completions.jsonl \
  --temperature 0.7 --top_p 0.9 --max_tokens 1024

# A.2 用 stage1 train_sft.py 在 Qwen3-1.7B 上 SFT
deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_sft.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --dataset_path  data/distilled/teacher_completions.jsonl \
  --output_dir    ckpts/student-warmup \
  --max_seq_length 4096 \
  --num_train_epochs 2 \
  --learning_rate 5e-5 \
  --bf16 --gradient_checkpointing \
  --swanlab_run_name stage4a-blackbox-warmup
```

## 2. Stage B — 离线 top-50 logits KL

```bash
# B.1 dump 教师 top-50 logits（demo ~30min；max_length=1024 不要再涨）
python stages/stage4_distillation/stage_b_logits_kl.py dump \
  --teacher_model ckpts/legalgpt-8b-grpo \
  --dataset_jsonl data/distilled/teacher_completions.jsonl \
  --cache_path    data/distilled/teacher_top50_cache.pt \
  --max_length 1024 --batch_size 4 --top_k 50

# B.2 学生加 CE+KL 训练
deepspeed --num_gpus=4 stages/stage4_distillation/stage_b_logits_kl.py train \
  --student_model ckpts/student-warmup \
  --cache_path    data/distilled/teacher_top50_cache.pt \
  --output_dir    ckpts/student-logits \
  --num_train_epochs 2 --learning_rate 2e-5 \
  --temperature 2.0 --alpha 0.3 \
  --logging_steps 10 \
  --swanlab_run_name stage4b-logits-kl
```

注意：CachedKLDataset 是把整个 cache `torch.load` 进 RAM。Demo 规模 5k × 1024 × top50
约 5GB，安全。简历规模（350k）必须改 chunked load，否则单机 RAM 装不下。

## 3. Stage C — On-policy KL

```bash
python stages/stage4_distillation/stage_c_onpolicy_kl.py \
  --teacher_model ckpts/legalgpt-8b-grpo \
  --student_model ckpts/student-logits \
  --prompts_jsonl data/processed/rlvr_demo/prompts.jsonl \
  --output_dir    ckpts/legalgpt-1.7b-distilled \
  --num_train_epochs 3 \
  --curriculum 128 256 512 \
  --learning_rate 1e-5 --temperature 1.0 \
  --logging_steps 10 \
  --swanlab_run_name stage4c-onpolicy-kl
```

## 4. 验证 90%+ 保留率（一行出数）

```bash
python stages/eval/run_eval.py \
  --models           ckpts/legalgpt-8b-grpo ckpts/legalgpt-1.7b-distilled \
  --eval_dataset     data/processed/rlvr_demo_test \
  --retain_baseline  ckpts/legalgpt-8b-grpo \
  --output           outputs/stage4_retain.json \
  --swanlab_run_name eval-stage4-retain
```

终端直接打印：

```
[eval] baseline = ckpts/legalgpt-8b-grpo: overall_mean = 0.7821
  ckpts/legalgpt-1.7b-distilled:
    overall_mean = 0.7234  Δ -5.87pt  retain = 92.49%
    crime_prediction_mean: retain = 93.12%
    contract_review_mean:  retain = 91.80%
```

`retain = 92.49%` 就是简历"保留教师 90%+ 性能"的来源。
swanlab 也写了 `compare/<student>/retain_pct` 可截图。

## 5. swanlab 监控

| Stage | 关键指标 | 看什么 |
|---|---|---|
| A | `train/loss`, `kd/teacher_match_top1` | warmup 完成度 |
| B | `loss/ce`, `loss/kl`, `kd/agreement_top1`, `kd/student_topk_prob_mass` | KL 主导 vs CE 主导平衡 |
| B | `kd/teacher_entropy`, `kd/student_entropy` | 学生熵应慢慢逼近教师 |
| C | `loss/forward_kl`, `kd/agreement_top1` | on-policy 收敛 |
| C | `rollout/length_mean`, `rollout/curriculum_max` | 课程是否生效（应看到三段台阶） |
| C | `rollout/teacher_logprob_mean` | 教师对学生的接受度（越高越好） |

详见 [`../SWANLAB_METRICS.md`](../SWANLAB_METRICS.md) Stage 4 节。

## 6. 工程要点

1. **教师推理**：vLLM 不直接吐 logits。Stage B 用 HF transformers + bf16 + flash_attn2，简单够快；想极致提速换 SGLang。
2. **显存预算**：H100 80GB 单卡可同时挂 Qwen3-8B 教师 (16GB) + Qwen3-1.7B 学生 (3.4GB) + Adam (~13GB)。4 卡 DDP 即可，不用 ZeRO-3。
3. **课程长度**：`--curriculum 128 256 512`，避免学生初期满屏垃圾让教师信号也学不到。
4. **温度**：B 用 T=2 软化；C 用 T=1，因为 on-policy 已经在采样了。
5. **CE shift**：stage B 的 compute_loss 已修：`student_logits[:, :-1]` 预测 `labels[:, 1:]`，
   不要回退老逻辑（老的把 last position 也算进 CE 等于学了下一句的开头是 EOS）。
