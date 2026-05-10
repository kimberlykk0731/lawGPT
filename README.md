# LegalGPT — 法律垂域全链路实验（2026）

围绕 Qwen3 全家桶完整复现 **Dense → MoE 分析 → MoE 自造 → 蒸馏部署** 的工业落地链路。

> 个人项目 / 4×H100 / Demo 规模 ~10 GPU·h，简历规模 ~150 GPU·h
> 数据源：HuggingFace `ShengbinYue/DISC-Law-SFT` · Reward：RLVR (JSON 集合 F1) · 跟踪：SwanLab · 训练：trl + DeepSpeed ZeRO-3

---

## 0. 一分钟跑通（demo 规模，能在 swanlab 出图）

```bash
# 准备代码 + 装依赖
git clone <repo> && cd lawGPT
pip install -e ".[vllm,deepspeed]"
pip install swanlab mergekit flash-attn

export SWANLAB_PROJECT=legalgpt-2026
export SWANLAB_API_KEY=<your-key>           # 或 export SWANLAB_MODE=local
export PYTHONPATH=$PWD:$PYTHONPATH

# 全部数据一键准备（HF 下载 + 多任务模板合成 + 长尾重采样 + 划分 train/test）
python stages/data_prep.py --out data
# → data/processed/sft_demo/ data/processed/rlvr_demo/ ... 全套就绪
```

后面按 §4 的 stage 命令依次跑即可。如果想跑简历规模（350k SFT / 20k RLVR），改成
`python stages/data_prep.py --full --out data`。

---

## 1. 项目目标（5 个 Stage）

| Stage | 内容 | 关键产物 / 数字 |
|---|---|---|
| 1 | Qwen3-8B 全参 SFT + GRPO 对齐（含 DAPO Clip-Higher + Dynamic Sampling），DPO 作为对照 | **GRPO 较 DPO +Δpt** on 罪名预测 |
| 2 | Qwen3-30B-A3B 路由行为剖析（128 expert / top-8）— 仅做 forward 分析，不做训练 | 5 域 × 各层路由方差 + top-5 expert 表 |
| 3 | mergekit-moe 把 Qwen3-1.7B FFN 复制 8 份验证 Dense→MoE 低成本路径 | 8-expert top-2 MoE + 路由收敛曲线 |
| 4 | 黑盒 + top-50 logits KL + on-policy 三段蒸馏到 Qwen3-1.7B | **保留教师 90%+ 性能** |
| 5 | 工程消融：扩词表 / GRPO vLLM rollout 加速 + 部署 bench | vLLM **吞吐 ×6** |

详细文档：每个 stage 自带 README，串起来在 [`stages/README.md`](stages/README.md)。

---

## 2. 仓库结构

```
stages/                              # 唯一的源码根（PYTHONPATH=.）
├── data_prep.py                     # ★ 一键数据准备（HF DISC-Law-SFT → 全套产物）
├── README.md                        # 5-stage 总览 + 数据流向图
├── SWANLAB_METRICS.md               # 全链路指标 schema
├── rewards/rlvr.py                  # 跨 stage 共享的 RLVR reward + metrics buffer
├── eval/run_eval.py                 # 跨 stage 通用评估（含 retain_pct 直出）
├── stage1_sft_grpo/                 # SFT + GRPO + DPO baseline
│   ├── configs/ds_zero3_bf16.json
│   ├── data/filter_dynamic_sampling.py    # DAPO Dynamic Sampling 离线过滤
│   ├── train_sft.py
│   ├── train_grpo.py                # 含 SwanLabLegalMetricsCallback + Clip-Higher
│   └── train_dpo.py                 # --build-preferences + 训练两 mode
├── stage2_moe_router/               # 30B-A3B 路由分析（forward only）
│   ├── analyze_router.py            # ★ 直接 swanlab log 各层 var / overlap
│   ├── domain_aware_aux_loss.py
│   └── train_with_aux.py            # 参考实现，主链路不跑
├── stage3_dense_to_moe/             # mergekit-moe Dense→MoE
│   ├── upcycle_qwen3_1_7b.yaml
│   ├── run_upcycle.sh
│   └── smoke_test.py                # ★ 上传 expert 占比到 swanlab
├── stage4_distillation/             # 三段蒸馏
│   ├── stage_a_blackbox_sft.py
│   ├── stage_b_logits_kl.py         # `dump` / `train` 两个 sub-command
│   └── stage_c_onpolicy_kl.py
└── stage5_engineering/              # bench + 两组消融
    ├── benchmark_throughput.py
    └── ablations/{vocab_extension,grpo_vllm_speedup}.md
```

---

## 3. 环境

```bash
# Python ≥ 3.10
pip install -e ".[vllm,deepspeed]"
pip install "swanlab" "mergekit" "flash-attn"

# 跟踪
export SWANLAB_PROJECT=legalgpt-2026
export SWANLAB_API_KEY=<your-key>
export SWANLAB_WORKSPACE=<your-org>          # 可选
# export SWANLAB_MODE=local                  # 离线模式：本地 dashboard 看图

# 让 stages 成为可 import 的顶层 package（必须！）
export PYTHONPATH=$PWD:$PYTHONPATH
```

硬件预算：**4× H100 80GB** 主线；Demo 规模在 4×A100-40G 上也能跑（学生侧 + 蒸馏没问题；
30B-A3B 分析建议 80G 卡或 device_map=auto + 8-bit）。

---

## 4. 端到端命令清单（demo 规模）

按依赖顺序排。Stage 2、3 是平行支线，不阻塞主链路。

### 4.0 数据准备（一次性，~5 min）

```bash
python stages/data_prep.py --out data
# 全部输出在 data/processed/{sft_demo, rlvr_demo, rlvr_demo_test, ...}/
```

### 4.1 Stage 1 · SFT + GRPO（主线）

```bash
# SFT (4× H100, demo ~3-5h)
deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_sft.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_path  data/processed/sft_demo \
  --output_dir    ckpts/legalgpt-8b-sft \
  --max_seq_length 4096 --num_train_epochs 3 \
  --learning_rate 2e-5 --warmup_ratio 0.03 \
  --bf16 --gradient_checkpointing \
  --deepspeed stages/stage1_sft_grpo/configs/ds_zero3_bf16.json \
  --swanlab_run_name stage1-sft-v1

# DAPO Dynamic Sampling 离线过滤
python stages/stage1_sft_grpo/data/filter_dynamic_sampling.py \
  --model_path     ckpts/legalgpt-8b-sft \
  --dataset_path   data/processed/rlvr_demo \
  --output_path    data/processed/rlvr_demo_filtered \
  --n_rollouts 8 --min_spread 0.1 \
  --report_path    outputs/dyn_sampling_report.json

# GRPO（3 卡训 + 1 卡 vLLM rollout, demo ~3-4h）
deepspeed --num_gpus=3 stages/stage1_sft_grpo/train_grpo.py \
  --model_path    ckpts/legalgpt-8b-sft \
  --dataset_path  data/processed/rlvr_demo_filtered \
  --output_dir    ckpts/legalgpt-8b-grpo \
  --use_vllm --vllm_device cuda:3 \
  --num_generations 8 --beta 0.04 \
  --epsilon 0.2 --epsilon_high 0.28 \
  --num_train_epochs 4 \
  --swanlab_run_name stage1-grpo-v1

# DPO baseline：先生成偏好对，再训练
python stages/stage1_sft_grpo/train_dpo.py --build-preferences \
  --model_path    ckpts/legalgpt-8b-sft \
  --rlvr_dataset  data/processed/rlvr_demo \
  --preference_jsonl data/processed/preferences.jsonl

deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_dpo.py \
  --model_path        ckpts/legalgpt-8b-sft \
  --preference_jsonl  data/processed/preferences.jsonl \
  --output_dir        ckpts/legalgpt-8b-dpo \
  --beta 0.1 --learning_rate 5e-7 \
  --swanlab_run_name  stage1-dpo-baseline

# 评估 GRPO vs DPO（产出 +Δpt 的那条数）
python stages/eval/run_eval.py \
  --models       ckpts/legalgpt-8b-sft ckpts/legalgpt-8b-grpo ckpts/legalgpt-8b-dpo \
  --eval_dataset data/processed/rlvr_demo_test \
  --output       outputs/stage1_eval.json \
  --swanlab_run_name eval-stage1-compare
```

### 4.2 Stage 2 · MoE Router 行为剖析（forward only，平行支线）

```bash
python stages/stage2_moe_router/analyze_router.py \
  --model       Qwen/Qwen3-30B-A3B \
  --eval_jsonl  data/processed/legal_eval_by_domain.jsonl \
  --top_k 8 --num_experts 128 \
  --output_json outputs/router_variance.json \
  --swanlab_run_name stage2-analyze-v1
```

跑完 swanlab 上能直接看到 `moe/expert_var/L*/{domain}` 折线 + `moe/cross_domain_overlap/L*`
+ `moe/top1_freq/L*/{domain}`。简历"刑事 token 在 expert {x, y, z} 上聚集"那条故事就是
读这个 JSON 写出来的。

### 4.3 Stage 3 · Dense→MoE Upcycle（平行支线）

```bash
# 1) mergekit-moe 离线 upcycle
./stages/stage3_dense_to_moe/run_upcycle.sh \
  stages/stage3_dense_to_moe/upcycle_qwen3_1_7b.yaml \
  ckpts/qwen3-1.7b-moe-8e

# 2) 冒烟测试（路由是否散开） — swanlab log expert 占比
python stages/stage3_dense_to_moe/smoke_test.py \
  --model ckpts/qwen3-1.7b-moe-8e \
  --swanlab_run_name stage3-smoke-test

# 3) 领域语料 fine-tune — 复用 stage1 SFT 入口
deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_sft.py \
  --model_name_or_path ckpts/qwen3-1.7b-moe-8e \
  --dataset_path       data/processed/sft_demo \
  --output_dir         ckpts/qwen3-1.7b-moe-8e-sft \
  --num_train_epochs 1 --learning_rate 1e-5 \
  --bf16 --gradient_checkpointing \
  --swanlab_run_name stage3-moe-sft-v1

# 4) fine-tune 后再跑一次 smoke test 对比
python stages/stage3_dense_to_moe/smoke_test.py \
  --model ckpts/qwen3-1.7b-moe-8e-sft \
  --swanlab_run_name stage3-smoke-test-after
```

### 4.4 Stage 4 · 蒸馏（依赖 Stage 1 GRPO 产物）

```bash
# Stage A: 教师离线生成
python stages/stage4_distillation/stage_a_blackbox_sft.py \
  --teacher_model ckpts/legalgpt-8b-grpo \
  --prompts_jsonl data/processed/distill_prompts.jsonl \
  --output_jsonl  data/distilled/teacher_completions.jsonl \
  --temperature 0.7 --top_p 0.9 --max_tokens 1024

# 学生在教师文本上 SFT (warmup)
deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_sft.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --dataset_path  data/distilled/teacher_completions.jsonl \
  --output_dir    ckpts/student-warmup \
  --num_train_epochs 2 --learning_rate 5e-5 \
  --bf16 --gradient_checkpointing \
  --swanlab_run_name stage4a-blackbox-warmup

# Stage B: 离线 top-50 logits KL
python stages/stage4_distillation/stage_b_logits_kl.py dump \
  --teacher_model ckpts/legalgpt-8b-grpo \
  --dataset_jsonl data/distilled/teacher_completions.jsonl \
  --cache_path    data/distilled/teacher_top50_cache.pt \
  --max_length 1024 --batch_size 4 --top_k 50

deepspeed --num_gpus=4 stages/stage4_distillation/stage_b_logits_kl.py train \
  --student_model ckpts/student-warmup \
  --cache_path    data/distilled/teacher_top50_cache.pt \
  --output_dir    ckpts/student-logits \
  --num_train_epochs 2 --learning_rate 2e-5 \
  --temperature 2.0 --alpha 0.3 \
  --swanlab_run_name stage4b-logits-kl

# Stage C: On-policy KL
python stages/stage4_distillation/stage_c_onpolicy_kl.py \
  --teacher_model ckpts/legalgpt-8b-grpo \
  --student_model ckpts/student-logits \
  --prompts_jsonl data/processed/rlvr_demo/prompts.jsonl \
  --output_dir    ckpts/legalgpt-1.7b-distilled \
  --num_train_epochs 3 --curriculum 128 256 512 \
  --learning_rate 1e-5 --temperature 1.0 \
  --swanlab_run_name stage4c-onpolicy-kl

# 验证 90%+ 保留率（直接打印 retain_pct）
python stages/eval/run_eval.py \
  --models           ckpts/legalgpt-8b-grpo ckpts/legalgpt-1.7b-distilled \
  --eval_dataset     data/processed/rlvr_demo_test \
  --retain_baseline  ckpts/legalgpt-8b-grpo \
  --output           outputs/stage4_retain.json \
  --swanlab_run_name eval-stage4-retain
```

终端会直接打印类似：

```
[eval] baseline = ckpts/legalgpt-8b-grpo: overall_mean = 0.7821
  ckpts/legalgpt-1.7b-distilled:
    overall_mean = 0.7234  Δ -5.87pt  retain = 92.49%
    crime_prediction_mean: retain = 93.12%
    contract_review_mean:  retain = 91.80%
```

那个 `retain = 92.49%` 就是简历里"保留教师 90%+ 性能"的来源。

### 4.5 Stage 5 · 工程消融 + 部署 Bench

```bash
python stages/stage5_engineering/benchmark_throughput.py \
  --models        ckpts/legalgpt-8b-grpo ckpts/legalgpt-1.7b-distilled \
  --prompts_jsonl data/processed/rlvr_demo/prompts.jsonl \
  --n_prompts 1000 --batch_size 64 --max_tokens 256
```

详见 [`stages/stage5_engineering/README.md`](stages/stage5_engineering/README.md)。

---

## 5. 训练 / 评测 / MoE 注意事项（踩过的坑）

### 通用

- **PYTHONPATH 必设**。所有命令默认从 repo 根目录起，`stages` 是顶层 package，
  忘了设会 `ImportError: stages.rewards.rlvr`。
- **swanlab API key 提前装好**。崩到一半 swanlab 写不进去比 OOM 还难排查；要么有 key，
  要么 `export SWANLAB_MODE=local` 走本地 dashboard。
- **不要 amend 已 push 的 checkpoint 目录**。HF Trainer 中断恢复要 `resume_from_checkpoint=True`，
  覆盖会丢 optimizer state。
- **只在 demo 规模下用 `padding="max_length"`**。Stage 4B/C 在简历规模会暴 padding token，
  浪费 30%+ 算力，要换 `pad_to_multiple_of=8` + dynamic padding。

### Stage 1 · SFT/GRPO/DPO

- **GRPO 起点必须是 SFT ckpt**，不是 base model。GRPO 假设 policy 已经能产出基本格式正确的
  JSON，否则 reward 噪声直接淹没信号（`legal/parse_failure_rate` 起步 90%+ 是这个症状）。
- **vLLM rollout 卡和训练卡不同卡**：`--use_vllm --vllm_device cuda:3` 配 `--num_gpus=3`，
  让 0/1/2 训练，3 跑 rollout。同卡会争显存。
- **Clip-Higher 验证**：`actor/clip_frac_high` 应明显高于 `actor/clip_frac_low`，否则
  `--epsilon_high 0.28` 没起作用，回去看 trl 版本是否 ≥ 0.13。
- **Dynamic Sampling 必跑**：`min_spread=0.1` 通常过滤掉 15-30%，能让 GRPO 训练快 1.3-1.5×。
- **DPO 偏好对生成**：`train_dpo.py --build-preferences` 是单独 step，先跑这个再训。
  preference pair 数量 ≈ RLVR 数量 × 70%（spread 不够的会被丢）。

### Stage 2 · MoE Router 分析

- **30B-A3B 显存**：bf16 60GB，单 H100 80G 装下；如显存紧 `--device_map auto` + 部分层
  offload to CPU。我们只 forward，不需要梯度。
- **pad token mask**：`analyze_router.py` 已经按 attention_mask 过滤；自己改要保留这步，
  不然 padding token 会被算进 expert 激活，把分布拉平。
- **Stage 2 不训练**：30B 全参 ZeRO-3 在 4×H100 上极易 OOM；`train_with_aux.py` 是参考
  实现，主链路不跑。简历的"激活率 0.xx vs 0.yy"故事直接读 `outputs/router_variance.json`
  写。
- **跨 layer 看趋势**：`moe/expert_var/L{layer}/{domain}` 在浅层基本都接近 uniform，
  深层才出现领域聚集；故事里挑深层 layer 讲。

### Stage 3 · Dense→MoE Upcycle

- **smoke_test 必须跑两次**：upcycle 直后 + fine-tune 后，对比 expert 占比的变化曲线，
  这是"router 自发分化"的证据图。
- **`--out-shard-size 5B --lazy-unpickle` 必带**，不带 mergekit-moe 在 1.7B 上会 OOM。
- **`gate_mode: random`**：故意用随机 router 起步，让训练看到收敛过程；用 `hidden`
  初始化反而看不到曲线变化。
- **expert 死掉**：smoke test 显示某 expert 0 token，回 yaml 加丰富 `positive_prompts`
  或换 `gate_mode: hidden`。

### Stage 4 · 蒸馏

- **教师 vLLM bf16**：vLLM 不直接吐 logits，所以 stage A（生文本）用 vLLM，stage B（dump
  logits）用 HF transformers + flash-attn 2。
- **stage B cache 内存**：demo 5k × 1024 token × top-50 ≈ 5GB，单机 RAM 装得下。简历规模
  必须改 chunked save（每 N batch 一个 .pt 文件），不要一次 `torch.load`。
- **stage C 显存**：Qwen3-8B 教师 bf16 (16GB) + 1.7B 学生 (3.4GB) + Adam (~13GB) ≈ 33GB，
  单 H100 80G 够。**不需要 ZeRO-3**，DDP 即可。
- **课程长度 128→256→512**：早期学生输出基本是噪声，让它在长序列上跟教师对齐反而学坏；
  曲线 `rollout/curriculum_max` 能在 swanlab 看到三段台阶。
- **retain_pct 直接出**：`run_eval.py --retain_baseline <teacher_path>` 自动算并打印
  per-task 保留率。截图就这一行。

### Stage 5 · Bench

- **吞吐 bench 在单 A10/L20 上跑**，多卡 vLLM tensor_parallel 会让 8B 教师和 1.7B 学生
  没法直接比。简历"单卡 A10 6×"的"单卡"是关键。
- **GRPO rollout 加速消融**：跑同样 1 epoch demo 数据，只比 wall-clock + 总 tokens/s。

---

## 6. RL 算法选型（GRPO + DAPO 补丁）

| 算法 | 决策 | 原因 |
|---|---|---|
| **GRPO** | ✅ 主线 | 短结构化输出 + 连续 F1 reward + dense 8B + trl 工具链最稳 |
| **GSPO** | ❌ 不上 | sequence-level importance ratio 主要解决长 CoT 和 MoE 训练稳定性，本任务两个优势都吃不到 |
| **DAPO** ➜ **Clip-Higher** | ✅ Cherry-pick | `epsilon_high=0.28 > epsilon_low=0.2`，trl 原生支持，1 行配置防 policy 过早崩 |
| **DAPO** ➜ **Dynamic Sampling** | ✅ Cherry-pick（离线版） | SFT 完成后用 vLLM 跑 N=8 rollout，丢弃 reward spread<0.1 的 prompt（advantage≈0 浪费算力）|
| **DAPO** Token-level loss / Overlong shaping | ❌ 不上 | 主要为长 reasoning 设计，工程复杂度 > 收益 |

---

## 7. Reward 设计

`stages/rewards/rlvr.py` 是全链路唯一 reward 入口。

| Task | Ground truth | 评分 | 备注 |
|---|---|---|---|
| `crime_prediction` | `{"crimes": [...]}` | F1 over set | JSON 解析失败 = -0.5（轻罚） |
| `contract_review` | `{"risks": [...]}`  | F1 over set | 同上 |
| 全部任务 | — | +0.1 format bonus | 上限 0.1 防 reward hacking |

`legal_reward_fn` 同时维护一个 `METRICS_BUFFER`：每次调用累计 parse 失败率、
per-task F1、format reward 占比等，由 `flush_rlvr_metrics()` 在 trainer callback
里 flush 到 SwanLab（见 `stages/stage1_sft_grpo/train_grpo.py` 的
`SwanLabLegalMetricsCallback`）。

---

## 8. SwanLab 监控

通过 SwanLab 的 wandb shim：`from swanlab.integration.wandb import wandb` 后代码里
仍写 `wandb.init/log` 但实际进 SwanLab；trl/HF Trainer 走 `report_to=["swanlab"]`。

完整指标 schema（每个 stage 该 log 什么、为什么）见 [`stages/SWANLAB_METRICS.md`](stages/SWANLAB_METRICS.md)。

Run name 命名约定：

| Stage | Run name |
|---|---|
| 1 | `stage1-sft-v1` / `stage1-grpo-v1` / `stage1-dpo-baseline` |
| 2 | `stage2-analyze-v1` |
| 3 | `stage3-smoke-test{,-after}` / `stage3-moe-sft-v1` |
| 4 | `stage4a-blackbox-warmup` / `stage4b-logits-kl` / `stage4c-onpolicy-kl` |
| 5 | `stage5-{ablation}-arm-{a,b}` |

---

## 9. 关键依赖与版本约束

| 组件 | 版本 | 说明 |
|---|---|---|
| transformers | ≥ 4.46 | `report_to=["swanlab"]` 需要 |
| trl | ≥ 0.13, < 0.15 | GRPOConfig 支持 `epsilon_high`（Clip-Higher）；0.15+ 砍掉 `vllm_device`（改 server-mode），暂未迁移 |
| vllm | ≥ 0.7 (< 0.8) | Dynamic Sampling / 蒸馏教师推理 / GRPO rollout；trl 0.13 的 GRPOTrainer 依赖 `vllm.sampling_params.GuidedDecodingParams`（首发 v0.6.5+），实测 0.7.x 最稳 |
| deepspeed | ≥ 0.15 | ZeRO-3 bf16 |
| swanlab | latest | wandb shim 在 `swanlab.integration.wandb` |
| mergekit | latest | Stage 3 mergekit-moe |
| flash-attn | latest | 全程开启 |
| datasets | ≥ 2.18 | `load_dataset("ShengbinYue/DISC-Law-SFT")` |

---

## 10. 不做的事

- **PT (continued pre-training)**：早期评估 SFT-only vs PT+SFT 差距 < 1pt（业界
  公认结论：法律语料 PT 只有在百亿 token 量级才显著），节省约 8 卡时，主线不做
- **扩词表**：消融 5.1 验证 < 0.5pt 差距，不扩
- **CoT 数据增强**：本项目 SFT 模板 reasoning slot 留空；teacher 自带的推理在 stage 4 蒸馏阶段通过教师文本 / logits 隐式传递给学生
- **RoPE 插值 / 长上下文工程**：Qwen3 系列原生 32K，stage1 `max_seq_length=4096` 远未满
- **GSPO**：见第 6 节；30B-A3B 上做 RL 时再考虑
- **PPO / RM 学习型奖励**：RLVR 已能覆盖项目目标的可验证任务，不需要额外训 RM
- **Stage 2 训练 30B-A3B**：4×H100 跑全参极易 OOM，且简历的"专家聚集"故事用 forward
  分析就够。`domain_aware_aux_loss.py` + `train_with_aux.py` 仅作参考实现

---

## 11. 数据来源说明

`stages/data_prep.py` 默认从 HF Hub 拉 `ShengbinYue/DISC-Law-SFT`（公开、~300k 条
中文法律 SFT、覆盖罪名/合同/法条/咨询多任务），处理出本仓库需要的所有产物：

| 产物 | 用于 | demo 规模 | full 规模 |
|---|---|---|---|
| `sft_demo/{clean,multitask,rebalanced}.jsonl` | Stage 1 SFT / Stage 3 fine-tune | ~30k | ~350k |
| `rlvr_demo/`（save_to_disk） | Stage 1 GRPO/DPO 训练 | 1.8k | 18k |
| `rlvr_demo_test/` | 全链路评估 | 0.2k | 2k |
| `rlvr_demo/prompts.jsonl` | Stage 4C on-policy KL | 同 train | 同 train |
| `distill_prompts.jsonl` | Stage 4A 教师生成 | 5k | 50k |
| `legal_eval_by_domain.jsonl` | Stage 2 路由分析 | 1k (5 域 × 200) | 同 demo |

合同条款数据 DISC-Law-SFT 没有现成的 risk-label 标注，所以 contract_review 部分用
确定性模板生成（8 类常见风险点）；只是为 GRPO 提供"另一个可验证任务"，让 reward
设计能在两个分布不同的任务上都 work。简历可以诚实地讲："罪名预测从公开数据，合同审查
做了模板合成"。

---

## 12. 风险提示

法律应用风险高。实际部署务必加：人工抽检、置信度阈值拒答、合规免责声明、**绝不**
直接对终端用户给法律建议。本仓库代码仅用于研究 / 评估场景。
