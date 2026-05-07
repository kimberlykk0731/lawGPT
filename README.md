# 中文法律大模型后训练工程（Legal-LM-CN）

面向中文法律案件分析、判决推理与法律咨询的**端到端后训练流水线**。本仓库由 `lawGPT` 与 `legal-llm-training` 合并演进而来，工程骨架沿用 lawGPT，蒸馏链路吸收 legal-llm-training，整体阶段划分参考 MedicalGPT。

默认基座：**`Qwen3.5-4B-Instruct`**（小参数、强中文，已具备相当的法律基础能力，因此**跳过继续预训练**，直接进入后训练；单卡 A100/4090 即可全流程跑通）。

> 配置中所有 `Qwen3.5-4B-Instruct` 字段均可直接替换为其他基座（Qwen2.5-7B / DeepSeek-V3-Lite / InternLM2.5-7B-chat 等），模板自动通过 `training/template.py` 注册中心选择。

---

## 1. 项目特点

- **不做 PT**：4B 基座中文能力足够，PT 投入产出比低；法条与最新案例通过 RAG 注入
- **单卡可训**：4B + LoRA + bf16 + flash-attention-2，单张 A100 40G / RTX 4090 24G 可全流程跑通
- **双层奖励体系**
  - 规则奖励：罪名匹配、法条 Jaccard、量刑相对误差等可验证子任务 → GRPO
  - 学习型奖励模型 (RM)：说理质量、风险提示、案件分析等开放式任务 → PPO / DPO
- **CoT 双蒸馏**
  - 离线蒸馏：DeepSeek-R1 / Qwen-Max 产出 `<think>` 推理数据
  - On-Policy Distillation (OPD)：SFT 模型自己生成 → 教师改写打分 → 回流训练
- **法条 RAG**：训练 / 推理两端均注入实时法条，避免参数化记忆失效
- **模板注册中心**：`training/template.py` 统一管理 Qwen / DeepSeek / GLM / InternLM 模板，换基座只改一行
- **学界 benchmark**：LawBench + CAIL2018 双基准
- **可交付**：LoRA 合并、OpenAI 兼容 API、Gradio Demo、vLLM 部署一站式
- **安全合规**：训练数据脱敏、推理护栏、强制风险提示

---

## 2. 与现有项目的差异

| 维度 | 本项目 | lawGPT(原) | legal-llm-training | MedicalGPT |
|---|---|---|---|---|
| PT | ❌（基座已具备） | ❌ | ❌ | ✅ |
| SFT | ✅ 三种 CoT 模式 | ✅ | ✅ | ✅ |
| 离线 CoT 蒸馏 | ✅ DeepSeek-R1 | ❌ | ✅ | ❌ |
| OPD（在线蒸馏） | ✅ | ❌ | ❌ | ✅ |
| 学习型 RM | ✅ | ❌ | ❌ | ✅ |
| PPO | ✅（基于 RM） | ❌ | ❌ | ✅ |
| GRPO + 规则奖励 | ✅（多领域） | ✅ | ✅ | ⚠️ 仅通用 |
| DPO / ORPO | ✅ | ✅ | ❌ | ✅ |
| 法条 RAG | ✅ | ❌ | ❌ | ❌ |
| LawBench / CAIL 评测 | ✅ | ⚠️ 自定义 | ⚠️ 部分 | ❌ |
| 部署链路 | ✅ | ❌ | ❌ | ✅ |
| 单卡可训 | ✅ (4B) | ⚠️ (7B) | ⚠️ (7B) | ❌ |

---

## 3. 目录结构

```text
configs/
  base.yaml                         # 模型/路径/系统提示词
  datasets.yaml                     # 数据源注册
  ds_zero2.json                     # DeepSpeed ZeRO-2（多卡推荐）
  ds_zero3.json                     # DeepSpeed ZeRO-3（仅大模型回退用）
  sft/qwen35_4b_lora.yaml
  opd/qwen35_4b_lora.yaml
  dpo/qwen35_4b_lora.yaml
  rm/qwen35_4b_lora.yaml
  grpo/qwen35_4b_lora.yaml
  ppo/qwen35_4b_lora.yaml
scripts/
  # 数据
  build_corpus.py                   # 法律语料清洗 + 脱敏
  build_statute_index.py            # 法条 RAG 向量索引
  build_sft_dataset.py              # SFT 数据构建（含 RAG 注入）
  distill_cot.py                    # 离线 CoT 蒸馏（DeepSeek-R1 等）
  build_opd_dataset.py              # OPD 数据生成（自采样 + 教师改写）
  build_preference_dataset.py       # DPO/ORPO 偏好对
  build_rm_dataset.py               # RM 训练数据
  build_grpo_dataset.py             # GRPO 可验证任务数据
  build_ppo_dataset.py              # PPO prompt 数据
  # 训练
  train_sft.py
  train_opd.py
  train_dpo.py
  train_rm.py
  train_grpo.py
  train_ppo.py
  merge_lora.py                     # LoRA 合并到基座
  # 评测
  run_eval.py                       # 调度 LawBench / CAIL
demo/
  inference.py                      # CLI 推理
  openai_api.py                     # OpenAI 兼容 API
  gradio_demo.py                    # 前端 demo
src/legal_lm/
  data/
    schema.py                       # 统一 LegalSample
    redact.py                       # PII 脱敏
    rag.py                          # 法条检索
    distill.py                      # CoT 蒸馏 client
    loaders/                        # cail / jec_qa / lecard / local
  training/
    template.py                     # 模板注册中心
    sft.py
    opd.py
    dpo.py
    rm.py
    grpo.py
    ppo.py
    common.py                       # TRL 兼容层
  rewards/
    common.py                       # 通用：法条/事实/结构/风险/格式
    criminal.py                     # 刑事
    civil.py                        # 民事/侵权
    administrative.py               # 行政
    learned_rm.py                   # 学习型 RM 包装
    composite.py                    # 规则 + 学习型组合
  eval/
    lawbench.py                     # LawBench runner
    cail.py                         # CAIL2018 runner
    report.py                       # Markdown 报告
  safety/
    guardrails.py                   # 推理护栏
data/
  raw/                              # 原始（判决书、法规、公开 QA）
  statute_index/                    # 法条 FAISS 索引
  processed/
    sft_brief/  sft_cot/  sft_answer_only/
    preference/  rm/  grpo/  ppo/  opd/
outputs/
  sft/  opd/  dpo/  rm/  grpo/  ppo/  merged/
  eval/
```

---

## 4. 环境准备

要求：Python ≥ 3.10、CUDA ≥ 12.1、PyTorch ≥ 2.3、显存 ≥ 24G（单卡 4090/A100 即可，多卡可加速并支持更长上下文）。

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[dev,vllm,serve]"
```

环境变量（`.env.example`）：

```bash
HF_TOKEN=
WANDB_PROJECT=legal-lm-cn
WANDB_ENTITY=
DEEPSEEK_API_KEY=                   # 离线蒸馏 / OPD 教师
QWEN_API_KEY=
CUDA_VISIBLE_DEVICES=0
VLLM_HOST=127.0.0.1
VLLM_PORT=8000
```

---

## 5. 数据流水线

### 5.1 统一 Schema（`src/legal_lm/data/schema.py`）

核心字段：`sample_id / domain / task_type / facts / issues / statutes / gold_answer / gold_reasoning / brief_reasoning / distilled_cot / citations / chosen / rejected / metadata`。

`domain ∈ {criminal, civil, administrative, general}`。

### 5.2 原始语料清洗 + 脱敏

数据来源建议：判决文书（OpenLaw / 文书网公开）、现行法律法规全文、司法解释、公开法律 QA（CrimeKgAssitant、LawGPT-zh-data 等）。

```bash
python3 scripts/build_corpus.py \
  --input-dir data/raw/judgments_raw \
  --output-dir data/raw/judgments_clean \
  --redact \
  --min-length 200 --max-length 16000
```

`--redact` 启用脱敏（`src/legal_lm/data/redact.py`）：
- 当事人姓名 → `[当事人A]` / `[当事人B]`
- 身份证号、银行卡号、手机号 → 全部 mask
- 案号保留前缀年份，去掉细节编号
- 详细住址 → 保留区县级

### 5.3 法条 RAG 索引

```bash
python3 scripts/build_statute_index.py \
  --statute-dir data/raw/statutes \
  --output-dir data/statute_index \
  --embed-model BAAI/bge-base-zh-v1.5 \
  --chunk-size 256
```

产出 FAISS 索引 + `statute_meta.jsonl`。训练 / 推理时通过 `from legal_lm.data.rag import StatuteRetriever` 调用，`top_k` 默认 5，注入到 prompt 的 `【参考法条】` 段。

### 5.4 SFT 数据构建（三种 CoT 模式）

```bash
# answer-only
python3 scripts/build_sft_dataset.py \
  --dataset-name local_jsonl_reasoning \
  --output-dir data/processed/sft_answer_only \
  --mode answer_only --inject-rag --rag-top-k 5 \
  --splits train validation test

# brief reasoning（推荐基线）
python3 scripts/build_sft_dataset.py \
  --dataset-name local_jsonl_reasoning \
  --output-dir data/processed/sft_brief \
  --mode brief_reasoning --inject-rag --rag-top-k 5 \
  --splits train validation test

# distilled CoT（依赖 5.5 蒸馏产物）
python3 scripts/build_sft_dataset.py \
  --dataset-name local_jsonl_reasoning \
  --output-dir data/processed/sft_cot \
  --mode distilled_cot --inject-rag --rag-top-k 5 \
  --splits train validation test
```

`--inject-rag`：构造时检索 top-k 法条拼到 prompt 中，让模型学会"看着法条说话"，减少幻觉。

### 5.5 离线 CoT 蒸馏（迁移自 legal-llm-training）

```bash
python3 scripts/distill_cot.py \
  --api deepseek --model deepseek-reasoner \
  --input data/processed/sft_brief/train.jsonl \
  --output data/processed/distilled/train.jsonl \
  --workers 8 --max-samples 50000 \
  --enforce-structure
```

`--enforce-structure`：强制 `<think>` 内出现"争点 / 大前提 / 小前提 / 结论"四段，否则丢弃，提升 CoT 质量。

### 5.6 OPD（On-Policy Distillation）数据

```bash
# 1) SFT 模型自采样
python3 scripts/build_opd_dataset.py sample \
  --model-path outputs/sft/qwen35_4b_lora/merged \
  --input data/processed/sft_brief/train.jsonl \
  --output data/processed/opd/student_samples.jsonl \
  --temperature 0.8 --num-samples-per-prompt 4

# 2) 教师模型改写 + 打分
python3 scripts/build_opd_dataset.py refine \
  --api deepseek --model deepseek-reasoner \
  --input data/processed/opd/student_samples.jsonl \
  --output data/processed/opd/train.jsonl \
  --keep-top 1
```

### 5.7 偏好数据 / RM / GRPO / PPO

```bash
# DPO/ORPO 偏好对
python3 scripts/build_preference_dataset.py \
  --dataset-name local_jsonl_reasoning \
  --output-dir data/processed/preference \
  --splits train validation

# RM 训练数据（与偏好同源，但格式不同）
python3 scripts/build_rm_dataset.py \
  --preference-dir data/processed/preference \
  --output-dir data/processed/rm

# GRPO 可验证子任务（罪名 / 法条 / 量刑）
python3 scripts/build_grpo_dataset.py \
  --dataset-name cail2018 \
  --output-dir data/processed/grpo \
  --task-types charge,article,sentencing

# PPO prompt 数据
python3 scripts/build_ppo_dataset.py \
  --dataset-name local_jsonl_reasoning \
  --output-dir data/processed/ppo
```

---

## 6. 训练流水线

### 6.1 推荐端到端路线

```
基座 Qwen3.5-4B-Instruct
        │
        ▼
[阶段 A] SFT（brief_reasoning + distilled_cot 混合，3:7）
        │
        ▼
[阶段 B] OPD（可选，开放式任务收益明显）
        │
        ├──▶ [阶段 C1] GRPO（可验证子任务，规则奖励）
        │
        ├──▶ [阶段 C2] DPO/ORPO（开放说理，偏好对）
        │
        └──▶ [阶段 C3] RM → PPO（开放说理，学习型奖励）
                    │
                    ▼
            合并 LoRA → 部署
```

资源紧张时推荐 **SFT → DPO**；追求质量上限推荐 **SFT → OPD → GRPO（子任务）+ DPO（说理）**；偏好数据 ≥ 10w 对时再上 **RM + PPO**。

### 6.2 SFT

配置：`configs/sft/qwen35_4b_lora.yaml`

```yaml
base_config: configs/base.yaml
dataset_config: configs/datasets.yaml
run:
  dataset_name: local_jsonl_reasoning
  sft_mode: distilled_cot
  mixed_modes:                      # 混合训练
    distilled_cot: 0.7
    brief_reasoning: 0.3
  output_dir: outputs/sft/qwen35_4b_lora
  max_seq_length: 4096
  packing: true
  inject_rag: true
trainer:
  learning_rate: 2.0e-5
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  bf16: true
  gradient_checkpointing: true
  attn_implementation: flash_attention_2
lora:
  enabled: true
  r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]
```

启动（单卡）：

```bash
python3 scripts/train_sft.py --config configs/sft/qwen35_4b_lora.yaml
```

启动（多卡 ZeRO-2）：

```bash
deepspeed --num_gpus 4 scripts/train_sft.py \
  --config configs/sft/qwen35_4b_lora.yaml \
  --deepspeed configs/ds_zero2.json
```

### 6.3 OPD

```bash
python3 scripts/train_opd.py --config configs/opd/qwen35_4b_lora.yaml
```

OPD 在 SFT 模型基础上继续训：loss = SFT loss(教师改写文本) + KL(student || teacher 在 student 采样上)，由 `src/legal_lm/training/opd.py` 实现。

### 6.4 DPO / ORPO

```bash
# DPO
python3 scripts/train_dpo.py --config configs/dpo/qwen35_4b_lora.yaml

# ORPO（无需参考模型，更省显存）
# 修改配置 run.alignment_method: orpo 后用同一脚本
python3 scripts/train_dpo.py --config configs/dpo/qwen35_4b_lora.yaml
```

关键参数：`beta: 0.1`，`loss_type: sigmoid`（DPO）/ `orpo_alpha: 0.1`（ORPO）。

### 6.5 奖励模型 (RM)

```bash
python3 scripts/train_rm.py --config configs/rm/qwen35_4b_lora.yaml
```

RM 基于 SFT 模型加 value-head，在偏好对（chosen/rejected）上训 ranking loss。产出用于 PPO 与离线打分。

### 6.6 GRPO（规则奖励，可验证子任务）

配置：`configs/grpo/qwen35_4b_lora.yaml`

```yaml
run:
  reward_profile: criminal          # criminal | civil | administrative | general
  use_vllm: true                    # vLLM 加速 rollout
  vllm_server_host: 127.0.0.1
trainer:
  learning_rate: 5.0e-7
  num_generations: 8
  beta: 0.04
  scale_rewards: batch
  temperature: 0.9
```

`reward_profile` 选定的规则奖励组合（例 criminal）：
- `charge_match_reward`（罪名）
- `article_match_reward`（法条）
- `sentencing_band_reward`（量刑区间）
- `element_coverage_reward`（构成要件覆盖）
- `format_reward`（结构合规）

启动：

```bash
# 先起 vLLM rollout 服务（4B 单卡即可）
python3 -m vllm.entrypoints.openai.api_server \
  --model outputs/sft/qwen35_4b_lora/merged --port 8000

# 再启动 GRPO
python3 scripts/train_grpo.py --config configs/grpo/qwen35_4b_lora.yaml
```

### 6.7 PPO（学习型 RM，开放式任务）

```bash
# vLLM rollout
python3 -m vllm.entrypoints.openai.api_server \
  --model outputs/sft/qwen35_4b_lora/merged --port 8000

# PPO
python3 scripts/train_ppo.py --config configs/ppo/qwen35_4b_lora.yaml
```

PPO 配置默认 `kl_coef: 0.05`、`reward_model_path: outputs/rm/qwen35_4b_lora`。RM 输出的标量奖励**可以叠加规则奖励的子集**（如 `format_reward + caution_compliance_reward`），权重在 `configs/ppo/*.yaml` 的 `reward.composite` 段配置。

### 6.8 合并 LoRA

```bash
python3 scripts/merge_lora.py \
  --base-model Qwen/Qwen3.5-4B-Instruct \
  --adapter outputs/sft/qwen35_4b_lora \
  --output outputs/sft/qwen35_4b_lora/merged \
  --dtype bfloat16
```

合并后产物可直接被 vLLM、Gradio、OpenAI API 加载，也是后续 OPD / GRPO / PPO 的起点。

### 6.9 多卡 / DeepSpeed

`configs/ds_zero2.json` 关键项（4B 默认推荐）：
- `zero_optimization.stage: 2`
- `bf16.enabled: true`
- `gradient_accumulation_steps: auto`

显存参考（bf16 + flash-attn-2 + gradient ckpt）：
- 单卡 24G（4090）：4B + LoRA r=32 + seq=2048 + ga=8 → 可跑
- 单卡 40G（A100）：4B + LoRA r=64 + seq=4096 + ga=8 → 推荐
- 单卡 80G（A100）：4B + 全参 + seq=4096 → 可跑（QLoRA 不必再开）
- 4×40G（A100）+ ZeRO-2：seq=8192 长判决书可承载

`configs/ds_zero3.json` 仅在切换到更大基座（≥ 14B）时启用。

---

## 7. 奖励体系

### 7.1 规则奖励（`src/legal_lm/rewards/`）

| 模块 | 奖励项 | 适用任务 |
|---|---|---|
| `common.py` | `citation_accuracy_reward` | 法条引用 Jaccard |
| | `fact_consistency_reward` | 事实一致性（NLI 校验，非 token overlap） |
| | `reasoning_structure_reward` | 结论/法律依据/推理摘要/风险提示 段落完整度 |
| | `caution_compliance_reward` | 风险提示关键词命中 |
| | `format_reward` | 强制结构 |
| `criminal.py` | `charge_match` / `article_match` / `sentencing_band` / `element_coverage` | 刑事 |
| `civil.py` | `liability_allocation` / `compensation_items` / `causation` / `burden_of_proof` / `statute_of_limitations` | 民事/侵权 |
| `administrative.py` | `legality_check` / `procedure_review` / `subject_qualification` / `remedy_path` | 行政 |

> **修订点**：`fact_consistency_reward` 不再用 `token_overlap × 2`，改用本地小型 NLI 模型做事实蕴含判断（环境无 NLI 时回退到改进版 token overlap）。

### 7.2 学习型奖励（`rewards/learned_rm.py`）

`LearnedRMReward(model_path, batch_size, normalize)` 包装 RM，输出 `[0,1]` 标量。

### 7.3 组合策略（`rewards/composite.py`）

```python
CompositeReward(
    rule_rewards=build_criminal_reward_functions(),
    learned_rm=LearnedRMReward("outputs/rm/qwen35_4b_lora"),
    weights={"rule": 0.4, "learned": 0.6},     # 参数化，不写死
)
```

GRPO 用纯规则；PPO 默认 `rule:learned = 0.3:0.7`；DPO 不直接调用 reward，但偏好数据可用 RM 离线打分二次过滤。

---

## 8. 评测（仅保留两个口径）

### 8.1 LawBench

LawBench（南京大学，20 个子任务，覆盖法律知识记忆 / 理解 / 应用）作为对外汇报主基准。

```bash
# 1) 拉取 LawBench
python3 scripts/run_eval.py prepare --benchmark lawbench \
  --output-dir data/raw/LawBench

# 2) 跑评测
python3 scripts/run_eval.py run \
  --benchmark lawbench \
  --model-path outputs/sft/qwen35_4b_lora/merged \
  --output-dir outputs/eval/lawbench/sft \
  --use-vllm \
  --inject-rag                       # 推理同步注入法条
```

输出：每个子任务的 score、`overall`、`by_capability`（记忆/理解/应用）、`report.md`。

### 8.2 CAIL2018

CAIL2018 三件套：罪名预测、法条推荐、刑期预测。

```bash
python3 scripts/run_eval.py run \
  --benchmark cail2018 \
  --model-path outputs/sft/qwen35_4b_lora/merged \
  --data-dir data/raw/CAIL2018 \
  --output-dir outputs/eval/cail/sft \
  --use-vllm
```

输出：`charge_macro_f1`、`article_macro_f1`、`sentencing_mae`（月）、`report.md`。

### 8.3 多模型对比 + CoT 消融

```bash
python3 scripts/run_eval.py compare \
  --runs outputs/eval/lawbench/sft outputs/eval/lawbench/dpo outputs/eval/lawbench/ppo \
  --output-dir outputs/eval/compare
```

CoT 消融通过分别训 `answer_only / brief_reasoning / distilled_cot` 三个 SFT，再分别评测对照。

---

## 9. 部署

### 9.1 CLI 推理

```bash
python3 demo/inference.py \
  --model-path outputs/sft/qwen35_4b_lora/merged \
  --use-vllm \
  --inject-rag --rag-top-k 5
```

### 9.2 OpenAI 兼容 API

```bash
python3 demo/openai_api.py \
  --model-path outputs/sft/qwen35_4b_lora/merged \
  --host 0.0.0.0 --port 8001 \
  --use-vllm \
  --inject-rag --enable-guardrails
```

调用方式与 OpenAI ChatCompletions 一致，便于上游业务接入。

### 9.3 Gradio Web Demo

```bash
python3 demo/gradio_demo.py \
  --api-base http://127.0.0.1:8001/v1 \
  --share
```

界面提供：案件事实输入框、自动检索的参考法条侧栏、结构化输出（结论 / 法律依据 / 推理摘要 / 风险提示）。

### 9.4 vLLM 直接部署

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model outputs/sft/qwen35_4b_lora/merged \
  --port 8000 \
  --max-model-len 32768
```

---

## 10. 安全与合规护栏

`src/legal_lm/safety/guardrails.py` 实现三层防护：

1. **System prompt 强制声明**："本回答仅供参考，不构成正式法律意见。"
2. **结构化拒答**：检测到以下情形直接拒答或转人工：
   - 涉及具体真实当事人姓名 / 案号且未脱敏
   - 涉及死刑量刑等高风险结论
   - 涉及最近一年内立法变更（由日期检测器触发）
3. **训练数据脱敏**：`build_corpus.py --redact` 在数据入口侧完成（详见 §5.2）。

风险提示由 `caution_compliance_reward` 在训练阶段持续强化；推理阶段如输出未包含风险提示，由 guardrails 自动追加。

---

## 11. 端到端最小复现路线

```bash
# 0) 基座
export BASE=Qwen/Qwen3.5-4B-Instruct

# 1) 数据
python3 scripts/build_corpus.py          --input-dir data/raw/judgments_raw --output-dir data/raw/judgments_clean --redact
python3 scripts/build_statute_index.py   --statute-dir data/raw/statutes    --output-dir data/statute_index
python3 scripts/build_sft_dataset.py     --dataset-name local_jsonl_reasoning --output-dir data/processed/sft_brief --mode brief_reasoning --inject-rag --splits train validation test
python3 scripts/distill_cot.py           --api deepseek --input data/processed/sft_brief/train.jsonl --output data/processed/distilled/train.jsonl --enforce-structure
python3 scripts/build_sft_dataset.py     --dataset-name local_jsonl_reasoning --output-dir data/processed/sft_cot --mode distilled_cot --inject-rag --splits train validation test
python3 scripts/build_preference_dataset.py --dataset-name local_jsonl_reasoning --output-dir data/processed/preference --splits train validation
python3 scripts/build_grpo_dataset.py    --dataset-name cail2018 --output-dir data/processed/grpo --task-types charge,article,sentencing

# 2) SFT
python3 scripts/train_sft.py             --config configs/sft/qwen35_4b_lora.yaml
python3 scripts/merge_lora.py            --base-model $BASE --adapter outputs/sft/qwen35_4b_lora --output outputs/sft/qwen35_4b_lora/merged

# 3) DPO（开放说理）
python3 scripts/train_dpo.py             --config configs/dpo/qwen35_4b_lora.yaml
python3 scripts/merge_lora.py            --base-model $BASE --adapter outputs/dpo/qwen35_4b_lora --output outputs/dpo/qwen35_4b_lora/merged

# 4) GRPO（可验证子任务）
python3 -m vllm.entrypoints.openai.api_server --model outputs/dpo/qwen35_4b_lora/merged --port 8000 &
python3 scripts/train_grpo.py            --config configs/grpo/qwen35_4b_lora.yaml
python3 scripts/merge_lora.py            --base-model $BASE --adapter outputs/grpo/qwen35_4b_lora --output outputs/grpo/qwen35_4b_lora/merged

# 5) 评测
python3 scripts/run_eval.py run --benchmark lawbench --model-path outputs/grpo/qwen35_4b_lora/merged --output-dir outputs/eval/lawbench/grpo --use-vllm --inject-rag
python3 scripts/run_eval.py run --benchmark cail2018 --model-path outputs/grpo/qwen35_4b_lora/merged --data-dir data/raw/CAIL2018 --output-dir outputs/eval/cail/grpo --use-vllm

# 6) 部署
python3 demo/openai_api.py --model-path outputs/grpo/qwen35_4b_lora/merged --port 8001 --use-vllm --inject-rag --enable-guardrails
python3 demo/gradio_demo.py --api-base http://127.0.0.1:8001/v1
```

如要进一步追求质量上限，在第 3 步后插入 OPD：

```bash
python3 scripts/build_opd_dataset.py sample  --model-path outputs/sft/qwen35_4b_lora/merged --input data/processed/sft_brief/train.jsonl --output data/processed/opd/student_samples.jsonl
python3 scripts/build_opd_dataset.py refine  --api deepseek --input data/processed/opd/student_samples.jsonl --output data/processed/opd/train.jsonl
python3 scripts/train_opd.py                 --config configs/opd/qwen35_4b_lora.yaml
```

如偏好数据规模 ≥ 10w 对，可启用 PPO 替代 DPO：

```bash
python3 scripts/train_rm.py   --config configs/rm/qwen35_4b_lora.yaml
python3 scripts/train_ppo.py  --config configs/ppo/qwen35_4b_lora.yaml
```

---

## 12. FAQ

**Q1：为什么不做 PT？**
4B 中文基座的法律覆盖度对后训练已足够，PT 收益小、成本高（百卡级）。法条与案例知识通过 RAG 注入更稳健、可更新。

**Q2：什么时候选 GRPO，什么时候选 PPO？**
- 任务有标准答案（罪名、法条编号、刑期数值）→ GRPO + 规则奖励
- 任务是开放式说理（案件分析、咨询答复）→ RM + PPO，或 DPO 替代

**Q3：DPO 与 PPO 二选一怎么选？**
偏好对 < 5w → DPO；5–10w 且需要在线探索更优解 → PPO；> 10w 且追求上限 → PPO + 离线 DPO 双轨。

**Q4：LawBench 跑一遍多久？**
20 子任务，单任务 1–3k 题，4B + vLLM 单卡约 1–2 小时。

**Q5：法条 RAG 的索引多久更新一次？**
建议每月一次，重大立法变更（如新司法解释）即时增量更新。`build_statute_index.py` 支持 `--incremental` 模式。

**Q6：训练数据涉及隐私怎么办？**
所有原始数据进入 `data/raw/` 之前必须通过 `build_corpus.py --redact`；线上推理同样开启 guardrails 检测真实姓名 / 案号。

**Q7：要换基座（如 Qwen2.5-7B、DeepSeek-V3-Lite）怎么做？**
- 改 `configs/base.yaml` 的 `model.model_name_or_path`
- `training/template.py` 注册中心已覆盖主流模型，自动选择对应 chat template
- LoRA r 视显存调整（7B 推荐 r=32，单卡 A100 40G 足够）

---

## 13. 路线图

- [ ] 长上下文：RoPE 插值到 32k–128k，长判决书全文输入
- [ ] 多 agent 推理：事实认定 / 争点识别 / 法条检索 / 量刑分析四 agent 协同
- [ ] 持续评测：CI 自动跑 LawBench 子集，写入 `outputs/eval/regression.csv`
- [ ] 行业适配：金融合规、知识产权、劳动争议等垂域微调

---

## 14. 注意事项

- 法律任务风险高，线上使用务必加人工抽检 + 拒答策略
- TRL 版本兼容：训练入口已通过 `instantiate_supported` 做参数白名单过滤，但仍以你本地版本为准
- 量化部署：4B → INT4 (AWQ/GPTQ) 后单卡 12G 即可推理，吞吐变化甚微
- 法条 RAG 不替代参数化记忆，二者互补；纯 RAG 在多步推理上仍有不足
