# LegalGPT — 5 Stage 全链路

| Stage | 内容 | 入口 |
|---|---|---|
| 0 | 一键数据准备（HF 下载 + 处理 + 划分 train/test） | [`data_prep.py`](data_prep.py) |
| 1 | Qwen3-8B 全参 SFT + GRPO (RLVR, 含 DAPO Clip-Higher + Dynamic Sampling) + DPO baseline | [`stage1_sft_grpo/`](stage1_sft_grpo/README.md) |
| 2 | Qwen3-30B-A3B router 行为剖析（forward only） | [`stage2_moe_router/`](stage2_moe_router/README.md) |
| 3 | mergekit-moe 把 Qwen3-1.7B 升级为 8-expert MoE | [`stage3_dense_to_moe/`](stage3_dense_to_moe/README.md) |
| 4 | 黑盒 + top-50 logits + on-policy 三段蒸馏到 Qwen3-1.7B | [`stage4_distillation/`](stage4_distillation/README.md) |
| 5 | 工程消融 + 部署 bench | [`stage5_engineering/`](stage5_engineering/README.md) |

## 数据流向

```
HF DISC-Law-SFT
      │
      ▼
data_prep.py ────┬── data/processed/sft_demo/         ──→ stage1 SFT ─┐
                 │                                                     ├─→ ckpts/legalgpt-8b-sft
                 ├── data/processed/rlvr_demo/        ──→ stage1 GRPO ─┘             │
                 ├── data/processed/rlvr_demo_test/   ──→ run_eval.py                ▼
                 │                                                       ckpts/legalgpt-8b-grpo
                 ├── data/processed/distill_prompts.jsonl ──→ stage4 蒸馏  │
                 │                                                          ▼
                 └── data/processed/legal_eval_by_domain.jsonl  ckpts/legalgpt-1.7b-distilled
                                          │                              │
                                          ▼                              ▼
                                stage2 analyze_router          stage5 vLLM bench (6×)

stage3 走平行支线（mergekit-moe upcycle Qwen3-1.7B）
```

## 共享依赖

- Reward 函数：`stages/rewards/rlvr.py`（GRPO + DPO + run_eval + filter_dynamic_sampling 都用这一份）
- 通用评估：`stages/eval/run_eval.py`（vLLM 跑指定 model list，按 task 分组打 F1，自动算 retain_pct）
- 启动约定：所有命令都假设 CWD=repo 根 + `export PYTHONPATH=$PWD`

## swanlab 监控

通过 swanlab 的 wandb shim：`from swanlab.integration.wandb import wandb`，
代码里仍写 `wandb.init/log` 但实际进 swanlab；trl/HF Trainer 走 `report_to=["swanlab"]`。

完整指标 schema 见 [`SWANLAB_METRICS.md`](SWANLAB_METRICS.md)。环境变量：

```bash
export SWANLAB_PROJECT=legalgpt-2026
export SWANLAB_API_KEY=<your-key>
export SWANLAB_WORKSPACE=your-org
# 离线模式（不上传，本地 dashboard 看图）
# export SWANLAB_MODE=local
```

每个训练入口都接受 `--swanlab_run_name` 参数，约定命名 `stage{N}-{phase}-{tag}`。

**Stage 1 GRPO 自定义 callback**（`SwanLabLegalMetricsCallback`）会把任务级 F1、parse 失败率、format reward 占比等额外指标自动 flush。**Stage 2 analyze_router** 直接把每域每层的方差 / overlap 写到 swanlab。**Stage 3 smoke_test** 把 expert 占比写到 swanlab，对比 upcycle 前后变化。**Stage 4 蒸馏**直接在训练循环里通过 shim 的 `wandb.log`（→ swanlab）记录教师/学生熵、agreement top-1、rollout 长度等 KD 专属指标。**run_eval.py** 跑评估时直接 swanlab log retain_pct。

## 一键串起来（4× H100, demo 规模）

参见顶层 [`README.md`](../README.md) §4。
