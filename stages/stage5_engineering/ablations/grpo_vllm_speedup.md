# Ablation: GRPO rollout — vLLM vs HF generate

## 假设
GRPO 的瓶颈是 rollout 阶段（每个 prompt 采 N=8 个 completion）。把 rollout 从
HF Trainer 内置 generate 切换到独立 vLLM server，吞吐 **3×**。

## 实验设置

| Arm | rollout backend | 卡分布 |
|---|---|---|
| A. HF generate | model.generate() in trainer | 4 GPU 全用于训练 + rollout |
| B. vLLM | 独立 vLLM server | 3 GPU 训练 + 1 GPU vLLM rollout |

启动：

```bash
# Arm A
deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_grpo.py \
  --use_vllm false ...

# Arm B
deepspeed --num_gpus=3 stages/stage1_sft_grpo/train_grpo.py \
  --use_vllm true --vllm_device cuda:3 ...
```

## 测量指标
- 单步 wall-clock（rollout + forward + backward）
- 等效 token/s
- 指标终值差异（应基本一致；如果不一致说明 vLLM/HF 数值差异引入了 bias）

## 预期
3× 速度提升，等效用 1 卡换回 2 卡纯训等价值，4 epoch 训练时间从 ~50h → ~17h。
