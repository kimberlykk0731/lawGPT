# Stage 3 — Dense → MoE Upcycle

验证低成本 Dense→MoE 路径：Qwen3-1.7B 的 FFN 复制 8 份变成 8-expert MoE（top-2），
不重新预训练，仅靠几千步领域 fine-tune 让 router + experts 收敛。

## 1. mergekit-moe 离线 upcycle

```bash
pip install mergekit
./stages/stage3_dense_to_moe/run_upcycle.sh \
  stages/stage3_dense_to_moe/upcycle_qwen3_1_7b.yaml \
  ckpts/qwen3-1.7b-moe-8e
```

## 2. 冒烟测试（路由是否散开）

```bash
python stages/stage3_dense_to_moe/smoke_test.py \
  --model ckpts/qwen3-1.7b-moe-8e
```

期望 8 个 expert 都被激活；若某个 expert 占比 > 50% 说明 router 卡死，
回到 mergekit 配置加调 `positive_prompts`。

## 3. 领域语料 fine-tune

复用 stage1 的 SFT 入口（demo 规模 sft_demo 子集即可，目的是让 router 收敛）：

```bash
deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_sft.py \
  --model_name_or_path ckpts/qwen3-1.7b-moe-8e \
  --dataset_path       data/processed/sft_demo \
  --output_dir         ckpts/qwen3-1.7b-moe-8e-sft \
  --max_seq_length 4096 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --bf16 --gradient_checkpointing \
  --swanlab_run_name stage3-moe-sft-v1
```

## 4. swanlab 监控

| 指标 | 看什么 |
|---|---|
| `train/loss` | 标准语言建模损失 |
| `moe/expert_load/std` | 早期高（随机），训练后应缓慢下降 |
| `moe/active_experts_per_token` | 应稳定在 2 左右（top-2 配置） |
| `moe/router_entropy` | 应**上升**（从随机/单 expert 走向多 expert 共担） |
| `moe/dead_experts` | 应保持 0 |

## 5. 与 Stage 4 教师的关系

Stage 3 产出 `qwen3-1.7b-moe-8e-sft` 是**自造 MoE** 路径的学生候选；
Stage 4 默认蒸馏到 dense Qwen3-1.7B。可以再做一个对照：把 stage 3 输出
也作为学生跑 stage 4 蒸馏，看 MoE 学生 vs dense 学生 92%+ 那条数差多少。
