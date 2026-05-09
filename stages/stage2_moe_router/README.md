# Stage 2 — Qwen3-30B-A3B Router 行为剖析（forward only）

观察 5 个法律子领域（民事/刑事/行政/商事/知产）下 Qwen3-30B-A3B（128 experts，
top-8 active）的路由行为。**主链路只做 forward 分析、不做训练**——简历"专家聚集"
的故事直接读分析输出写。`domain_aware_aux_loss.py` + `train_with_aux.py` 作为
参考实现保留，被问到再讲。

## 1. 数据

`stages/data_prep.py` 会自动产 `data/processed/legal_eval_by_domain.jsonl`：

```jsonl
{"domain": "刑事", "text": "..."}
{"domain": "民事", "text": "..."}
{"domain": "行政", "text": "..."}
{"domain": "商事", "text": "..."}
{"domain": "知产", "text": "..."}
```

每域 200 条，按关键词启发式分配（`stages/data_prep.py: DOMAIN_KEYWORDS`）。

## 2. 路由行为分析（一次性，离线，主入口）

```bash
python stages/stage2_moe_router/analyze_router.py \
  --model       Qwen/Qwen3-30B-A3B \
  --eval_jsonl  data/processed/legal_eval_by_domain.jsonl \
  --top_k       8 \
  --num_experts 128 \
  --output_json outputs/router_variance.json \
  --swanlab_run_name stage2-analyze-v1
```

输出：

- **JSON 文件** `outputs/router_variance.json`：每个 (domain, layer) 的方差 + top-5 expert + top-1 频率 + 跨域 top-5 重合率
- **swanlab 图表**：
  - `moe/expert_var/L{layer}/{domain}` — 每域每层路由方差折线
  - `moe/top1_freq/L{layer}/{domain}` — 每域 top-1 expert 激活频率
  - `moe/cross_domain_overlap/L{layer}` — 跨域 top-5 expert 集合重合率（深层应明显 < 浅层）
  - `moe/mean_variance/{domain}` — 每域跨层平均方差汇总

## 3. 显存

bf16 加载 ~60GB，单卡 H100 80G 够。如有压力：

```bash
python stages/stage2_moe_router/analyze_router.py \
  --model Qwen/Qwen3-30B-A3B \
  --device_map auto       # 跨卡分片
  ...
```

## 4. 写故事

跑完 `outputs/router_variance.json` 是这样：

```json
{
  "刑事": {
    "L20": {"variance": 0.0034, "top_experts": [17, 42, 89, 11, 56], "top1_freq": 0.18},
    ...
  },
  ...
}
```

简历那句"刑事 token 在 expert {x, y, z} 上聚集（激活率 0.xx vs 平均 0.yy）"
就是挑某一深层 layer，把 `top_experts[:3]` 和 `top1_freq` vs 5 域平均值写出来。

## 5. 不做的事 / 需要时再开

- **不做训练**：30B 全参 ZeRO-3 在 4×H100 上极易 OOM；面试被问"领域 aux loss 怎么训"，
  指 `train_with_aux.py` 解释结构 + 说明实际产物从 analyze 出图即可
- 想真的训练验证 aux loss 收益，把 `--model_name_or_path` 指向 stage 3 的 1.7B-MoE
  上下游会顺很多
