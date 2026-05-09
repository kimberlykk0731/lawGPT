# swanlab 指标 Schema（5-Stage 全链路）

> 用 swanlab 的 wandb shim：`from swanlab.integration.wandb import wandb` 后，
> `wandb.init/log/run` 全部代理到 swanlab。trl/HF 通过 `report_to=["swanlab"]`
> 走原生 swanlab callback。代码示例里仍写 `wandb.log({...})`——shim 后等价于
> 调用 swanlab。

命名约定按业务域分组：`train/*` 训练动力学，`legal/*` 法律任务质量，`kd/*` 蒸馏对齐，`moe/*` 路由健康度，`val/*` 评估集，`sys/*` 系统。

trl 自带 `train/loss` / `train/learning_rate` / `train/grad_norm` / `rewards/<fn>` / `kl` / `completion_length` / `clip_ratio` 等默认指标，**不重命名**，直接用。下面只列**需要自己埋点**的部分。

---

## Stage 1A — SFT (Qwen3-8B 350k 全参)

```python
wandb.log({
    # 基础（HuggingFace Trainer 自动）
    "train/loss":          loss,
    "train/learning_rate": lr,
    "train/grad_norm":     grad_norm,
    "train/epoch":         epoch,

    # 自定义评估
    "val/loss":            eval_loss,
    "val/perplexity":      math.exp(eval_loss),

    # 法律任务级（每 N 步在 holdout 集上 generate + 解析）
    "legal/charge_f1":     charge_f1,            # 罪名预测 F1
    "legal/article_jacc":  article_jaccard,      # 法条 Jaccard
    "legal/sentencing_mae":sentencing_mae,       # 量刑相对误差

    # 数据分布
    "data/template_mix":   wandb.Histogram(template_idxs),  # 多任务模板使用比例
})
```

## Stage 1B — GRPO (RLVR)

```python
wandb.log({
    # === 用户给的核心几项（trl 自带）===
    "actor/pg_loss":              actor_pg_loss,
    "actor/kl_loss":              actor_kl_loss,
    "actor/entropy_loss":         actor_entropy_loss,
    "critic/rewards/mean":        mean_reward,
    "response_length/mean":       mean_response_length,
    "response_length/clip_ratio": clip_ratio,
    "val/test_score":             test_score,

    # === 项目目标补的（自定义 callback，见 train_grpo.py）===
    "critic/rewards/std":           reward_std,            # GRPO group baseline 方差，监控学习稳定
    "critic/rewards/min":           reward_min,
    "critic/rewards/max":           reward_max,
    "critic/main_reward/mean":      main_reward_mean,      # 拆出主奖励
    "critic/format_reward/mean":    format_reward_mean,    # 拆出格式奖励，防 hacking
    "critic/format_reward/frac":    format_reward_frac,    # 占比 ≈1 时说明只学格式没学内容

    "legal/crime_f1":               crime_f1,              # 任务级 F1
    "legal/contract_f1":            contract_f1,
    "legal/parse_failure_rate":     parse_fail_rate,       # JSON 解析失败比例（关键！）
    "legal/empty_pred_rate":        empty_pred_rate,       # 空预测比例

    # 训练动力学
    "actor/grad_norm":              grad_norm,
    "actor/clip_frac":              ppo_clip_frac,         # ratio clip 触发频率
    "actor/approx_kl":              approx_kl,             # 实际 KL（与目标 beta 比对）

    # rollout 健康度
    "rollout/length_max":           length_max,
    "rollout/length_min":           length_min,
    "rollout/diversity":            diversity,             # 同 prompt 不同 completion 的 token 重合度
})
```

## Stage 1C — DPO baseline (对照 GRPO)

```python
wandb.log({
    # trl DPOTrainer 自带
    "train/loss":                 dpo_loss,
    "train/rewards/chosen":       chosen_reward,
    "train/rewards/rejected":     rejected_reward,
    "train/rewards/margin":       margin,
    "train/rewards/accuracies":   accuracy,                # chosen reward > rejected 比例

    # 自加（与 GRPO 同一 eval set 对齐对比）
    "val/crime_f1":               crime_f1,
    "val/contract_f1":            contract_f1,
    "val/test_score":             test_score,
    "compare/grpo_minus_dpo_pt":  grpo_score - dpo_score,  # 项目目标 +4.2pt 这条数
})
```

## Stage 2 — MoE Router 训练（领域感知 aux loss）

```python
wandb.log({
    "loss/lm_ce":                  lm_ce,
    "loss/intra_aux":              intra_aux,              # 域内均衡（KL to uniform）
    "loss/inter_js":               inter_js,               # 域间分化（JS divergence，越大越好）
    "loss/total":                  total,

    # 路由健康度（每个 layer × domain）
    "moe/expert_var/L{layer}/{domain}":  expert_variance,  # 专家利用方差，目标 X→Y
    "moe/expert_top1_freq/{domain}":     top1_freq,        # 每域 top-1 专家活跃度
    "moe/cross_domain_overlap":          overlap,          # 域间 top-5 专家集合重合率
    "moe/dead_experts":                  dead_count,       # 0 token 专家数
    "moe/load_balance_score":            lb_score,
})
```

## Stage 3 — Dense→MoE Upcycle 后续 fine-tune

```python
wandb.log({
    "train/loss":                 loss,
    "moe/expert_load/std":        load_std,
    "moe/active_experts_per_token": active_n,             # 实际激活的专家数（理论 top-2）
    "moe/dead_experts":           dead_count,
    "moe/router_entropy":         router_entropy,         # 早期低（随机）→训练后升高
})
```

## Stage 4A — 黑盒 SFT 蒸馏

```python
wandb.log({
    # 同 Stage 1A SFT 基础指标，外加：
    "kd/teacher_match_top1":      top1_match,             # 学生 argmax token 与教师答案重合率
    "kd/teacher_text_bleu":       bleu_to_teacher,        # 学生输出对教师输出的 BLEU（采样温度下）
})
```

## Stage 4B — 离线 top-50 logits KL 蒸馏

```python
wandb.log({
    "loss/total":                 total,
    "loss/ce":                    ce_loss,                 # alpha 分支
    "loss/kl":                    kl_loss,                 # (1-alpha) 分支
    "loss/alpha":                 alpha_value,

    "kd/teacher_entropy":         teacher_entropy,         # 教师分布锐度
    "kd/student_entropy":         student_entropy,
    "kd/agreement_top1":          agreement_top1,          # student top1 ∈ teacher top50 比例
    "kd/student_topk_prob_mass":  student_mass_in_topk,    # 学生在教师 top50 上的概率质量

    "kd/grad_norm":               grad_norm,
})
```

## Stage 4C — On-policy KL 蒸馏

```python
wandb.log({
    "loss/forward_kl":            kl_loss,

    # rollout 监控
    "rollout/length_mean":        rollout_len_mean,
    "rollout/length_max":         rollout_len_max,
    "rollout/curriculum_max":     curriculum_max_new_tokens,
    "rollout/teacher_logprob_mean": teacher_logprob,        # 教师对学生 rollout 的 logprob

    # 蒸馏对齐质量
    "kd/teacher_entropy_on_rollout": teacher_entropy,
    "kd/student_entropy_on_rollout": student_entropy,
    "kd/agreement_top1":             agreement,

    # 关键产出指标（每 N 步在 holdout 上对教师 / 学生跑 generate）
    "kd/student_vs_teacher_charge_f1":   student_f1 / teacher_f1,  # 项目目标 92%+
    "kd/student_vs_teacher_contract_f1": s_f1 / t_f1,
    "kd/retain_pct":                     overall_retain_pct,
})
```

## Stage 5 — 工程消融

每个 ablation 单独一个 swanlab run，用 tags 区分：

```python
wandb.init(    # shim → swanlab.init
    project="legalgpt",
    name=f"ablation-{arm_name}",
    tags=["stage5", "ablation", arm_name],
)

# 主指标只看终值（消融比的是收敛后差距）
wandb.log({
    "final/crime_f1":              crime_f1,
    "final/contract_f1":           contract_f1,
    "final/test_score":            test_score,
    "compare/delta_vs_baseline":   delta_pt,              # 项目目标记 <1pt / <0.5pt 那两条
})

# vLLM 吞吐 bench
wandb.log({
    "sys/tokens_per_s":            tokens_per_s,
    "sys/req_per_s":                req_per_s,
    "sys/speedup_vs_teacher":      speedup,                # 项目目标 6×
})
```

## 通用 swanlab run 配置

每个 stage 的训练入口约定：

```python
from swanlab.integration.wandb import wandb   # 走 shim
wandb.init(
    project="legalgpt-2026",
    name=f"stage{N}-{phase}-{tag}",   # 例: stage1-grpo-v1
    tags=[f"stage{N}", phase, model_tag],
    config={
        "model": model_name_or_path,
        "data": dataset_path,
        "lr": lr,
        # ... full args
    },
)
```

环境变量统一在 `.env`：

```bash
export SWANLAB_PROJECT=legalgpt-2026
export SWANLAB_API_KEY=...
export SWANLAB_WORKSPACE=your-org
```

> swanlab 提供 SaaS（swanlab.cn）和**完全离线模式**（`swanlab.init(mode="local")`）。
> 离线模式不需要 API key，本地起一个 dashboard 服务看图，适合内网/隔离训练机。
