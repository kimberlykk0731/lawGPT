# Ablation: 是否扩词表

## 假设
中文法律领域 OOV 不严重（"罪名/法条/裁定"等都在 Qwen tokenizer 词表里），
扩词表对最终指标增益 < 0.5pt，但训练复杂度（embedding 重训、checkpoint 体积）显著上升。

## 实验设置

| Arm | tokenizer | embedding 处理 |
|---|---|---|
| A. 不扩 | 原 Qwen3 152K vocab | 全部冻结/参与训练，无变化 |
| B. 扩 200 token | 用 BPE 在法律语料上挖 200 个高频复合词 | 新增 embedding 行随机初始化，前 1k step 单独高 LR |

## 评估
- 主指标：罪名预测 F1（同 stage1 RLVR eval）
- 副指标：平均 token/汉字 比（看分词效率）
- 副指标：训练吞吐（扩词表后 embedding gather 是否变慢）

## 结论填空
预期：F1 差距 < 0.5pt，token/字比从 0.X → 0.Y（提升约 Z%），不值得扩。
