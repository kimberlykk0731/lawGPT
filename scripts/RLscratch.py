import torch
from torch import nn
from torch.nn import functional as F

# PPO
class PPO:
    def __init__(self, clip = 0.2, gamma=1, lam=0.95):
        self.clip = clip
        self.gamma = gamma
        self.lam = lam

    def mask_mean(self, loss, mask, dim=1):
        # 形状 [B, T, ...]、dim=1：对每个 batch 样本，在长度 T 上求带 mask 的平均，结果形状是 [B, ...]。
        # 在 mask 为 0 的位置把 loss 乘成 0，相当于忽略这些位置，再除以mask非零位置个数得到平均值
        return(loss * mask ).sum(dim=dim) / mask.sum(dim=dim)

    def advantage_estimate(self, rewards, values):
        # GAE广义优势估计
        # 序列长度本质上是「时间步数」，既等于 reward 步数，也应对齐 value 的步数
        seq_len = values.shape[1]
        # 0作为占位符，新建一个张量和rewards相同形状，值全为0
        advantages = torch.zeros_like(rewards)
        gae = 0
        for i in range(seq_len-1, -1, -1):
            #i 等于seq_len-1时，已经是序列最后一个时间步，所以next_value等于0
            next_value = values[:, i+1] if i < seq_len-1 else 0
            delta = rewards[:, i] + self.gamma * next_value - values[:, i]
            gae = delta + self.gamma * self.lam * gae
            advantages[:, i] = gae
            returns = advantages + values
        return advantages, returns

    def policy_loss(self, new_probs, old_probs, advantages, mask):
        # 策略损失
        # 形状 [B, T, ...]、dim=1：对每个 batch 样本，在长度 T 上求带 mask 的平均，结果形状是 [B, ...]。
        # 在 mask 为 0 的位置把 loss 乘成 0，相当于忽略这些位置，再除以mask非零位置个数得到平均值
        ratio = torch.exp(new_probs - old_probs)
        # 裁剪比例，防止比例过大或过小
        clipped_ratio = torch.clamp(ratio, 1-self.clip, 1+self.clip)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        # 传入mask函数得到有效位置loss平均值
        return self.mask_mean(policy_loss, mask)

    def value_loss(self, new_values, returns, mask):
        # 价值损失
        # 形状 [B, T, ...]、dim=1：对每个 batch 样本，在长度 T 上求带 mask 的平均，结果形状是 [B, ...]。
        # 在 mask 为 0 的位置把 loss 乘成 0，相当于忽略这些位置，再除以mask非零位置个数得到平均值
        value_loss = (new_values - returns) ** 2
        return self.mask_mean(value_loss, mask)

    



