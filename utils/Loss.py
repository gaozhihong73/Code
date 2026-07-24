import torch
from torch import nn


class MSEPCCLoss(nn.Module):
    def __init__(self, pcc_weight=0.0):
        super().__init__()
        self.pcc_weight = pcc_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        pred: [Batch, Phenotype, Time]
        target: [Batch, Phenotype, Time]
        """
        # 1. MSE Loss (数值准确性)
        mse_loss = self.mse(pred, target)

        # 2. PCC Loss (排序准确性) - 修正版
        # 我们需要在 Batch 维度 (dim=0) 计算相关性
        # 也就是：对于每一个 (Phenotype, Time) 组合，计算 400 个样本的 PCC

        # 减去每个 (P, T) 的 Batch 均值
        # mean shape: [1, P, T]
        pred_mean = torch.mean(pred, dim=0, keepdim=True)
        target_mean = torch.mean(target, dim=0, keepdim=True)

        vx = pred - pred_mean  # [B, P, T]
        vy = target - target_mean  # [B, P, T]

        # 计算分子分母 (在 Batch 维度求和)
        # sum shape: [P, T]
        numerator = torch.sum(vx * vy, dim=0)
        denominator = torch.sqrt(torch.sum(vx**2, dim=0)) * torch.sqrt(
            torch.sum(vy**2, dim=0)
        )

        # 计算每个 (P, T) 的 PCC
        # 添加 eps 防止除零 (如果某时刻所有样本值都一样，std为0)
        pcc_per_point = numerator / (denominator + 1e-8)

        # 对所有表型和时间点取平均，得到最终 PCC Loss
        # 我们希望 PCC 接近 1，所以 Loss = 1 - PCC
        pcc_loss = 1.0 - torch.mean(pcc_per_point)

        # 打印调试 (可选)
        # print(f"Batch PCC: {torch.mean(pcc_per_point).item():.4f}")

        return mse_loss + self.pcc_weight * pcc_loss


class MSEOffsetLoss(nn.Module):
    """MSE + 个体时间均值 MSE + 可选 PCC, 针对 TrueGV 跨时间常数的场景

    核心思想: 对时间维度取均值压缩噪声 (√T 倍), 让梯度直指个体差异 (TrueGV),
    避免逐点 MSE 被时间点噪声淹没。
    """

    def __init__(self, offset_weight: float = 0.5, pcc_weight: float = 0.0):
        super().__init__()
        self.offset_weight = offset_weight
        self.pcc_weight = pcc_weight
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, P, T]
            target: [B, P, T]
        """
        # 1. 逐点 MSE: 保留时间结构信息
        mse_loss = self.mse(pred, target)

        # 2. 个体偏移 MSE: 对时间维度取均值, 直指个体遗传差异
        pred_indiv = pred.mean(dim=-1)  # [B, P]
        target_indiv = target.mean(dim=-1)  # [B, P]
        offset_loss = self.mse(pred_indiv, target_indiv)

        loss = mse_loss + self.offset_weight * offset_loss

        # 3. 可选 PCC: 加强跨个体排序
        if self.pcc_weight > 0:
            pred_mean = pred.mean(dim=0, keepdim=True)  # [1, P, T]
            target_mean = target.mean(dim=0, keepdim=True)
            vx = pred - pred_mean
            vy = target - target_mean
            num = (vx * vy).sum(dim=0)  # [P, T]
            den = (vx**2).sum(dim=0).sqrt() * (vy**2).sum(dim=0).sqrt()
            pcc_per_point = num / (den + 1e-8)
            pcc_loss = 1.0 - pcc_per_point.mean()
            loss = loss + self.pcc_weight * pcc_loss

        return loss
