"""ConvAE 统一训练/验证目标，以及按 seed、split、sample_id 固定的验证掩码。"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from model.ConvAE import expand_padding_mask, sample_patch_mask

RECONSTRUCTION_POLICY = "selected_observed_nonpadding_snp_mean_v1"
EMPTY_TARGET_POLICY = "zero_loss_zero_gradient"


def fixed_reconstruction_masks(padding_mask: torch.Tensor, *, patch_size: int, sample_ids,
                               seed: int, split_fingerprint: str, mask_ratio: float) -> torch.Tensor:
    """每个 sample_id 独立的 CPU RNG；与 batch 顺序、batch_size 和 epoch 无关。"""
    padding = torch.as_tensor(padding_mask, dtype=torch.bool, device="cpu")
    if padding.ndim != 2 or not len(sample_ids):
        raise ValueError("固定验证掩码需要 [chunks, length] padding mask 和非空 sample_id")
    extra = (-padding.shape[-1]) % patch_size
    valid = ~F.pad(padding, (0, extra), value=True).reshape(padding.shape[0], -1, patch_size).all(dim=-1)
    masks = []
    for sample in sample_ids:
        if isinstance(sample, np.generic):
            sample = sample.item()
        payload = json.dumps([int(seed), split_fingerprint, sample], ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
        masks.append(sample_patch_mask(valid, mask_ratio, generator=generator))
    return torch.stack(masks)


@dataclass
class MAELossTerms:
    recon_sum: torch.Tensor
    recon_count: int
    pheno_sum: torch.Tensor
    pheno_count: int

    @property
    def recon_loss(self):
        return self.recon_sum / max(1, self.recon_count)

    @property
    def pheno_loss(self):
        return self.pheno_sum / max(1, self.pheno_count)

    def total(self, recon_weight: float, pheno_weight: float):
        return recon_weight * self.recon_loss + pheno_weight * self.pheno_loss


def convae_loss_terms(outputs, codes: torch.Tensor, phenotype: torch.Tensor,
                     padding_mask: torch.Tensor, *, use_pheno_help: bool) -> MAELossTerms:
    """CE 仅监督已遮挡且非 padding 的观测类别1/2/3；无目标返回可反传的零。"""
    _, prediction, logits, selected = outputs
    if logits is None:
        raise ValueError("训练和验证损失都必须启用重建，不能用恒为零的验证重建占位")
    padding = expand_padding_mask(codes, padding_mask)
    rows, patches, patch_size, classes = logits.shape
    if classes != 4 or rows != codes.shape[0] * codes.shape[1]:
        raise ValueError("重建 logits 形状与输入不一致")
    extra = patches * patch_size - codes.shape[-1]
    labels = F.pad(codes.masked_fill(padding, 0), (0, extra)).reshape(rows, patches, patch_size)
    padding = F.pad(padding, (0, extra), value=True).reshape_as(labels)
    targets = selected[..., None] & ~padding & (labels > 0)
    count = int(targets.sum())
    recon = (F.cross_entropy(logits[targets].float(), labels[targets].long(), reduction="sum")
             if count else logits.sum() * 0)
    if use_pheno_help:
        if prediction is None or prediction.shape != phenotype.shape:
            raise ValueError("表型预测与原始配置必须一致为 [batch, n_phenotypes, n_time_points]")
        pheno = F.mse_loss(prediction.float(), phenotype.float(), reduction="sum")
        pheno_count = phenotype.numel()
    else:
        if prediction is not None:
            raise ValueError("关闭表型辅助时不应构建或调用表型头")
        pheno, pheno_count = logits.sum() * 0, 0
    return MAELossTerms(recon, count, pheno, pheno_count)
