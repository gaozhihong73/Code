"""ConvAE：显式表型维度、结构性 padding 感知的 masked autoencoder。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.Genotype import model_code_tensor

MODEL_VERSION = "convae_padding_mask_v2"
MASK_POLICY = "valid_patch_ceil_keep_context_v1"


def sample_patch_mask(valid: torch.Tensor, ratio: float, *, generator=None) -> torch.Tensor:
    """仅从含有效 SNP 的 patch 抽样；单 patch 可全遮挡，其他行至少保留一个上下文。"""
    if valid.ndim != 2 or valid.dtype != torch.bool or not 0 <= ratio <= 1:
        raise ValueError("valid 必须为二维 bool patch mask，mask_ratio 必须在 [0,1]")
    counts = valid.sum(dim=1)
    masked = torch.ceil(counts.float() * ratio).long()
    masked = torch.minimum(masked, torch.where(counts > 1, counts - 1, counts))
    noise = torch.rand(valid.shape, device=valid.device, generator=generator)
    noise = noise.masked_fill(~valid, float("inf"))
    ranks = noise.argsort(dim=1).argsort(dim=1)
    return (ranks < masked[:, None]) & valid


def expand_padding_mask(x: torch.Tensor, padding_mask=None) -> torch.Tensor:
    """None 显式表示没有结构性 padding；类别0绝不用于推测 padding。"""
    if padding_mask is None:
        return torch.zeros_like(x, dtype=torch.bool)
    mask = torch.as_tensor(padding_mask, device=x.device)
    if mask.dtype != torch.bool or mask.shape not in (x.shape, x.shape[1:]):
        raise ValueError("padding_mask 必须为 bool [chunks, length] 或 [batch, chunks, length]")
    return mask.expand_as(x)


class SNPFeatureExtractor(nn.Module):
    """结构性 padding 的 embedding 贡献为零；真实缺失仍使用类别0的 embedding。"""

    def __init__(self, embed_dim=8, patch_size=20, chunk_size=12000, d_model=128):
        super().__init__()
        self.patch_size, self.chunk_size = patch_size, chunk_size
        self.num_patches = math.ceil(chunk_size / patch_size)
        self.snp_embedding = nn.Embedding(4, embed_dim)
        self.patch_project = nn.Linear(patch_size * embed_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x, padding_mask=None):
        if x.ndim != 2 or x.shape[1] != self.chunk_size:
            raise ValueError("SNPFeatureExtractor 输入长度与 chunk_size 不一致")
        padding = torch.zeros_like(x, dtype=torch.bool) if padding_mask is None else padding_mask
        if padding.shape != x.shape or padding.dtype != torch.bool:
            raise ValueError("特征提取器 padding mask 形状或类型错误")
        # 先屏蔽结构性 padding 数值，再校验有效位点；padding 内容不影响投影。
        codes = model_code_tensor(x.masked_fill(padding, 0))
        embedded = self.snp_embedding(codes).masked_fill(padding[..., None], 0)
        extra = self.num_patches * self.patch_size - self.chunk_size
        embedded = F.pad(embedded, (0, 0, 0, extra))
        tokens = embedded.reshape(x.shape[0], self.num_patches, -1)
        return self.patch_project(tokens) + self.pos_embedding


class ConvAE(nn.Module):
    """保留现有类名和 Transformer MAE；输出表型为 [B, P, T] 或 None。

    四项返回值为 chunk 表征、表型预测、重建 logits、被选中重建的 bool patch mask。
    eval 也会在提供 reconstruction_mask 或正 mask_ratio 时执行重建。
    """

    def __init__(self, chunk_size=12000, patch_size=20, d_model=128, d_ext=166,
                 num_chunks=300, *, n_phenotypes=1, n_time_points=14, use_pheno_help=True,
                 num_heads=4, encoder_layers=4, decoder_layers=2, dropout=0.1):
        super().__init__()
        sizes = (chunk_size, patch_size, d_model, d_ext, num_chunks, n_phenotypes,
                 n_time_points, num_heads, encoder_layers, decoder_layers)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in sizes):
            raise ValueError("ConvAE 维度及层数必须为正整数")
        if d_model % num_heads or not 0 <= dropout < 1:
            raise ValueError("d_model 必须能被 num_heads 整除，dropout 必须在 [0,1)")
        self.chunk_size, self.patch_size = chunk_size, patch_size
        self.num_patches = math.ceil(chunk_size / patch_size)
        self.num_chunks, self.d_model, self.d_ext = num_chunks, d_model, d_ext
        self.n_phenotypes, self.n_time_points = n_phenotypes, n_time_points
        self.use_pheno_help = use_pheno_help
        self._config = dict(chunk_size=chunk_size, patch_size=patch_size, d_model=d_model,
                            d_ext=d_ext, num_chunks=num_chunks, n_phenotypes=n_phenotypes,
                            n_time_points=n_time_points, use_pheno_help=use_pheno_help,
                            num_heads=num_heads, encoder_layers=encoder_layers,
                            decoder_layers=decoder_layers, dropout=dropout)
        self.feature_extractor = SNPFeatureExtractor(patch_size=patch_size, chunk_size=chunk_size, d_model=d_model)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=256,
                                            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, encoder_layers, enable_nested_tensor=False)
        self.compress_layer = nn.Sequential(nn.Linear(d_model, d_ext), nn.LayerNorm(d_ext), nn.ReLU())
        layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=256,
                                            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.normal_(self.decoder_pos_embedding, std=0.02)
        self.reconstruct_head = nn.Linear(d_model, patch_size * 4)
        self.phenotype_head = (nn.Sequential(
            nn.Linear(num_chunks * d_ext, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_phenotypes * n_time_points),
        ) if use_pheno_help else None)

    def model_config(self) -> dict:
        return {"model_version": MODEL_VERSION, **self._config}

    def forward(self, x, mask_ratio=0.5, *, padding_mask=None, reconstruction_mask=None, generator=None):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.num_chunks, self.chunk_size):
            raise ValueError("ConvAE 输入必须为 [batch, num_chunks, chunk_size]，并与模型配置一致")
        padding = expand_padding_mask(x, padding_mask)
        batch, chunks, length = x.shape
        rows = batch * chunks
        flat = x.reshape(rows, length)
        flat_padding = padding.reshape(rows, length)
        features = self.feature_extractor(flat, flat_padding)
        extra = self.num_patches * self.patch_size - length
        padded = F.pad(flat_padding, (0, extra), value=True).reshape(rows, self.num_patches, self.patch_size)
        weights = (~padded).sum(dim=-1)
        valid = weights > 0
        if reconstruction_mask is None:
            selected = sample_patch_mask(valid, mask_ratio, generator=generator)
        else:
            selected = torch.as_tensor(reconstruction_mask, device=x.device)
            if selected.shape == (batch, chunks, self.num_patches):
                selected = selected.reshape(rows, self.num_patches)
            if selected.dtype != torch.bool or selected.shape != valid.shape or (selected & ~valid).any():
                raise ValueError("reconstruction_mask 只能选择有效 patch，不能选择结构性 padding")
        visible = valid & ~selected
        active = visible.any(dim=1)
        width = max(1, int(visible.sum(dim=1).max()))
        order = (~visible).int().argsort(dim=1, stable=True)[:, :width]
        memory_padding = ~visible.gather(1, order)
        packed = features.gather(1, order[..., None].expand(-1, -1, self.d_model))
        # 全 padding 行不进入注意力，避免全 key 被遮挡时 softmax NaN。
        memory = features.new_zeros(rows, width, self.d_model) + features.sum() * 0
        if active.any():
            memory[active] = self.encoder(packed[active], src_key_padding_mask=memory_padding[active])
        pool_weights = weights.gather(1, order) * ~memory_padding
        pooled = (memory * pool_weights[..., None]).sum(dim=1) / pool_weights.sum(dim=1).clamp_min(1)[:, None]
        compressed = self.compress_layer(pooled).masked_fill(~active[:, None], 0)
        compressed = compressed.reshape(batch, chunks, self.d_ext)
        phenotype = (self.phenotype_head(compressed.reshape(batch, -1)).reshape(
            batch, self.n_phenotypes, self.n_time_points) if self.phenotype_head is not None else None)
        logits = None
        if reconstruction_mask is not None or mask_ratio > 0:
            # masked patch 使用 mask token；不可见的结构性 patch 不提供 key/value。
            restored = features.new_zeros(features.shape) + features.sum() * 0
            restored.scatter_(1, order[..., None].expand(-1, -1, self.d_model), memory)
            decoder_input = torch.where(selected[..., None], self.mask_token, restored) + self.decoder_pos_embedding
            decoder_active = valid.any(dim=1)
            logits = features.new_zeros(rows, self.num_patches, self.patch_size, 4) + features.sum() * 0
            if decoder_active.any():
                # 单个有效 patch 全被遮挡时使用非 SNP 的零上下文；结构性 padding 仍不参与注意力。
                decoder_memory = memory.clone()
                decoder_memory_padding = memory_padding.clone()
                decoder_memory_padding[~active, 0] = False
                decoded = self.decoder(
                    decoder_input[decoder_active], decoder_memory[decoder_active],
                    tgt_key_padding_mask=~valid[decoder_active],
                    memory_key_padding_mask=decoder_memory_padding[decoder_active],
                )
                logits[decoder_active] = self.reconstruct_head(decoded).reshape(-1, self.num_patches, self.patch_size, 4)
            logits = logits.masked_fill(padded[..., None], 0)
        return compressed, phenotype, logits, selected
