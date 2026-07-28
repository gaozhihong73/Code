"""ConvAE 卷积自编码器模型定义：Masked Autoencoder + 表型预测联合训练。

本模块实现 ConvAE 的完整网络结构，包含：
- SNPFeatureExtractor: 将离散 SNP 序列转换为 Patch 向量并添加位置编码
- ConvAE: 主模型，通过 MAE 风格掩码重建学习 SNP 压缩表征，
  同时使用表型预测头进行监督学习，压缩结果可供下游 TFPP 模型使用
"""

import torch
from torch import nn


class SNPFeatureExtractor(nn.Module):
    """将离散 SNP 序列转换为稠密 Patch 向量并添加位置编码。

    Parameters
    ----------
    embed_dim : int
        SNP 类别嵌入维度（将 0/1/2/3 映射到的连续空间维度）。
    patch_size : int
        每个 Patch 包含的 SNP 数量。
    chunk_size : int
        单个分块内的总 SNP 数。
    d_model : int
        Transformer 隐层维度（Patch 投影的目标维度）。
    """

    def __init__(self, embed_dim=8, patch_size=20, chunk_size=12000, d_model=128):
        """初始化 SNP 特征提取器。

        Parameters
        ----------
        embed_dim : int
            SNP 类别嵌入维度。
        patch_size : int
            每个 Patch 包含的 SNP 数量。
        chunk_size : int
            单个分块内的总 SNP 数。
        d_model : int
            Transformer 隐层维度。
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = chunk_size // patch_size

        self.snp_embedding = nn.Embedding(4, embed_dim)

        self.patch_project = nn.Linear(patch_size * embed_dim, d_model)

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        """提取 Patch 特征并添加位置编码。

        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状 (B * num_chunks, chunk_size)，整数索引 0-3。

        Returns
        -------
        torch.Tensor
            嵌入后的 Patch 特征，形状 (B * num_chunks, num_patches, d_model)。
        """
        B_chunks, L = x.shape

        # 1. SNP 类别嵌入: (B * num_chunks, chunk_size) -> (B * num_chunks, chunk_size, embed_dim)
        x = self.snp_embedding(x)

        # 2. 划分为 Patches: (B * num_chunks, chunk_size, embed_dim) -> (B * num_chunks, num_patches, patch_size * embed_dim)
        x = x.view(B_chunks, self.num_patches, self.patch_size * x.shape[-1])

        # 3. 线性投影: (B * num_chunks, num_patches, patch_size * embed_dim) -> (B * num_chunks, num_patches, d_model)
        x = self.patch_project(x)

        # 4. 添加可学习位置编码
        x = x + self.pos_embedding
        return x


class ConvAE(nn.Module):
    """卷积自编码器：Masked Autoencoder + 表型预测联合训练。

    通过自监督掩码重建学习 SNP 序列的压缩表征，
    同时使用表型预测头进行监督学习，实现表征与下游任务的联合优化。

    Parameters
    ----------
    chunk_size : int
        单个分块内的总 SNP 数。
    patch_size : int
        每个 Patch 包含的 SNP 数量。
    d_model : int
        Transformer 隐层维度。
    d_ext : int
        每个分块的压缩特征维度（Bottleneck 输出维度）。
    num_chunks : int
        分块总数。
    """

    def __init__(
        self, chunk_size=12000, patch_size=20, d_model=128, d_ext=166, num_chunks=300
    ):
        """初始化 ConvAE 模型。

        Parameters
        ----------
        chunk_size : int
            单个分块内的总 SNP 数。
        patch_size : int
            每个 Patch 包含的 SNP 数量。
        d_model : int
            Transformer 隐层维度。
        d_ext : int
            每个分块的压缩特征维度。
        num_chunks : int
            分块总数。
        """
        super().__init__()
        self.num_chunks = num_chunks
        self.patch_size = patch_size
        self.num_patches = chunk_size // patch_size
        self.d_model = d_model
        self.d_ext = d_ext

        # ==============================================================================
        # 1. 特征输入与 Patch 化模块
        # ==============================================================================
        self.feature_extractor = SNPFeatureExtractor(
            patch_size=patch_size, chunk_size=chunk_size, d_model=d_model
        )

        # ==============================================================================
        # 2. Transformer 编码器 (Encoder)
        # ==============================================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # ==============================================================================
        # 3. 压缩映射层 (Bottleneck)
        # ==============================================================================
        self.compress_layer = nn.Sequential(
            nn.Linear(d_model, d_ext), nn.LayerNorm(d_ext), nn.ReLU()
        )

        # ==============================================================================
        # 4. Transformer 解码器 (Decoder)
        # ==============================================================================
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.normal_(self.decoder_pos_embedding, std=0.02)

        # 重建输出：每个 SNP 的 4 类预测 (20 个 SNP * 4 类 = 80)
        self.reconstruct_head = nn.Linear(d_model, patch_size * 4)

        # ==============================================================================
        # 5. 表型预测引导分支
        # ==============================================================================
        self.phenotype_head = nn.Sequential(
            nn.Linear(num_chunks * d_ext, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 14),
        )

    def random_masking(self, x, mask_ratio=0.5):
        """对 Patch 维度进行随机掩码（MAE 风格）。

        Parameters
        ----------
        x : torch.Tensor
            输入 Patch 序列，形状 (B_chunks, num_patches, d_model)。
        mask_ratio : float
            掩码比例，范围 (0, 1)。

        Returns
        -------
        x_masked : torch.Tensor
            保留的可见 Patch，形状 (B_chunks, len_keep, d_model)。
        mask : torch.Tensor
            二进制掩码，形状 (B_chunks, num_patches)，1 表示被掩码。
        ids_restore : torch.Tensor
            用于还原顺序的索引，形状 (B_chunks, num_patches)。
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio=0.5):
        """ConvAE 前向传播：SNP 特征提取 → 掩码 → 编码 → 压缩 → 表型预测 → 重建。

        Parameters
        ----------
        x : torch.Tensor
            原始 SNP 输入，整数索引 0-3，形状 (batch_size, num_chunks, chunk_size)。
        mask_ratio : float
            训练时的掩码比例，推理时设为 0.0。

        Returns
        -------
        compressed_final_X : torch.Tensor
            压缩特征矩阵，形状 (batch_size, num_chunks, d_ext)。
        pred_phenotype : torch.Tensor
            表型预测值，形状 (batch_size, n_times)。
        pred_snps : torch.Tensor | None
            SNP 重建 logits，形状 (B * num_chunks, num_patches, patch_size, 4)。
            推理时为 None。
        mask : torch.Tensor | None
            掩码矩阵，推理时为 None。
        """
        B, Num_Chunks, L = x.shape

        # 合并 batch 和 chunk 维度: (B, num_chunks, chunk_size) -> (B * num_chunks, chunk_size)
        x_flattened = x.view(B * Num_Chunks, L)

        # 1. 提取 Patch 特征: (B * num_chunks, chunk_size) -> (B * num_chunks, num_patches, d_model)
        features = self.feature_extractor(x_flattened)

        # 2. 掩码操作（仅训练时启用）
        if self.training and mask_ratio > 0.0:
            x_encoder, mask, ids_restore = self.random_masking(features, mask_ratio)
        else:
            x_encoder = features
            mask, ids_restore = None, None

        # 3. Transformer 编码: (B * num_chunks, len_keep, d_model) -> (B * num_chunks, len_keep, d_model)
        x_encoded = self.encoder(x_encoder)

        # 4. 压缩特征提取
        # 对可见 Patch 取均值: (B * num_chunks, len_keep, d_model) -> (B * num_chunks, d_model)
        pooled_feat = x_encoded.mean(dim=1)
        # 映射到压缩维度: (B * num_chunks, d_model) -> (B * num_chunks, d_ext)
        compressed_chunk_feat = self.compress_layer(pooled_feat)

        # 还原 batch 维度: (B * num_chunks, d_ext) -> (batch_size, num_chunks, d_ext)
        compressed_final_X = compressed_chunk_feat.view(B, Num_Chunks, self.d_ext)

        # 5. 表型预测
        # 展平所有分块: (batch_size, num_chunks * d_ext) -> (batch_size, n_times)
        phenotype_input = compressed_final_X.view(B, -1)
        pred_phenotype = self.phenotype_head(phenotype_input)

        # 6. 解码器重建分支（仅训练时计算重建 Loss）
        pred_snps = None
        if self.training and ids_restore is not None:
            # 补齐掩码占位符: (B * num_chunks, len_keep, d_model) + (B * num_chunks, num_masked, d_model) -> (B * num_chunks, num_patches, d_model)
            mask_tokens = self.mask_token.repeat(
                x_encoded.shape[0], ids_restore.shape[1] - x_encoded.shape[1], 1
            )
            x_concat = torch.cat([x_encoded, mask_tokens], dim=1)

            # 还原物理顺序
            x_decoder_input = torch.gather(
                x_concat,
                dim=1,
                index=ids_restore.unsqueeze(-1).repeat(1, 1, x_encoded.shape[-1]),
            )

            # 添加解码器位置编码: (B * num_chunks, num_patches, d_model)
            x_decoder_input = x_decoder_input + self.decoder_pos_embedding

            # 解码: (B * num_chunks, num_patches, d_model)
            x_decoded = self.decoder(x_decoder_input, memory=x_encoded)

            # SNP 分类预测: (B * num_chunks, num_patches, d_model) -> (B * num_chunks, num_patches, patch_size * 4)
            logits = self.reconstruct_head(x_decoded)

            # 还原为 (B * num_chunks, num_patches, patch_size, 4)
            pred_snps = logits.view(
                B * Num_Chunks, self.num_patches, self.patch_size, 4
            )

        return compressed_final_X, pred_phenotype, pred_snps, mask
