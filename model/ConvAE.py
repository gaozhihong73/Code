import torch
from torch import nn


class SNPFeatureExtractor(nn.Module):
    """
    负责将离散的 SNP 序列转换为稠密 Patch 向量，并添加位置编码
    """

    def __init__(self, embed_dim=8, patch_size=20, chunk_size=12000, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = chunk_size // patch_size  # 12000 / 20 = 600 个 patches

        # 离散类别嵌入：将 0,1,2,3 映射到 8 维连续空间
        self.snp_embedding = nn.Embedding(4, embed_dim)

        # 局部特征聚合：将一个 Patch 内的全部 SNP 级联后映射到 Transformer 隐层维度
        # 20 * 8 = 160 维 -> 128 维
        self.patch_project = nn.Linear(patch_size * embed_dim, d_model)

        # 可学习的位置编码，用于捕捉 SNP 在分块内的物理绝对位置
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        # x 形状: (B * Num_Chunks, Chunk_Size) 即 (B * 300, 12000)
        B_chunks, L = x.shape

        # 1. 类别嵌入 -> (B * 300, 12000, 8)
        x = self.snp_embedding(x)

        # 2. 划分成 Patches -> (B * 300, 600, 20 * 8)
        x = x.view(B_chunks, self.num_patches, self.patch_size * x.shape[-1])

        # 3. 线性映射 -> (B * 300, 600, 128)
        x = self.patch_project(x)

        # 4. 加入位置信息
        x = x + self.pos_embedding
        return x


class ConvAE(nn.Module):
    def __init__(
        self, chunk_size=12000, patch_size=20, d_model=128, d_ext=166, num_chunks=300
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.patch_size = patch_size
        self.num_patches = chunk_size // patch_size  # 600
        self.d_model = d_model
        self.d_ext = d_ext  # 每个分块提取的特征数：166

        # 1. 特征输入与 Patch 化模块
        self.feature_extractor = SNPFeatureExtractor(
            patch_size=patch_size, chunk_size=chunk_size, d_model=d_model
        )

        # 2. Transformer 编码器 (Encoder)
        # 只处理可见(未掩码)的 Patch，4层，4个注意力头
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 3. 压缩映射层 (Bottleneck)
        # 将整个分块的所有 Patch 特征聚合，并压缩至 166 维
        self.compress_layer = nn.Sequential(
            nn.Linear(d_model, d_ext), nn.LayerNorm(d_ext), nn.ReLU()
        )

        # 4. Transformer 解码器 (Decoder) - 仅在训练阶段重建 SNP 时使用
        # 2层，用于根据压缩表征和掩码标记还原被遮挡的信息
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # 可学习的掩码占位符 Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # 解码器位置编码
        self.decoder_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.normal_(self.decoder_pos_embedding, std=0.02)

        # 重建输出层：预测每个 Patch 里的 20 个 SNP 分别属于 4 个类别的概率
        # 输出维度: 20 * 4 = 80 维
        self.reconstruct_head = nn.Linear(d_model, patch_size * 4)

        # 5. 表型预测引导分支 (监督学习核心)
        # 输入维度: 300 个分块 * 166 维 = 49800 维 (约5万维)
        self.phenotype_head = nn.Sequential(
            nn.Linear(num_chunks * d_ext, 128),
            nn.ReLU(),
            nn.Dropout(0.4),  # 小样本强制加高 Dropout 防止过拟合
            nn.Linear(128, 14),  # 假设表型为连续型变量（回归问题）
        )

    def random_masking(self, x, mask_ratio=0.5):
        """
        对 Patch 进行随机掩码
        x: (B_chunks, 600, 128)
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # 随机生成每个位置的噪声强度
        noise = torch.rand(N, L, device=x.device)

        # 排序以获取随机索引
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 保留前 len_keep 个最不活跃的噪声点作为可见 Patch
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 生成二进制掩码矩阵 (1代表被掩码，0代表可见)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio=0.5):
        # 输入 x 形状: (B, 300, 12000)
        B, Num_Chunks, L = x.shape

        # 为了高效并行，将 Batch 和 Chunks 合并处理
        # 变换后形状: (B * 300, 12000)
        x_flattened = x.view(B * Num_Chunks, L)

        # 1. 提取 Patch 特征 -> (B * 300, 600, 128)
        features = self.feature_extractor(x_flattened)

        # 2. 掩码操作 (仅在模型训练模式下进行掩码，推理时 mask_ratio=0.0)
        if self.training and mask_ratio > 0.0:
            x_encoder, mask, ids_restore = self.random_masking(features, mask_ratio)
        else:
            x_encoder = features
            mask, ids_restore = None, None

        # 3. 编码器前向计算 -> (B * 300, 300, 128) [假设隐蔽率50%]
        # 由于只计算可见 Patch，4090 显存开销直接减半
        x_encoded = self.encoder(x_encoder)

        # 4. 提取用于后续预测模块的【压缩特征】
        # 对可见 Patch 的特征取均值 Pooling -> (B * 300, 128)
        pooled_feat = x_encoded.mean(dim=1)
        # 映射至指定的 166 维 -> (B * 300, 166)
        compressed_chunk_feat = self.compress_layer(pooled_feat)

        # 还原回标准批次形状 -> (B, 300, 166)
        # 这就是您需要的输入到后续表型预测模块的 X 矩阵
        compressed_final_X = compressed_chunk_feat.view(B, Num_Chunks, self.d_ext)

        # 5. 表型预测（联合损失引导核心）
        # 将 300 个块的特征全连接展平 -> (B, 300 * 166) 即 (B, 49800)
        phenotype_input = compressed_final_X.view(B, -1)
        pred_phenotype = self.phenotype_head(phenotype_input)

        # 6. 解码器重建分支（仅在训练时用于计算自编码重建 Loss）
        pred_snps = None
        if self.training and ids_restore is not None:
            # 补齐掩码占位符 Token -> (B * 300, 600, 128)
            mask_tokens = self.mask_token.repeat(
                x_encoded.shape[0], ids_restore.shape[1] - x_encoded.shape[1], 1
            )
            x_concat = torch.cat([x_encoded, mask_tokens], dim=1)
            # 重新排列回原物理顺序
            x_decoder_input = torch.gather(
                x_concat,
                dim=1,
                index=ids_restore.unsqueeze(-1).repeat(1, 1, x_encoded.shape[-1]),
            )

            # 加入解码器专属位置编码
            x_decoder_input = x_decoder_input + self.decoder_pos_embedding

            # 解码器计算 -> (B * 300, 600, 128)
            x_decoded = self.decoder(x_decoder_input, memory=x_encoded)

            # 预测原 SNP 的分类概率 -> (B * 300, 600, 80)
            logits = self.reconstruct_head(x_decoded)

            # 整理形状，将最后的 80 维还原为 (20个SNP, 4个分类概率)
            # 最终重建输出形状: (B * 300, 600, 20, 4)
            pred_snps = logits.view(
                B * Num_Chunks, self.num_patches, self.patch_size, 4
            )

        return compressed_final_X, pred_phenotype, pred_snps, mask
