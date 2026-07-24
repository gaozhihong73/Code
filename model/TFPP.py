import math

import torch
import torch.nn.functional as F
from torch import nn


class LinearSelfAttention(nn.Module):
    """
    线性自注意力模块实现，基于核函数的线性注意力机制，相比传统注意力计算更高效。
    该模块使用ELU+1作为核函数，实现了O(N)时间复杂度的注意力计算。
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        初始化线性自注意力模块
        参数:
            d_model: 模型的输入维度
            num_heads: 注意力头的数量
            dropout: dropout概率, 默认为0.1
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # RoPE 旋转位置编码: 预计算频率参数
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def feature_map(self, x):
        return F.elu(x) + 1

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, q, k, seq_len):
        device = q.device
        position = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = torch.outer(position, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, 1, seq_len, -1)
        sin = emb.sin().view(1, 1, seq_len, -1)
        q_embed = q * cos + self._rotate_half(q) * sin
        k_embed = k * cos + self._rotate_half(k) * sin
        return q_embed, k_embed

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # 1. 线性映射 & 分头 -> [B, N, H, D] -> [B, H, N, D]
        q = self.linear_q(x).view(B, N, H, D).transpose(1, 2)
        k = self.linear_k(x).view(B, N, H, D).transpose(1, 2)
        v = self.linear_v(x).view(B, N, H, D).transpose(1, 2)

        # 0. 旋转位置编码 (RoPE): 在核函数之前注入相对位置信息
        q, k = self._apply_rope(q, k, N)

        # 2. 应用核函数
        q = self.feature_map(q)
        k = self.feature_map(k)

        # 3. 线性注意力核心计算 Q(K^T V)
        # -----------------------------------------------------------
        # 传统 Attention: softmax(Q K^T) V  -> 先算 N*N (爆炸)
        # 线性 Attention: Q (K^T V)         -> 先算 D*D (极快)
        # -----------------------------------------------------------

        # (1) 计算 KV 聚合矩阵: K^T [B, H, D, N] @ V [B, H, N, D] -> [B, H, D, D]
        # 这个矩阵只有 D*D 大小 (比如 32*32)，非常小，且包含了所有序列的信息
        kv = torch.matmul(k.transpose(-2, -1), v)

        # (2) 计算归一化因子 Z: Q [B, H, N, D] @ K_sum [B, H, D, 1] -> [B, H, N, 1]
        k_sum = k.sum(dim=-2, keepdim=True).transpose(
            -2, -1
        )  # Sum over N -> [B, H, D, 1]
        z = torch.matmul(q, k_sum) + 1e-6  # 避免除零

        # (3) 计算最终输出: Q @ KV -> [B, H, N, D]
        # 此时只需 O(N) 的复杂度
        output = torch.matmul(q, kv) / z

        # 4. 恢复维度 [B, N, d_model]
        output = output.transpose(1, 2).contiguous().reshape(B, N, C)

        return self.out_proj(self.dropout(output))


class LinearTransformerLayer(nn.Module):
    """
    使用线性注意力的 Transformer Encoder 层
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = LinearSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN (保持不变)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-Norm 结构 (通常更稳定)
        x2 = self.norm1(x)
        x = x + self.dropout(self.attn(x2))

        x2 = self.norm2(x)
        x = x + self.dropout(self.ffn(x2))
        return x


class LinearTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LinearTransformerLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ==============================================================================
# 2. 辅助组件 (保持原有逻辑)
# ==============================================================================
class LegendreWideLayer(nn.Module):
    """
    【增强型宽通路】高维 SNP → BatchNorm → 软特征选择 → Linear 系数投影 → Legendre 基矩阵。
    base_linear 负责表型均值 (0 阶), dynamic_linear 负责时间动态 (高阶)。
    """

    def __init__(
        self,
        snp_dim,
        num_phenotypes,
        time_coordinates,
        degree=3,
        dropout=0.3,
    ):
        super().__init__()
        t_raw = torch.tensor(time_coordinates, dtype=torch.float32)
        t_min, t_max = t_raw.min(), t_raw.max()
        if t_max - t_min > 1e-6:
            t_norm = 2 * (t_raw - t_min) / (t_max - t_min) - 1
        else:
            t_norm = torch.zeros_like(t_raw)

        self.num_phenotypes = num_phenotypes
        self.num_time_points = len(time_coordinates)
        self.degree = degree

        t_basis = torch.zeros(degree + 1, len(t_norm))
        t_basis[0] = 1.0
        if degree >= 1:
            t_basis[1] = t_norm
        for n in range(2, degree + 1):
            term1 = (2 * n - 1) * t_norm * t_basis[n - 1]
            term2 = (n - 1) * t_basis[n - 2]
            t_basis[n] = (term1 - term2) / n
        self.register_buffer("t_basis", t_basis)

        self.dropout = nn.Dropout(p=dropout)
        self.input_norm = nn.BatchNorm1d(snp_dim, affine=False)

        # 逐 SNP 软特征选择
        self.snp_importance = nn.Parameter(torch.ones(1, snp_dim))

        # 解耦的多项式系数投影
        self.base_linear = nn.Linear(snp_dim, num_phenotypes)
        if degree > 0:
            self.dynamic_linear = nn.Linear(snp_dim, num_phenotypes * degree)

        self._init_enhanced_weights()

    def forward(self, x):
        x = self.input_norm(x)
        x = x * self.snp_importance
        x = self.dropout(x)

        base_coeffs = self.base_linear(x).unsqueeze(-1)

        if self.degree > 0:
            dyn_coeffs = self.dynamic_linear(x).view(
                -1, self.num_phenotypes, self.degree
            )
            coeffs = torch.cat([base_coeffs, dyn_coeffs], dim=-1)
        else:
            coeffs = base_coeffs

        trend = torch.matmul(coeffs, self.t_basis)
        return trend

    def _init_enhanced_weights(self):
        # BatchNorm1d(affine=False) 无学习参数, 无需初始化

        nn.init.normal_(self.base_linear.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.base_linear.bias, 0.0)
        if self.degree > 0:
            nn.init.normal_(self.dynamic_linear.weight, mean=0.0, std=1e-4)
            nn.init.constant_(self.dynamic_linear.bias, 0.0)


class TimeAwarePredictor(nn.Module):
    """
    [重构版] Deep Path 的预测头
    """

    def __init__(self, d_model, time_coordinates, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.time_coordinates = time_coordinates

        # 1. 时间编码 (保持不变)
        init_encoding = self._get_real_time_encoding()
        self.time_queries = nn.Parameter(init_encoding)

        # 2. 交叉注意力 (Cross Attention)
        # batch_first=True: 输入输出格式为 [Batch, Seq_Len, Feature]
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN (保持不变)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # 3. 基础预测层
        # 输出原始的预测值
        self.output_proj = nn.Linear(d_model, 1, bias=True)

    def _get_real_time_encoding(self):
        """保持原有的时间编码生成逻辑不变"""
        position = torch.tensor(self.time_coordinates, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(len(self.time_coordinates), self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, genome_features, return_attn=False):
        """
        Args:
            genome_features: [Batch, Genome_Len, D_Model]
            return_attn: bool, 是否返回注意力权重矩阵 (用于绘图)
        """
        B, L, D = genome_features.shape

        # 扩展时间查询向量到 Batch 维度
        queries = self.time_queries.repeat(B, 1, 1)

        # 1. 执行 Cross Attention
        # need_weights=True: 必须开启以获取权重
        # average_attn_weights=False: [关键技巧] 设置为 False 可以返回 [Batch, Heads, Time, Genome]
        # 这样您可以单独分析每一个头的注意力，选择最清晰的那个头来画图，而不是看混乱的平均值
        attn_out, attn_weights = self.cross_attn(
            query=queries,
            key=genome_features,
            value=genome_features,
            need_weights=True,
            average_attn_weights=False,  # PyTorch 1.11+ 支持。如果是旧版本，默认返回平均值
        )

        # 残差连接与归一化
        x = self.norm1(queries + attn_out)
        x = self.norm2(x + self.ffn(x))  # x shape: [Batch, Time_Len, D_Model]

        # 2. 计算基础预测值
        final_out = self.output_proj(x).squeeze(-1)  # [Batch, Time_Len]

        if return_attn:
            # 返回: 预测值, 注意力权重
            # 权重形状: [Batch, Num_Heads, Time_Len, Genome_Len]
            return final_out, attn_weights

        return final_out


# ==============================================================================
# 3. 主模型 TFPP (Linear Version)
# ==============================================================================
class TFPP(nn.Module):
    def __init__(
        self,
        snp_dim,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        num_embeddings,
        time_coordinates,
        dropout,
        phenotype_names,
        degree=3,
        ablation_mode="none",
    ):
        super().__init__()
        self.num_phenotypes = len(phenotype_names)
        self.time_points = len(time_coordinates)

        self.ablation_mode = ablation_mode

        # -----------------------------------------------------------
        # [修改点 1] Embedding: 不压缩，全长输入
        # -----------------------------------------------------------
        # 输入 0,1,2 -> 向量
        self.snp_emb = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=d_model
        )

        # 局部上下文卷积: 捕获邻近 SNP 的 haplotype block 模式
        self.local_conv = nn.Conv1d(
            d_model, d_model, kernel_size=7, padding=3, groups=d_model
        )

        # -----------------------------------------------------------
        # [修改点 2] Encoder: 替换为 Linear Transformer Encoder
        # -----------------------------------------------------------
        self.encoder = LinearTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.predictor = nn.ModuleDict(
            {
                name: TimeAwarePredictor(
                    d_model, time_coordinates, num_heads, dropout=dropout
                )
                for name in phenotype_names
            }
        )

        # Wide Path
        self.wide_layer = LegendreWideLayer(
            snp_dim=snp_dim,
            num_phenotypes=self.num_phenotypes,
            time_coordinates=time_coordinates,
            degree=degree,
            dropout=dropout,
        )

        # [新增] 动态门控网络
        # 输入：d_model (深通路的特征维度)
        # 输出：1 (全局权重) 或 num_phenotypes (每个表型独立的权重)
        self.gating_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.num_phenotypes),  # 为每个表型单独生成权重
            nn.Sigmoid(),  # 关键！把权重限制在 0~1 之间
        )

        self._init_weights()

    def forward(self, x):
        """
        x: [Batch, SNP_Dim]
        """
        # 1. 宽通路 (Linear)
        x_norm = x / 2.0
        out_wide = self.wide_layer(x_norm)

        # 只输出宽通路的结果
        if self.ablation_mode == "wide":
            return out_wide

        # 2. 深通路 (Linear Attention)
        # [B, 50000] -> [B, 50000, D]
        x_emb = self.snp_emb(x.long())  # [B, N, d_model]
        # 局部卷积: 为每个位点注入 ±3 邻居的 haplotype 上下文
        x_local = x_emb.transpose(1, 2)  # [B, d_model, N]
        x_local = F.gelu(self.local_conv(x_local))  # [B, d_model, N]
        x_deep = x_emb + x_local.transpose(1, 2)  # 残差连接，保留原始位点值信息
        # RoPE 已在 LinearSelfAttention 内部注入相对位置信息，无需额外 PE
        # Linear Attention 处理 50k 序列，显存占用 O(N)，速度极快
        x_deep = self.encoder(x_deep)

        # 预测头
        deep_outputs = []
        for name in self.predictor:
            # TimeAwarePredictor 内部做 Cross Attention 聚合
            pred = self.predictor[name](x_deep)
            deep_outputs.append(pred)

        out_deep = torch.stack(deep_outputs, dim=1)

        # 只输出深通路的结果
        if self.ablation_mode == "deep":
            return out_deep

        # 计算门控
        # 我们需要先做一个全局池化拿到 "特征向量" 用来计算门控
        global_feat = x_deep.mean(dim=1)  # [B, D]
        # =======================================================
        # [修改点] 动态计算门控值
        # =======================================================
        # gate shape: [B, P] -> 扩充到 [B, P, 1] 以匹配时间维度
        gate = self.gating_net(global_feat).unsqueeze(-1)

        # 融合：Wide + Gate * Deep
        if self.ablation_mode == "add":
            # 去掉门控 (直接相加)
            final_out = out_wide + out_deep
        else:
            # 完整模型 (Full Model)
            final_out = out_wide + gate * out_deep

        return final_out

    def _init_weights(self):
        """深通路 + 门控网络的初始化。
        宽通路由 LegendreWideLayer._init_enhanced_weights 自行管理, 此处不干预。
        """
        for name, m in self.named_modules():
            # 宽通路全部跳过 —— 已由 _init_enhanced_weights() 处理
            if "wide_layer" in name:
                continue

            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # === Gating Net 特殊处理 ===
        for name, m in self.gating_net.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-5)
                if m.bias is not None and m.out_features == self.num_phenotypes:
                    nn.init.constant_(m.bias, -5.0)  # sigmoid(-5) ≈ 0.007
                elif m.bias is not None:
                    nn.init.zeros_(m.bias)
