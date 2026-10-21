"""TFPP 主模型定义：Wide & Deep 架构的时序基因组表型预测模型。

本模块实现 TFPP (Time-aware Feature Projection & Prediction) 的完整网络结构，包含：
- LinearSelfAttention: 基于核函数的线性自注意力 (O(D²) 复杂度，避免 O(N²) 显存瓶颈)
- LinearTransformerEncoder: 多层 Pre-Norm 线性 Transformer 编码器 (Deep 通路)
- LegendreWideLayer: 勒让德多项式宽通路，将 SNP 直接映射为时间曲线 (Wide 通路)
- TimeAwarePredictor: 交叉注意力预测头，聚合编码特征后预测各时间点表型
- TFPP: 主模型，通过动态门控网络融合 Wide & Deep 输出
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from utils.Genotype import model_code_tensor, tfpp_input_views


class LinearSelfAttention(nn.Module):
    """基于核函数的线性自注意力模块。

    使用 ELU+1 作为核函数，将传统 O(N²) 复杂度的注意力计算
    转化为 O(N·D²) 的线性注意力，适用于超长序列处理。

    Parameters
    ----------
    d_model : int
        模型隐藏维度，必须能被 num_heads 整除。
    num_heads : int
        注意力头数量。
    dropout : float
        dropout 比率。
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        """初始化线性自注意力模块。

        Parameters
        ----------
        d_model : int
            模型隐藏维度，必须能被 num_heads 整除。
        num_heads : int
            注意力头数量。
        dropout : float
            dropout 比率，默认 0.1。
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

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
        """核函数: ELU + 1，保证输出非负。"""
        return F.elu(x) + 1

    @staticmethod
    def _rotate_half(x):
        """RoPE 辅助函数：将 x 的后半部分取反后与前半部分拼接。"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, q, k, seq_len):
        """对 Q/K 应用旋转位置编码 (RoPE)。

        Parameters
        ----------
        q : torch.Tensor
            查询张量，形状 (B, H, N, D_head)。
        k : torch.Tensor
            键张量，形状 (B, H, N, D_head)。
        seq_len : int
            序列长度。

        Returns
        -------
        q_embed : torch.Tensor
            旋转后的查询张量。
        k_embed : torch.Tensor
            旋转后的键张量。
        """
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
        """线性自注意力前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入序列，形状 (batch_size, seq_len, d_model)。

        Returns
        -------
        torch.Tensor
            注意力输出，形状 (batch_size, seq_len, d_model)。
        """
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # 1. 线性映射 & 分头: (B, N, d_model) -> (B, N, H, D_head) -> (B, H, N, D_head)
        q = self.linear_q(x).view(B, N, H, D).transpose(1, 2)
        k = self.linear_k(x).view(B, N, H, D).transpose(1, 2)
        v = self.linear_v(x).view(B, N, H, D).transpose(1, 2)

        # 2. 注入 RoPE 位置信息
        q, k = self._apply_rope(q, k, N)

        # 3. 核函数
        q = self.feature_map(q)
        k = self.feature_map(k)

        # ==============================================================================
        # 4. 线性注意力核心: Q (K^T V)
        #    传统 Attention: softmax(Q K^T) V → 先算 N*N (O(N²)，序列长时显存爆炸)
        #    线性 Attention: Q (K^T V)       → 先算 D*D (O(D²)，与序列长度 N 无关)
        # ==============================================================================

        # (1) KV 聚合: K^T @ V -> (B, H, D_head, N) @ (B, H, N, D_head) -> (B, H, D_head, D_head)
        kv = torch.matmul(k.transpose(-2, -1), v)

        # (2) 归一化因子: Q @ K_sum -> (B, H, N, D_head) @ (B, H, D_head, 1) -> (B, H, N, 1)
        k_sum = k.sum(dim=-2, keepdim=True).transpose(-2, -1)
        z = torch.matmul(q, k_sum) + 1e-6

        # (3) 最终输出: Q @ KV / Z -> (B, H, N, D_head)
        output = torch.matmul(q, kv) / z

        # 5. 恢复维度: (B, H, N, D_head) -> (B, N, d_model)
        output = output.transpose(1, 2).contiguous().reshape(B, N, C)

        return self.out_proj(self.dropout(output))


class LinearTransformerLayer(nn.Module):
    """使用线性注意力的 Transformer Encoder 层（Pre-Norm 结构）。

    Parameters
    ----------
    d_model : int
        模型隐藏维度。
    num_heads : int
        注意力头数量。
    d_ff : int
        前馈网络隐藏层维度。
    dropout : float
        dropout 比率。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """初始化 Transformer Encoder 层。

        Parameters
        ----------
        d_model : int
            模型隐藏维度。
        num_heads : int
            注意力头数量。
        d_ff : int
            前馈网络隐藏层维度。
        dropout : float
            dropout 比率。
        """
        super().__init__()
        self.attn = LinearSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Pre-Norm Transformer 层前向计算。

        Parameters
        ----------
        x : torch.Tensor
            输入，形状 (batch_size, seq_len, d_model)。

        Returns
        -------
        torch.Tensor
            输出，形状 (batch_size, seq_len, d_model)。
        """
        x2 = self.norm1(x)
        x = x + self.dropout(self.attn(x2))

        x2 = self.norm2(x)
        x = x + self.dropout(self.ffn(x2))
        return x


class LinearTransformerEncoder(nn.Module):
    """线性 Transformer 编码器栈。

    Parameters
    ----------
    d_model : int
        模型隐藏维度。
    num_heads : int
        注意力头数量。
    num_layers : int
        编码器层数。
    d_ff : int
        前馈网络隐藏层维度。
    dropout : float
        dropout 比率。
    """

    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        """初始化编码器栈。

        Parameters
        ----------
        d_model : int
            模型隐藏维度。
        num_heads : int
            注意力头数量。
        num_layers : int
            编码器层数。
        d_ff : int
            前馈网络隐藏层维度。
        dropout : float
            dropout 比率。
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LinearTransformerLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """逐层执行 Transformer 编码。

        Parameters
        ----------
        x : torch.Tensor
            输入，形状 (batch_size, seq_len, d_model)。

        Returns
        -------
        torch.Tensor
            编码输出，形状 (batch_size, seq_len, d_model)。
        """
        for layer in self.layers:
            x = layer(x)
        return x


# ==============================================================================
# 2. 辅助组件：Legendre 多项式宽通路
# ==============================================================================
class LegendreWideLayer(nn.Module):
    """Legendre 多项式宽通路：高维 SNP → BatchNorm → 特征选择 → 系数投影 → 时间曲线。

    使用 Legendre 多项式基函数将 SNP 直接映射为时间曲线。
    base_linear 负责表型均值（0 阶），dynamic_linear 负责时间动态（高阶）。

    Parameters
    ----------
    snp_dim : int
        输入 SNP 维度。
    num_phenotypes : int
        表型数量。
    time_coordinates : list[float]
        采样时间点列表。
    degree : int
        Legendre 多项式的最高次数。
    dropout : float
        dropout 比率。
    """

    def __init__(
        self,
        snp_dim,
        num_phenotypes,
        time_coordinates,
        degree=3,
        dropout=0.3,
    ):
        """初始化 Legendre 宽通路。

        Parameters
        ----------
        snp_dim : int
            输入 SNP 维度。
        num_phenotypes : int
            表型数量。
        time_coordinates : list[float]
            采样时间点列表。
        degree : int
            Legendre 多项式的最高次数。
        dropout : float
            dropout 比率。
        """
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

        # 预计算 Legendre 基函数矩阵: (degree + 1, num_time_points)
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

        # 逐 SNP 软特征选择权重
        self.snp_importance = nn.Parameter(torch.ones(1, snp_dim))

        # 解耦的多项式系数投影：base 输出常量项，dynamic 输出高阶系数
        self.base_linear = nn.Linear(snp_dim, num_phenotypes)
        if degree > 0:
            self.dynamic_linear = nn.Linear(snp_dim, num_phenotypes * degree)

        self._init_enhanced_weights()

    def forward(self, x):
        """宽通路前向计算：SNP → Legendre 系数 → 时间曲线。

        Parameters
        ----------
        x : torch.Tensor
            SNP 矩阵，形状 (batch_size, snp_dim)。

        Returns
        -------
        torch.Tensor
            时间曲线预测值，形状 (batch_size, num_phenotypes, num_time_points)。
        """
        # BatchNorm + 特征选择: (B, snp_dim)
        x = self.input_norm(x)
        x = x * self.snp_importance
        x = self.dropout(x)

        # base 系数: (B, snp_dim) -> (B, num_phenotypes, 1)
        base_coeffs = self.base_linear(x).unsqueeze(-1)

        if self.degree > 0:
            # dynamic 系数: (B, snp_dim) -> (B, num_phenotypes, degree)
            dyn_coeffs = self.dynamic_linear(x).view(
                -1, self.num_phenotypes, self.degree
            )
            # 拼接系数: (B, num_phenotypes, degree + 1)
            coeffs = torch.cat([base_coeffs, dyn_coeffs], dim=-1)
        else:
            coeffs = base_coeffs

        # 系数 @ 基矩阵: (B, P, degree+1) @ (degree+1, T) -> (B, P, T)
        trend = torch.matmul(coeffs, self.t_basis)
        return trend

    def _init_enhanced_weights(self):
        nn.init.normal_(self.base_linear.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.base_linear.bias, 0.0)
        if self.degree > 0:
            nn.init.normal_(self.dynamic_linear.weight, mean=0.0, std=1e-4)
            nn.init.constant_(self.dynamic_linear.bias, 0.0)


class TimeAwarePredictor(nn.Module):
    """Deep Path 预测头：Cross-Attention 聚合编码特征后预测各时间点的表型值。

    Parameters
    ----------
    d_model : int
        编码器输出维度。
    time_coordinates : list[float]
        采样时间点列表。
    num_heads : int
        Cross-Attention 的注意力头数。
    dropout : float
        dropout 比率。
    """

    def __init__(self, d_model, time_coordinates, num_heads=4, dropout=0.1):
        """初始化时间感知预测头。

        Parameters
        ----------
        d_model : int
            编码器输出维度。
        time_coordinates : list[float]
            采样时间点列表。
        num_heads : int
            Cross-Attention 注意力头数。
        dropout : float
            dropout 比率。
        """
        super().__init__()
        self.d_model = d_model
        self.time_coordinates = time_coordinates

        init_encoding = self._get_real_time_encoding()
        self.time_queries = nn.Parameter(init_encoding)

        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.output_proj = nn.Linear(d_model, 1, bias=True)

    def _get_real_time_encoding(self):
        """生成基于真实采样时间点的正弦位置编码。

        Returns
        -------
        torch.Tensor
            位置编码，形状 (1, num_time_points, d_model)。
        """
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
        """执行 Cross-Attention 聚合并预测各时间点表型值。

        Parameters
        ----------
        genome_features : torch.Tensor
            编码后的基因组特征，形状 (batch_size, genome_len, d_model)。
        return_attn : bool
            是否返回注意力权重矩阵（用于可视化）。

        Returns
        -------
        torch.Tensor or tuple
            return_attn=False 时返回预测值 (batch_size, num_time_points)。
            return_attn=True 时返回 (预测值, 注意力权重) 元组，
            权重形状为 (batch_size, num_heads, num_time_points, genome_len)。
        """
        B, L, D = genome_features.shape

        # 扩展时间查询到 batch 维度: (1, T, d_model) -> (B, T, d_model)
        queries = self.time_queries.repeat(B, 1, 1)

        # Cross Attention: queries 与 genome_features 交叉注意
        # average_attn_weights=False 保留每个头独立的权重，便于分析不同头的行为
        attn_out, attn_weights = self.cross_attn(
            query=queries,
            key=genome_features,
            value=genome_features,
            need_weights=True,
            average_attn_weights=False,
        )

        x = self.norm1(queries + attn_out)
        x = self.norm2(x + self.ffn(x))  # (B, T, d_model)

        # 输出投影: (B, T, d_model) -> (B, T, 1) -> (B, T)
        final_out = self.output_proj(x).squeeze(-1)

        if return_attn:
            return final_out, attn_weights

        return final_out


# ==============================================================================
# 3. 主模型 TFPP (Transformer + Legendre Wide & Deep Architecture)
# ==============================================================================
class TFPP(nn.Module):
    """TFPP 模型：Wide & Deep 架构的时序表型预测模型。

    Wide 通路使用 Legendre 多项式直接从 SNP 预测时间曲线；
    Deep 通路使用 Linear Transformer 编码全长序列，
    经 Cross Attention 聚合后预测各时间点表型；
    两路通过动态门控网络融合。

    Parameters
    ----------
    snp_dim : int
        输入 SNP 维度。
    d_model : int
        Transformer 隐藏维度。
    num_heads : int
        注意力头数量。
    num_layers : int
        Deep 通路编码器层数。
    d_ff : int
        前馈网络隐藏层维度。
    num_embeddings : int
        SNP 类别嵌入维度。
    time_coordinates : list[float]
        采样时间点列表。
    dropout : float
        dropout 比率。
    phenotype_names : list[str]
        表型名称列表。
    degree : int
        Wide 通路 Legendre 多项式次数。
    ablation_mode : str
        消融实验模式 ("none" / "wide" / "deep" / "add")。
    """

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
        wide_input_mode="dosage_v1",
        wide_fill_values=None,
    ):
        """初始化 TFPP 模型。

        Parameters
        ----------
        snp_dim : int
            输入 SNP 维度。
        d_model : int
            Transformer 隐藏维度。
        num_heads : int
            注意力头数量。
        num_layers : int
            编码器层数。
        d_ff : int
            前馈网络隐藏层维度。
        num_embeddings : int
            SNP 类别嵌入维度。
        time_coordinates : list[float]
            采样时间点列表。
        dropout : float
            dropout 比率。
        phenotype_names : list[str]
            表型名称列表。
        degree : int
            Legendre 多项式次数。
        ablation_mode : str
            消融模式。
        """
        super().__init__()
        self.num_phenotypes = len(phenotype_names)
        self.time_points = len(time_coordinates)

        self.ablation_mode = ablation_mode
        if wide_input_mode not in ("dosage_v1", "legacy_categorical"):
            raise ValueError("未知的 Wide 输入语义")
        self.wide_input_mode = wide_input_mode
        if wide_input_mode == "dosage_v1":
            fill = torch.as_tensor(
                [] if wide_fill_values is None else wide_fill_values, dtype=torch.float32
            ).clone()
            if fill.numel() and (fill.shape != (snp_dim,) or not torch.isfinite(fill).all()
                                 or not ((fill >= 0) & (fill <= 2)).all()):
                raise ValueError("Wide 填补参数不合法")
            self.register_buffer("wide_fill_values", fill)

        # ==============================================================================
        # Deep Path: SNP 嵌入 + 局部卷积 + Linear Transformer 编码
        # ==============================================================================
        # SNP 类别嵌入: 输入整数 0-3 -> 输出 d_model 维向量
        self.snp_emb = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=d_model
        )

        # 局部上下文卷积: 捕获邻近 SNP 的 haplotype block 模式 (kernel_size=7)
        self.local_conv = nn.Conv1d(
            d_model, d_model, kernel_size=7, padding=3, groups=d_model
        )

        self.encoder = LinearTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # 每个表型对应一个独立的预测头
        self.predictor = nn.ModuleDict(
            {
                name: TimeAwarePredictor(
                    d_model, time_coordinates, num_heads, dropout=dropout
                )
                for name in phenotype_names
            }
        )

        # ==============================================================================
        # Wide Path: Legendre 多项式宽通路
        # ==============================================================================
        self.wide_layer = LegendreWideLayer(
            snp_dim=snp_dim,
            num_phenotypes=self.num_phenotypes,
            time_coordinates=time_coordinates,
            degree=degree,
            dropout=dropout,
        )

        # ==============================================================================
        # 动态门控网络: 融合 Wide & Deep 的输出
        # ==============================================================================
        self.gating_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.num_phenotypes),
            nn.Sigmoid(),
        )

        self._init_weights()

    def forward(self, x):
        """TFPP 前向传播：Wide → Deep → Gating 融合。

        Parameters
        ----------
        x : torch.Tensor
            原始 SNP 输入（整数 0-3），形状 (batch_size, snp_dim)。

        Returns
        -------
        torch.Tensor
            表型预测值，形状 (batch_size, num_phenotypes, num_time_points)。
        """
        # ==============================================================================
        # 1. Wide Path: SNP -> Legendre 系数 -> 时间曲线
        # ==============================================================================
        model_codes = model_code_tensor(x)
        if self.ablation_mode != "deep" or self.wide_input_mode == "legacy_categorical":
            _, wide_input = tfpp_input_views(
                model_codes, getattr(self, "wide_fill_values", None),
                legacy_categorical=self.wide_input_mode == "legacy_categorical",
            )
            out_wide = self.wide_layer(wide_input)

        if self.ablation_mode == "wide":
            return out_wide

        # ==============================================================================
        # 2. Deep Path: SNP -> Embedding -> 局部卷积 -> Linear Transformer -> 预测头
        # ==============================================================================
        # SNP 嵌入: (B, snp_dim) -> (B, snp_dim, d_model)
        x_emb = self.snp_emb(model_codes)

        # 局部卷积: (B, snp_dim, d_model) -> (B, d_model, snp_dim) -> conv -> (B, snp_dim, d_model)
        x_local = x_emb.transpose(1, 2)
        x_local = F.gelu(self.local_conv(x_local))
        x_deep = x_emb + x_local.transpose(1, 2)

        # Linear Transformer 编码: (B, snp_dim, d_model) -> 同形状
        x_deep = self.encoder(x_deep)

        # 各表型独立预测头
        deep_outputs = []
        for name in self.predictor:
            pred = self.predictor[name](x_deep)
            deep_outputs.append(pred)

        out_deep = torch.stack(
            deep_outputs, dim=1
        )  # (B, num_phenotypes, num_time_points)

        if self.ablation_mode == "deep":
            return out_deep

        # ==============================================================================
        # 3. Gate 融合: final = Wide + Gate * Deep
        # ==============================================================================
        global_feat = x_deep.mean(dim=1)  # (B, d_model)

        # 门控值: (B, d_model) -> (B, num_phenotypes) -> (B, num_phenotypes, 1)
        gate = self.gating_net(global_feat).unsqueeze(-1)

        if self.ablation_mode == "add":
            final_out = out_wide + out_deep
        else:
            final_out = out_wide + gate * out_deep

        return final_out

    def _init_weights(self):
        """Deep 通路 + 门控网络权重初始化。

        Wide 通路权重由 LegendreWideLayer._init_enhanced_weights 独立管理。
        """
        for name, m in self.named_modules():
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

        # 门控网络特殊初始化：初始偏向 0（sigmoid(-5) ≈ 0.007），先让 Wide 通路主导
        for _, m in self.gating_net.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-5)
                if m.bias is not None and m.out_features == self.num_phenotypes:
                    nn.init.constant_(m.bias, -5.0)
                elif m.bias is not None:
                    nn.init.zeros_(m.bias)
