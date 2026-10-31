"""E3 单时间点深度模型的网络结构。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DNNGPSNPModel(nn.Module):
    """三层 CNN、单个 BatchNorm、双 Dropout 的 SNP-only DNNGP 适配。"""

    def __init__(
        self,
        *,
        channels: tuple[int, int, int] = (8, 16, 32),
        pooled_length: int = 16,
        dense_units: int = 64,
        dropout_input: float = 0.25,
        dropout_dense: float = 0.25,
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.conv1 = nn.Conv1d(1, c1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(c3)
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(pooled_length)
        self.dropout_input = nn.Dropout(dropout_input)
        self.dense = nn.Linear(c3 * pooled_length, dense_units)
        self.dropout_dense = nn.Dropout(dropout_dense)
        self.output = nn.Linear(dense_units, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.pool(F.relu(self.conv1(values)))
        values = self.pool(F.relu(self.conv2(values)))
        values = F.relu(self.batch_norm(self.conv3(values)))
        values = self.adaptive_pool(values).flatten(1)
        values = self.dropout_input(values)
        values = F.relu(self.dense(values))
        return self.output(self.dropout_dense(values)).squeeze(-1)


class ISRU(nn.Module):
    """DLGWAS 使用的 inverse square root unit。"""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values / torch.sqrt(1.0 + self.alpha * values.square())


class SamePadConv1d(nn.Module):
    """对偶数卷积核显式执行左右非对称 same padding。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        total = self.kernel_size - 1
        left = total // 2
        return self.conv(F.pad(values, (left, total - left)))


class DLGWASModel(nn.Module):
    """核大小 4/20 的双流残差 CNN。"""

    def __init__(
        self,
        *,
        channels: int = 16,
        pooled_length: int = 16,
        dropout: float = 0.75,
    ):
        super().__init__()
        self.stacked_short = SamePadConv1d(4, channels, kernel_size=4)
        self.stacked_long = SamePadConv1d(channels, channels, kernel_size=20)
        self.residual = SamePadConv1d(4, channels, kernel_size=4)
        self.feature_conv = SamePadConv1d(channels, channels * 2, kernel_size=4)
        self.activation = ISRU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(pooled_length)
        self.output = nn.Linear(channels * 2 * pooled_length, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        stacked = self.activation(self.stacked_short(values))
        stacked = self.activation(self.stacked_long(self.dropout(stacked)))
        residual = self.activation(self.residual(values))
        values = self.activation(self.feature_conv(stacked + residual))
        values = self.pool(self.dropout(values)).flatten(1)
        return self.output(values).squeeze(-1)


class WheatGPModel(nn.Module):
    """五段共享三层 CNN，随后以 LSTM 聚合局部特征。"""

    def __init__(
        self,
        *,
        segment_count: int = 5,
        segment_tokens: int = 16,
        hidden_size: int = 128,
        cnn_dropout: float = 0.5,
        lstm_dropout: float = 0.3,
    ):
        super().__init__()
        self.segment_count = segment_count
        self.segment_tokens = segment_tokens
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(2, 4, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=9, padding="same"),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(segment_tokens)
        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(lstm_dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        segments = torch.tensor_split(values, self.segment_count, dim=-1)
        encoded = [self.pool(self.cnn(segment)) for segment in segments]
        sequence = torch.cat(encoded, dim=-1).transpose(1, 2)
        sequence, _ = self.lstm(sequence)
        return self.output(self.dropout(sequence[:, -1])).squeeze(-1)


class CropformerModel(nn.Module):
    """CNN tokenization + 多头自注意力的 Cropformer 适配。"""

    def __init__(
        self,
        *,
        input_snps: int,
        embedding_dim: int = 32,
        num_heads: int = 4,
        max_tokens: int = 128,
        hidden_dropout: float = 0.5,
        attention_dropout: float = 0.5,
    ):
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("Cropformer embedding_dim 必须被 num_heads 整除")
        self.token_count = min(max_tokens, max(1, (input_snps + 1) // 2))
        self.cnn = nn.Conv1d(
            1, embedding_dim, kernel_size=3, stride=2, padding=1
        )
        self.pool = nn.AdaptiveAvgPool1d(self.token_count)
        self.position = nn.Parameter(torch.zeros(1, self.token_count, embedding_dim))
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(hidden_dropout)
        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self.pool(F.relu(self.cnn(values))).transpose(1, 2)
        tokens = tokens + self.position
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + self.dropout(attended))
        tokens = self.norm2(tokens + self.feed_forward(tokens))
        return self.output(self.dropout(tokens.mean(dim=1))).squeeze(-1)
