"""反归一化预测矩阵的 E3 指标。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from utils.Calculate_Metrics import METRICS_PROTOCOL, compute_metrics_vectorized


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def compute_prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    train_target: np.ndarray,
    time_coordinates: np.ndarray,
) -> dict[str, Any]:
    """按冻结协议计算整体和逐时间点指标。"""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    train_target = np.asarray(train_target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction/target 必须是形状相同的二维矩阵")
    if train_target.ndim != 2 or train_target.shape[1] != target.shape[1]:
        raise ValueError("train_target 与测试矩阵的时间维必须一致")
    if len(time_coordinates) != target.shape[1]:
        raise ValueError("time_coordinates 长度不一致")
    scale = float(np.percentile(train_target, 95) - np.percentile(train_target, 5))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("训练表型 P95-P5 必须为正的有限值")

    raw = compute_metrics_vectorized(prediction, target)
    overall = {
        "mse": _finite(raw["mse"]),
        "rmse": _finite(raw["rmse"]),
        "mae": _finite(raw["mae"]),
        "pcc": _finite(raw["pcc"]),
        "time_adjusted_r2": _finite(raw["r2"]),
        "nrmse": _finite(raw["rmse"] / scale),
        "pcc_valid_timepoints": int(raw["pcc_valid_count"]),
        "pcc_total_timepoints": int(raw["pcc_total_count"]),
        "train_p95_p5": scale,
        "test_individual_count": int(target.shape[0]),
    }
    by_time = []
    for index, coordinate in enumerate(time_coordinates):
        pred = prediction[:, index]
        truth = target[:, index]
        error = pred - truth
        rmse = float(np.sqrt(np.mean(np.square(error))))
        pred_sd = float(np.std(pred))
        truth_sd = float(np.std(truth))
        pcc = None
        if pred_sd > 1e-12 and truth_sd > 1e-12:
            pcc = _finite(np.corrcoef(pred, truth)[0, 1])
        spearman = None
        if pred_sd > 1e-12 and truth_sd > 1e-12:
            spearman = _finite(spearmanr(pred, truth).statistic)
        by_time.append(
            {
                "time_index": index,
                "time": float(coordinate),
                "rmse": rmse,
                "nrmse": rmse / scale,
                "mae": float(np.mean(np.abs(error))),
                "pcc": pcc,
                "spearman": spearman,
                "valid_n": int(len(truth)),
            }
        )
    return {
        "metrics_protocol": METRICS_PROTOCOL,
        "normalization": "original_phenotype_scale",
        "overall": overall,
        "by_time": by_time,
    }
