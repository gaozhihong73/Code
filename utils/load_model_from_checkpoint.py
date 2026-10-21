"""画图脚本的公共函数: 模型加载、推理、反归一化。"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_model_from_checkpoint(
    model_dir: str | Path, device: str = "cpu", *, legacy: bool = False
):
    """统一委托严格模型加载器；旧实验必须显式 legacy=True。

    Parameters
    ----------
    model_dir : str or Path
        模型输出目录, 应包含 ``models/best_model.pt``。
    device : str, optional
        加载模型的设备, 默认 ``"cpu"``。
    legacy : bool, optional
        仅诊断性兼容旧实验，结果禁止进入论文正式汇总。

    Returns
    -------
    model : TFPP
        已加载权重并设为 eval 模式的 TFPP 实例。
    info : dict
        包含 ``snp_dim``, ``time_points``, ``degree``, ``d_model``,
        ``num_layers``, ``d_ff``, ``phenotype_names`` 等信息。
    """
    from .model_loader import load_tfpp_model

    return load_tfpp_model(model_dir, device=device, legacy=legacy)


def load_scaler_and_mean_curve(model_dir: str | Path):
    """加载 residual_global 反归一化所需的 StandardScaler 与 mean_curve。"""
    scaler_path = Path(model_dir) / "scalers" / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        pkg = pickle.load(f)
    scaler = pkg["pheno_meta"]["scalers"]["sim_pheno"]
    mean_curve = pkg["pheno_meta"]["mean_curve"]  # shape (1, 1, T)
    return scaler, mean_curve


def denormalize_predictions(
    pred_norm: np.ndarray, scaler, mean_curve: np.ndarray
) -> np.ndarray:
    """residual_global 逆归一化: zscored -> raw = scaler_inv(flat) + mean_curve。

    Parameters
    ----------
    pred_norm : np.ndarray
        模型原始输出, shape ``(batch_size, n_pheno, n_time)``。
    scaler : sklearn.preprocessing.StandardScaler
        仅含 ``mean ~= 0, std ~= sigma`` 的 StandardScaler。
    mean_curve : np.ndarray
        训练集时间均值曲线, shape ``(1, n_pheno, n_time)``。

    Returns
    -------
    np.ndarray
        反归一化后的预测值, shape ``(batch_size, n_pheno, n_time)``。
    """
    B, P, T = pred_norm.shape
    flat = pred_norm.reshape(-1, 1)
    inv = scaler.inverse_transform(flat).reshape(B, P, T)
    return inv + mean_curve


def load_test_csvs(scenario: str, p: int, n: int, h: float, rep: int):
    """加载指定实验的测试集 genotype / phenotype csv。

    Parameters
    ----------
    scenario : str
        场景标识, ``"S0"`` 或 ``"S1"``。
    p : int
        SNP 数。
    n : int
        样本数。
    h : float
        遗传力。
    rep : int
        重复编号。

    Returns
    -------
    X : np.ndarray
        基因型矩阵, shape ``(n_test, snp_dim)``, dtype ``int64``。
    y : np.ndarray
        表型矩阵, shape ``(n_test, n_time)``, dtype ``float32``。
    """
    dirname = f"P{p}_N{n}_H{h}"
    prefix = f"DynaGS_{scenario}_P{p}_H{h}_N{n}_Rep{rep:02d}"
    base = Path("dataset/simulated_data") / scenario / dirname
    if not base.exists():
        raise FileNotFoundError(f"数据目录不存在: {base}")
    geno = pd.read_csv(base / f"{prefix}_Test_Genotype.csv", index_col=0)
    pheno = pd.read_csv(base / f"{prefix}_Test_Phenotype.csv", index_col=0)
    return geno.values.astype(np.int64), pheno.values.astype(np.float32)


def run_inference(
    model, X: np.ndarray, device: str, *, artifacts=None, input_stage=None
):
    """运行模型推理并返回预测值。

    Parameters
    ----------
    model : TFPP
        已加载权重的模型实例 (eval 模式)。
    X : np.ndarray
        基因型矩阵, shape ``(n_samples, snp_dim)``。
    device : str
        推理设备 (``"cuda"`` 或 ``"cpu"``)。

    Returns
    -------
    pred_norm : np.ndarray
        模型原始输出, shape ``(n_samples, n_pheno, n_time)``。
    """
    if model.wide_input_mode == "dosage_v1":
        from .Inference import prepare_inference_input

        if artifacts is None:
            raise ValueError(
                "新格式推理必须传入加载器返回的 artifacts 和显式 input_stage"
            )
        X_t = prepare_inference_input(X, artifacts, input_stage=input_stage).to(device)
    else:
        # 显式 legacy 加载的旧画图流程保持原有类别输入约定。
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_norm = model(X_t).cpu().numpy()  # [N, 1, T]
    return pred_norm


def prepare_plot_data(
    model_dir: str | Path,
    scenario: str,
    p: int,
    n: int,
    h: float,
    rep: int,
    device: str = "cpu",
):
    """一站式: 加载模型 + scaler + 测试数据 + 运行推理 + 反归一化。

    Parameters
    ----------
    model_dir : str or Path
        模型输出目录。
    scenario : str
        场景标识, ``"S0"`` 或 ``"S1"``。
    p : int
        SNP 数。
    n : int
        样本数。
    h : float
        遗传力。
    rep : int
        重复编号。
    device : str, optional
        推理设备, 默认 ``"cpu"``。

    Returns
    -------
    pred : np.ndarray
        反归一化后的预测值, shape ``(n_samples, 1, n_time)``。
    y : np.ndarray
        真实标签, shape ``(n_samples, 1, n_time)``。
    time_coords : np.ndarray
        时间坐标数组, ``np.arange(1, n_time + 1)``。
    avg_metrics : dict
        包含 ``mse``, ``rmse``, ``mae``, ``pcc``, ``r2`` 的平均指标。
    """
    print("=" * 60)
    print("加载模型...")
    model, info = load_model_from_checkpoint(model_dir, device)

    print("加载 scaler...")
    scaler, mean_curve = load_scaler_and_mean_curve(model_dir)

    print("加载测试数据...")
    X, y_raw = load_test_csvs(scenario, p, n, h, rep)
    T = info["time_points"]

    print(f"  测试集样本数: {X.shape[0]},  SNP 数: {X.shape[1]},  时间点: {T}")

    print("运行推理...")
    pred_norm = run_inference(model, X, device)
    y = y_raw.reshape(-1, 1, T)
    pred = denormalize_predictions(pred_norm, scaler, mean_curve)

    time_coords = np.arange(1, T + 1)

    from utils.Calculate_Metrics import (
        calculate_metrics_by_average,
        calculate_phenotype_metrics_by_time,
    )

    pred_t = torch.tensor(pred.reshape(-1, T))
    true_t = torch.tensor(y.reshape(-1, T))
    metrics = calculate_phenotype_metrics_by_time(
        pred_t.unsqueeze(1), true_t.unsqueeze(1), info["phenotype_names"]
    )
    avg = calculate_metrics_by_average(metrics)
    print("测试集指标 (反归一化):")
    print(
        f"  MSE={avg['mse']:.6f}  RMSE={avg['rmse']:.4f}  MAE={avg['mae']:.4f}  "
        f"PCC={avg['pcc']:.4f}  R²={avg['r2']:.4f}"
    )

    return pred, y, time_coords, avg
