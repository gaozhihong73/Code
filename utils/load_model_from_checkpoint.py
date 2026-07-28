"""画图脚本的公共函数: 模型加载、推理、反归一化。"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.TFPP import TFPP


def load_model_from_checkpoint(model_dir: str | Path, device: str = "cpu"):
    """从 checkpoint 反推所有超参并重建 TFPP 模型。

    Parameters
    ----------
    model_dir : str or Path
        模型输出目录, 应包含 ``models/best_model.pt``。
    device : str, optional
        加载模型的设备, 默认 ``"cpu"``。

    Returns
    -------
    model : TFPP
        已加载权重并设为 eval 模式的 TFPP 实例。
    info : dict
        包含 ``snp_dim``, ``time_points``, ``degree``, ``d_model``,
        ``num_layers``, ``d_ff``, ``phenotype_names`` 等信息。
    """
    model_dir = Path(model_dir)
    ckpt_path = model_dir / "models" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    t_basis = ckpt["wide_layer.t_basis"]
    degree = int(t_basis.shape[0] - 1)
    time_points = int(t_basis.shape[1])
    snp_dim = int(ckpt["wide_layer.snp_importance"].shape[1])
    d_model = int(ckpt["snp_emb.weight"].shape[1])
    d_ff = int(ckpt["encoder.layers.0.ffn.0.weight"].shape[0])
    num_layers = sum(
        1
        for k in ckpt
        if k.startswith("encoder.layers.") and k.endswith(".norm1.weight")
    )
    num_embeddings = 3
    num_heads = 4  # wide-only 模式下 deep 通路不执行, 值不影响推理
    phenotype_names = ["sim_pheno"]

    print("从 checkpoint 反推的模型参数:")
    print(
        f"  d_model={d_model}  num_heads={num_heads}  num_layers={num_layers}  d_ff={d_ff}"
    )
    print(f"  snp_dim={snp_dim}  degree={degree}  time_points={time_points}")

    model = TFPP(
        snp_dim=snp_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        num_embeddings=num_embeddings,
        time_coordinates=list(range(1, time_points + 1)),
        dropout=0.25,
        phenotype_names=phenotype_names,
        degree=degree,
        ablation_mode="wide",
    ).to(device)

    model.load_state_dict(ckpt)
    model.eval()

    info = {
        "snp_dim": snp_dim,
        "time_points": time_points,
        "degree": degree,
        "d_model": d_model,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "phenotype_names": phenotype_names,
    }
    return model, info


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


def run_inference(model, X: np.ndarray, device: str):
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
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
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
