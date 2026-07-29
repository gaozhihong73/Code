"""TFPP 模型通用加载器。

从 checkpoint (.pt) 自动反推模型结构参数并重建 TFPP 实例，
支持真实数据与模拟数据训练产出的所有模型。

优先级:
1. 若模型目录下存在 ``model_config.json``，直接读取（训练时自动保存）。
2. 否则从 state_dict 的张量形状反推参数（兼容旧 checkpoint）。

用法::

    from utils.model_loader import load_tfpp_model

    model, info = load_tfpp_model("outputs/tfpp/my_exp")
    # 或直接指定 .pt 文件
    model, info = load_tfpp_model("outputs/tfpp/my_exp/models/best_model.pt")
"""

import json
from pathlib import Path

import torch

from model.TFPP import TFPP


def _infer_params_from_state_dict(ckpt: dict) -> dict:
    """从 state_dict 张量形状反推模型结构参数。"""
    params = {}

    # snp_dim: wide_layer.snp_importance shape (n_pheno, snp_dim)
    if "wide_layer.snp_importance" in ckpt:
        params["snp_dim"] = int(ckpt["wide_layer.snp_importance"].shape[1])
    elif "wide_layer.base_linear.weight" in ckpt:
        params["snp_dim"] = int(ckpt["wide_layer.base_linear.weight"].shape[1])

    # d_model / num_embeddings: snp_emb.weight shape (num_embeddings, d_model)
    if "snp_emb.weight" in ckpt:
        params["num_embeddings"] = int(ckpt["snp_emb.weight"].shape[0])
        params["d_model"] = int(ckpt["snp_emb.weight"].shape[1])

    # num_layers: 统计 encoder.layers.{i}.norm1.weight
    params["num_layers"] = sum(
        1
        for k in ckpt
        if k.startswith("encoder.layers.") and k.endswith(".norm1.weight")
    )

    # d_ff: encoder.layers.0.ffn.0.weight shape (d_ff, d_model)
    if "encoder.layers.0.ffn.0.weight" in ckpt:
        params["d_ff"] = int(ckpt["encoder.layers.0.ffn.0.weight"].shape[0])

    # degree / time_points: wide_layer.t_basis shape (degree+1, time_points)
    if "wide_layer.t_basis" in ckpt:
        t_basis = ckpt["wide_layer.t_basis"]
        params["degree"] = int(t_basis.shape[0] - 1)
        params["time_points"] = int(t_basis.shape[1])

    # phenotype_names: 从 predictor ModuleDict 的 key 中提取
    pheno_names = set()
    for k in ckpt:
        if k.startswith("predictor.") and ".time_queries" in k:
            name = k.split(".")[1]
            pheno_names.add(name)
    if pheno_names:
        params["phenotype_names"] = sorted(pheno_names)

    # num_phenotypes: wide_layer.base_linear.weight shape (n_pheno, snp_dim)
    if "wide_layer.base_linear.weight" in ckpt:
        params["num_phenotypes"] = int(ckpt["wide_layer.base_linear.weight"].shape[0])

    # ablation_mode: 根据存在的子模块推断
    has_wide = any(k.startswith("wide_layer.") for k in ckpt)
    has_deep = any(k.startswith("encoder.layers.") for k in ckpt)
    has_gating = any(k.startswith("gating_net.") for k in ckpt)
    if has_wide and has_deep:
        params["ablation_mode"] = "all" if has_gating else "add"
    elif has_wide:
        params["ablation_mode"] = "wide"
    elif has_deep:
        params["ablation_mode"] = "deep"
    else:
        params["ablation_mode"] = "all"

    return params


def _load_config_json(model_dir: Path) -> dict | None:
    """尝试加载训练时保存的 model_config.json。"""
    config_path = model_dir / "model_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_tfpp_model(
    path: str | Path,
    device: str = "cpu",
    *,
    num_heads: int | None = None,
    dropout: float = 0.0,
    time_coordinates: list[float] | None = None,
    phenotype_names: list[str] | None = None,
    ablation_mode: str | None = None,
    verbose: bool = True,
) -> tuple[TFPP, dict]:
    """通用 TFPP 模型加载器：自动推测参数、重建模型、加载权重。

    Parameters
    ----------
    path : str or Path
        模型目录（含 ``models/best_model.pt``）或直接指向 ``.pt`` 文件。
    device : str, optional
        加载设备，默认 ``"cpu"``。
    num_heads : int, optional
        注意力头数。无法从权重推断，若未提供且无 config.json 则默认 4。
    dropout : float, optional
        Dropout 比例，不影响推理结果，默认 0.0。
    time_coordinates : list[float], optional
        时间坐标列表。若未提供，默认为 ``[1, 2, ..., time_points]``。
    phenotype_names : list[str], optional
        表型名称列表。若未提供，从 state_dict 的 predictor keys 中自动提取。
    ablation_mode : str, optional
        消融模式覆盖。若未提供，从 state_dict 自动推断。
    verbose : bool, optional
        是否打印反推的参数信息，默认 True。

    Returns
    -------
    model : TFPP
        已加载权重并设为 eval 模式的模型实例。
    info : dict
        模型元信息字典，包含所有结构参数。

    Raises
    ------
    FileNotFoundError
        当找不到 .pt 文件时抛出。
    """
    path = Path(path)

    # 确定 .pt 文件路径和模型目录
    if path.is_file() and path.suffix == ".pt":
        ckpt_path = path
        model_dir = path.parent.parent  # models/best_model.pt -> exp_dir
    elif path.is_dir():
        model_dir = path
        ckpt_path = model_dir / "models" / "best_model.pt"
    else:
        raise FileNotFoundError(f"无法定位模型文件: {path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")

    # 加载权重
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # 优先从 config.json 读取，否则从 state_dict 推断
    saved_config = _load_config_json(model_dir)
    inferred = _infer_params_from_state_dict(ckpt)

    # 合并参数: saved_config > inferred > 函数参数默认值
    snp_dim = (saved_config or {}).get("snp_dim") or inferred.get("snp_dim")
    d_model = (saved_config or {}).get("d_model") or inferred.get("d_model")
    num_layers = (saved_config or {}).get("num_layers") or inferred.get("num_layers")
    d_ff = (saved_config or {}).get("d_ff") or inferred.get("d_ff")
    num_embeddings = (saved_config or {}).get("num_embeddings") or inferred.get(
        "num_embeddings", 4
    )
    degree = (saved_config or {}).get("degree", inferred.get("degree", 3))
    time_points = (saved_config or {}).get("time_points") or inferred.get(
        "time_points", 1
    )

    # num_heads 无法从权重推断
    if num_heads is None:
        num_heads = (saved_config or {}).get("num_heads", 4)

    # ablation_mode
    if ablation_mode is None:
        ablation_mode = (saved_config or {}).get(
            "ablation_mode", inferred.get("ablation_mode", "all")
        )

    # phenotype_names
    if phenotype_names is None:
        phenotype_names = (saved_config or {}).get(
            "phenotype_names", inferred.get("phenotype_names", ["pheno_0"])
        )

    # time_coordinates
    if time_coordinates is None:
        time_coordinates = (saved_config or {}).get("time_coordinates")
        if time_coordinates is None:
            time_coordinates = list(range(1, time_points + 1))

    # 构建模型
    model = TFPP(
        snp_dim=snp_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        num_embeddings=num_embeddings,
        time_coordinates=time_coordinates,
        dropout=dropout,
        phenotype_names=phenotype_names,
        degree=degree,
        ablation_mode=ablation_mode,
    ).to(device)

    model.load_state_dict(ckpt)
    model.eval()

    info = {
        "snp_dim": snp_dim,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "num_embeddings": num_embeddings,
        "degree": degree,
        "time_points": time_points,
        "time_coordinates": time_coordinates,
        "phenotype_names": phenotype_names,
        "ablation_mode": ablation_mode,
        "ckpt_path": str(ckpt_path),
        "config_source": "model_config.json" if saved_config else "state_dict_inference",
    }

    if verbose:
        print("=" * 60)
        print("TFPP 模型加载完成")
        print(f"  来源: {info['config_source']}")
        print(f"  权重: {ckpt_path}")
        print(
            f"  d_model={d_model}  num_heads={num_heads}  "
            f"num_layers={num_layers}  d_ff={d_ff}"
        )
        print(
            f"  snp_dim={snp_dim}  degree={degree}  "
            f"time_points={time_points}  num_embeddings={num_embeddings}"
        )
        print(f"  ablation_mode={ablation_mode}")
        print(f"  phenotype_names={phenotype_names}")
        print("=" * 60)

    return model, info


def save_model_config(config, model_dir: str | Path) -> None:
    """训练时调用：将模型结构参数保存为 model_config.json，供加载器直接读取。

    Parameters
    ----------
    config : TFPPConfig or SimTFPPConfig
        训练配置对象。
    model_dir : str or Path
        模型输出根目录（如 ``outputs/tfpp/exp_name``）。
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "snp_dim": None,  # 由 trainer 在筛选后填入
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "d_ff": config.d_ff,
        "num_embeddings": config.num_embeddings,
        "degree": config.degree,
        "ablation_mode": config.ablation_mode,
        "time_coordinates": config.time_coordinates,
        "time_points": len(config.time_coordinates),
        "phenotype_names": config.phenotype_names,
    }

    config_path = model_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)
