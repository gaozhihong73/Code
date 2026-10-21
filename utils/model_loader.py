"""TFPP 模型通用加载器。

默认严格校验模型配置与完整的普通训练预处理产物。
旧 checkpoint 只有显式 ``legacy=True`` 才允许反推参数，并警告禁止用于正式汇总。

用法::

    from utils.model_loader import load_tfpp_model

    model, info = load_tfpp_model("outputs/tfpp/my_exp")
    # 或直接指定 .pt 文件
    model, info = load_tfpp_model("outputs/tfpp/my_exp/models/best_model.pt")
"""

import json
import warnings
from pathlib import Path

import torch

from model.TFPP import TFPP
from .Genotype import GENOTYPE_ENCODING_VERSION
from .Screening import SCREENING_ALGORITHM_VERSION


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


def _load_config_json(ckpt_path: Path) -> dict | None:
    """尝试加载与 .pt 同级的 model_config.json。"""
    config_path = ckpt_path.parent / "model_config.json"
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
    legacy: bool = False,
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
    legacy : bool, optional
        显式允许不完整的旧实验；新格式损坏时仍失败，不绕过校验。

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
    saved_config = _load_config_json(ckpt_path)
    inferred = _infer_params_from_state_dict(ckpt)
    from .Inference import load_inference_artifacts

    is_new = ("wide_fill_values" in ckpt
              or (model_dir / "preprocessing" / "preprocessing.json").exists()
              or (saved_config or {}).get("wide_input_mode") == "dosage_v1")
    artifacts = None
    if is_new:
        if saved_config is None:
            raise FileNotFoundError("新格式实验缺少 models/model_config.json，禁止从权重猜测配置")
        artifacts = load_inference_artifacts(model_dir, saved_config)
        from .ArtifactIntegrity import CHECKPOINT_PROTOCOL, verify_checkpoint
        if (artifacts.metadata.get("checkpoint_protocol") == CHECKPOINT_PROTOCOL
                or (model_dir / "models/checkpoint_integrity.json").exists()
                or (model_dir / "test_evaluation.json").exists()):
            verify_checkpoint(model_dir)
        for key in ("snp_dim", "d_model", "num_layers", "d_ff", "num_embeddings", "degree", "time_points"):
            if saved_config.get(key) != inferred.get(key):
                raise ValueError(f"checkpoint 与 model_config 不一致: {key}")
        if (set(saved_config["phenotype_names"]) != set(inferred.get("phenotype_names", []))
                or len(saved_config["phenotype_names"]) != inferred.get("num_phenotypes")):
            raise ValueError("checkpoint 的表型名称或数量与模型配置不一致")
        fill = ckpt.get("wide_fill_values")
        if fill is None or not torch.equal(fill.cpu(), torch.as_tensor(artifacts.wide_fill_values)):
            raise ValueError("checkpoint 缺少 Wide 填充值或与训练预处理记录不一致")
        for key, override in (("num_heads", num_heads), ("time_coordinates", time_coordinates),
                              ("phenotype_names", phenotype_names), ("ablation_mode", ablation_mode)):
            if override is not None and override != saved_config.get(key):
                raise ValueError(f"严格推理禁止覆盖训练配置: {key}")
    elif not legacy:
        raise ValueError("旧 checkpoint 缺少完整训练产物；仅可显式使用 legacy=True / --legacy 进入兼容模式")
    else:
        warnings.warn("!!! LEGACY：旧 checkpoint 无法保证预处理可复现；结果禁止进入论文正式实验汇总 !!!",
                      RuntimeWarning, stacklevel=2)

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

    # 新 checkpoint 自带训练期 Wide 填补 buffer；旧权重显式保留旧输入语义。
    wide_input_mode = "dosage_v1" if "wide_fill_values" in ckpt else "legacy_categorical"
    recorded_mode = (saved_config or {}).get("wide_input_mode", wide_input_mode)
    if recorded_mode != wide_input_mode:
        raise ValueError("checkpoint 的 Wide 输入语义与 model_config.json 不一致")

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
        wide_input_mode=wide_input_mode,
        wide_fill_values=ckpt.get("wide_fill_values"),
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
        "wide_input_mode": wide_input_mode,
        "ckpt_path": str(ckpt_path),
        "config_source": "model_config.json" if saved_config else "state_dict_inference",
        "artifacts": artifacts,
        "legacy": not is_new,
        "paper_eligible": bool(is_new and artifacts.metadata.get("checkpoint_protocol") == "checkpoint_sha256_v1"),
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
        "screen_k": config.screen_k,
        "gwas_k": config.gwas_k,
        "screen_multivariate": config.screen_multivariate,
        "screen_missing_strategy": config.screen_missing_strategy,
        "genotype_encoding_version": GENOTYPE_ENCODING_VERSION,
        "screening_algorithm_version": SCREENING_ALGORITHM_VERSION,
        "wide_input_mode": "legacy_categorical" if getattr(config, "use_kfold", True) else "dosage_v1",
        "wide_missing_strategy": getattr(config, "wide_missing_strategy", "legacy"),
        "fold_index": getattr(config, "fold_index", None),
        "n_folds": getattr(config, "n_folds", None),
        "kfold_val_ratio": getattr(config, "kfold_val_ratio", None),
        "seed": config.seed,
    }

    config_path = model_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)
