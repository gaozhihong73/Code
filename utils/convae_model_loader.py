"""ConvAE v2 严格 checkpoint 加载；不对旧权重猜测表型维度或迁移 padding 语义。"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from config.ConvAE_Config import ConvAEConfig
from model.ConvAE import ConvAE, MASK_POLICY, MODEL_VERSION
from .ConvAE_Objective import EMPTY_TARGET_POLICY, RECONSTRUCTION_POLICY, fixed_reconstruction_masks
from .ConvAE_Preprocessing import load_convae_preprocessing
from .Inference import file_sha256


def load_convae_model(path: str | Path, device: str = "cpu") -> tuple[ConvAE, dict]:
    """加载实验目录或 models/best_model.pt，校验配置、数据契约和固定验证 mask。"""
    path = Path(path)
    checkpoint = path / "models/best_model.pt" if path.is_dir() else path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ConvAE checkpoint 不存在: {checkpoint}")
    package = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(package, dict) or package.get("format_version") != 2:
        raise ValueError("旧 ConvAE checkpoint 不受此入口支持：旧14维表型头/全量归一化/未排除padding不可等价迁移；请保留旧版本诊断或使用新实验从头训练")
    if not {"model_config", "metadata", "state_dict"}.issubset(package):
        raise ValueError("ConvAE v2 checkpoint 缺少必需字段")
    root = checkpoint.parent.parent
    artifacts = load_convae_preprocessing(root)
    config_path = root / "config.json"
    config = ConvAEConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    model_config, metadata = package["model_config"], package["metadata"]
    from .ArtifactIntegrity import CHECKPOINT_PROTOCOL, verify_checkpoint
    for name, expected in (("model_config.json", model_config), ("checkpoint_metadata.json", metadata)):
        sidecar = checkpoint.parent / name
        if not sidecar.is_file():
            raise FileNotFoundError(f"缺少 ConvAE checkpoint 元数据: {sidecar}")
        if json.loads(sidecar.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"ConvAE checkpoint 与 {name} 不一致")
    if model_config.get("model_version") != MODEL_VERSION or metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("ConvAE 模型版本不兼容")
    for key in ("chunk_size", "patch_size", "d_model", "d_ext", "n_phenotypes", "n_time_points",
                "use_pheno_help", "num_heads", "encoder_layers", "decoder_layers", "dropout"):
        if model_config.get(key) != getattr(config, key):
            raise ValueError(f"ConvAE 模型参数与保存配置不一致: {key}")
    if (model_config.get("num_chunks") != artifacts.metadata["num_chunks"]
            or config.n_phenotypes != len(artifacts.metadata["phenotype_names"])
            or config.n_time_points != artifacts.metadata["time_points"]):
        raise ValueError("ConvAE chunk/表型/时间点与预处理契约不一致")
    expected_policies = {
        "schema_version": 3, "selection_metric": config.selection_metric,
        "selection_definition": config.selection_definition, "use_pheno_help": config.use_pheno_help,
        "recon_weight": config.recon_weight, "pheno_weight": config.pheno_weight,
        "mask_policy": MASK_POLICY, "reconstruction_policy": RECONSTRUCTION_POLICY,
        "empty_target_policy": EMPTY_TARGET_POLICY,
        "empty_training_batch_policy": "skip_optimizer_step_if_no_weighted_target",
        "reconstruction_loss_definition": config.reconstruction_loss_definition,
        "phenotype_loss_definition": config.phenotype_loss_definition,
        "train_mask_ratio": config.train_mask_ratio, "val_mask_ratio": config.val_mask_ratio,
    }
    for key, expected in expected_policies.items():
        if metadata.get(key) != expected:
            raise ValueError(f"ConvAE checkpoint 目标/模型选择口径不一致: {key}")
    losses = [metadata.get(key) for key in ("val_total_loss", "val_recon_loss", "val_pheno_loss", "best_selection_loss")]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in losses):
        raise ValueError("ConvAE checkpoint 缺少有限的验证分项或选择损失")
    total, recon, pheno, selected = losses
    expected_total = config.recon_weight * recon + config.pheno_weight * pheno
    if (not math.isclose(total, expected_total, rel_tol=1e-6, abs_tol=1e-7)
            or not math.isclose(selected, total if config.use_pheno_help else recon, rel_tol=1e-6, abs_tol=1e-7)
            or (not config.use_pheno_help and pheno != 0)):
        raise ValueError("ConvAE checkpoint 验证损失不符合固定的模型选择定义")
    if metadata.get("preprocessing_sha256") != file_sha256(root / "preprocessing/preprocessing.json"):
        raise ValueError("ConvAE checkpoint 预处理哈希不一致")
    mask_path = root / "preprocessing/validation_masks.npz"
    if not mask_path.is_file():
        raise FileNotFoundError("缺少 ConvAE 固定验证掩码 validation_masks.npz")
    if metadata.get("validation_masks_sha256") != file_sha256(mask_path):
        raise ValueError("ConvAE 固定验证掩码哈希不一致")
    ids = np.asarray(artifacts.split.metadata["sample_ids"])[artifacts.split.val_idx]
    expected_masks = fixed_reconstruction_masks(
        torch.as_tensor(artifacts.chunk_padding_mask), patch_size=config.patch_size, sample_ids=ids,
        seed=config.seed, split_fingerprint=artifacts.metadata["split_fingerprint"], mask_ratio=config.val_mask_ratio,
    )
    with np.load(mask_path, allow_pickle=False) as saved:
        if (not np.array_equal(ids, saved["sample_ids"])
                or not np.array_equal(expected_masks, saved["patch_masks"])):
            raise ValueError("ConvAE 验证掩码与 seed/split/sample_id 不一致")
    if metadata.get("checkpoint_protocol") == CHECKPOINT_PROTOCOL or (root / "models/checkpoint_integrity.json").exists():
        verify_checkpoint(root)
    parameters = {key: value for key, value in model_config.items() if key != "model_version"}
    model = ConvAE(**parameters).to(device)
    try:
        model.load_state_dict(package["state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("ConvAE 权重形状或辅助头开关不匹配；禁止静默迁移，请使用匹配的 v2 配置") from error
    model.eval()
    return model, {"model_config": model_config, "metadata": metadata, "artifacts": artifacts,
                   "paper_eligible": metadata.get("checkpoint_protocol") == CHECKPOINT_PROTOCOL}
