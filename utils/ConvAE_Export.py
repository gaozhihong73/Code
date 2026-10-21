"""仅用 SNP 和严格最佳 checkpoint 导出可追溯的二维潜在特征。"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .Genotype import dosage_to_model_codes, model_code_tensor
from .Inference import file_sha256
from .Split import load_split
from .convae_model_loader import load_convae_model

EXPORT_SCHEMA = 1
LATENT_LAYOUT = "chunk_major_flatten_C_order"
ROLES = ("train", "val", "test")


class SNPExportDataset(Dataset):
    """只读取原始 dosage，不打开表型文件；sample_id 是有记录的原始行索引。"""

    def __init__(self, dosage: np.ndarray, indices: np.ndarray, metadata: dict):
        self.dosage, self.indices, self.metadata = dosage, indices, metadata

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row = int(self.indices[item])
        codes = dosage_to_model_codes(self.dosage[row:row + 1])[0]
        codes = np.pad(codes, (0, self.metadata["target_length"] - len(codes)))
        chunks = model_code_tensor(codes[None])[0].unfold(
            0, self.metadata["chunk_size"], self.metadata["step_size"]).contiguous()
        return chunks, row


def _id_tokens(ids) -> list[str]:
    values = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else np.asarray(ids).tolist()
    if not isinstance(values, list):
        raise ValueError("sample_ids 必须为一维序列")
    tokens = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, str)) or value == "":
            raise ValueError("sample_id 必须为整数或非空字符串，禁止猜测/截断 ID")
        tokens.append(json.dumps(value, ensure_ascii=False))
    return tokens


@torch.inference_mode()
def encode_feature_batches(model, batches: Iterable, *, padding_mask: torch.Tensor,
                           expected_sample_ids) -> np.ndarray:
    """只接收 (SNP, sample_id) 批次；按 manifest 恢复行序并拒绝重复/缺失/额外 ID。"""
    model.eval()
    expected = _id_tokens(expected_sample_ids)
    positions = {key: i for i, key in enumerate(expected)}
    if len(positions) != len(expected):
        raise ValueError("manifest sample_id 重复")
    result = np.empty((len(expected), model.num_chunks * model.d_ext), dtype=np.float32)
    seen = set()
    device = next(model.parameters()).device
    for batch in batches:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("编码批次必须仅包含 (SNP, sample_id)，不得包含 phenotype/scaler")
        x, ids = batch
        tokens = _id_tokens(ids)
        if len(tokens) != len(x):
            raise ValueError("sample_id 数量与 SNP batch 不一致")
        for token in tokens:
            if token not in positions:
                raise ValueError("导出出现额外 sample_id")
            if token in seen:
                raise ValueError("导出出现重复 sample_id")
            seen.add(token)
        features = model.encode_features(x.to(device), padding_mask=padding_mask.to(device))
        values = features.cpu().numpy().astype(np.float32, copy=False)
        if values.shape != (len(tokens), result.shape[1]) or not np.isfinite(values).all():
            raise ValueError("潜在特征 shape 不合法或含非有限值")
        result[[positions[key] for key in tokens]] = values
    if seen != set(expected):
        raise ValueError("导出缺失 manifest sample_id")
    return result


def _git_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                           capture_output=True, text=True).stdout.strip()
    return {"git_commit": commit, "git_dirty": bool(dirty)}


def load_exported_features(directory: str | Path) -> tuple[dict, dict]:
    """落盘后重新读取并校验 shape、dtype、有限值、ID 顺序与每个文件内容哈希。"""
    root = Path(directory)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("export_schema_version") != EXPORT_SCHEMA or meta.get("latent_layout") != LATENT_LAYOUT:
        raise ValueError("ConvAE 导出 schema/layout 不兼容")
    arrays = {}
    for role in ROLES:
        record = meta["arrays"][role]
        loaded = []
        for name in (f"{role}.npy", f"{role}_sample_ids.npy"):
            path = root / "features" / name
            if file_sha256(path) != record["files"][name]:
                raise ValueError(f"导出内容哈希不匹配: {name}")
            loaded.append(np.load(path, allow_pickle=False))
        features, ids = loaded
        if (features.shape != tuple(record["shape"]) or features.ndim != 2
                or features.shape != (len(ids), meta["latent_dim"])
                or str(features.dtype) != record["dtype"] or not np.isfinite(features).all()
                or list(ids.shape) != record["sample_ids_shape"] or str(ids.dtype) != record["sample_ids_dtype"]
                or ids.ndim != 1 or ids.tolist() != record["sample_ids"]
                or len(set(_id_tokens(ids))) != len(ids)):
            raise ValueError(f"导出 shape/dtype/sample_id 顺序不一致: {role}")
        arrays[role] = (features, ids)
    return arrays, meta


def export_convae_features(model_path: str | Path, *, split_file: str | Path,
                          snp_file: str | Path | None = None, output_root: str | Path | None = None,
                          batch_size: int = 32, device: str = "cpu",
                          expected_checkpoint_sha256: str | None = None,
                          expected_model_config: dict | None = None,
                          loader_factory: Callable | None = None) -> Path:
    """严格加载当前 split 的最佳模型，保存三集合特征；已存在的导出绝不覆盖。

    loader_factory(dataset, batch_size) 可改变内部顺序，不改变 manifest 行序。
    本入口不构建 ConvAEDataset、不读表型 CSV、不 fit/transform scaler。
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    path = Path(model_path).resolve()
    checkpoint = path / "models/best_model.pt" if path.is_dir() else path
    if checkpoint.name != "best_model.pt" or checkpoint.parent.name != "models":
        raise ValueError("仅允许严格加载实验 models/best_model.pt 验证集选出的最佳 checkpoint")
    experiment = checkpoint.parent.parent
    checkpoint_hash = file_sha256(checkpoint)
    if expected_checkpoint_sha256 is not None and checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError("当前导出任务 checkpoint 哈希不匹配")
    model, info = load_convae_model(checkpoint, device=device)
    artifacts = info["artifacts"]
    meta, split = artifacts.metadata, artifacts.split
    if expected_model_config is not None and info["model_config"] != expected_model_config:
        raise ValueError("当前导出任务模型配置不匹配")
    split_file = Path(split_file).resolve()
    if file_sha256(split_file) != meta["split_sha256"]:
        raise ValueError("当前导出任务 split/fold 文件哈希与 checkpoint 不匹配")
    current = load_split(split_file)
    if (current.metadata != split.metadata or any(not np.array_equal(
            getattr(current, role + "_idx"), getattr(split, role + "_idx")) for role in ROLES)):
        raise ValueError("当前导出任务 split/fold 不一致")
    if meta["sample_id_source"] != "row_index" or split.metadata["sample_ids"] != list(range(meta["n_samples"])):
        raise ValueError("当前 SNP 文件无生物学 ID，仅支持训练记录明确的 row_index 顺序")
    config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
    source = Path(snp_file if snp_file is not None else config["snp_file"]).resolve()
    data_hash = file_sha256(source)
    dosage = np.load(source, mmap_mode="r", allow_pickle=False)
    if dosage.shape != (meta["n_samples"], meta["original_length"]):
        raise ValueError("导出 SNP 个体数/原始 SNP 维度与训练元数据不一致")
    base = Path(output_root) if output_root is not None else experiment / "exports"
    destination = base / meta["split_fingerprint"] / checkpoint_hash / data_hash
    if destination.exists():
        raise FileExistsError(f"导出目录已存在，禁止覆盖；复现实验请指定不同 --output_root: {destination}")
    features, ids_by_role = {}, {}
    for role in ROLES:
        indices = getattr(split, role + "_idx")
        ids = np.asarray(split.metadata["sample_ids"], dtype=np.int64)[indices]
        dataset = SNPExportDataset(dosage, indices, meta)
        loader = (DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                  if loader_factory is None else loader_factory(dataset, batch_size))
        features[role] = encode_feature_batches(model, loader,
            padding_mask=torch.as_tensor(artifacts.chunk_padding_mask), expected_sample_ids=ids)
        ids_by_role[role] = ids
    # 防止导出过程中源文件或 checkpoint 被其他进程替换。
    if file_sha256(source) != data_hash or file_sha256(checkpoint) != checkpoint_hash:
        raise ValueError("导出期间 SNP 或 checkpoint 内容已变化")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        directory = destination / "features"
        directory.mkdir()
        records = {}
        for role in ROLES:
            values, ids = features[role], ids_by_role[role]
            np.save(directory / f"{role}.npy", values, allow_pickle=False)
            np.save(directory / f"{role}_sample_ids.npy", ids, allow_pickle=False)
            records[role] = {"shape": list(values.shape), "dtype": str(values.dtype), "sample_ids": ids.tolist(),
                "sample_ids_shape": list(ids.shape), "sample_ids_dtype": str(ids.dtype),
                "files": {name: file_sha256(directory / name) for name in (f"{role}.npy", f"{role}_sample_ids.npy")}}
        metadata = {
            "export_schema_version": EXPORT_SCHEMA, **_git_provenance(),
            "preprocessing_schema_version": meta["schema_version"],
            "preprocessing_sha256": info["metadata"]["preprocessing_sha256"],
            "split_file": str(split_file), "split_sha256": meta["split_sha256"],
            "split_fingerprint": meta["split_fingerprint"], "fold_metadata": meta["fold_metadata"],
            "sample_id_source": meta["sample_id_source"], "sample_ids_sha256": meta["sample_ids_sha256"],
            "original_snp_dim": meta["original_length"], "target_length": meta["target_length"],
            "structural_padding_count": int(artifacts.padding_mask.sum()),
            "chunk_padding_mask_sha256": meta["chunk_padding_mask_sha256"],
            "chunk_size": meta["chunk_size"], "overlap": meta["overlap"], "num_chunks": meta["num_chunks"],
            "d_ext": model.d_ext, "latent_dim": model.num_chunks * model.d_ext,
            "latent_layout": LATENT_LAYOUT, "flatten_rule": "D = chunk_index * d_ext + feature_index; chunks in increasing SNP start order",
            "phenotype_names": meta["phenotype_names"], "time_points": meta["time_points"],
            "time_labels": meta["time_labels"], "time_coordinates": meta["time_coordinates"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "model_config": info["model_config"], "model_selection": info["metadata"],
            "use_pheno_help": model.use_pheno_help, "seed": config["seed"],
            "genotype_encoding_version": meta["genotype_encoding_version"],
            "snp_file": str(source), "snp_file_sha256": data_hash,
            "reconstruction_mask_ratio": 0, "encoder_only": True, "batch_size": batch_size,
            "device": str(device), "torch_version": str(torch.__version__),
            "arrays": records,
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        load_exported_features(destination)
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination
