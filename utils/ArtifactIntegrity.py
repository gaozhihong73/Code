"""训练结束时封存 checkpoint 与关键预处理文件；推理只核验，不补建。"""

import json
from pathlib import Path

from .Inference import file_sha256

CHECKPOINT_PROTOCOL = "checkpoint_sha256_v1"


def seal_checkpoint(root: str | Path, *, extra_files=()) -> None:
    root = Path(root)
    names = ["models/best_model.pt", "models/model_config.json", "preprocessing/preprocessing.json",
             "preprocessing/split.npz", "scalers/scaler.pkl", *extra_files]
    record = {"protocol": CHECKPOINT_PROTOCOL, "files": {name: file_sha256(root / name) for name in names}}
    (root / "models/checkpoint_integrity.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def verify_checkpoint(root: str | Path) -> None:
    root = Path(root)
    path = root / "models/checkpoint_integrity.json"
    if not path.is_file():
        raise FileNotFoundError("缺少 checkpoint_integrity.json；禁止以旧格式回退绕过 checkpoint 哈希校验")
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {"models/best_model.pt", "models/model_config.json", "preprocessing/preprocessing.json",
                "preprocessing/split.npz", "scalers/scaler.pkl"}
    if record.get("protocol") != CHECKPOINT_PROTOCOL or not required.issubset(record.get("files", {})):
        raise ValueError("checkpoint 完整性清单不完整")
    for name, expected in record["files"].items():
        target = root / name
        if root.resolve() not in target.resolve().parents or file_sha256(target) != expected:
            raise ValueError(f"checkpoint/预处理完整性哈希不匹配: {name}")
