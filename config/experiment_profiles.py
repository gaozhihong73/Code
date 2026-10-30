"""实验配置文件读取工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .TFPP_Config import TFPPConfig


def load_tfpp_profile(path: str | Path) -> dict[str, Any]:
    """读取并校验 TFPP 的 JSON 实验配置。"""
    profile_path = Path(path).resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"未找到实验配置文件: {profile_path}")
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"实验配置不是有效 JSON: {profile_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("实验配置根节点必须是 JSON 对象")
    unknown = set(payload) - set(TFPPConfig.model_fields)
    if unknown:
        raise ValueError(f"实验配置包含未知字段: {', '.join(sorted(unknown))}")
    return payload


def build_tfpp_config(
    profile_path: str | Path,
    **overrides: Any,
) -> TFPPConfig:
    """从配置文件构建 TFPPConfig；仅动态数据路径可在调用处覆盖。"""
    payload = load_tfpp_profile(profile_path)
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return TFPPConfig(**payload)
