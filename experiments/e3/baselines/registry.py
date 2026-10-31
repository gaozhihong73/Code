"""模型实现状态、来源许可证与适配器工厂。"""

from __future__ import annotations

from typing import Any

from .adapters import (
    GBLUPAdapter,
    MeanAdapter,
    RandomForestAdapter,
    RidgeAdapter,
    XGBoostAdapter,
)
from .deep_adapters import SingleTimePointDeepAdapter
from .rr_gblup import RRGBLUPAdapter


IMPLEMENTATION_REGISTRY: dict[str, dict[str, Any]] = {
    "Mean": {"status": "runnable", "source": "independent", "license": "project"},
    "Ridge-All": {"status": "runnable", "source": "scikit-learn", "license": "BSD-3-Clause"},
    "Ridge-A+D": {"status": "runnable", "source": "scikit-learn", "license": "BSD-3-Clause"},
    "GBLUP-All": {"status": "runnable", "source": "independent VanRaden kernel implementation", "license": "project"},
    "RR-GBLUP": {"status": "runnable", "source": "independent separable genomic/Legendre kernel implementation", "license": "project", "adaptation": "K_genomic tensor-product Legendre time kernel; validation selects degree/lambda"},
    "Random Forest": {"status": "runnable", "source": "scikit-learn", "license": "BSD-3-Clause"},
    "XGBoost": {"status": "runnable", "source": "xgboost", "license": "Apache-2.0"},
    "DNNGP-SNP": {"status": "runnable", "source": "independent implementation from Molecular Plant 2023 paper, DOI 10.1016/j.molp.2022.11.004", "license": "project implementation; official repository code not copied", "adaptation": "SNP-only three-layer CNN; adaptive pooling bounds the dense head"},
    "DLGWAS": {"status": "runnable", "source": "independent implementation from Frontiers in Genetics 2019 paper, DOI 10.3389/fgene.2019.01091", "license": "project implementation; official repository code not copied", "adaptation": "four-channel one-hot, kernel 4/20 dual stream, bounded adaptive pooling"},
    "WheatGP": {"status": "runnable", "source": "https://github.com/Breed-AI/WheatGP", "source_commit": "a061b8be1d231e21fc89ed7179ae58c2a3048819", "license": "Apache-2.0", "adaptation": "five SNP segments and shared 1/3/9-kernel CNN retained; pooled local tokens form the LSTM sequence; no upstream split or pretrained weights"},
    "Cropformer": {"status": "runnable", "source": "https://github.com/jiekesen/Cropformer", "source_commit": "e0a77cf699b034d956b249f41b2e1f357f486f06", "license": "MIT", "adaptation": "CNN plus actual multi-head token attention on train-standardized 0/1/2 dosage; no MIC selection, PED 0-9 conversion, upstream split, or pretrained weights"},
    "DynaGS": {"status": "runnable_reference", "source": "existing E2 Screened run", "license": "project"},
}


def build_adapter(name: str, *, seed: int, options: dict[str, Any]):
    status = IMPLEMENTATION_REGISTRY.get(name, {}).get("status")
    if status == "placeholder":
        raise NotImplementedError(f"{name} 目前仅注册接口，尚无可运行适配器")
    if name == "Mean":
        return MeanAdapter(seed=seed, options=options)
    if name == "Ridge-All":
        return RidgeAdapter(seed=seed, options=options, dominance=False)
    if name == "Ridge-A+D":
        return RidgeAdapter(seed=seed, options=options, dominance=True)
    if name == "GBLUP-All":
        return GBLUPAdapter(seed=seed, options=options)
    if name == "RR-GBLUP":
        return RRGBLUPAdapter(seed=seed, options=options)
    if name == "Random Forest":
        return RandomForestAdapter(seed=seed, options=options)
    if name == "XGBoost":
        return XGBoostAdapter(seed=seed, options=options)
    if name in ("DNNGP-SNP", "DLGWAS", "WheatGP", "Cropformer"):
        return SingleTimePointDeepAdapter(
            name=name, seed=seed, options=options
        )
    if name == "DynaGS":
        raise ValueError("DynaGS 由 runner 的既有 E2 只读引用接口处理")
    raise ValueError(f"未知 E3 模型: {name}")
