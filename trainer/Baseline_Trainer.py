"""动态非线性遗传场景的传统模型与机器学习模型比较。"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils import apply_residual_global_norm, denormalize_phenotype
from utils.Calculate_Metrics import METRICS_PROTOCOL, compute_metrics_vectorized
from utils.Split import load_split


SCENARIO_LABEL = "动态非线性遗传场景"
ORACLE_SCENARIO_LABEL = "动态非线性遗传场景（Oracle 20 QTL）"
REPORT_COLUMNS = (
    "model",
    "PCC ↑",
    "R² ↑",
    "RMSE ↓",
    "MAE ↓",
    "True-GV PCC ↑",
    "Runtime (s) ↓",
)
SUPPORTED_MODELS = (
    "Ridge-All",
    "Ridge-A+D",
    "GBLUP-All",
    "RF",
    "XGBoost",
)


@dataclass
class BaselineConfig:
    """同一固定 split、同一指标协议下的比较实验配置。"""

    data_dir: str
    output_dir: str = "outputs/baseline_comparison/scenario_tfpp_dynamic_rep_01"
    seed: int = 73
    scenario_label: str = SCENARIO_LABEL
    alphas: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.logspace(0, 0, num=1).tolist())
    )
    models: tuple[str, ...] = SUPPORTED_MODELS
    rf_n_estimators: int = 200
    rf_max_features: float | str = 0.1
    rf_min_samples_leaf: int = 1
    rf_n_jobs: int = -1
    xgb_n_estimators: int = 200
    xgb_learning_rate: float = 0.005
    xgb_max_depth: int = 4
    xgb_subsample: float = 0.5
    xgb_colsample_bytree: float = 0.3
    xgb_early_stopping_rounds: int = 10
    xgb_n_jobs: int = -1
    xgb_device: str = "cuda"
    genotype_file: str | None = None
    phenotype_file: str | None = None
    sample_ids_file: str | None = None
    split_file: str | None = None
    preselected_snp_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        unknown = sorted(set(self.models).difference(SUPPORTED_MODELS))
        if unknown:
            raise ValueError(f"不支持的模型: {unknown}; 可选模型: {SUPPORTED_MODELS}")
        if len(self.models) != len(set(self.models)):
            raise ValueError("models 不得包含重复项")
        if self.scenario_label not in (SCENARIO_LABEL, ORACLE_SCENARIO_LABEL):
            raise ValueError("不支持的比较实验场景标签")
        if not self.alphas or any(value <= 0 for value in self.alphas):
            raise ValueError("alphas 必须包含正数")
        is_cuda_device = self.xgb_device == "cuda" or (
            self.xgb_device.startswith("cuda:")
            and self.xgb_device.removeprefix("cuda:").isdigit()
        )
        if self.xgb_device != "cpu" and not is_cuda_device:
            raise ValueError("xgb_device 仅支持 'cpu'、'cuda' 或 'cuda:<设备编号>'")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BaselineTrainer:
    """训练集拟合、验证集选参/早停、测试集一次评估。"""

    def __init__(self, config: BaselineConfig):
        self.config = config
        self.data_dir = Path(config.data_dir).resolve()
        self.output_dir = Path(config.output_dir).resolve()
        self.genotype_path = Path(config.genotype_file or self.data_dir / "genotype.npy")
        self.phenotype_path = Path(
            config.phenotype_file or self.data_dir / "cleaned_pheno/sim_pheno.csv"
        )
        self.sample_ids_path = Path(config.sample_ids_file or self.data_dir / "sample_ids.csv")
        self.manifest_path = self.data_dir / "data_manifest.json"
        self.split_path = Path(
            config.split_file or self.data_dir / "splits" / f"split_seed{config.seed}.npz"
        )

        required = (
            self.manifest_path,
            self.genotype_path,
            self.phenotype_path,
            self.sample_ids_path,
            self.split_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("比较实验缺少输入文件:\n" + "\n".join(missing))
        if self.output_dir.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖: {self.output_dir}")

        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.sample_ids = (
            pd.read_csv(self.sample_ids_path)["SampleID"].astype(str).tolist()
        )
        self.split = load_split(self.split_path, sample_ids=self.sample_ids)
        if self.split.metadata["seed"] != config.seed:
            raise ValueError("固定 split 中的 seed 与配置不一致")

        genotype = np.load(self.genotype_path, mmap_mode="r", allow_pickle=False)
        phenotype_frame = pd.read_csv(self.phenotype_path)
        if "SampleID" in phenotype_frame.columns:
            phenotype_ids = phenotype_frame.pop("SampleID").astype(str).tolist()
            if phenotype_ids != self.sample_ids:
                raise ValueError("表型文件的 SampleID 顺序与 sample_ids.csv 不一致")
        self.phenotype = phenotype_frame.to_numpy(dtype=np.float32)
        if genotype.ndim != 2 or self.phenotype.ndim != 2:
            raise ValueError("基因型和表型必须均为二维矩阵")
        if genotype.shape[0] != len(self.sample_ids):
            raise ValueError("基因型行数与 SampleID 数量不一致")
        if self.phenotype.shape[0] != len(self.sample_ids):
            raise ValueError("表型行数与 SampleID 数量不一致")
        if not np.isin(genotype, (0, 1, 2)).all():
            raise ValueError("基因型必须使用 0/1/2 剂量编码")
        if not np.isfinite(self.phenotype).all():
            raise ValueError("表型必须全部为有限值")

        self.raw_snp_count = int(genotype.shape[1])
        if config.preselected_snp_indices is None:
            self.genotype = genotype
        else:
            indices = np.asarray(config.preselected_snp_indices, dtype=np.int64)
            if (
                indices.ndim != 1
                or not len(indices)
                or len(np.unique(indices)) != len(indices)
                or (indices < 0).any()
                or (indices >= self.raw_snp_count).any()
            ):
                raise ValueError("预选 SNP 索引必须为互异且有效的原始列索引")
            self.genotype = genotype[:, np.sort(indices)]

        train = self.phenotype[self.split.train_idx, None, :]
        val = self.phenotype[self.split.val_idx, None, :]
        test = self.phenotype[self.split.test_idx, None, :]
        train_n, val_n, test_n, meta = apply_residual_global_norm(
            train, val, test, ["sim_pheno"], StandardScaler
        )
        self.y_train = train_n[:, 0, :]
        self.y_val = val_n[:, 0, :]
        self.y_test = test_n[:, 0, :]
        self.target_test = self.phenotype[self.split.test_idx]
        self.scalers = {
            "phenotype_names": ["sim_pheno"],
            "norm_method": "residual_global",
            "pheno_meta": meta,
            "snp_dim": self.genotype.shape[1],
        }

    def _dosage_splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读取所有原始 SNP；不加载任何筛选索引。"""
        return tuple(
            np.asarray(self.genotype[indices], dtype=np.float32)
            for indices in (
                self.split.train_idx,
                self.split.val_idx,
                self.split.test_idx,
            )
        )

    @staticmethod
    def _standardize_from_train(
        train: np.ndarray, val: np.ndarray, test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
        keep = scale > 1e-12
        if not keep.any():
            raise ValueError("训练集没有可用的变异特征")
        mean = mean[keep]
        scale = scale[keep]
        transformed = []
        for block in (train, val, test):
            values = block[:, keep]
            values -= mean
            values /= scale
            transformed.append(values)
        return *transformed, int(keep.sum())

    def _standardized_features(
        self, kind: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        train, val, test = self._dosage_splits()
        train, val, test, additive_count = self._standardize_from_train(
            train, val, test
        )
        if kind == "Ridge-All":
            return train, val, test, additive_count
        if kind != "Ridge-A+D":
            raise ValueError(f"未知 Ridge 类型: {kind}")

        raw_train, raw_val, raw_test = self._dosage_splits()
        dominance = tuple(
            (block == 1).astype(np.float32)
            for block in (raw_train, raw_val, raw_test)
        )
        dom_train, dom_val, dom_test, dominance_count = self._standardize_from_train(
            *dominance
        )
        return (
            np.concatenate((train, dom_train), axis=1),
            np.concatenate((val, dom_val), axis=1),
            np.concatenate((test, dom_test), axis=1),
            additive_count + dominance_count,
        )

    def _fit_ridge(self, kind: str) -> tuple[np.ndarray, dict, list[dict], int]:
        x_train, x_val, x_test, feature_count = self._standardized_features(kind)
        search = []
        best = None
        for alpha in self.config.alphas:
            model = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", tol=1e-5)
            model.fit(x_train, self.y_train)
            pred = model.predict(x_val)
            mse = float(np.mean(np.square(pred - self.y_val)))
            search.append({"alpha": alpha, "val_mse": mse})
            if best is None or mse < best[0]:
                best = (mse, alpha, model)
        assert best is not None
        return (
            best[2].predict(x_test),
            {"alpha": float(best[1])},
            search,
            feature_count,
        )

    def _fit_gblup(self) -> tuple[np.ndarray, dict, list[dict], int]:
        train_g = np.asarray(
            self.genotype[self.split.train_idx], dtype=np.float32
        )
        p = train_g.mean(axis=0, dtype=np.float64) / 2.0
        keep = (p > 0.0) & (p < 1.0)
        p = p[keep].astype(np.float32)
        z_train = train_g[:, keep] - 2.0 * p
        denominator = float(np.sum(2.0 * p * (1.0 - p)))
        if denominator <= 0:
            raise ValueError("GBLUP 无可用多态 SNP")
        k_train = (z_train @ z_train.T) / denominator
        eigenvalues, eigenvectors = eigh(k_train, check_finite=False)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projected_y = eigenvectors.T @ self.y_train

        val_g = np.asarray(self.genotype[self.split.val_idx], dtype=np.float32)
        k_val = ((val_g[:, keep] - 2.0 * p) @ z_train.T) / denominator
        search = []
        best = None
        for regularization in self.config.alphas:
            weights = eigenvectors @ (
                projected_y / (eigenvalues[:, None] + regularization)
            )
            pred = k_val @ weights
            mse = float(np.mean(np.square(pred - self.y_val)))
            search.append({"lambda": regularization, "val_mse": mse})
            if best is None or mse < best[0]:
                best = (mse, regularization)
        assert best is not None

        regularization = float(best[1])
        weights = eigenvectors @ (
            projected_y / (eigenvalues[:, None] + regularization)
        )
        test_g = np.asarray(self.genotype[self.split.test_idx], dtype=np.float32)
        k_test = ((test_g[:, keep] - 2.0 * p) @ z_train.T) / denominator
        return (
            k_test @ weights,
            {"lambda": regularization},
            search,
            int(keep.sum()),
        )

    def _fit_rf(self) -> tuple[np.ndarray, dict, list[dict], int]:
        x_train, _, x_test = self._dosage_splits()
        model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_features=self.config.rf_max_features,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=self.config.seed,
            n_jobs=self.config.rf_n_jobs,
        )
        model.fit(x_train, self.y_train)
        parameters = {
            "n_estimators": self.config.rf_n_estimators,
            "max_features": self.config.rf_max_features,
            "min_samples_leaf": self.config.rf_min_samples_leaf,
        }
        return model.predict(x_test), parameters, [], x_train.shape[1]

    def _fit_xgboost(self) -> tuple[np.ndarray, dict, list[dict], int]:
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError("运行 XGBoost 前请先执行 `uv sync`") from error

        x_train, x_val, x_test = self._dosage_splits()
        multi_strategy = (
            "one_output_per_tree"
            if self.config.xgb_device.startswith("cuda")
            else "multi_output_tree"
        )
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            multi_strategy=multi_strategy,
            n_estimators=self.config.xgb_n_estimators,
            learning_rate=self.config.xgb_learning_rate,
            max_depth=self.config.xgb_max_depth,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            early_stopping_rounds=self.config.xgb_early_stopping_rounds,
            random_state=self.config.seed,
            n_jobs=self.config.xgb_n_jobs,
            device=self.config.xgb_device,
        )
        model.fit(
            x_train,
            self.y_train,
            eval_set=[(x_val, self.y_val)],
            verbose=False,
        )
        best_iteration = getattr(model, "best_iteration", None)
        validation = []
        results = model.evals_result()
        for metric_name, values in results.get("validation_0", {}).items():
            validation.extend(
                {"iteration": index, "metric": metric_name, "value": value}
                for index, value in enumerate(values)
            )
        parameters = {
            "n_estimators_limit": self.config.xgb_n_estimators,
            "best_iteration": best_iteration,
            "learning_rate": self.config.xgb_learning_rate,
            "max_depth": self.config.xgb_max_depth,
            "subsample": self.config.xgb_subsample,
            "colsample_bytree": self.config.xgb_colsample_bytree,
            "device": self.config.xgb_device,
            "multi_strategy": multi_strategy,
        }
        return model.predict(x_test), parameters, validation, x_train.shape[1]

    def _denormalize(self, values: np.ndarray) -> np.ndarray:
        return denormalize_phenotype(values[:, None, :], self.scalers).numpy()[:, 0, :]

    def _truth_pcc(self, prediction: np.ndarray) -> float:
        path = self.data_dir / "truth/true_gv.csv"
        if not path.is_file():
            return float("nan")
        truth_frame = pd.read_csv(path)
        if "SampleID" not in truth_frame.columns:
            raise ValueError("true_gv.csv 必须包含 SampleID 列")
        truth_ids = truth_frame.pop("SampleID").astype(str).tolist()
        if truth_ids != self.sample_ids:
            raise ValueError("true_gv.csv 的 SampleID 顺序与固定 split 不一致")
        truth = truth_frame.to_numpy(dtype=np.float64)[self.split.test_idx]
        if truth.shape != prediction.shape:
            raise ValueError("True-GV 与测试预测的形状不一致")
        return compute_metrics_vectorized(prediction, truth)["pcc"]

    def _fit_model(self, name: str) -> tuple[np.ndarray, dict, list[dict], int]:
        if name == "Train-Mean":
            return np.zeros_like(self.y_test), {}, [], 0
        if name in ("Ridge-All", "Ridge-A+D"):
            return self._fit_ridge(name)
        if name == "GBLUP-All":
            return self._fit_gblup()
        if name == "RF":
            return self._fit_rf()
        if name == "XGBoost":
            return self._fit_xgboost()
        raise ValueError(f"未知模型: {name}")

    def run(self) -> pd.DataFrame:
        """运行比较并返回仅含预注册报告字段的测试集汇总表。"""
        experiment_started = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        rows = []
        model_names = ("Train-Mean", *self.config.models)
        for name in model_names:
            model_started = time.perf_counter()
            pred_norm, selected, search, feature_count = self._fit_model(name)
            pred_denorm = self._denormalize(np.asarray(pred_norm, dtype=np.float32))
            runtime_seconds = time.perf_counter() - model_started

            metrics = compute_metrics_vectorized(pred_denorm, self.target_test)
            true_gv_pcc = self._truth_pcc(pred_denorm)
            model_dir = self.output_dir / name
            model_dir.mkdir()
            pd.DataFrame(search).to_csv(
                model_dir / "validation_history.csv", index=False
            )
            np.savez_compressed(
                model_dir / "test_predictions.npz",
                sample_idx=self.split.test_idx,
                sample_ids=np.asarray(self.sample_ids)[self.split.test_idx],
                prediction=pred_denorm.astype(np.float32),
                target=self.target_test.astype(np.float32),
            )
            details = {
                "scenario": self.config.scenario_label,
                "model": name,
                "raw_snp_count": self.raw_snp_count,
                "model_input_snp_count": int(self.genotype.shape[1]),
                "effective_feature_count": feature_count,
                "selected_parameters": selected,
                "test_metrics": metrics,
                "true_gv_pcc": true_gv_pcc,
                "runtime_seconds": runtime_seconds,
                "split_sha256": _sha256(self.split_path),
            }
            (model_dir / "metrics.json").write_text(
                json.dumps(details, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
                encoding="utf-8",
            )
            rows.append(
                {
                    "model": name,
                    "PCC ↑": metrics["pcc"],
                    "R² ↑": metrics["r2"],
                    "RMSE ↓": metrics["rmse"],
                    "MAE ↓": metrics["mae"],
                    "True-GV PCC ↑": true_gv_pcc,
                    "Runtime (s) ↓": runtime_seconds,
                }
            )
            print(
                f"{name}: PCC={metrics['pcc']:.3f}, R²={metrics['r2']:.3f}, "
                f"Runtime={runtime_seconds:.3f}s"
            )

        summary = pd.DataFrame(rows, columns=REPORT_COLUMNS)
        summary.to_csv(self.output_dir / "comparison_summary.csv", index=False)
        protocol = {
            "schema_version": 2,
            "scenario": self.config.scenario_label,
            "scope_statement": (
                "结果仅适用于动态非线性遗传场景，不代表所有遗传结构。"
            ),
            "config": asdict(self.config),
            "data_manifest": str(self.manifest_path),
            "genotype_sha256": _sha256(self.genotype_path),
            "phenotype_sha256": _sha256(self.phenotype_path),
            "split_path": str(self.split_path),
            "split_sha256": _sha256(self.split_path),
            "split_sizes": {
                "train": len(self.split.train_idx),
                "val": len(self.split.val_idx),
                "test": len(self.split.test_idx),
            },
            "raw_snp_policy": (
                "Oracle 模式只使用真值 QTL 输入；普通模式除 Train-Mean 外使用全部原始 SNP。"
                if self.config.preselected_snp_indices is not None
                else "除不使用 SNP 的 Train-Mean 外，所有模型直接读取全部原始 SNP；"
                "不读取或使用任何预筛选 SNP 索引。树模型内部的随机列采样属于模型训练。"
            ),
            "fit_scope": "preprocessing=train; tuning_or_early_stop=val; test=evaluate_once",
            "metrics_protocol": METRICS_PROTOCOL,
            "true_gv_usage": "post-fit test diagnostic only",
            "report_columns": list(REPORT_COLUMNS),
            "elapsed_seconds": time.perf_counter() - experiment_started,
        }
        (self.output_dir / "protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
