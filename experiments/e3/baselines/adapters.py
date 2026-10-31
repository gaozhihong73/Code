"""首批 E3 传统统计与机器学习适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


@dataclass
class FitResult:
    best_parameters: dict[str, Any]
    best_epoch: int | None
    validation_history: list[dict[str, Any]]
    effective_feature_count: int
    parameter_count: int | None
    best_epochs_by_time: list[int] | None = None


class BaseAdapter:
    name = ""
    uses_genotype = True

    def __init__(self, *, seed: int, options: dict[str, Any]):
        self.seed = seed
        self.options = options

    def fit(
        self,
        x_train: np.ndarray | None,
        y_train: np.ndarray,
        x_val: np.ndarray | None,
        y_val: np.ndarray,
    ) -> FitResult:
        raise NotImplementedError

    def predict(self, x_test: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError

    def preprocessing_arrays(self) -> dict[str, np.ndarray]:
        return {}

    def save_artifacts(self, output_dir) -> dict[str, Any]:
        return {}

    def resource_metrics(self) -> dict[str, float | None]:
        return {}


class MeanAdapter(BaseAdapter):
    name = "Mean"
    uses_genotype = False

    def fit(self, x_train, y_train, x_val, y_val) -> FitResult:
        self.mean_ = np.mean(y_train, axis=0, dtype=np.float64).astype(np.float32)
        prediction = np.broadcast_to(self.mean_, y_val.shape)
        return FitResult({}, None, [{"candidate": 0, "val_rmse": _rmse(prediction, y_val)}], 0, 0)

    def predict(self, x_test) -> np.ndarray:
        if x_test is not None:
            raise ValueError("Mean 不得接收 SNP")
        return np.broadcast_to(self.mean_, (self.test_size, len(self.mean_))).copy()


class _FeatureScaler:
    def fit(self, values: np.ndarray) -> "_FeatureScaler":
        self.mean_ = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.scale_ = values.std(axis=0, dtype=np.float64).astype(np.float32)
        self.keep_ = self.scale_ > 1e-12
        if not self.keep_.any():
            raise ValueError("训练集没有可用的变异 SNP")
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (
            (values[:, self.keep_].astype(np.float32) - self.mean_[self.keep_])
            / self.scale_[self.keep_]
        )


class RidgeAdapter(BaseAdapter):
    def __init__(self, *, seed: int, options: dict[str, Any], dominance: bool):
        super().__init__(seed=seed, options=options)
        self.dominance = dominance
        self.name = "Ridge-A+D" if dominance else "Ridge-All"

    def _features(self, values: np.ndarray, *, fit: bool) -> np.ndarray:
        if fit:
            self.additive_scaler = _FeatureScaler().fit(values)
        additive = self.additive_scaler.transform(values)
        if not self.dominance:
            return additive
        dominance = (values == 1).astype(np.float32)
        if fit:
            self.dominance_scaler = _FeatureScaler().fit(dominance)
        return np.concatenate((additive, self.dominance_scaler.transform(dominance)), axis=1)

    def fit(self, x_train, y_train, x_val, y_val) -> FitResult:
        train = self._features(x_train, fit=True)
        val = self._features(x_val, fit=False)
        history, best = [], None
        for alpha in self.options.get("alphas", [1.0]):
            model = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr", tol=1e-5)
            model.fit(train, y_train)
            score = _rmse(model.predict(val), y_val)
            history.append({"alpha": float(alpha), "val_rmse": score})
            if best is None or score < best[0]:
                best = (score, float(alpha), model)
        self.model = best[2]
        parameter_count = int(self.model.coef_.size + self.model.intercept_.size)
        return FitResult(
            {"alpha": best[1]}, None, history, int(train.shape[1]), parameter_count
        )

    def predict(self, x_test) -> np.ndarray:
        return self.model.predict(self._features(x_test, fit=False))

    def preprocessing_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "additive_mean": self.additive_scaler.mean_,
            "additive_scale": self.additive_scaler.scale_,
            "additive_keep": self.additive_scaler.keep_,
        }
        if self.dominance:
            arrays.update(
                dominance_mean=self.dominance_scaler.mean_,
                dominance_scale=self.dominance_scaler.scale_,
                dominance_keep=self.dominance_scaler.keep_,
            )
        return arrays


class GBLUPAdapter(BaseAdapter):
    name = "GBLUP-All"

    def fit(self, x_train, y_train, x_val, y_val) -> FitResult:
        self.mean_ = y_train.mean(axis=0, dtype=np.float64)
        p = x_train.mean(axis=0, dtype=np.float64) / 2.0
        self.keep = (p > 0.0) & (p < 1.0)
        p = p[self.keep].astype(np.float32)
        self.p = p
        self.z_train = x_train[:, self.keep].astype(np.float32) - 2.0 * p
        self.denominator = float(np.sum(2.0 * p * (1.0 - p)))
        if self.denominator <= 0:
            raise ValueError("GBLUP 没有可用多态 SNP")
        kernel = (self.z_train @ self.z_train.T) / self.denominator
        eigenvalues, eigenvectors = eigh(kernel, check_finite=False)
        self.eigenvalues = np.maximum(eigenvalues, 0.0)
        self.eigenvectors = eigenvectors
        self.projected_y = eigenvectors.T @ (y_train - self.mean_)
        k_val = ((x_val[:, self.keep] - 2.0 * p) @ self.z_train.T) / self.denominator
        history, best = [], None
        for value in self.options.get("lambdas", [1.0]):
            regularization = float(value)
            weights = eigenvectors @ (
                self.projected_y / (self.eigenvalues[:, None] + regularization)
            )
            score = _rmse(k_val @ weights + self.mean_, y_val)
            history.append({"lambda": regularization, "val_rmse": score})
            if best is None or score < best[0]:
                best = (score, regularization, weights)
        self.weights = best[2]
        return FitResult(
            {"lambda": best[1]}, None, history, int(self.keep.sum()), int(self.weights.size)
        )

    def predict(self, x_test) -> np.ndarray:
        kernel = ((x_test[:, self.keep] - 2.0 * self.p) @ self.z_train.T) / self.denominator
        return kernel @ self.weights + self.mean_

    def preprocessing_arrays(self) -> dict[str, np.ndarray]:
        return {
            "polymorphic_snp_mask": self.keep,
            "training_allele_frequency": self.p,
            "training_time_mean": self.mean_,
            "kernel_denominator": np.asarray(self.denominator),
        }


class RandomForestAdapter(BaseAdapter):
    name = "Random Forest"

    def fit(self, x_train, y_train, x_val, y_val) -> FitResult:
        candidates = self.options.get("candidates") or [
            {"n_estimators": 200, "max_features": 0.1, "min_samples_leaf": 1}
        ]
        history, best = [], None
        for index, parameters in enumerate(candidates):
            model = RandomForestRegressor(
                **parameters,
                random_state=self.seed,
                n_jobs=int(self.options.get("n_jobs", -1)),
            )
            model.fit(x_train, y_train)
            score = _rmse(model.predict(x_val), y_val)
            history.append({"candidate": index, **parameters, "val_rmse": score})
            if best is None or score < best[0]:
                best = (score, dict(parameters), model)
        self.model = best[2]
        nodes = sum(tree.tree_.node_count for tree in self.model.estimators_)
        return FitResult(best[1], None, history, int(x_train.shape[1]), int(nodes))

    def predict(self, x_test) -> np.ndarray:
        return self.model.predict(x_test)


class XGBoostAdapter(BaseAdapter):
    name = "XGBoost"

    def fit(self, x_train, y_train, x_val, y_val) -> FitResult:
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError("运行 XGBoost 前请安装锁定依赖") from error
        candidates = self.options.get("candidates") or [
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4}
        ]
        history, best = [], None
        for index, parameters in enumerate(candidates):
            model = XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                multi_strategy="one_output_per_tree",
                early_stopping_rounds=int(self.options.get("early_stopping_rounds", 20)),
                random_state=self.seed,
                n_jobs=int(self.options.get("n_jobs", -1)),
                device=self.options.get("device", "cpu"),
                **parameters,
            )
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            score = _rmse(model.predict(x_val), y_val)
            best_iteration = getattr(model, "best_iteration", None)
            row = {
                "candidate": index,
                **parameters,
                "best_iteration": best_iteration,
                "val_rmse": score,
            }
            history.append(row)
            if best is None or score < best[0]:
                best = (score, row, model)
        self.model = best[2]
        selected = {key: value for key, value in best[1].items() if key != "val_rmse"}
        epoch = selected.get("best_iteration")
        return FitResult(
            selected,
            None if epoch is None else int(epoch) + 1,
            history,
            int(x_train.shape[1]),
            None,
        )

    def predict(self, x_test) -> np.ndarray:
        return self.model.predict(x_test)
