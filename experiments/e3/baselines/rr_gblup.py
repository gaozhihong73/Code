"""基于 Legendre 时间核的 random-regression GBLUP。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import eigh

from .adapters import BaseAdapter, FitResult, _rmse


class RRGBLUPAdapter(BaseAdapter):
    """用 ``K_genomic ⊗ K_time`` 的可分离核实现动态 RR-GBLUP。"""

    name = "RR-GBLUP"

    def _genomic_kernel_fit(self, x_train: np.ndarray) -> np.ndarray:
        frequency = x_train.mean(axis=0, dtype=np.float64) / 2.0
        self.keep = (frequency > 0.0) & (frequency < 1.0)
        self.frequency = frequency[self.keep]
        self.z_train = x_train[:, self.keep].astype(np.float64) - 2.0 * self.frequency
        self.denominator = float(
            np.sum(2.0 * self.frequency * (1.0 - self.frequency))
        )
        if self.denominator <= 0:
            raise ValueError("RR-GBLUP 没有可用多态 SNP")
        return (self.z_train @ self.z_train.T) / self.denominator

    def _cross_kernel(self, values: np.ndarray) -> np.ndarray:
        centered = values[:, self.keep].astype(np.float64) - 2.0 * self.frequency
        return (centered @ self.z_train.T) / self.denominator

    @staticmethod
    def _time_kernel(n_time: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.linspace(-1.0, 1.0, n_time, dtype=np.float64)
        basis = np.polynomial.legendre.legvander(coordinates, degree)
        norms = np.sqrt(np.sum(np.square(basis), axis=0, keepdims=True))
        basis = basis / np.where(norms > 0, norms, 1.0)
        return basis @ basis.T, basis

    @staticmethod
    def _fit_coefficients(
        genomic_values: np.ndarray,
        genomic_vectors: np.ndarray,
        time_kernel: np.ndarray,
        response: np.ndarray,
        regularization: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_values, time_vectors = eigh(time_kernel, check_finite=False)
        time_values = np.maximum(time_values, 0.0)
        projected = genomic_vectors.T @ response @ time_vectors
        denominator = (
            genomic_values[:, None] * time_values[None, :] + regularization
        )
        coefficients = genomic_vectors @ (projected / denominator) @ time_vectors.T
        return coefficients, time_values, time_vectors

    def fit(self, x_train, y_train, x_val, y_val) -> FitResult:
        self.time_mean = y_train.mean(axis=0, dtype=np.float64)
        centered_y = y_train.astype(np.float64) - self.time_mean
        train_kernel = self._genomic_kernel_fit(x_train)
        genomic_values, genomic_vectors = eigh(train_kernel, check_finite=False)
        genomic_values = np.maximum(genomic_values, 0.0)
        val_kernel = self._cross_kernel(x_val)
        history: list[dict[str, Any]] = []
        best = None
        degrees = [int(value) for value in self.options.get("degrees", [3])]
        lambdas = [float(value) for value in self.options.get("lambdas", [1.0])]
        for degree in degrees:
            if degree < 0 or degree >= y_train.shape[1]:
                raise ValueError("RR-GBLUP degree 必须在 [0, n_time-1] 内")
            time_kernel, basis = self._time_kernel(y_train.shape[1], degree)
            for regularization in lambdas:
                if regularization <= 0:
                    raise ValueError("RR-GBLUP lambda 必须为正数")
                coefficients, _, _ = self._fit_coefficients(
                    genomic_values,
                    genomic_vectors,
                    time_kernel,
                    centered_y,
                    regularization,
                )
                prediction = val_kernel @ coefficients @ time_kernel + self.time_mean
                score = _rmse(prediction, y_val)
                history.append(
                    {
                        "degree": degree,
                        "lambda": regularization,
                        "val_rmse": score,
                    }
                )
                if best is None or score < best[0]:
                    best = (
                        score,
                        degree,
                        regularization,
                        coefficients,
                        time_kernel,
                        basis,
                    )
        assert best is not None
        _, degree, regularization, self.coefficients, self.time_kernel, self.basis = best
        return FitResult(
            best_parameters={"degree": degree, "lambda": regularization},
            best_epoch=None,
            validation_history=history,
            effective_feature_count=int(self.keep.sum()),
            parameter_count=int(self.coefficients.size),
        )

    def predict(self, x_test) -> np.ndarray:
        return self._cross_kernel(x_test) @ self.coefficients @ self.time_kernel + self.time_mean

    def preprocessing_arrays(self) -> dict[str, np.ndarray]:
        return {
            "polymorphic_snp_mask": self.keep,
            "training_allele_frequency": self.frequency,
            "kernel_denominator": np.asarray(self.denominator),
            "training_time_mean": self.time_mean,
            "legendre_basis": self.basis,
            "time_kernel": self.time_kernel,
            "dual_coefficients": self.coefficients,
        }
