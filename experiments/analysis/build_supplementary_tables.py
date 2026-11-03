"""Build read-only E1/E2/E3 supplementary result tables.

This script never writes into dataset or training-output directories. It reads
the frozen run artifacts, validates their identities, derives preregistered
metrics, and writes unrounded tidy CSV files plus a reproducibility manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.e3.schemas.e3 import E3RunManifest
from utils.Calculate_Metrics import METRICS_PROTOCOL, compute_metrics_vectorized


DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "dataset/experiments"
DEFAULT_E1_ROOT = REPOSITORY_ROOT / "outputs/simulation/experiment1_oracle"
DEFAULT_E2_ROOT = REPOSITORY_ROOT / "outputs/simulation/experiment2_screened"
DEFAULT_E3_ROOT = REPOSITORY_ROOT / "outputs/paper_experiments/runs/E3"
DEFAULT_TABLE_DIR = REPOSITORY_ROOT / "outputs/paper_experiments/tables"
DEFAULT_MANIFEST_DIR = REPOSITORY_ROOT / "outputs/paper_experiments/manifests"

P_LEVELS = (5000, 10000)
NTR_LEVELS = (100, 200, 400, 800, 1600)
H2_LEVELS = (0.1, 0.4, 0.8)
REPETITIONS = tuple(range(1, 11))
E3_MODELS = (
    "Mean",
    "Ridge-All",
    "Ridge-A+D",
    "GBLUP-All",
    "RR-GBLUP",
    "Random Forest",
    "XGBoost",
    "DNNGP-SNP",
    "DLGWAS",
    "WheatGP",
    "Cropformer",
    "DynaGS",
)
HIGHER_IS_BETTER = {"pcc", "time_adjusted_r2"}
LOWER_IS_BETTER = {"mse", "rmse", "mae", "nrmse"}
STATE_DICT_BUFFERS = {
    "wide_fill_values",
    "wide_layer.t_basis",
    "wide_layer.input_norm.running_mean",
    "wide_layer.input_norm.running_var",
    "wide_layer.input_norm.num_batches_tracked",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition(p: int, ntr: int, h2: float) -> str:
    return f"P{p}_NTR{ntr}_H{h2:g}"


def _parse_condition(condition: str) -> tuple[int, int, float]:
    match = re.fullmatch(r"P(\d+)_NTR(\d+)_H([0-9.]+)", condition)
    if match is None:
        raise ValueError(f"Invalid condition identifier: {condition}")
    return int(match.group(1)), int(match.group(2)), float(match.group(3))


def _slug(model: str) -> str:
    return model.lower().replace("+", "_plus_").replace("-", "_").replace(" ", "_")


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _percentile_scale(train_target: np.ndarray) -> float:
    scale = float(np.percentile(train_target, 95) - np.percentile(train_target, 5))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Training phenotype P95-P5 must be positive and finite")
    return scale


@dataclass(frozen=True)
class DatasetBundle:
    condition: str
    repetition: int
    p: int
    ntr: int
    h2: float
    data_dir: Path
    manifest_path: Path
    manifest_sha256: str
    split_path: Path
    split_sha256: str
    sample_ids: np.ndarray
    phenotype: np.ndarray
    latent: np.ndarray
    time_coordinates: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_scale: float
    simulation_seed: int
    qtl_indices: np.ndarray


def _split_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        aliases = (
            ("train_idx", "train_indices"),
            ("val_idx", "validation_indices", "val_indices"),
            ("test_idx", "test_indices"),
        )
        arrays = []
        for choices in aliases:
            key = next((name for name in choices if name in archive.files), None)
            if key is None:
                raise ValueError(f"Missing split array {choices}: {path}")
            arrays.append(np.asarray(archive[key], dtype=np.int64))
    combined = np.concatenate(arrays)
    if len(combined) != len(np.unique(combined)):
        raise ValueError(f"Split indices overlap: {path}")
    return arrays[0], arrays[1], arrays[2]


def _load_dataset(data_root: Path, condition: str, repetition: int) -> DatasetBundle:
    p, ntr, h2 = _parse_condition(condition)
    data_dir = data_root / condition / f"rep_{repetition:02d}"
    manifest_path = data_dir / "data_manifest.json"
    manifest = _json(manifest_path)
    split_path = data_dir / "splits/split_seed73.npz"
    train_idx, val_idx, test_idx = _split_arrays(split_path)
    if (len(train_idx), manifest["split"]["train_n"]) != (ntr, ntr):
        raise ValueError(f"NTR mismatch: {condition}/rep_{repetition:02d}")

    sample_frame = pd.read_csv(data_dir / "sample_ids.csv")
    if list(sample_frame.columns) != ["SampleID"]:
        raise ValueError(f"Invalid sample_ids.csv schema: {data_dir}")
    sample_ids = sample_frame["SampleID"].astype(str).to_numpy()
    phenotype_frame = pd.read_csv(data_dir / "sim_pheno.csv")
    latent_frame = pd.read_csv(data_dir / "truth/true_curve.csv")
    for name, frame in (("phenotype", phenotype_frame), ("latent", latent_frame)):
        if frame.columns[0] != "SampleID":
            raise ValueError(f"{name} first column is not SampleID: {data_dir}")
        if not np.array_equal(frame["SampleID"].astype(str).to_numpy(), sample_ids):
            raise ValueError(f"{name} SampleID order mismatch: {data_dir}")
    phenotype = phenotype_frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    latent = latent_frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    time_coordinates = np.asarray(manifest["time_coordinates"], dtype=np.float64)
    expected_shape = (len(sample_ids), len(time_coordinates))
    if phenotype.shape != expected_shape or latent.shape != expected_shape:
        raise ValueError(f"Phenotype/latent shape mismatch: {data_dir}")
    if not np.isfinite(phenotype).all() or not np.isfinite(latent).all():
        raise ValueError(f"Non-finite phenotype/latent values: {data_dir}")

    simulation = _json(data_dir / "simulation_config.json")
    causal = pd.read_csv(data_dir / "truth/causal_effects.csv")
    qtl_indices = np.sort(causal["snp_index"].to_numpy(dtype=np.int64) - 1)
    if len(qtl_indices) != 20 or len(np.unique(qtl_indices)) != 20:
        raise ValueError(f"Expected 20 unique QTL: {data_dir}")
    return DatasetBundle(
        condition=condition,
        repetition=repetition,
        p=p,
        ntr=ntr,
        h2=h2,
        data_dir=data_dir.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest_sha256=_sha256(manifest_path),
        split_path=split_path.resolve(),
        split_sha256=_sha256(split_path),
        sample_ids=sample_ids,
        phenotype=phenotype,
        latent=latent,
        time_coordinates=time_coordinates,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_scale=_percentile_scale(phenotype[train_idx]),
        simulation_seed=int(simulation["seeds"]["simulation"]),
        qtl_indices=qtl_indices,
    )


def _load_e1_e2_prediction(path: Path, bundle: DatasetBundle) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        sample_idx = np.asarray(archive["sample_idx"], dtype=np.int64)
        prediction = np.asarray(archive["predictions"][:, 0, :], dtype=np.float64)
        target = np.asarray(archive["targets"][:, 0, :], dtype=np.float64)
        times = np.asarray(archive["time_coordinates"], dtype=np.float64)
    if not np.array_equal(sample_idx, bundle.test_idx):
        raise ValueError(f"Prediction test indices differ from frozen split: {path}")
    if not np.array_equal(times, bundle.time_coordinates):
        raise ValueError(f"Prediction time coordinates differ from manifest: {path}")
    if not np.allclose(target, bundle.phenotype[sample_idx], rtol=0, atol=5e-6):
        raise ValueError(f"Prediction targets differ from source phenotype: {path}")
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid prediction matrix: {path}")
    return sample_idx, prediction


def _load_e3_prediction(path: Path, bundle: DatasetBundle) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        sample_idx = np.asarray(archive["sample_idx"], dtype=np.int64)
        prediction = np.asarray(archive["prediction"], dtype=np.float64)
        target = np.asarray(archive["target"], dtype=np.float64)
        times = np.asarray(archive["time_coordinates"], dtype=np.float64)
    if not np.array_equal(sample_idx, bundle.test_idx):
        raise ValueError(f"E3 prediction test indices differ from split: {path}")
    if not np.array_equal(times, bundle.time_coordinates):
        raise ValueError(f"E3 time coordinates differ from manifest: {path}")
    if not np.allclose(target, bundle.phenotype[sample_idx], rtol=0, atol=5e-6):
        raise ValueError(f"E3 targets differ from source phenotype: {path}")
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid E3 prediction matrix: {path}")
    return sample_idx, prediction


def _run_fields(
    run_id: str,
    experiment: str,
    bundle: DatasetBundle,
    model: str,
    mode: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment": experiment,
        "condition": bundle.condition,
        "p": bundle.p,
        "ntr": bundle.ntr,
        "heritability": bundle.h2,
        "repetition": bundle.repetition,
        "simulation_seed": bundle.simulation_seed,
        "model": model,
        "mode": mode,
    }


def _metric_rows(
    fields: dict[str, Any],
    target_name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    scale: float,
    times: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = compute_metrics_vectorized(prediction, target)
    values = {
        "mse": raw["mse"],
        "rmse": raw["rmse"],
        "mae": raw["mae"],
        "pcc": raw["pcc"],
        "time_adjusted_r2": raw["r2"],
        "nrmse": raw["rmse"] / scale,
    }
    overall = []
    for metric, value in values.items():
        pcc = metric == "pcc"
        overall.append(
            {
                **fields,
                "target": target_name,
                "metric": metric,
                "value": _finite(value),
                "valid_n": int(raw["pcc_valid_count"] if pcc else target.size),
                "total_n": int(raw["pcc_total_count"] if pcc else target.size),
                "train_p95_p5": scale,
                "metric_protocol": METRICS_PROTOCOL,
            }
        )

    by_time = []
    for index, time in enumerate(times):
        pred = prediction[:, index]
        truth = target[:, index]
        error = pred - truth
        rmse = float(np.sqrt(np.mean(np.square(error))))
        pred_variable = float(np.std(pred)) > 1e-12
        truth_variable = float(np.std(truth)) > 1e-12
        pcc = None
        spearman = None
        if pred_variable and truth_variable:
            pcc = _finite(np.corrcoef(pred, truth)[0, 1])
            spearman = _finite(spearmanr(pred, truth).statistic)
        time_values = {
            "rmse": rmse,
            "nrmse": rmse / scale,
            "mae": float(np.mean(np.abs(error))),
            "pcc": pcc,
            "spearman": spearman,
        }
        for metric, value in time_values.items():
            correlation = metric in {"pcc", "spearman"}
            by_time.append(
                {
                    **fields,
                    "target": target_name,
                    "time_index": index,
                    "time": float(time),
                    "metric": metric,
                    "value": value,
                    "valid_n": int(len(truth) if not correlation or value is not None else 0),
                    "total_n": int(len(truth)),
                    "train_p95_p5": scale,
                }
            )
    return overall, by_time


def _ccc(prediction: np.ndarray, target: np.ndarray) -> float | None:
    pred_var = float(np.var(prediction))
    target_var = float(np.var(target))
    denominator = pred_var + target_var + float(np.mean(prediction) - np.mean(target)) ** 2
    if denominator <= 1e-12:
        return None
    covariance = float(np.mean((prediction - prediction.mean()) * (target - target.mean())))
    return _finite(2.0 * covariance / denominator)


def _shape_error(prediction: np.ndarray, target: np.ndarray) -> float | None:
    pred_sd = float(np.std(prediction))
    target_sd = float(np.std(target))
    if pred_sd <= 1e-12 or target_sd <= 1e-12:
        return None
    pred_z = (prediction - prediction.mean()) / pred_sd
    target_z = (target - target.mean()) / target_sd
    return float(np.sqrt(np.mean(np.square(pred_z - target_z))))


def _inflection(
    values: np.ndarray,
    times: np.ndarray,
    scale: float,
) -> tuple[float | None, float | None, str]:
    if float(np.ptp(values)) < 0.01 * scale:
        return None, None, "amplitude_below_one_percent_train_scale"
    dense = np.linspace(float(times[0]), float(times[-1]), 1000)
    interpolator = PchipInterpolator(times, values, extrapolate=False)
    derivative = np.asarray(interpolator.derivative()(dense), dtype=np.float64)
    margin = 0.05 * float(times[-1] - times[0])
    internal = np.flatnonzero((dense >= times[0] + margin) & (dense <= times[-1] - margin))
    if len(internal) < 3:
        return None, None, "insufficient_internal_grid"
    local_index = int(np.argmax(derivative[internal]))
    if local_index in (0, len(internal) - 1):
        return None, None, "maximum_derivative_on_internal_boundary"
    index = int(internal[local_index])
    rate = float(derivative[index])
    if not math.isfinite(rate) or rate <= 0:
        return None, None, "nonpositive_maximum_derivative"
    return float(dense[index]), rate, "valid"


def _curve_rows(
    fields: dict[str, Any],
    sample_idx: np.ndarray,
    sample_ids: np.ndarray,
    prediction: np.ndarray,
    observed: np.ndarray,
    latent: np.ndarray,
    times: np.ndarray,
    scale: float,
) -> list[dict[str, Any]]:
    rows = []
    dense = np.linspace(float(times[0]), float(times[-1]), 1000)
    interval = float(np.median(np.diff(times)))
    for row_index, source_index in enumerate(sample_idx):
        pred = prediction[row_index]
        pred_dense = PchipInterpolator(times, pred, extrapolate=False)(dense)
        pred_inflection, pred_rate, pred_reason = _inflection(pred, times, scale)
        for target_name, matrix in (("observed", observed), ("latent", latent)):
            truth = matrix[row_index]
            truth_dense = PchipInterpolator(times, truth, extrapolate=False)(dense)
            curve_nrmse = float(np.sqrt(np.mean(np.square(pred - truth))) / scale)
            niae = float(np.trapz(np.abs(pred_dense - truth_dense), dense) / ((times[-1] - times[0]) * scale))
            truth_inflection = truth_rate = None
            truth_reason = "not_computed_for_observed_reference"
            both_valid = False
            error = abs_error = within = rate_relative_error = None
            if target_name == "latent":
                truth_inflection, truth_rate, truth_reason = _inflection(truth, times, scale)
                both_valid = truth_inflection is not None and pred_inflection is not None
                if both_valid:
                    error = float(pred_inflection - truth_inflection)
                    abs_error = abs(error)
                    within = float(abs_error <= interval)
                    rate_relative_error = float(abs(pred_rate - truth_rate) / abs(truth_rate))
            rows.append(
                {
                    **fields,
                    "sample_idx": int(source_index),
                    "sample_id": str(sample_ids[row_index]),
                    "target": target_name,
                    "curve_nrmse": curve_nrmse,
                    "ccc": _ccc(pred, truth),
                    "niae": niae,
                    "shape_error": _shape_error(pred, truth),
                    "predicted_inflection_time": pred_inflection if target_name == "latent" else None,
                    "true_inflection_time": truth_inflection,
                    "inflection_error": error,
                    "inflection_abs_error": abs_error,
                    "within_one_sampling_interval": within,
                    "predicted_max_growth_rate": pred_rate if target_name == "latent" else None,
                    "true_max_growth_rate": truth_rate,
                    "max_growth_rate_relative_error": rate_relative_error,
                    "predicted_inflection_status": pred_reason if target_name == "latent" else "not_computed_for_observed_reference",
                    "true_inflection_status": truth_reason,
                    "inflection_both_valid": both_valid,
                    "train_p95_p5": scale,
                    "sampling_interval": interval,
                }
            )
    return rows


def _parameter_count(checkpoint: Path) -> int:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not a state_dict: {checkpoint}")
    return int(
        sum(
            value.numel()
            for key, value in state.items()
            if isinstance(value, torch.Tensor) and key not in STATE_DICT_BUFFERS
        )
    )


def _screening_rows(
    fields: dict[str, Any],
    bundle: DatasetBundle,
    selected_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    selected = np.asarray(np.load(selected_path, allow_pickle=False), dtype=np.int64)
    if selected.shape != (20,) or len(np.unique(selected)) != 20:
        raise ValueError(f"Expected 20 unique screened SNPs: {selected_path}")
    qtl = bundle.qtl_indices
    exact = np.isin(qtl, selected)

    genotype = np.load(bundle.data_dir / "sim_snp.npy", mmap_mode="r", allow_pickle=False)
    columns = np.unique(np.concatenate((qtl, selected)))
    train = np.asarray(genotype[np.ix_(bundle.train_idx, columns)], dtype=np.float64)
    lookup = {int(column): index for index, column in enumerate(columns)}
    qtl_values = train[:, [lookup[int(column)] for column in qtl]]
    selected_values = train[:, [lookup[int(column)] for column in selected]]
    combined = np.concatenate((qtl_values, selected_values), axis=1)
    for column in range(combined.shape[1]):
        values = combined[:, column]
        missing = ~np.isfinite(values) | (values < 0)
        if missing.any():
            observed = values[~missing].astype(np.int64)
            if not len(observed):
                raise ValueError(f"All training dosages missing for a screening SNP: {selected_path}")
            counts = np.bincount(observed, minlength=3)
            values[missing] = int(np.flatnonzero(counts == counts.max())[0])
    qtl_values = combined[:, : len(qtl)]
    selected_values = combined[:, len(qtl) :]
    qtl_centered = qtl_values - qtl_values.mean(axis=0)
    selected_centered = selected_values - selected_values.mean(axis=0)
    numerator = qtl_centered.T @ selected_centered
    denominator = np.sqrt(
        np.sum(np.square(qtl_centered), axis=0)[:, None]
        * np.sum(np.square(selected_centered), axis=0)[None, :]
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    r2 = np.square(correlation)
    best_selected_position = np.argmax(r2, axis=1)
    max_r2 = r2[np.arange(len(qtl)), best_selected_position]
    # An exact selected QTL is always a recovered QTL. This keeps LD-aware
    # recall at least as large as exact recall even when a QTL is monomorphic
    # in a small training split and its empirical correlation is undefined.
    ld_hit = exact | (max_r2 >= 0.8)
    exact_hits = int(exact.sum())
    ld_hits = int(ld_hit.sum())
    k = int(len(selected))
    exact_recall = exact_hits / len(qtl)
    precision = exact_hits / k
    f1 = 0.0 if exact_recall + precision == 0 else 2 * exact_recall * precision / (exact_recall + precision)
    expected_random_hits = k * len(qtl) / bundle.p
    summary = {
        **fields,
        "k": k,
        "qtl_count": int(len(qtl)),
        "exact_hit_count": exact_hits,
        "ld_hit_count": ld_hits,
        "exact_recall": exact_recall,
        "precision": precision,
        "f1": f1,
        "ld_aware_recall": ld_hits / len(qtl),
        "random_expected_hit_count": expected_random_hits,
        "random_enrichment_fold": exact_hits / expected_random_hits,
        "ld_r2_threshold": 0.8,
        "ld_estimation_scope": "training_individuals_only",
        "ranking_available": False,
        "ranking_unavailable_reason": "intermediate_screening_scores_not_saved",
    }
    qtl_rows = []
    for index, qtl_index in enumerate(qtl):
        selected_index = int(selected[best_selected_position[index]])
        qtl_rows.append(
            {
                **fields,
                "qtl_index_zero_based": int(qtl_index),
                "qtl_index_one_based": int(qtl_index + 1),
                "exact_hit": bool(exact[index]),
                "ld_hit": bool(ld_hit[index]),
                "best_tag_snp_zero_based": selected_index,
                "best_tag_snp_one_based": selected_index + 1,
                "best_tag_train_r2": float(max_r2[index]),
                "ld_r2_threshold": 0.8,
            }
        )
    selected_rows = []
    for storage_position, selected_index in enumerate(selected):
        qtl_position = int(np.argmax(r2[:, storage_position]))
        selected_rows.append(
            {
                **fields,
                "storage_position": storage_position,
                "storage_position_is_rank": False,
                "snp_index_zero_based": int(selected_index),
                "snp_index_one_based": int(selected_index + 1),
                "exact_qtl": bool(selected_index in set(qtl.tolist())),
                "best_qtl_index_zero_based": int(qtl[qtl_position]),
                "best_qtl_index_one_based": int(qtl[qtl_position] + 1),
                "best_qtl_train_r2": float(r2[qtl_position, storage_position]),
                "tags_any_qtl_at_r2_0_8": bool((r2[:, storage_position] >= 0.8).any()),
            }
        )
    return summary, qtl_rows, selected_rows


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _summaries(frame: pd.DataFrame, group_columns: list[str], value_column: str, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(group[value_column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        low, high = _bootstrap_interval(finite, rng, n_boot)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "mean": float(np.mean(finite)) if len(finite) else None,
                "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
                "median": float(np.median(finite)) if len(finite) else None,
                "q1": float(np.percentile(finite, 25)) if len(finite) else None,
                "q3": float(np.percentile(finite, 75)) if len(finite) else None,
                "ci_low": low,
                "ci_high": high,
                "valid_n": int(len(finite)),
                "total_n": int(len(values)),
                "bootstrap_replicates": n_boot,
                "bootstrap_seed": seed,
                "bootstrap_unit": "independent_simulation_repetition",
            }
        )
    return pd.DataFrame(rows)


def _curve_run_summaries(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_aggregation = {
        "curve_nrmse": "mean",
        "ccc": "median",
        "niae": "mean",
        "shape_error": "mean",
        "inflection_abs_error": "mean",
        "inflection_error": "mean",
        "within_one_sampling_interval": "mean",
        "max_growth_rate_relative_error": "mean",
    }
    identifiers = [
        "run_id", "experiment", "condition", "p", "ntr", "heritability",
        "repetition", "model", "mode", "target",
    ]
    rows = []
    for keys, group in curves.groupby(identifiers, dropna=False, sort=True):
        base = dict(zip(identifiers, keys))
        for metric, aggregation in metric_aggregation.items():
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            if aggregation == "median":
                run_value = float(np.median(finite)) if len(finite) else None
            else:
                run_value = float(np.mean(finite)) if len(finite) else None
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "run_value": run_value,
                    "aggregation_across_individuals": aggregation,
                    "mean": float(np.mean(finite)) if len(finite) else None,
                    "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
                    "median": float(np.median(finite)) if len(finite) else None,
                    "q1": float(np.percentile(finite, 25)) if len(finite) else None,
                    "q3": float(np.percentile(finite, 75)) if len(finite) else None,
                    "valid_n": int(len(finite)),
                    "total_n": int(len(values)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(
        [
            {"metric": metric, "aggregation_across_individuals": aggregation}
            for metric, aggregation in metric_aggregation.items()
        ]
    )


def _paired_tables(metrics: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["condition", "p", "ntr", "heritability", "repetition", "target", "metric"]
    e12 = metrics[metrics["experiment"].isin(["E1", "E2"])].pivot(index=key, columns="experiment", values="value").reset_index()
    e12 = e12.dropna(subset=["E1", "E2"], how="any")
    e12["comparison"] = "Oracle vs Screened"
    e12["reference_model"] = "Oracle"
    e12["comparison_model"] = "Screened"
    e12["reference_value"] = e12["E1"]
    e12["comparison_value"] = e12["E2"]
    e12["delta_raw_reference_minus_comparison"] = e12["E1"] - e12["E2"]
    e12["advantage_reference"] = np.where(
        e12["metric"].isin(HIGHER_IS_BETTER),
        e12["E1"] - e12["E2"],
        e12["E2"] - e12["E1"],
    )
    e12["positive_advantage_means"] = "Oracle_better"
    e12["experiment"] = "E1_vs_E2"

    e3 = metrics[(metrics["experiment"] == "E3") & metrics["model"].isin(E3_MODELS)]
    e3_index = ["condition", "p", "ntr", "heritability", "repetition", "target", "metric"]
    wide = e3.pivot(index=e3_index, columns="model", values="value").reset_index()
    e3_rows = []
    for model in E3_MODELS:
        if model == "DynaGS":
            continue
        subset = wide.dropna(subset=["DynaGS", model], how="any").copy()
        subset["comparison"] = f"DynaGS vs {model}"
        subset["reference_model"] = "DynaGS"
        subset["comparison_model"] = model
        subset["reference_value"] = subset["DynaGS"]
        subset["comparison_value"] = subset[model]
        subset["delta_raw_reference_minus_comparison"] = subset["DynaGS"] - subset[model]
        subset["advantage_reference"] = np.where(
            subset["metric"].isin(HIGHER_IS_BETTER),
            subset["DynaGS"] - subset[model],
            subset[model] - subset["DynaGS"],
        )
        subset["positive_advantage_means"] = "DynaGS_better"
        subset["experiment"] = "E3"
        e3_rows.append(subset)
    paired = pd.concat([e12, *e3_rows], ignore_index=True, sort=False)
    keep = [
        "experiment", "condition", "p", "ntr", "heritability", "repetition",
        "target", "metric", "comparison", "reference_model", "comparison_model",
        "reference_value", "comparison_value", "delta_raw_reference_minus_comparison",
        "advantage_reference", "positive_advantage_means",
    ]
    paired = paired[keep]
    summary = _summaries(
        paired,
        ["experiment", "condition", "p", "ntr", "heritability", "target", "metric", "comparison", "reference_model", "comparison_model", "positive_advantage_means"],
        "advantage_reference",
        n_boot,
        seed,
    )
    return paired, summary


def _write_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {"path": str(path.resolve()), "rows": int(len(frame)), "sha256": _sha256(path)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    screening_rows: list[dict[str, Any]] = []
    screening_qtl_rows: list[dict[str, Any]] = []
    selected_snp_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    training_history_rows: list[pd.DataFrame] = []
    e3_validation_rows: list[pd.DataFrame] = []
    prediction_cache: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}

    for p in P_LEVELS:
        for ntr in NTR_LEVELS:
            for h2 in H2_LEVELS:
                condition = _condition(p, ntr, h2)
                for repetition in REPETITIONS:
                    bundle = _load_dataset(args.data_root, condition, repetition)
                    for experiment, mode, root in (
                        ("E1", "Oracle", args.e1_root),
                        ("E2", "Screened", args.e2_root),
                    ):
                        run_dir = root / condition / f"rep_{repetition:02d}"
                        run_id = f"{experiment}_{condition}_rep{repetition:02d}_dynags_{mode.lower()}_seed73"
                        fields = _run_fields(run_id, experiment, bundle, "DynaGS", mode)
                        preprocessing = _json(run_dir / "preprocessing/preprocessing.json")
                        complete = _json(run_dir / "run_complete.json")
                        summary = _json(run_dir / "training_history/training_summary.json")
                        prediction_path = run_dir / "test_predictions.npz"
                        sample_idx, prediction = _load_e1_e2_prediction(prediction_path, bundle)
                        prediction_cache[(experiment, repetition, condition)] = (sample_idx, prediction)
                        run_rows.append(
                            {
                                **fields,
                                "status": "succeeded",
                                "data_dir": str(bundle.data_dir),
                                "dataset_manifest_path": str(bundle.manifest_path),
                                "dataset_manifest_sha256": bundle.manifest_sha256,
                                "split_path": str(bundle.split_path),
                                "split_sha256": bundle.split_sha256,
                                "sample_ids_sha256": preprocessing["sample_ids_sha256"],
                                "train_indices_sha256": preprocessing["train_indices_sha256"],
                                "predictions_path": str(prediction_path.resolve()),
                                "predictions_sha256": _sha256(prediction_path),
                                "raw_snp_count": preprocessing["original_snp_dim"],
                                "model_input_snp_count": preprocessing["snp_dim"],
                                "test_individual_count": len(sample_idx),
                                "time_point_count": len(bundle.time_coordinates),
                                "model_initialization_seed": preprocessing["model_initialization_seed"],
                                "ablation_mode": _json(run_dir / "models/model_config.json")["ablation_mode"],
                                "source_identity_note": "condition/rep/model reconstructed from directory and verified against data_dir and hashes",
                            }
                        )
                        observed = bundle.phenotype[sample_idx]
                        latent = bundle.latent[sample_idx]
                        for target_name, target in (("observed", observed), ("latent", latent)):
                            overall, by_time = _metric_rows(fields, target_name, prediction, target, bundle.train_scale, bundle.time_coordinates)
                            metric_rows.extend(overall)
                            time_rows.extend(by_time)
                        curve_rows.extend(
                            _curve_rows(
                                fields,
                                sample_idx,
                                bundle.sample_ids[sample_idx],
                                prediction,
                                observed,
                                latent,
                                bundle.time_coordinates,
                                bundle.train_scale,
                            )
                        )
                        resource_rows.append(
                            {
                                **fields,
                                "training_time_seconds": float(summary["training_time_seconds"]),
                                "screening_time_seconds": None,
                                "peak_cpu_rss_mb": None,
                                "peak_gpu_memory_mb": None,
                                "parameter_count": _parameter_count(run_dir / "models/best_model.pt"),
                                "best_epoch": int(summary["best_epoch"]),
                                "total_epochs": int(summary["total_epochs"]),
                                "model_input_snp_count": preprocessing["snp_dim"],
                                "timing_scope": "training_loop_only_excludes_screening_and_initialization",
                            }
                        )
                        history = pd.read_csv(run_dir / "training_history/training_history_metrics.csv")
                        for key, value in fields.items():
                            history[key] = value
                        history["best_epoch"] = int(summary["best_epoch"])
                        training_history_rows.append(history)
                        if experiment == "E2":
                            screening, qtl_detail, selected_detail = _screening_rows(
                                fields,
                                bundle,
                                run_dir / "preprocessing/screened_indices.npy",
                            )
                            screening_rows.append(screening)
                            screening_qtl_rows.extend(qtl_detail)
                            selected_snp_rows.extend(selected_detail)

    e3_condition = "P10000_NTR400_H0.8"
    for repetition in REPETITIONS:
        bundle = _load_dataset(args.data_root, e3_condition, repetition)
        reference_target: np.ndarray | None = None
        reference_idx: np.ndarray | None = None
        for model in E3_MODELS:
            run_dir = args.e3_root / e3_condition / f"rep_{repetition:02d}" / _slug(model)
            manifest_path = run_dir / "run_manifest.json"
            manifest = E3RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            if manifest.status != "succeeded":
                raise ValueError(f"E3 run is not succeeded: {manifest.run_id}")
            if manifest.data.data_manifest_sha256 != bundle.manifest_sha256 or manifest.data.split.sha256 != bundle.split_sha256:
                raise ValueError(f"E3 dataset/split hash mismatch: {manifest.run_id}")
            fields = _run_fields(manifest.run_id, "E3", bundle, model, "End-to-end")
            prediction_path = Path(manifest.prediction.path)
            sample_idx, prediction = _load_e3_prediction(prediction_path, bundle)
            observed = bundle.phenotype[sample_idx]
            latent = bundle.latent[sample_idx]
            if reference_target is None:
                reference_idx = sample_idx.copy()
                reference_target = observed.copy()
            elif not np.array_equal(sample_idx, reference_idx) or not np.array_equal(observed, reference_target):
                raise ValueError(f"E3 pairing mismatch: rep={repetition}, model={model}")
            run_rows.append(
                {
                    **fields,
                    "status": manifest.status,
                    "data_dir": manifest.data.data_dir,
                    "dataset_manifest_path": manifest.data.data_manifest_path,
                    "dataset_manifest_sha256": manifest.data.data_manifest_sha256,
                    "split_path": manifest.data.split.path,
                    "split_sha256": manifest.data.split.sha256,
                    "sample_ids_sha256": manifest.data.split.sample_ids_sha256,
                    "train_indices_sha256": manifest.data.split.train_indices_sha256,
                    "predictions_path": str(prediction_path.resolve()),
                    "predictions_sha256": manifest.prediction.sha256,
                    "raw_snp_count": manifest.data.raw_snp_count,
                    "model_input_snp_count": manifest.data.model_input_snp_count,
                    "test_individual_count": len(sample_idx),
                    "time_point_count": manifest.data.n_time_points,
                    "model_initialization_seed": manifest.seed,
                    "ablation_mode": "add" if model == "DynaGS" else None,
                    "source_identity_note": "explicit E3 formal run manifest",
                }
            )
            for target_name, target in (("observed", observed), ("latent", latent)):
                overall, by_time = _metric_rows(fields, target_name, prediction, target, bundle.train_scale, bundle.time_coordinates)
                metric_rows.extend(overall)
                time_rows.extend(by_time)
            curve_rows.extend(
                _curve_rows(
                    fields,
                    sample_idx,
                    bundle.sample_ids[sample_idx],
                    prediction,
                    observed,
                    latent,
                    bundle.time_coordinates,
                    bundle.train_scale,
                )
            )
            history_path = Path(manifest.training.validation_history_path) if manifest.training.validation_history_path else None
            if model != "DynaGS" and history_path is not None:
                history = pd.read_csv(history_path)
                for key, value in fields.items():
                    history[key] = value
                e3_validation_rows.append(history)

    run_index = pd.DataFrame(run_rows)
    metrics = pd.DataFrame(metric_rows)
    metrics_by_time = pd.DataFrame(time_rows)
    curves = pd.DataFrame(curve_rows)
    screening = pd.DataFrame(screening_rows)
    screening_qtl = pd.DataFrame(screening_qtl_rows)
    selected_snps = pd.DataFrame(selected_snp_rows)
    resources = pd.DataFrame(resource_rows)
    histories = pd.concat(training_history_rows, ignore_index=True)
    e3_histories = pd.concat(e3_validation_rows, ignore_index=True, sort=False)

    if len(run_index) != 720 or run_index["run_id"].nunique() != 720:
        raise ValueError("Unified run index must contain exactly 720 unique runs")
    if set(resources["experiment"]) != {"E1", "E2"} or len(resources) != 600:
        raise ValueError("Resource table must contain E1/E2 only; E3 timing is excluded")
    if len(screening) != 300 or not (screening["k"] == 20).all():
        raise ValueError("Screening table must contain 300 final-K=20 runs")
    if not np.allclose(screening["exact_recall"], screening["precision"]):
        raise ValueError("Exact recall and precision should match because QTL count equals K")
    if not np.allclose(screening["exact_recall"], screening["f1"]):
        raise ValueError("Exact recall and F1 should match because QTL count equals K")

    condition_summary = _summaries(
        metrics,
        ["experiment", "condition", "p", "ntr", "heritability", "model", "mode", "target", "metric"],
        "value",
        args.bootstrap_replicates,
        args.seed,
    )
    curve_run_summary, curve_aggregation = _curve_run_summaries(curves)
    curve_condition_summary = _summaries(
        curve_run_summary,
        [
            "experiment", "condition", "p", "ntr", "heritability", "model",
            "mode", "target", "metric", "aggregation_across_individuals",
        ],
        "run_value",
        args.bootstrap_replicates,
        args.seed,
    )
    time_summary = _summaries(
        metrics_by_time,
        [
            "experiment", "condition", "p", "ntr", "heritability", "model",
            "mode", "target", "time_index", "time", "metric",
        ],
        "value",
        args.bootstrap_replicates,
        args.seed,
    )
    screening_long = screening.melt(
        id_vars=[
            "run_id", "experiment", "condition", "p", "ntr", "heritability",
            "repetition", "model", "mode",
        ],
        value_vars=[
            "exact_hit_count", "ld_hit_count", "exact_recall", "precision", "f1",
            "ld_aware_recall", "random_expected_hit_count", "random_enrichment_fold", "k",
        ],
        var_name="metric",
        value_name="value",
    )
    screening_summary = _summaries(
        screening_long,
        ["experiment", "condition", "p", "ntr", "heritability", "model", "mode", "metric"],
        "value",
        args.bootstrap_replicates,
        args.seed,
    )
    paired, paired_summary = _paired_tables(metrics, args.bootstrap_replicates, args.seed)

    output_frames = {
        "run_index.csv": run_index,
        "metrics_overall_long.csv": metrics,
        "metrics_by_time.csv": metrics_by_time,
        "metrics_by_time_summary.csv": time_summary,
        "curve_metrics_individual.csv": curves,
        "curve_metrics_run_summary.csv": curve_run_summary,
        "curve_metrics_condition_summary.csv": curve_condition_summary,
        "curve_metric_aggregation.csv": curve_aggregation,
        "screening_metrics.csv": screening,
        "screening_summary.csv": screening_summary,
        "screening_qtl_detail.csv": screening_qtl,
        "screened_snps.csv": selected_snps,
        "resource_metrics_e1_e2.csv": resources,
        "condition_summary.csv": condition_summary,
        "paired_deltas.csv": paired,
        "paired_delta_summary.csv": paired_summary,
        "training_history_e1_e2.csv": histories,
        "validation_history_e3.csv": e3_histories,
    }
    outputs = {
        name: _write_csv(frame, args.table_dir / name)
        for name, frame in output_frames.items()
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__)),
        "source_roots": {
            "data": str(args.data_root.resolve()),
            "e1": str(args.e1_root.resolve()),
            "e2": str(args.e2_root.resolve()),
            "e3": str(args.e3_root.resolve()),
        },
        "source_batch_manifest_sha256": {
            "dataset": _sha256(args.data_root / "dataset_manifest.json"),
            "e1": _sha256(args.e1_root / "_batch/batch_manifest.json"),
            "e2": _sha256(args.e2_root / "_batch/batch_manifest.json"),
            "e3": _sha256(args.e3_root / "_batch/batch_manifest.json"),
        },
        "replicate_unit": "independent_simulation_repetition",
        "metric_scale": "denormalized_original_phenotype_scale",
        "metrics_protocol": METRICS_PROTOCOL,
        "nrmse_denominator": "training_individuals_observed_phenotype_global_P95_minus_P5",
        "curve_reference": "latent_truth_primary; observed_reference_also_exported",
        "inflection_protocol": "shared_PCHIP_1000_grid; interior_5_percent_excluded; maximum_first_derivative",
        "screening_protocol": {
            "exact": "zero_based_selected_index_equals_zero_based_QTL_index",
            "ld_aware": "maximum training-individual dosage correlation squared >= 0.8",
            "ranking": "unavailable because intermediate screening scores were not saved",
            "storage_position_is_not_rank": True,
        },
        "resource_scope": {
            "included_experiments": ["E1", "E2"],
            "excluded_experiments": ["E3"],
            "e3_exclusion_reason": "user requested no E3 timing statistics",
            "screening_time": "unavailable_not_saved",
            "peak_memory_e1_e2": "unavailable_not_saved",
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "interval": "percentile_95_percent_on_repetition_level_mean",
        },
        "outputs": outputs,
    }
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_dir / "supplementary_tables_metadata.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--e1-root", type=Path, default=DEFAULT_E1_ROOT)
    parser.add_argument("--e2-root", type=Path, default=DEFAULT_E2_ROOT)
    parser.add_argument("--e3-root", type=Path, default=DEFAULT_E3_ROOT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        parser.error("--bootstrap-replicates must be at least 1000")
    for name in ("data_root", "e1_root", "e2_root", "e3_root"):
        setattr(args, name, getattr(args, name).resolve())
    args.table_dir = args.table_dir.resolve()
    args.manifest_dir = args.manifest_dir.resolve()
    return args


def main() -> None:
    manifest = build(parse_args())
    print(f"manifest: {manifest['manifest_path']}")
    for name, record in manifest["outputs"].items():
        print(f"{name}: rows={record['rows']} sha256={record['sha256']}")


if __name__ == "__main__":
    main()
