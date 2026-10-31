"""E3 配置、统一结果与 run manifest schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


E3ModelName = Literal[
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
]


class E3ExperimentConfig(BaseModel):
    """冻结条件与开发期调参配置。"""

    schema_version: Literal[1] = 1
    protocol_version: str = "e3_endpoint_v1"
    protocol_status: Literal["development_smoke_only", "formal_frozen"]
    condition: str
    n_snps: int = Field(gt=0)
    n_train: int = Field(gt=0)
    heritability: float = Field(gt=0, le=1)
    qtl_count: int = Field(gt=0)
    time_points: int = Field(gt=1)
    seed: int = 73
    repetitions: list[int]
    models: list[E3ModelName]
    data_root: str
    output_root: str
    dynags_source_root: str
    selection_metric: Literal["validation_rmse_denormalized"]
    model_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notes: str = ""

    @field_validator("repetitions")
    @classmethod
    def _validate_repetitions(cls, values: list[int]) -> list[int]:
        if not values or len(values) != len(set(values)) or any(v < 1 or v > 10 for v in values):
            raise ValueError("repetitions 必须是 1..10 内的非空互异列表")
        return values

    @field_validator("models")
    @classmethod
    def _validate_models(cls, values: list[str]) -> list[str]:
        if len(values) != 12 or len(values) != len(set(values)):
            raise ValueError("E3 必须恰好包含冻结的12个互异模型")
        return values

    @model_validator(mode="after")
    def _validate_frozen_condition(self) -> "E3ExperimentConfig":
        expected = ("P10000_NTR400_H0.8", 10000, 400, 0.8, 20, 20, 73)
        observed = (
            self.condition,
            self.n_snps,
            self.n_train,
            self.heritability,
            self.qtl_count,
            self.time_points,
            self.seed,
        )
        if observed != expected:
            raise ValueError(f"E3冻结条件不匹配: {observed!r}")
        return self


class SplitRecord(BaseModel):
    path: str
    sha256: str = Field(pattern="^[0-9a-f]{64}$")
    sample_ids_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    train_indices_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]

    @model_validator(mode="after")
    def _disjoint(self) -> "SplitRecord":
        groups = [self.train_indices, self.validation_indices, self.test_indices]
        combined = [value for group in groups for value in group]
        if any(not group for group in groups) or len(combined) != len(set(combined)):
            raise ValueError("train/validation/test 索引必须非空、互异且无交集")
        return self


class DataRecord(BaseModel):
    data_dir: str
    data_manifest_path: str
    data_manifest_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    genotype_path: str | None = None
    genotype_sha256: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    phenotype_path: str
    phenotype_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    simulation_config_path: str | None = None
    simulation_config_sha256: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    simulation_seed: int | None = None
    raw_snp_count: int = Field(ge=0)
    model_input_snp_count: int = Field(ge=0)
    n_samples: int = Field(gt=0)
    n_time_points: int = Field(gt=0)
    split: SplitRecord


class TrainingRecord(BaseModel):
    fit_scope: Literal[
        "train_only_validation_selection_test_once",
        "existing_e2_screened_run_reference",
    ]
    selection_metric: str
    best_parameters: dict[str, Any] = Field(default_factory=dict)
    best_epoch: int | None = Field(default=None, ge=1)
    best_epochs_by_time: list[int] | None = None
    validation_history_path: str | None = None
    effective_feature_count: int | None = Field(default=None, ge=0)
    parameter_count: int | None = Field(default=None, ge=0)
    preprocessing_artifacts: dict[str, Any] = Field(default_factory=dict)
    model_artifacts: dict[str, Any] = Field(default_factory=dict)
    implementation: dict[str, Any] = Field(default_factory=dict)


class PredictionRecord(BaseModel):
    path: str
    sha256: str = Field(pattern="^[0-9a-f]{64}$")
    test_evaluation_count: Literal[1] = 1
    metrics_path: str
    metrics_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    prediction_source: Literal["new_model_prediction", "frozen_existing_prediction"]


class ResourceRecord(BaseModel):
    train_seconds: float = Field(ge=0)
    predict_seconds: float = Field(ge=0)
    end_to_end_seconds: float = Field(ge=0)
    peak_cpu_rss_mb: float | None = Field(default=None, ge=0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0)
    hardware: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, str] = Field(default_factory=dict)
    measurement_notes: dict[str, str] = Field(default_factory=dict)


class FailureRecord(BaseModel):
    exception_type: str
    message: str
    traceback_path: str | None = None


class E3RunManifest(BaseModel):
    """一个模型、一个数据重复的不可歧义记录。"""

    schema_version: Literal[1] = 1
    protocol_version: str
    run_id: str
    experiment: Literal["E3"] = "E3"
    condition: str
    repetition: int = Field(ge=1, le=10)
    model: E3ModelName
    seed: int
    status: Literal["running", "succeeded", "failed"]
    started_at_utc: str
    completed_at_utc: str | None = None
    parsed_config_path: str
    git_commit: str | None = None
    data: DataRecord | None = None
    training: TrainingRecord | None = None
    prediction: PredictionRecord | None = None
    resources: ResourceRecord | None = None
    failure: FailureRecord | None = None

    @model_validator(mode="after")
    def _validate_terminal_state(self) -> "E3RunManifest":
        if self.status == "succeeded" and any(
            value is None for value in (self.data, self.training, self.prediction, self.resources)
        ):
            raise ValueError("成功 run 必须包含 data/training/prediction/resources")
        if self.status == "succeeded":
            if self.model == "Mean" and self.data.model_input_snp_count != 0:
                raise ValueError("Mean 不得使用 SNP")
            if self.model not in ("Mean", "DynaGS") and (
                self.data.model_input_snp_count != self.data.raw_snp_count
            ):
                raise ValueError("E3 非 DynaGS 比较模型必须使用全部原始 SNP")
            deep_models = {"DNNGP-SNP", "DLGWAS", "WheatGP", "Cropformer"}
            if self.model in deep_models and (
                self.training.best_epochs_by_time is None
                or len(self.training.best_epochs_by_time) != self.data.n_time_points
            ):
                raise ValueError("外部深度模型必须记录每个时间点的独立最佳 epoch")
        if self.status == "failed" and self.failure is None:
            raise ValueError("失败 run 必须记录 failure")
        return self
