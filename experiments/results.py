"""运行状态和成功结果清单的严格 schema v1。"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class RunStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    state: Literal["running", "success", "failed"]
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt: int = Field(ge=1)
    test_evaluation_count: int = Field(ge=0, le=1)
    error_type: str | None = None
    error: str | None = None


class CodeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool
    working_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResultManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    code: CodeIdentity
    axes: dict
    subset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_ids: dict[str, list[int]]
    actual_input_dim: int = Field(ge=1)
    model_parameter_count: int = Field(ge=1)
    times_seconds: dict[str, float]
    peak_cpu_rss_bytes: int = Field(ge=1)
    cpu_memory_scope: str
    peak_gpu_allocated_bytes: int = Field(ge=0)
    test_evaluation_count: Literal[1]
    history: str
    model_config_artifact: str
    model_artifact: str
    feature_artifact: str
    split_artifact: str
    predictions: str
    predictions_csv: str
    metrics: str
