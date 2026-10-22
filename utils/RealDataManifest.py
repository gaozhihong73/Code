"""只读真实数据清单：描述统计不产生筛选索引，不拟合任何训练变换。"""

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.spec import code_identity, digest, file_digest

SCHEMA_VERSION = 1
ID_NAMES = {"id", "sample_id", "individual_id", "plant_id", "genotype_id", "accession", "taxa"}
AUXILIARY_NAMES = {"all_phenotype_h2.csv", "gene_snp_mapping.csv"}
EXCLUDED_DIRS = {"snp_3", "pheno_3", "cleaned_pheno", "simulated_data", "snp_zip"}
POLICY = "descriptive_only_no_selection_no_imputation_fit; observed_diploid_MAF; population_variance_ddof0"


class RealDataManifest(BaseModel):
    """v1 private scan artifact; nested statistics are JSON-safe and content-hashed."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    created_at_utc: str
    code: dict
    environment: dict
    parameters: dict
    status: Literal["ready", "blocked"]
    errors: list[str]
    genotype: dict | None
    phenotypes: list[dict]
    identity: dict | None
    id_discovery: dict
    auxiliary: list[dict]
    statistics_policy: str
    artifacts: dict
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def ready_contract(self):
        if self.status == "ready":
            if self.errors or not self.genotype or not self.identity or not self.phenotypes:
                raise ValueError("ready 清单必须具有完整数据和身份记录")
            identity = self.identity
            g = self.genotype
            n = g["n_samples"]
            if (g["shape"] != [n, g["n_snps"]] or identity["n_samples"] != n
                    or len(self.phenotypes) != self.parameters["expected_traits"]
                    or any(p["shape"][0] != n or p["nonfinite_count"] for p in self.phenotypes)
                    or any(p["time"] != self.phenotypes[0]["time"] for p in self.phenotypes)):
                raise ValueError("ready 清单的样本数、性状、时间网格或有限值契约不一致")
            expected = {"row_index": ("confirmed_by_user_source_order", "user_source_order_attestation"),
                        "biological_id": ("verified_by_biological_id", "explicit_snp_row_ids_and_all_phenotype_ids")}
            if (identity.get("alignment_status"), identity.get("evidence_type")) != expected.get(identity.get("sample_id_source")):
                raise ValueError("样本身份来源与证据类型不匹配")
            scope = identity["confirmation_scope"]
            if (scope["snp"] != self.genotype["sha256"]
                    or scope["phenotypes"] != {Path(p["path"]).name: p["sha256"] for p in self.phenotypes}):
                raise ValueError("行对齐确认未绑定当前全部源文件哈希")
        return self


def manifest_hash(payload: dict) -> str:
    return digest({key: value for key, value in payload.items() if key != "manifest_sha256"})


def _signature(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _file_ref(path):
    path = Path(path).resolve()
    before = _signature(path)
    result = {"path": str(path), "sha256": file_digest(path), "size_bytes": before[2]}
    if before != _signature(path):
        raise ValueError(f"扫描时文件变化: {path.name}")
    return result


def time_grid(columns):
    """返回自然时间顺序及原列排列；不移动数据，乱序调用方必须停止。"""
    if not columns or len(columns) != len(set(columns)) or any(not c.strip() for c in columns):
        raise ValueError("时间列缺失/重复/空名称")
    values = []
    for name in columns:
        match = re.fullmatch(r"(?:X|T|time|day|DAP)?[_ -]?(\d+(?:\.\d+)?)", name, re.I)
        if match is None:
            raise ValueError(f"无法解析时间列: {name}")
        values.append(float(match.group(1)))
    if not np.isfinite(values).all() or len(values) != len(set(values)):
        raise ValueError("时间坐标重复或非有限")
    order = sorted(range(len(columns)), key=lambda i: values[i])
    return {"original_labels": list(columns), "natural_labels": [columns[i] for i in order],
            "coordinates": [values[i] for i in order], "natural_permutation": order,
            "already_natural": order == list(range(len(columns))),
            "coordinate_source": "numeric_label_parse", "coordinate_unit": "unconfirmed"}


def _ids(values):
    ids = [str(v).strip() for v in values]
    if any(not v or v.lower() in ("nan", "none") for v in ids) or len(ids) != len(set(ids)):
        raise ValueError("生物学 ID 缺失或重复")
    return ids


def check_ids(expected, actual):
    expected, actual = _ids(expected), _ids(actual)
    if set(expected) != set(actual):
        raise ValueError("生物学 ID 集合不一致")
    if expected != actual:
        raise ValueError("生物学 ID 错序；禁止自动重排")


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    return {"finite_count": int(len(valid)), "undefined_count": int(values.size - len(valid)),
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
            "mean": float(valid.mean()) if len(valid) else None,
            "std_ddof0": float(valid.std()) if len(valid) else None,
            "quantiles": dict(zip(("q00", "q25", "q50", "q75", "q100"),
                                  np.quantile(valid, [0, .25, .5, .75, 1]).tolist())) if len(valid) else {}}


def inspect_phenotype(path, *, progress=None):
    ref = _file_ref(path)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError(f"{Path(path).name}: CSV 原始重复列名")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    id_columns = [name for name in header if name.lower() in ID_NAMES]
    if len(id_columns) > 1:
        raise ValueError("存在多个 ID 列，需显式澄清")
    ids = _ids(frame[id_columns[0]].tolist()) if id_columns else None
    columns = [name for name in header if name not in id_columns]
    grid = time_grid(columns)
    result = {**ref, "trait": Path(path).stem, "source_shape": list(frame.shape),
              "shape": [len(frame), len(columns)], "dtype": "float64", "source_dtypes": {c: "CSV text" for c in header},
              "time": grid, "id_column": id_columns[0] if id_columns else None}
    if progress is not None:
        progress.update(result)
    if not grid["already_natural"]:
        raise ValueError(f"{Path(path).name}: 时间列非自然顺序 {grid['natural_labels']}，禁止静默重排")
    try:
        values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{Path(path).name}: phenotype 含非数值/缺失字符串") from error
    missing = np.isnan(values)
    nonfinite = ~np.isfinite(values)
    result.update(missing_count=int(missing.sum()), nonfinite_count=int(nonfinite.sum()),
                  missing_rate=float(missing.mean()) if values.size else 1.,
                  nonfinite_rate=float(nonfinite.mean()) if values.size else 1.,
                  per_time={name: _summary(values[:, i]) for i, name in enumerate(columns)},
                  candidate_note="待用户/导师确认生物学意义、主要性状和预先声明质量门槛；未按预测结果排序")
    if file_digest(path) != ref["sha256"]:
        raise ValueError("phenotype 在扫描期间发生变化")
    if progress is not None:
        progress.update(result)
    return result, ids


def inspect_genotype(path, n_samples, *, block_columns=16384, progress=None):
    """只读 mmap；仅一个 N×block_columns 工作块及 O(P) 描述向量。"""
    if block_columns < 1:
        raise ValueError("block_columns 必须为正整数")
    path = Path(path)
    ref = _file_ref(path)
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        offset = handle.tell()
    result = {**ref, "shape": list(shape), "dtype": str(dtype), "fortran_order": bool(fortran),
              "order": "F" if fortran else "C", "header_version": list(version), "header_bytes": offset,
              "loading": "numpy_mmap_read_only_column_blocks", "block_columns": block_columns,
              "encoding": {"screening_dosage": [0, 1, 2], "missing": ["NaN", -1],
                           "model_codes_not_accepted": "1/2/3; no implicit +1/-1"}}
    if progress is not None:
        progress.update(result)
    if len(shape) != 2 or min(shape) < 1 or dtype.kind not in "iuf":
        raise ValueError(f"SNP 必须为非空二维实数剂量矩阵: shape={shape}, dtype={dtype}")
    if shape[0] != n_samples:
        hint = "疑似转置；禁止自动转置" if shape[1] == n_samples else "与 phenotype 行数均不匹配"
        raise ValueError(f"SNP shape={shape}, phenotype rows={n_samples}: {hint}")
    if path.stat().st_size != offset + int(np.prod(shape)) * dtype.itemsize:
        raise ValueError("NPY 文件长度与 header 不一致")
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(a, np.memmap) or a.flags.writeable:
        raise ValueError("禁止非只读 mmap 加载大数组")
    n, p = a.shape
    missing_rate, maf, variance = (np.empty(p, dtype=np.float64) for _ in range(3))
    totals = {"0": 0, "1": 0, "2": 0, "minus_one": 0, "nan": 0}
    all_missing = constant = 0
    for start in range(0, p, block_columns):
        stop = min(p, start + block_columns)
        block = a[:, start:stop]
        nan = np.isnan(block)
        absent = nan | (block == -1)
        valid = (block == 0) | (block == 1) | (block == 2)
        if not (valid | absent).all():
            raise ValueError(f"SNP 包含非法剂量/Inf（列块 {start}:{stop}），只允许0/1/2及NaN/-1")
        counts = [np.count_nonzero(block == value, axis=0) for value in (0, 1, 2)]
        observed = sum(counts)
        observed_safe = np.maximum(observed, 1)
        mean = (counts[1] + 2 * counts[2]) / observed_safe
        var = np.maximum((counts[1] + 4 * counts[2]) / observed_safe - mean ** 2, 0)
        frequency = mean / 2
        column_maf = np.minimum(frequency, 1 - frequency)
        column_maf[observed == 0] = np.nan
        var[observed == 0] = np.nan
        missing_rate[start:stop] = 1 - observed / n
        maf[start:stop], variance[start:stop] = column_maf, var
        for value in (0, 1, 2):
            totals[str(value)] += int(counts[value].sum())
        totals["nan"] += int(nan.sum())
        totals["minus_one"] += int(np.count_nonzero(block == -1))
        all_missing += int((observed == 0).sum())
        constant += int(((var == 0) & (observed > 0)).sum())
    if file_digest(path) != ref["sha256"]:
        raise ValueError("SNP 在扫描期间内容发生变化")
    missing_count = totals["nan"] + totals["minus_one"]
    result.update(n_samples=n, n_snps=p, dosage_counts=totals,
                  missing_count=missing_count, missing_rate=missing_count / (n * p),
                  column_missing_rate=_summary(missing_rate), column_maf=_summary(maf),
                  column_variance=_summary(variance), constant_columns=constant, all_missing_columns=all_missing,
                  workspace_bound="three float64 P-vectors + O(N*block_columns); no full matrix copy",
                  selection_indices_generated=False)
    if progress is not None:
        progress.update(result)
    return result


def discover_ids(data_root, pheno_dir):
    """仅搜索源目录元数据候选；辅助表和排除管线绝不作为训练输入。"""
    root, pheno_dir = Path(data_root).resolve(), Path(pheno_dir).resolve()
    candidates, inspected, auxiliary = [], [], []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in sorted(dirs) if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for name in sorted(files):
            path = Path(base) / name
            if path.parent == pheno_dir or path.suffix.lower() not in (".csv", ".json", ".txt", ".tsv"):
                continue
            relative = str(path.relative_to(root))
            if name in AUXILIARY_NAMES:
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    header = next(csv.reader(handle), []) if path.suffix.lower() == ".csv" else []
                auxiliary.append({**_file_ref(path), "role": "auxiliary_only_not_training_input", "columns": header})
                continue
            id_headers = []
            if path.suffix.lower() == ".csv":
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    id_headers = [c for c in next(csv.reader(handle), []) if c.lower() in ID_NAMES]
            if id_headers or re.search(r"(?:id|mapping|order|sample|individual|accession)", name, re.I):
                candidates.append(relative)
            inspected.append(relative)
    return {"search_root": str(root), "excluded_directories": sorted(EXCLUDED_DIRS),
            "metadata_files_examined": inspected, "unresolved_candidates": candidates,
            "biological_id_verification_claimed": False}, auxiliary


def resources(output_root):
    dest = Path(output_root).resolve()
    while not dest.exists():
        dest = dest.parent
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                memory[key + "_bytes"] = int(value.split()[0]) * 1024
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    cpu = next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                if s.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor()
    return {"python": platform.python_version(), "pytorch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "gpu": gpu.stdout.strip() if gpu and gpu.returncode == 0 else "unavailable",
            "cpu": cpu, "logical_cpus": os.cpu_count(), "memory": memory,
            "disk_free_bytes": shutil.disk_usage(dest).free, "platform": platform.platform()}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_manifest(payload, output_root, *, sample_ids=None):
    """目录名为 canonical payload 哈希；禁止覆盖已有任何扫描。"""
    payload = dict(payload)
    payload["artifacts"] = {}
    ids_bytes = None
    if sample_ids is not None:
        ids_bytes = json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":")).encode()
        payload["artifacts"]["sample_ids.json"] = hashlib.sha256(ids_bytes).hexdigest()
    payload["manifest_sha256"] = manifest_hash(payload)
    RealDataManifest.model_validate(payload)
    dest = Path(output_root).resolve() / payload["manifest_sha256"]
    dest.mkdir(parents=True, exist_ok=False)
    if ids_bytes is not None:
        (dest / "sample_ids.json").write_bytes(ids_bytes)
    _write_json(dest / "real_data_manifest.json", payload)
    rows = ["trait,time_points,missing_rate,nonfinite_rate,decision"]
    for p in payload["phenotypes"]:
        rows.append(f"{p['trait']},{p.get('shape', [None, None])[1]},{p.get('missing_rate', 'unknown')},{p.get('nonfinite_rate', 'unknown')},pending_user_supervisor")
    (dest / "trait_candidates.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    report = {"status": payload["status"], "manifest_sha256": payload["manifest_sha256"],
              "errors": payload["errors"], "row_alignment": payload["identity"],
              "statistics_policy": POLICY, "training_or_screening_performed": False,
              "main_traits_selected": False}
    _write_json(dest / "scan_report.json", report)
    return dest


def scan_data(snp_file, pheno_dir, output_root, *, expected_traits=24, block_columns=16384,
              attestation_date=None, snp_ids_file=None, data_root=None, attestation_file=None):
    snp_file, pheno_dir = Path(snp_file).resolve(), Path(pheno_dir).resolve()
    output_root = Path(output_root).resolve()
    if (output_root == pheno_dir or pheno_dir in output_root.parents
            or output_root == snp_file.parent or snp_file.parent in output_root.parents):
        raise ValueError("output_root 不能位于原始输入目录内")
    payload = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "code": code_identity(), "environment": resources(output_root),
               "parameters": {"snp_file": str(snp_file), "pheno_dir": str(pheno_dir),
                              "expected_traits": expected_traits, "block_columns": block_columns},
               "status": "blocked", "errors": [], "genotype": None, "phenotypes": [], "identity": None,
               "id_discovery": {}, "auxiliary": [], "statistics_policy": POLICY, "artifacts": {}}
    sample_ids = None
    try:
        paths = sorted((p for p in pheno_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"), key=lambda p: p.name)
        if len(paths) != expected_traits or any(p.name in AUXILIARY_NAMES for p in paths):
            raise ValueError(f"正式 phenotype 应为 {expected_traits} 个性状 CSV，实际 {len(paths)}；禁止辅助表混入")
        if len({p.stem for p in paths}) != len(paths):
            raise ValueError("重复性状名")
        inventory = {p.name: _file_ref(p) for p in paths}
        payload["parameters"]["phenotype_inventory"] = inventory
        payload["parameters"]["snp_source"] = _file_ref(snp_file)
        discovery, auxiliary = discover_ids(data_root or pheno_dir.parent, pheno_dir)
        payload["id_discovery"], payload["auxiliary"] = discovery, auxiliary
        role_ids = []
        for path in paths:
            info = {"trait": path.stem, **inventory[path.name]}
            payload["phenotypes"].append(info)
            info, ids = inspect_phenotype(path, progress=info)
            role_ids.append(ids)
            if info["nonfinite_count"] or info["shape"][0] < 1:
                raise ValueError(f"{path.name}: phenotype 缺失/非有限或空数据，停止；禁止自动填补")
            if info["time"] != payload["phenotypes"][0]["time"] or info["shape"] != payload["phenotypes"][0]["shape"]:
                raise ValueError("不同性状行数/时间网格不一致")
        n_samples = payload["phenotypes"][0]["shape"][0]
        if snp_ids_file is not None:
            mapping = pd.read_csv(snp_ids_file, dtype=str, keep_default_na=False)
            if mapping.shape != (n_samples, 1):
                raise ValueError("SNP ID 映射必须为带表头的单列 CSV，行数等于 SNP 个体数")
            sample_ids = _ids(mapping.iloc[:, 0].tolist())
            for ids in role_ids:
                if ids is None:
                    raise ValueError("使用 SNP 生物学 ID 时所有 phenotype 必须有 ID 列")
                check_ids(sample_ids, ids)
            payload["identity"] = {"sample_id_source": "biological_id", "alignment_status": "verified_by_biological_id",
                                   "evidence_type": "explicit_snp_row_ids_and_all_phenotype_ids", "id_mapping": _file_ref(snp_ids_file)}
        else:
            if any(ids is not None for ids in role_ids) or discovery["unresolved_candidates"]:
                raise ValueError("存在生物学 ID 或外部映射候选；请先检查并显式提供 --snp_ids_file，不能降级行号")
            sample_ids = list(range(n_samples))
            scope = {"snp": payload["parameters"]["snp_source"]["sha256"],
                     "phenotypes": {p.name: inventory[p.name]["sha256"] for p in paths}}
            previous_scopes = []
            for frozen in output_root.glob("*/real_data_manifest.json"):
                previous = json.loads(frozen.read_text())
                if (previous.get("identity") or {}).get("evidence_type") != "user_source_order_attestation":
                    continue
                if previous.get("parameters", {}).get("snp_file") == str(snp_file) and previous["parameters"].get("pheno_dir") == str(pheno_dir):
                    if manifest_hash(previous) != previous.get("manifest_sha256"):
                        raise ValueError("既有源顺序确认清单损坏，必须人工检查")
                    previous_scopes.append(previous["identity"]["confirmation_scope"])
            if attestation_file is not None:
                renewed = json.loads(Path(attestation_file).read_text())
                if (renewed.get("evidence_type") != "user_source_order_attestation"
                        or renewed.get("source_hashes") != scope):
                    raise ValueError("重新确认凭据必须明确绑定当前全部源文件哈希")
                attestation_date = renewed.get("attestation_date")
            elif previous_scopes and scope not in previous_scopes:
                raise ValueError("当前源哈希不同于已有用户确认；旧日期不能续用，必须重新确认并提供 --attestation_file")
            if attestation_date is None:
                raise ValueError("只有行号，缺少用户源顺序确认，禁止声称已验证行对齐")
            datetime.strptime(attestation_date, "%Y-%m-%d")
            payload["identity"] = {"sample_id_source": "row_index", "alignment_status": "confirmed_by_user_source_order",
                "evidence_type": "user_source_order_attestation", "attestation_date": attestation_date,
                "limitation": "只能由源数据顺序及用户确认保证对应；未从数据本身证明生物学身份或行对齐"}
        if attestation_file is not None and snp_ids_file is None:
            payload["identity"]["attestation_file"] = _file_ref(attestation_file)
        payload["identity"].update(n_samples=n_samples, sample_ids_sha256=digest(sample_ids),
            confirmation_scope={"snp": payload["parameters"]["snp_source"]["sha256"],
                                "phenotypes": {p.name: inventory[p.name]["sha256"] for p in paths}},
            invalidation="任一源文件内容或顺序/清单变化，确认立即失效，必须重新确认")
        payload["genotype"] = {}
        payload["genotype"] = inspect_genotype(snp_file, n_samples, block_columns=block_columns, progress=payload["genotype"])
        for ref in [payload["parameters"]["snp_source"], *inventory.values(), *auxiliary]:
            if file_digest(ref["path"]) != ref["sha256"]:
                raise ValueError("扫描期间源数据变化，禁止冻结")
        payload["status"] = "ready"
    except (ValueError, OSError, StopIteration) as error:
        payload["errors"].append(str(error))
    return write_manifest(payload, output_root, sample_ids=sample_ids)


def verify_manifest(path):
    """读取时重新核对内容哈希与输入清单；原确认不能跨数据版本复用。"""
    path = Path(path).resolve()
    payload = RealDataManifest.model_validate_json(path.read_text()).model_dump()
    if payload["manifest_sha256"] != manifest_hash(payload) or path.parent.name != payload["manifest_sha256"]:
        raise ValueError("manifest canonical 哈希/目录名不匹配")
    if payload["status"] != "ready":
        raise ValueError("扫描存在阻塞项，不能用于实验")
    parameters = payload["parameters"]
    current = {p.name for p in Path(parameters["pheno_dir"]).iterdir() if p.is_file() and p.suffix.lower() == ".csv"}
    if current != set(parameters["phenotype_inventory"]):
        raise ValueError("phenotype 文件清单变化，源顺序确认已失效")
    refs = [parameters["snp_source"], *parameters["phenotype_inventory"].values(), *payload["auxiliary"]]
    if payload["identity"].get("id_mapping"):
        refs.append(payload["identity"]["id_mapping"])
    if payload["identity"].get("attestation_file"):
        refs.append(payload["identity"]["attestation_file"])
    for ref in refs:
        if file_digest(ref["path"]) != ref["sha256"]:
            raise ValueError("源文件哈希变化，行对齐确认已失效，必须重新确认")
    for name, expected in payload["artifacts"].items():
        artifact = path.parent / name
        if path.parent not in artifact.resolve().parents or file_digest(artifact) != expected:
            raise ValueError("manifest 附属文件内容哈希不匹配")
    ids = json.loads((path.parent / "sample_ids.json").read_text())
    if payload["identity"]["sample_id_source"] == "row_index" and ids != list(range(payload["genotype"]["n_samples"])):
        raise ValueError("row_index 必须为原始连续行索引0..N-1")
    if digest(ids) != payload["identity"]["sample_ids_sha256"]:
        raise ValueError("sample_id 顺序哈希不匹配")
    return payload


def experiment_template(payload):
    """只生成待补全模板，不创建 split、不 dry-run 或运行训练。"""
    if payload["status"] != "ready":
        raise ValueError("blocked manifest 不生成实验模板")
    if payload["identity"]["sample_id_source"] != "row_index":
        raise ValueError("当前 experiment runner 只支持 row_index；生物学 ID manifest 不可静默降级")
    return {"schema_version": 1, "data": {"snp": {k: payload["genotype"][k] for k in ("path", "sha256")},
        "phenotypes": {p["trait"]: {k: p[k] for k in ("path", "sha256")} for p in payload["phenotypes"]},
        "n_samples": payload["genotype"]["n_samples"], "n_snps": payload["genotype"]["n_snps"],
        "sample_id_source": payload["identity"]["sample_id_source"],
        "sample_ids_sha256": payload["identity"]["sample_ids_sha256"],
        "time_labels": payload["phenotypes"][0]["time"]["natural_labels"],
        "time_coordinates": payload["phenotypes"][0]["time"]["coordinates"]},
        "splits": {"mode": "regular", "manifests": [{"path": "<FILL_SHARED_SPLIT_OR_FOLD>", "sha256": "<FILL_SPLIT_SHA256>"}]},
        "experiments": [{"route": "random_k", "seed": "<FILL_SEED>", "traits": ["<FILL_PRIMARY_TRAITS>"],
                         "k": "<FILL_CANDIDATE_SNP_COUNT>", "gwas_k": "<FILL_GWAS_K>", "ablation": "full",
                         "tfpp": {"screen_missing_strategy": "<FILL_error_OR_train_only_mode>"}}],
        "output_root": "<FILL_OUTPUT_ROOT>"}


def redacted_example(_private_payload=None):
    """公开示例只用固定虚构值；不复制真实路径、哈希、性状、统计或个体信息。"""
    value = {"schema_version": 1, "created_at_utc": "2000-01-01T00:00:00+00:00",
             "code": {"commit": "0" * 40, "dirty": False}, "environment": {},
             "parameters": {"snp_file": "<DATA_ROOT>/snp.npy", "pheno_dir": "<DATA_ROOT>/pheno"},
             "status": "blocked", "errors": ["Illustrative synthetic values only; not a real-data approval"],
             "genotype": {"path": "<DATA_ROOT>/snp.npy", "shape": [8, 12], "dtype": "int8"},
             "phenotypes": [], "identity": {"sample_id_source": "row_index",
                "evidence_type": "user_source_order_attestation", "alignment_status": "unconfirmed"},
             "id_discovery": {}, "auxiliary": [], "statistics_policy": POLICY, "artifacts": {}}
    value["manifest_sha256"] = manifest_hash(value)
    return RealDataManifest.model_validate(value).model_dump()
