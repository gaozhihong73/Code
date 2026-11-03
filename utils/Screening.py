"""SNP 特征筛选共享模块。

正式入口执行训练集质量控制、PCA 校正的加性+显性联合 GWAS，再仅在 GWAS
候选内执行 DC-SIS 并保留实验指定数量。底层旧函数仅供历史工程入口重放。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.preprocessing import StandardScaler

from .Genotype import GENOTYPE_ENCODING_VERSION, validate_dosage


SCREENING_ALGORITHM_VERSION = "qc_add_dom_gwas_candidate_dcsis_v1"
SCREEN_MIN_MAC = 5
SCREEN_MAX_MISSING_RATE = 0.20
SCREEN_GWAS_BLOCK_COLUMNS = 16384


def array_fingerprint(array: np.ndarray) -> str:
    """按形状、dtype 和内容计算稳定哈希，不依赖全数据文件的 mtime。"""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    raw = memoryview(array).cast("B")
    for start in range(0, len(raw), 1024 * 1024):
        digest.update(raw[start:start + 1024 * 1024])
    return digest.hexdigest()


def training_indices_hash(indices: np.ndarray) -> str:
    """保留训练行顺序，以显式小端 int64 编码记录训练个体。"""
    values = np.asarray(indices)
    if (values.ndim != 1 or values.dtype.kind not in "iu" or not values.size
            or (values < 0).any() or len(np.unique(values)) != len(values)):
        raise ValueError("训练索引必须为一维互异整数")
    return array_fingerprint(values.astype("<i8"))


def _log(logger, msg: str) -> None:
    """统一日志输出: 有 logger 时走 logger.info, 否则 print。"""
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# 缓存工具
# ---------------------------------------------------------------------------


def compute_cache_key(
    snp_file: str,
    pheno_source: str,
    phenotype_names: list[str],
    gwas_k: int,
    seed: int,
    split_info: str = "",
    multivariate: bool = False,
    missing_strategy: str = "error",
    gwas_population_structure: str = "pca",
    gwas_n_pcs: int = 5,
    gwas_pca_panel_size: int | None = None,
    gwas_pca_panel_seed: int | None = None,
    gwas_pca_algorithm_version: str = "sklearn_centered_all_train_markers_v1",
    screen_final_k: int = 20,
    time_coordinates: list[float] | None = None,
    *,
    train_indices: np.ndarray | None = None,
    train_genotype: np.ndarray | None = None,
    train_phenotype: np.ndarray | None = None,
) -> str:
    """普通/折内训练用训练个体内容指纹；模拟调用保留路径+mtime 缓存。

    提供 train_indices 时不读取全数据文件 mtime，不受 held-out 数据修改影响。
    注意: multivariate 参与哈希, 避免多元/逐表型两种筛选结果互相污染缓存。

    Parameters
    ----------
    snp_file : str
        SNP 数据文件路径。
    pheno_source : str
        表型文件路径或目录路径。
    phenotype_names : list[str]
        表型名称列表。
    gwas_k : int
        加性+显性联合 GWAS 候选池大小。
    seed : int
        随机种子（影响 train/val/test 划分）。
    split_info : str, optional
        额外划分信息（如 kfold fold index），默认空字符串。
    multivariate : bool, optional
        是否使用多元 DC-SIS 模式（多表型拼接联合筛选），默认 ``False``。
    missing_strategy : str, optional
        显式缺失策略标识，参与缓存键；编码/算法版本也参与缓存键。
    screen_final_k : int, optional
        GWAS 候选内 DC-SIS 的最终 SNP 数，参与缓存键。
    time_coordinates : list[float], optional
        GWAS 动态性状使用的实际时间坐标，参与缓存键。
    train_indices, train_genotype, train_phenotype : np.ndarray, optional
        普通/折内训练必须同时传入训练原始行号及训练子矩阵；只哈希这些内容。

    Returns
    -------
    str
        32 位十六进制 MD5 哈希字符串。
    """
    hasher = hashlib.md5()
    cached_time_coordinates = (
        None
        if time_coordinates is None
        else np.asarray(time_coordinates, dtype=np.float64).tolist()
    )
    # 编码/算法版本以及训练感知缓存命名空间使旧筛选缓存失效。
    hasher.update(
        f"{GENOTYPE_ENCODING_VERSION}|{SCREENING_ALGORITHM_VERSION}|{missing_strategy}".encode()
    )

    if train_indices is not None:
        if train_genotype is None or train_phenotype is None:
            raise ValueError("训练感知缓存必须提供训练 SNP 和表型指纹")
        if len(train_indices) != len(train_genotype) or len(train_indices) != len(train_phenotype):
            raise ValueError("缓存训练数据与训练索引行数不一致")
        # 普通模式仅包含训练内容；修改 held-out 行或文件 mtime 不影响缓存键。
        payload = {
            "cache_scope": "train_only_v1",
            "train_indices_sha256": training_indices_hash(train_indices),
            "genotype_sha256": array_fingerprint(train_genotype),
            "phenotype_sha256": array_fingerprint(train_phenotype),
            "phenotype_names": list(phenotype_names),
            "gwas_k": gwas_k, "seed": seed,
            "multivariate": multivariate, "split_info": split_info,
            "gwas_population_structure": gwas_population_structure,
            "gwas_n_pcs": gwas_n_pcs,
            "gwas_pca_panel_size": gwas_pca_panel_size,
            "gwas_pca_panel_seed": gwas_pca_panel_seed,
            "gwas_pca_algorithm_version": gwas_pca_algorithm_version,
            "screen_final_k": screen_final_k,
            "screen_min_mac": SCREEN_MIN_MAC,
            "screen_max_missing_rate": SCREEN_MAX_MISSING_RATE,
            "time_coordinates": cached_time_coordinates,
        }
        hasher.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        return hasher.hexdigest()
    if train_genotype is not None or train_phenotype is not None:
        raise ValueError("训练指纹不能缺少 train_indices")

    # 文件路径 + 修改时间
    for fpath in [snp_file, pheno_source]:
        p = Path(fpath)
        hasher.update(fpath.encode())
        if p.exists():
            hasher.update(str(p.stat().st_mtime).encode())

    # 表型名称（排序保证顺序无关）
    hasher.update("|".join(sorted(phenotype_names)).encode())

    # 筛选参数
    hasher.update(f"{gwas_k}_{seed}_{split_info}".encode())
    hasher.update(json.dumps({
        "gwas_population_structure": gwas_population_structure,
        "gwas_n_pcs": gwas_n_pcs,
        "gwas_pca_panel_size": gwas_pca_panel_size,
        "gwas_pca_panel_seed": gwas_pca_panel_seed,
        "gwas_pca_algorithm_version": gwas_pca_algorithm_version,
        "screen_final_k": screen_final_k,
        "screen_min_mac": SCREEN_MIN_MAC,
        "screen_max_missing_rate": SCREEN_MAX_MISSING_RATE,
        "time_coordinates": cached_time_coordinates,
    }, sort_keys=True, separators=(",", ":")).encode())

    # 筛选方法标识（多元联合 vs 单表型）
    hasher.update(
        f"method={'mv_gwas_sis_v2' if multivariate else 'single_phenotype'}".encode()
    )

    return hasher.hexdigest()


def _load_cache(cache_key: str, cache_dir: str, logger=None) -> np.ndarray | None:
    """尝试从磁盘加载缓存的筛选索引。"""
    cache_path = Path(cache_dir) / f"{cache_key}.npy"
    if cache_path.exists():
        indices = np.load(cache_path)
        _log(logger, f"  [缓存命中] 从 {cache_path} 加载 {len(indices)} 个 SNP 索引")
        return indices
    return None


def _save_cache(
    cache_key: str, cache_dir: str, indices: np.ndarray, logger=None
) -> None:
    """将筛选索引保存到磁盘。"""
    cache_path = Path(cache_dir) / f"{cache_key}.npy"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, indices)
    _log(logger, f"  [缓存写入] 已保存 {len(indices)} 个 SNP 索引到 {cache_path}")


# ---------------------------------------------------------------------------
# 基础筛选函数
# ---------------------------------------------------------------------------


def _rint_transform(x: np.ndarray) -> np.ndarray:
    """基于秩次的逆正态变换 (Rank-Based Inverse Normal Transformation)。

    将任意分布映射到标准正态, GWAS 中常用于减小偏离正态的影响。
    """
    ranks = stats.rankdata(x, method="average")
    probs = (ranks - 0.5) / len(x)
    return stats.norm.ppf(probs).astype(np.float64)


def _multivariate_pheno_matrix(
    pheno_train: np.ndarray, logger=None
) -> np.ndarray:
    """将多表型数据变换为多元 DC-SIS 所需的扁平化 Y 矩阵。

    对每个表型展平后做 z-score 标准化（消除量纲差异, 避免量纲大的表型
    主导欧氏距离）, 再按表型维度拼接为 ``(n_sample, n_pheno * n_time)``。

    Parameters
    ----------
    pheno_train : np.ndarray
        训练集表型数据, 形状为 ``(n_sample, n_pheno, n_time)``。

    Returns
    -------
    np.ndarray
        标准化并拼接后的表型矩阵, 形状为 ``(n_sample, n_pheno * n_time)``。
    """
    N = pheno_train.shape[0]
    n_pheno = pheno_train.shape[1]
    flat_columns = []
    for i in range(n_pheno):
        pheno_i = pheno_train[:, i, :].reshape(N, -1)  # [N, T]
        # z-score 标准化（按整个表型展平统计, 保持各表型量纲一致）
        mean = pheno_i.mean()
        std = pheno_i.std()
        if std > 1e-12:
            pheno_i = (pheno_i - mean) / std
        else:
            pheno_i = pheno_i - mean
        flat_columns.append(pheno_i)
    return np.concatenate(flat_columns, axis=1).astype(np.float32)


def dc_sis_screen(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    k: int,
    verbose: bool = True,
    logger=None,
) -> tuple[np.ndarray, np.ndarray]:
    """DC-SIS 特征筛选（距离相关系数）。

    仅用训练集计算每个 SNP 与表型 Y 的距离相关系数，返回 top-k SNP 列索引。
    利用 SNP 取值仅为 ``{0, 1, 2}`` 的特性，用矩阵-向量积 + 计数闭式代替
    N x N 矩阵分配，O(N) per SNP，适合 P >> N 场景。

    Parameters
    ----------
    snp_train : np.ndarray
        训练集基因型矩阵，形状为 ``(n_sample, n_snp)``，取值 ``{0, 1, 2}``。
    pheno_train : np.ndarray
        训练集表型数据，形状为 ``(n_sample, n_time)`` 或 ``(n_sample,)``。
    k : int
        要保留的 SNP 数目。
    verbose : bool, optional
        是否打印筛选进度与统计信息，默认 ``True``。

    Returns
    -------
    indices : np.ndarray
        排序后的列索引，形状为 ``(k,)``，dtype ``int64``，按 dcor 从大到小排列。
    dcor : np.ndarray
        每个 SNP 的距离相关系数，形状为 ``(n_snp,)``，dtype ``float32``。
    """
    snp_train = validate_dosage(snp_train)
    N, P = snp_train.shape
    y_2d = pheno_train.reshape(N, -1)  # [N, T]

    # ---- 1. Y 的距离矩阵 D_Y [N, N] (只算一次) ----
    diff = y_2d[:, None, :] - y_2d[None, :, :]
    D_Y = np.sqrt((diff**2).sum(axis=-1)).astype(np.float32)

    n = N
    nn2 = float(n * n)
    nn3 = float(nn2 * n)
    _eps = 1e-20

    # ---- dVar²(Y) / S2_Y / rY (只算一次) ----
    S1_YY = (D_Y * D_Y).sum() / nn2
    S2_Y = D_Y.mean()
    rY = D_Y.sum(axis=1).astype(np.float64)
    dvar2_Y = S1_YY + S2_Y * S2_Y - 2.0 * (rY * rY).sum() / nn3

    # ---- 2. 逐 SNP 扫描 ----
    dcor = np.zeros(P, dtype=np.float32)
    sqrt_dvar2_Y = np.sqrt(max(dvar2_Y, 0.0))

    for p in range(P):
        x = snp_train[:, p]
        is0 = np.asarray(x == 0, dtype=np.float32)
        is1 = np.asarray(x == 1, dtype=np.float32)
        is2 = np.asarray(x == 2, dtype=np.float32)

        cnt0 = is0.sum()
        cnt1 = is1.sum()
        cnt2 = is2.sum()

        if cnt0 == n or cnt1 == n or cnt2 == n:
            dcor[p] = 0.0
            continue

        u1 = D_Y @ is1
        u2 = D_Y @ is2

        sm01 = np.dot(is0, u1)
        sm02 = np.dot(is0, u2)
        sm12 = np.dot(is1, u2)

        sum_DY_DX = 2.0 * (sm01 + sm12 + 2.0 * sm02)

        S1_YX = sum_DY_DX / nn2
        sum_DX = 2.0 * (cnt0 * cnt1 + cnt1 * cnt2 + 2.0 * cnt0 * cnt2)
        S2_YX = S2_Y * sum_DX / nn2
        rX = is0 * (cnt1 + 2.0 * cnt2) + is1 * (cnt0 + cnt2) + is2 * (2.0 * cnt0 + cnt1)
        S3_YX = np.dot(rY, rX.astype(np.float64)) / nn3
        dcov2_YX = S1_YX + S2_YX - 2.0 * S3_YX

        S1_XX = 2.0 * (cnt0 * cnt1 + cnt1 * cnt2 + 4.0 * cnt0 * cnt2) / nn2
        mean_DX = sum_DX / nn2
        S2_XX = mean_DX * mean_DX
        S3_XX = (
            cnt0 * (cnt1 + 2.0 * cnt2) ** 2
            + cnt1 * (cnt0 + cnt2) ** 2
            + cnt2 * (2.0 * cnt0 + cnt1) ** 2
        ) / nn3
        dvar2_X = max(S1_XX + S2_XX - 2.0 * S3_XX, 0.0)

        denom = sqrt_dvar2_Y * np.sqrt(max(dvar2_X, 0.0))
        if denom > 1e-10:
            dcor[p] = np.sqrt(max(dcov2_YX, 0.0)) / np.sqrt(max(denom, _eps))
        else:
            dcor[p] = 0.0

    # ---- 3. Top-k ----
    k = min(k, P)
    idx = np.argsort(dcor)[::-1][:k].astype(np.int64)
    if verbose:
        _log(logger, f"    DC-SIS 筛选完成: 按 dCor 保留 {k}/{P} SNPs")
        _log(
            logger,
            f"    dCor 分布: min={dcor[idx[-1]]:.4f}  max={dcor[idx[0]]:.4f}  "
            f"mean={dcor.mean():.4f}  median={np.median(dcor):.4f}",
        )

    return idx, dcor


def _extract_dynamic_traits(
    y_2d: np.ndarray,
    time_coordinates: list[float] | np.ndarray,
    use_rint: bool = True,
) -> np.ndarray:
    """从单表型时序数据提取 3 个动态性状 (Mean / AUC / Slope)。

    Parameters
    ----------
    y_2d : np.ndarray
        单表型数据, 形状为 ``(n_sample, n_time)``。
    time_coordinates : list[float] | np.ndarray
        与表型列一一对应、按采样先后严格递增的实际时间坐标。
    use_rint : bool, optional
        是否对每个性状做秩次逆正态变换（RINT），默认 ``True``。

    Returns
    -------
    np.ndarray
        动态性状矩阵, 形状为 ``(n_sample, 3)``，列顺序为 Mean, AUC, Slope。
    """
    y_2d = np.asarray(y_2d)
    if y_2d.ndim != 2:
        raise ValueError("动态性状输入必须为二维 [个体, 时间]")
    time_points = np.asarray(time_coordinates, dtype=np.float64)
    if (
        time_points.ndim != 1
        or len(time_points) != y_2d.shape[1]
        or not np.isfinite(time_points).all()
        or (len(time_points) > 1 and (np.diff(time_points) <= 0).any())
    ):
        raise ValueError("time_coordinates 必须与表型列一一对应且严格递增")
    if len(time_points) < 2:
        raise ValueError("动态 AUC 和斜率至少需要两个实际时间点")

    feat_mean = y_2d.mean(axis=1)
    feat_auc = np.trapz(y_2d, x=time_points, axis=1)

    t_bar = time_points.mean()
    t_dev = time_points - t_bar
    y_bar = y_2d.mean(axis=1, keepdims=True)
    feat_slope = ((y_2d - y_bar) * t_dev[None, :]).sum(axis=1) / (t_dev**2).sum()

    traits = np.column_stack([feat_mean, feat_auc, feat_slope])
    if use_rint:
        traits = np.column_stack(
            [_rint_transform(traits[:, j]) for j in range(traits.shape[1])]
        )
    return traits


def quality_control_snps(
    raw_dosage: np.ndarray,
    *,
    min_mac: int = SCREEN_MIN_MAC,
    max_missing_rate: float = SCREEN_MAX_MISSING_RATE,
    verbose: bool = True,
    logger=None,
) -> np.ndarray:
    """仅依据训练基因型保留缺失率合格且次要等位基因计数充足的 SNP。"""
    dosage = np.asarray(raw_dosage)
    if dosage.ndim != 2 or not dosage.shape[0] or not dosage.shape[1]:
        raise ValueError("质量控制输入必须为非空二维训练剂量矩阵")
    missing = np.isnan(dosage) | (dosage == -1)
    valid = missing | (dosage == 0) | (dosage == 1) | (dosage == 2)
    if not valid.all():
        raise ValueError("质量控制输入含非 -1/0/1/2 的非法剂量")
    observed_n = (~missing).sum(axis=0)
    allele_count = np.where(missing, 0, dosage).sum(axis=0, dtype=np.float64)
    minor_allele_count = np.minimum(allele_count, 2 * observed_n - allele_count)
    missing_rate = missing.mean(axis=0)
    keep = (
        (observed_n > 0)
        & (minor_allele_count >= min_mac)
        & (missing_rate <= max_missing_rate)
    )
    indices = np.flatnonzero(keep).astype(np.int64)
    if not len(indices):
        raise ValueError("训练集 SNP 质量控制后没有可用位点")
    if verbose:
        _log(
            logger,
            f"训练集 SNP 质量控制: {dosage.shape[1]} -> {len(indices)}, "
            f"MAC>={min_mac}, missing_rate<={max_missing_rate:.0%}",
        )
    return indices


def additive_dominance_gwas_screen(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    k: int,
    time_coordinates: list[float] | np.ndarray,
    n_pcs: int = 5,
    multivariate: bool = False,
    block_columns: int = SCREEN_GWAS_BLOCK_COLUMNS,
    verbose: bool = True,
    logger=None,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA 校正的加性+显性二自由度联合 GWAS，按最小动态性状 P 值排序。"""
    dosage = validate_dosage(snp_train)
    n_samples, n_snps = dosage.shape
    if k < 1 or block_columns < 1:
        raise ValueError("GWAS 候选数和分块列数必须为正整数")
    if multivariate:
        traits = np.concatenate(
            [
                _extract_dynamic_traits(
                    pheno_train[:, i, :], time_coordinates, use_rint=True
                )
                for i in range(pheno_train.shape[1])
            ],
            axis=1,
        )
    else:
        traits = _extract_dynamic_traits(
            pheno_train.reshape(n_samples, -1), time_coordinates, use_rint=True
        )

    n_pc = min(n_pcs, n_samples - 3, n_snps)
    if n_pc < 1:
        raise ValueError("训练样本或 SNP 数不足以执行 PCA 校正 GWAS")
    pca_input = dosage.astype(np.float32, copy=False)
    pcs = PCA(
        n_components=n_pc,
        svd_solver="randomized",
        random_state=42,
    ).fit_transform(pca_input)
    design = np.column_stack([np.ones(n_samples), pcs])
    q, _ = np.linalg.qr(design)
    y_residual = traits - q @ (q.T @ traits)
    y_sum_squares = np.sum(y_residual**2, axis=0)
    residual_df = n_samples - design.shape[1] - 2
    if residual_df <= 0:
        raise ValueError("训练样本不足以执行加性+显性联合 GWAS")

    min_p = np.ones(n_snps, dtype=np.float64)
    for start in range(0, n_snps, block_columns):
        stop = min(start + block_columns, n_snps)
        additive = dosage[:, start:stop].astype(np.float64)
        dominance = (additive == 1).astype(np.float64)
        additive -= q @ (q.T @ additive)
        dominance -= q @ (q.T @ dominance)

        g_aa = np.sum(additive * additive, axis=0)
        g_ad = np.sum(additive * dominance, axis=0)
        g_dd = np.sum(dominance * dominance, axis=0)
        determinant = g_aa * g_dd - g_ad**2
        c_a = additive.T @ y_residual
        c_d = dominance.T @ y_residual
        valid = determinant > 1e-12
        explained = np.zeros_like(c_a)
        explained[valid] = (
            g_dd[valid, None] * c_a[valid] ** 2
            - 2 * g_ad[valid, None] * c_a[valid] * c_d[valid]
            + g_aa[valid, None] * c_d[valid] ** 2
        ) / determinant[valid, None]
        explained = np.clip(explained, 0.0, y_sum_squares[None, :])
        rss = np.maximum(y_sum_squares[None, :] - explained, 1e-20)
        f_stat = (explained / 2.0) / (rss / residual_df)
        block_p = stats.f.sf(f_stat, 2, residual_df)
        block_p[~valid] = 1.0
        min_p[start:stop] = np.nan_to_num(
            block_p, nan=1.0, posinf=1.0, neginf=1.0
        ).min(axis=1)

    keep = min(k, n_snps)
    indices = np.argsort(min_p, kind="stable")[:keep].astype(np.int64)
    if verbose:
        _log(
            logger,
            f"加性+显性联合 GWAS: {n_snps} -> {keep}, "
            f"PCs={n_pc}, traits={traits.shape[1]}",
        )
    return indices, min_p


def gwas_screen(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    k: int,
    time_coordinates: list[float] | np.ndarray,
    n_pcs: int = 5,
    use_rint: bool = True,
    multivariate: bool = False,
    verbose: bool = True,
    logger=None,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化 GWAS 特征筛选（动态性状 + PCA 协变量 + min-P 策略）。

    使用 Frisch-Waugh-Lovell 定理将逐 SNP 独立 OLS 回归转换成全 P 维
    向量化线性代数运算。单表型模式下对每个 SNP 与 3 个性状（Mean / AUC /
    Slope）计算 OLS P 值并取最小；多元模式下将 ``(N, n_pheno, T)`` 中每个
    表型各提取 3 个动态性状拼接为 ``(N, n_pheno*3)`` 特征矩阵一次性回归，
    对每个 SNP 取所有表型-性状组合的最小 P 值（即 SNP 与多表型联合关联）。

    Parameters
    ----------
    snp_train : np.ndarray
        训练集基因型矩阵，形状为 ``(n_sample, n_snp)``，取值 ``{0, 1, 2}``。
    pheno_train : np.ndarray
        训练集表型数据。单表型: ``(n_sample, n_time)`` 或 ``(n_sample,)``；
        多元模式 (``multivariate=True``): ``(n_sample, n_pheno, n_time)``。
    k : int
        保留的 SNP 数目。
    time_coordinates : list[float] | np.ndarray
        与表型列对应的实际采样坐标；AUC 和斜率均在此坐标上计算。
    n_pcs : int, optional
        PCA 主成分个数，作为群体结构校正协变量，默认 ``5``。
    use_rint : bool, optional
        是否对性状做秩次逆正态变换（RINT），默认 ``True``。
    multivariate : bool, optional
        是否为多元模式（多表型联合筛选），默认 ``False``。
    verbose : bool, optional
        是否打印筛选进度与统计信息，默认 ``True``。

    Returns
    -------
    indices : np.ndarray
        按 min-P 从小到大排序的列索引，形状为 ``(k,)``，dtype ``int64``。
    min_p : np.ndarray
        每个 SNP 的最小 P 值，形状为 ``(n_snp,)``，dtype ``float64``。
    """
    snp_train = validate_dosage(snp_train)
    N, P = snp_train.shape

    # ---- 1. 构造动态性状特征矩阵 [N, F] ----
    if multivariate:
        n_pheno = pheno_train.shape[1]
        trait_cols = [
            _extract_dynamic_traits(
                pheno_train[:, i, :], time_coordinates, use_rint
            )
            for i in range(n_pheno)
        ]
        Y = np.concatenate(trait_cols, axis=1)  # [N, n_pheno*3]
        trait_names = [
            f"P{i + 1}_{t}" for i in range(n_pheno) for t in ("Mean", "AUC", "Slope")
        ]
    else:
        y_2d = pheno_train.reshape(N, -1)  # [N, T]
        Y = _extract_dynamic_traits(y_2d, time_coordinates, use_rint)  # [N, 3]
        trait_names = ["Mean", "AUC", "Slope"]
    F = Y.shape[1]

    # ---- 2. PCA 群体结构校正 ----
    snp_f64 = snp_train.astype(np.float64)
    n_pc = min(n_pcs, min(N, P) - 2)
    pca = PCA(n_components=n_pc, random_state=42)
    pcs = pca.fit_transform(snp_f64)  # [N, n_pc]

    # ---- 3. 协变量矩阵 A = [1, pcs] ----
    A = np.column_stack([np.ones(N), pcs])
    k_full = A.shape[1] + 1

    # ---- 4. QR 分解 ----
    Q, _ = np.linalg.qr(A)

    # ---- 5. 残差化所有 SNP ----
    Qt_snp = Q.T @ snp_f64
    snp_res = snp_f64 - Q @ Qt_snp
    snp_res_sq = (snp_res**2).sum(axis=0)
    snp_res_sq = np.maximum(snp_res_sq, 1e-20)

    # ---- 6. 残差化所有性状并向量化回归 ----
    df = N - k_full
    Y_res = Y - Q @ (Q.T @ Y)  # [N, F]

    num = snp_res.T @ Y_res  # [P, F]
    beta = num / snp_res_sq[:, None]

    y_res_sq = (Y_res**2).sum(axis=0)  # [F]
    rss = y_res_sq[None, :] - beta * num  # [P, F]
    rss = np.maximum(rss, 1e-20)

    sigma2 = rss / df
    se = np.sqrt(np.maximum(sigma2 / snp_res_sq[:, None], 1e-30))

    t_stat = beta / np.maximum(se, 1e-30)
    p_vals = 2 * stats.t.sf(np.abs(t_stat), df=df)  # [P, F]
    p_vals = np.nan_to_num(p_vals, nan=1.0, posinf=1.0, neginf=1.0)

    min_p = p_vals.min(axis=1)  # 每个 SNP 取所有表型-性状组合的最小 P
    if verbose:
        for j, name in enumerate(trait_names):
            _log(
                logger,
                f"    GWAS 性状 {name}: "
                f"min p={p_vals[:, j].min():.2e}  median p={np.median(p_vals[:, j]):.2e}",
            )

    # ---- 7. Top-k by 最小 min-P ----
    k = min(k, P)
    idx = np.argsort(min_p)[:k].astype(np.int64)
    if verbose:
        _log(logger, f"    GWAS 筛选完成: 按 min-P 保留 {k}/{P} SNPs")
        _log(
            logger, f"    min-P 分布: min={min_p[idx[0]]:.2e}  max={min_p[idx[-1]]:.2e}"
        )

    return idx, min_p


def sparse_multitask_refine(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    final_k: int,
    verbose: bool = True,
    logger=None,
) -> np.ndarray:
    """以固定多任务 Elastic Net 在候选内保留实验指定数量的 SNP。

    固定正则化避免每个重复都进行大规模交叉验证，使模拟实验的筛选协议
    保持唯一且运行时间可控。
    """
    dosage = validate_dosage(snp_train).astype(np.float64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if (
        candidates.ndim != 1
        or not candidates.size
        or np.unique(candidates).size != candidates.size
        or (candidates < 0).any()
        or (candidates >= dosage.shape[1]).any()
    ):
        raise ValueError("稀疏精筛候选必须是有效、互异的原始SNP索引")
    if final_k < 1:
        raise ValueError("稀疏精筛 final_k 必须为正整数")
    if final_k > len(candidates):
        raise ValueError(
            f"稀疏精筛候选数 {len(candidates)} 少于要求的 final_k={final_k}"
        )

    candidate_dosage = dosage[:, candidates]
    features = np.concatenate(
        [candidate_dosage, (candidate_dosage == 1).astype(np.float64)], axis=1
    )
    target = np.asarray(pheno_train, dtype=np.float64).reshape(len(dosage), -1)
    if not np.isfinite(target).all():
        raise ValueError("稀疏精筛表型必须为有限值")

    # 仅在训练集拟合。alpha=0.03、l1_ratio=0.80 在标准化后的加性/显性
    # 特征上提供稳定稀疏性，且不引入额外的交叉验证分支。
    alpha = 0.03
    l1_ratio = 0.80
    feature_scaler = StandardScaler().fit(features)
    y_mean = target.mean(axis=0, keepdims=True)
    y_scale = target.std(axis=0, ddof=0, keepdims=True)
    y_scale = np.where(y_scale > 1e-12, y_scale, 1.0)
    estimator = MultiTaskElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=3000,
        tol=1e-3,
        selection="cyclic",
        random_state=73,
    ).fit(feature_scaler.transform(features), (target - y_mean) / y_scale)

    count = len(candidates)
    coefficient = estimator.coef_
    group_score = np.sqrt(
        np.sum(coefficient[:, :count] ** 2 + coefficient[:, count:] ** 2, axis=0)
    )
    keep = final_k
    local = np.argsort(-group_score, kind="stable")[:keep]
    selected = np.sort(candidates[local])
    if verbose:
        _log(
            logger,
            f"稀疏联合精筛: candidates={count}, final={keep}, "
            f"alpha={alpha:.6g}, l1_ratio={l1_ratio:.2f}, "
            f"nonzero_groups={np.count_nonzero(group_score):d}",
        )
    return selected


def conditional_residual_union_screen(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    *,
    gwas_k: int,
    screen_k: int,
    time_coordinates: list[float] | np.ndarray,
    condition_k: int = 15,
    n_pcs: int = 5,
    multivariate: bool = True,
    verbose: bool = True,
    logger=None,
) -> np.ndarray:
    """边际GWAS/DC-SIS后对条件残差再筛一次，以解除强位点掩盖。"""
    dosage = validate_dosage(snp_train)
    n_samples, n_snps = dosage.shape
    if min(gwas_k, screen_k, condition_k) < 1:
        raise ValueError("条件筛选的 gwas_k、screen_k、condition_k 必须为正整数")
    if n_samples <= 2 * condition_k + 2:
        raise ValueError("训练样本不足以拟合条件位点的加性和显性效应")

    def marginal_scores(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, gwas_p = gwas_screen(
            dosage,
            target,
            k=n_snps,
            time_coordinates=time_coordinates,
            n_pcs=n_pcs,
            multivariate=multivariate,
            verbose=False,
            logger=logger,
        )
        dc_target = (
            _multivariate_pheno_matrix(target)
            if multivariate
            else target[:, 0, :]
        )
        _, dc_score = dc_sis_screen(
            dosage,
            dc_target,
            k=n_snps,
            verbose=False,
            logger=logger,
        )
        return gwas_p, dc_score

    first_gwas, first_dc = marginal_scores(pheno_train)
    rank_gwas = np.empty(n_snps, dtype=np.int64)
    rank_gwas[np.argsort(first_gwas, kind="stable")] = np.arange(n_snps)
    rank_dc = np.empty(n_snps, dtype=np.int64)
    rank_dc[np.argsort(-first_dc, kind="stable")] = np.arange(n_snps)
    conditioning = np.argsort(np.minimum(rank_gwas, rank_dc), kind="stable")[
        :min(condition_k, n_snps)
    ]

    condition_dosage = dosage[:, conditioning].astype(np.float64)
    condition_design = np.concatenate(
        [
            np.ones((n_samples, 1), dtype=np.float64),
            condition_dosage,
            (condition_dosage == 1).astype(np.float64),
        ],
        axis=1,
    )
    flat_pheno = pheno_train.reshape(n_samples, -1)
    coefficients, _, _, _ = np.linalg.lstsq(condition_design, flat_pheno, rcond=None)
    residual_pheno = (flat_pheno - condition_design @ coefficients).reshape(
        pheno_train.shape
    )
    residual_gwas, residual_dc = marginal_scores(residual_pheno)

    first_candidates = set(np.argsort(first_gwas, kind="stable")[:gwas_k].tolist())
    first_candidates.update(np.argsort(-first_dc, kind="stable")[:screen_k].tolist())
    residual_candidates = set(
        np.argsort(residual_gwas, kind="stable")[:gwas_k].tolist()
    )
    residual_candidates.update(
        np.argsort(-residual_dc, kind="stable")[:screen_k].tolist()
    )
    selected = np.array(sorted(first_candidates | residual_candidates), dtype=np.int64)
    if verbose:
        _log(
            logger,
            f"条件残差筛选: 第一轮={len(first_candidates)}, "
            f"条件位点={len(conditioning)}, 新增={len(residual_candidates - first_candidates)}, "
            f"最终={len(selected)} SNPs",
        )
    return selected


def gwas_dc_sis_screen_mmap(
    snp_file: str | Path,
    train_indices: np.ndarray,
    pheno_train: np.ndarray,
    *,
    gwas_k: int,
    screen_k: int,
    time_coordinates: list[float] | np.ndarray,
    block_columns: int = 16384,
    n_pcs: int = 5,
    pca_marker_count: int = 20000,
    multivariate: bool = True,
    logger=None,
) -> tuple[np.ndarray, dict]:
    """训练行上的 mmap 分块 GWAS→DC-SIS；不构造完整 SNP 副本或全长分数。

    GWAS 与 :func:`gwas_screen` 使用相同动态性状、FWL 回归和 min-P 口径。
    PCA 使用按原始列顺序确定性抽取的 label-free marker panel，避免构造全矩阵
    或执行 O(N^2 P) 的精确 Gram 累加；实际 marker 数与算法版本写入返回元数据。
    每列缺失众数仅由 ``train_indices`` 拟合；DC-SIS 只物化 GWAS 候选列。
    """
    import time

    if min(gwas_k, screen_k, block_columns, pca_marker_count) < 1:
        raise ValueError("gwas_k、screen_k、block_columns 和 pca_marker_count 必须为正整数")
    train_indices = np.asarray(train_indices, dtype=np.int64)
    raw = np.load(snp_file, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 2 or not isinstance(raw, np.memmap) or raw.flags.writeable:
        raise ValueError("真实筛选要求二维只读 NPY mmap")
    if (train_indices < 0).any() or (train_indices >= raw.shape[0]).any():
        raise ValueError("训练索引越界")
    n, p = len(train_indices), raw.shape[1]
    if pheno_train.shape[0] != n:
        raise ValueError("训练 phenotype 行数与训练索引不一致")

    def dosage_block(start, stop):
        block = np.asarray(raw[train_indices, start:stop])
        missing = np.isnan(block) | (block == -1)
        valid = (block == 0) | (block == 1) | (block == 2)
        if not (missing | valid).all():
            raise ValueError(f"SNP 列块 {start}:{stop} 含非法剂量")
        counts = np.stack([np.count_nonzero(block == value, axis=0) for value in (0, 1, 2)])
        if (counts.sum(axis=0) == 0).any():
            raise ValueError("训练个体中存在全缺失 SNP，mode 无法拟合")
        fill = counts.argmax(axis=0).astype(np.int8)
        return np.where(missing, fill[None, :], block).astype(np.float64, copy=False)

    started = time.perf_counter()
    panel_count = min(p, pca_marker_count)
    panel_indices = np.linspace(0, p - 1, panel_count, dtype=np.int64)
    panel = np.asarray(raw[np.ix_(train_indices, panel_indices)])
    panel_missing = np.isnan(panel) | (panel == -1)
    panel_valid = (panel == 0) | (panel == 1) | (panel == 2)
    if not (panel_missing | panel_valid).all():
        raise ValueError("PCA marker panel 含非法剂量")
    panel_counts = np.stack([
        np.count_nonzero(panel == value, axis=0) for value in (0, 1, 2)
    ])
    if (panel_counts.sum(axis=0) == 0).any():
        raise ValueError("PCA marker panel 含训练集全缺失列")
    panel_fills = panel_counts.argmax(axis=0)
    panel = np.where(panel_missing, panel_fills[None, :], panel).astype(np.float64)
    panel_center = panel.mean(axis=0)
    panel_variances = panel.var(axis=0)
    n_pc = min(n_pcs, min(n, p) - 2)
    pcs = PCA(n_components=n_pc, svd_solver="randomized", random_state=42).fit_transform(panel)
    pca_seconds = time.perf_counter() - started

    if multivariate:
        traits = np.concatenate([
            _extract_dynamic_traits(
                pheno_train[:, i, :], time_coordinates, True
            )
            for i in range(pheno_train.shape[1])], axis=1)
    else:
        traits = _extract_dynamic_traits(
            pheno_train.reshape(n, -1), time_coordinates, True
        )
    design = np.column_stack([np.ones(n), pcs])
    q, _ = np.linalg.qr(design)
    y_res = traits - q @ (q.T @ traits)
    y_res_sq = (y_res ** 2).sum(axis=0)
    df = n - design.shape[1] - 1
    if df <= 0:
        raise ValueError("训练样本不足以执行 GWAS")

    selected_idx = np.empty(0, dtype=np.int64)
    selected_score = np.empty(0, dtype=np.float64)
    gwas_started = time.perf_counter()
    keep = min(gwas_k, p)
    for start in range(0, p, block_columns):
        stop = min(p, start + block_columns)
        block = dosage_block(start, stop)
        residual = block - q @ (q.T @ block)
        residual_sq = np.maximum((residual ** 2).sum(axis=0), 1e-20)
        numerator = residual.T @ y_res
        beta = numerator / residual_sq[:, None]
        rss = np.maximum(y_res_sq[None, :] - beta * numerator, 1e-20)
        se = np.sqrt(np.maximum((rss / df) / residual_sq[:, None], 1e-30))
        p_values = 2 * stats.t.sf(np.abs(beta / np.maximum(se, 1e-30)), df=df)
        scores = np.nan_to_num(p_values, nan=1., posinf=1., neginf=1.).min(axis=1)
        indices = np.arange(start, stop, dtype=np.int64)
        combined_score = np.concatenate([selected_score, scores])
        combined_idx = np.concatenate([selected_idx, indices])
        if len(combined_score) > keep:
            chosen = np.argpartition(combined_score, keep - 1)[:keep]
            selected_score, selected_idx = combined_score[chosen], combined_idx[chosen]
        else:
            selected_score, selected_idx = combined_score, combined_idx
    ranked = np.argsort(selected_score, kind="stable")
    gwas_indices = selected_idx[ranked]
    gwas_seconds = time.perf_counter() - gwas_started

    candidates = np.asarray(raw[np.ix_(train_indices, gwas_indices)])
    missing = np.isnan(candidates) | (candidates == -1)
    counts = np.stack([np.count_nonzero(candidates == value, axis=0) for value in (0, 1, 2)])
    if (counts.sum(axis=0) == 0).any():
        raise ValueError("GWAS 候选含训练集全缺失列")
    fills = counts.argmax(axis=0)
    candidates = np.where(missing, fills[None, :], candidates).astype(np.float32)
    dc_target = (_multivariate_pheno_matrix(pheno_train, logger=logger)
                 if multivariate else pheno_train[:, 0, :])
    dc_started = time.perf_counter()
    local, _ = dc_sis_screen(candidates, dc_target, min(screen_k, len(gwas_indices)),
                             verbose=True, logger=logger)
    final = gwas_indices[local].astype(np.int64)
    dc_seconds = time.perf_counter() - dc_started
    candidate_original_order = np.argsort(gwas_indices)
    stable_candidate_scores = np.round(
        selected_score[ranked][candidate_original_order], decimals=12)
    return final, {
        "algorithm": "streaming_train_mode_gwas_fwl_dc_sis_v2",
        "mmap": True, "block_columns": block_columns, "n_train": n,
        "training_indices_sha256": training_indices_hash(train_indices),
        "gwas_population_structure": "pca",
        "gwas_n_pcs": n_pc,
        "pca_marker_count": panel_count,
        "gwas_pca_panel_size": pca_marker_count,
        "gwas_pca_panel_seed": None,
        "gwas_pca_algorithm_version": "sklearn_randomized_centered_even_panel_v1",
        "pca_marker_selection": "evenly_spaced_original_columns_label_free_v1",
        "pca_scaling": "center_only_no_standard_deviation",
        "pca_zero_variance_marker_count": int(np.count_nonzero(panel_variances == 0)),
        "pca_train_mode_fill_sha256": array_fingerprint(panel_fills.astype("<i8")),
        "pca_train_center_sha256": array_fingerprint(panel_center.astype("<f8")),
        "pca_panel_indices": panel_indices.tolist(),
        "pca_panel_indices_sha256": array_fingerprint(panel_indices.astype("<i8")),
        "pca_scores_sha256": array_fingerprint(pcs.astype("<f8")),
        "gwas_candidate_scores_sha256": array_fingerprint(
            stable_candidate_scores.astype("<f8")),
        "gwas_score_hash_round_decimals": 12,
        "original_snp_dim": p, "gwas_candidates": len(gwas_indices),
        "selected_count": len(final), "pca_seconds": pca_seconds,
        "gwas_seconds": gwas_seconds, "dc_sis_seconds": dc_seconds,
        "full_matrix_copy_created": False,
    }


# ---------------------------------------------------------------------------
# 正式筛选入口（质量控制 + 加性/显性 GWAS + 候选内 DC-SIS）
# ---------------------------------------------------------------------------


def screen_snps(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    time_coordinates: list[float] | np.ndarray,
    gwas_k: int = 0,
    qc_dosage: np.ndarray | None = None,
    multivariate: bool = False,
    cache_key: str | None = None,
    cache_dir: str = "dataset/screen_cache",
    verbose: bool = True,
    logger=None,
    n_pcs: int = 5,
    final_k: int = 20,
) -> np.ndarray | None:
    """正式 SNP 筛选入口：QC → 加性+显性 GWAS → 候选内 DC-SIS → final-K。

    ``snp_train`` 必须是仅由训练个体拟合缺失值后的完整剂量；``qc_dosage``
    保留填补前缺失标记，仅用于训练集 MAC 与缺失率质量控制。

    Parameters
    ----------
    snp_train : np.ndarray
        完整的离散剂量0/1/2，形状为 ``(n_sample, n_snp)``；拒绝缺失与模型类别码。
    pheno_train : np.ndarray
        训练集表型数据，形状为 ``(n_sample, n_pheno, n_time)``。
    time_coordinates : list[float] | np.ndarray
        与表型时间列一一对应的实际采样坐标，用于 GWAS 的 AUC 和斜率。
    gwas_k : int, optional
        加性+显性联合 GWAS 候选池大小；``0`` 表示关闭内部筛选。
    multivariate : bool, optional
        是否使用多元联合筛选（GWAS 与 DC-SIS 均多表型拼接联合筛选）。
        ``False`` 仅用于单表型数据，默认 ``False``。
    cache_key : str | None, optional
        缓存哈希键。传入时启用磁盘缓存；``None`` 表示不缓存，默认 ``None``。
    cache_dir : str, optional
        缓存文件存放目录，默认 ``'dataset/screen_cache'``。
    verbose : bool, optional
        是否打印筛选信息，默认 ``True``。
    logger : optional
        Logger 实例，传入时日志写入训练日志文件；``None`` 时输出到 stdout。

    Returns
    -------
    np.ndarray | None
        排序后的原始 SNP 列索引数组；若 ``gwas_k=0`` 则返回 ``None``。
    """
    # 即使缓存命中或筛选关闭，也不能让非法类别码/缺失绕过剂量契约。
    snp_train = validate_dosage(snp_train)
    if gwas_k == 0:
        return None
    if gwas_k < 1 or final_k < 1 or final_k > gwas_k:
        raise ValueError("筛选要求 1 <= final_k <= gwas_k")
    pheno_train = np.asarray(pheno_train)
    if pheno_train.ndim != 3 or pheno_train.shape[0] != len(snp_train):
        raise ValueError("筛选表型必须为 [个体, 表型, 时间] 且与 SNP 行数一致")
    if not multivariate and pheno_train.shape[1] != 1:
        raise ValueError("multivariate=false 仅支持单表型筛选")
    # 在缓存读取前验证实际时间轴，避免无效坐标绕过动态性状契约。
    _extract_dynamic_traits(
        pheno_train[:, 0, :], time_coordinates, use_rint=False
    )

    _, P = snp_train.shape
    n_pheno = pheno_train.shape[1]
    qc_indices = quality_control_snps(
        snp_train if qc_dosage is None else qc_dosage,
        verbose=verbose,
        logger=logger,
    )
    if qc_dosage is not None and np.asarray(qc_dosage).shape != snp_train.shape:
        raise ValueError("填补前后筛选剂量矩阵形状必须一致")
    if len(qc_indices) < final_k:
        raise ValueError(
            f"质量控制后仅有 {len(qc_indices)} 个 SNP，少于 final_k={final_k}"
        )

    # ---- 尝试加载缓存 ----
    if cache_key is not None:
        cached = _load_cache(cache_key, cache_dir, logger=logger)
        if cached is not None:
            if (
                cached.shape != (final_k,)
                or cached.dtype.kind not in "iu"
                or np.unique(cached).size != final_k
                or (cached < 0).any()
                or (cached >= P).any()
                or not np.isin(cached, qc_indices).all()
            ):
                raise ValueError("筛选缓存与 final_k 或当前 SNP 维度不一致")
            return cached

    if verbose:
        _log(logger, "=" * 60)
        _log(logger, "SNP 特征筛选")
        _log(logger, f"  编码版本: {GENOTYPE_ENCODING_VERSION}")
        _log(logger, f"  筛选算法版本: {SCREENING_ALGORITHM_VERSION}")
        _log(logger, f"  原始特征数: {P}")
        _log(logger, f"  QC后特征数: {len(qc_indices)}")
        _log(logger, f"  表型数量:   {n_pheno}")
        _log(logger, f"  gwas_k:     {gwas_k}")
        _log(logger, f"  final_k:    {final_k}")
        _log(logger, f"  时间坐标:  {list(time_coordinates)}")
        _log(logger, f"  筛选模式: {'多元联合 (GWAS+DC-SIS)' if multivariate else '单表型'}")
        _log(logger, "=" * 60)

    qc_matrix = snp_train[:, qc_indices]
    gwas_local, _ = additive_dominance_gwas_screen(
        qc_matrix,
        pheno_train,
        time_coordinates=time_coordinates,
        k=min(gwas_k, len(qc_indices)),
        n_pcs=n_pcs,
        multivariate=multivariate,
        verbose=verbose,
        logger=logger,
    )
    candidate_indices = qc_indices[gwas_local]
    candidate_dosage = snp_train[:, candidate_indices]
    dc_target = (
        _multivariate_pheno_matrix(pheno_train, logger=logger)
        if multivariate
        else pheno_train[:, 0, :]
    )
    dc_local, _ = dc_sis_screen(
        candidate_dosage,
        dc_target,
        k=final_k,
        verbose=verbose,
        logger=logger,
    )
    screened_idx = np.sort(candidate_indices[dc_local]).astype(np.int64)
    if verbose:
        _log(
            logger,
            f"正式筛选完成: raw={P}, qc={len(qc_indices)}, "
            f"gwas={len(candidate_indices)}, final={len(screened_idx)}",
        )
    if cache_key is not None:
        _save_cache(cache_key, cache_dir, screened_idx, logger=logger)
    return screened_idx
