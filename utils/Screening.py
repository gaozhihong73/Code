"""SNP 特征筛选共享模块。

提供 DC-SIS（距离相关系数）和 GWAS（OLS + PCA 协变量校正 min-P）两种筛选方法，
支持多表型（默认逐表型筛选后取并集；多元模式下将多表型拼接联合筛选）、
两阶段串联（GWAS 粗筛 → DC-SIS 精筛）、以及基于文件哈希的磁盘缓存（相同数据+参数直接加载索引，跳过计算）。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from .Genotype import GENOTYPE_ENCODING_VERSION, validate_dosage


SCREENING_ALGORITHM_VERSION = "gwas_pca_rint_minp_dcsis_v3"


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
    screen_k: int,
    gwas_k: int,
    seed: int,
    split_info: str = "",
    multivariate: bool = False,
    missing_strategy: str = "error",
    *,
    train_indices: np.ndarray | None = None,
    train_genotype: np.ndarray | None = None,
    train_phenotype: np.ndarray | None = None,
) -> str:
    """普通训练用训练个体内容指纹；旧K折/模拟调用保留路径+mtime 缓存。

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
    screen_k : int
        DC-SIS 保留数目。
    gwas_k : int
        GWAS 保留数目。
    seed : int
        随机种子（影响 train/val/test 划分）。
    split_info : str, optional
        额外划分信息（如 kfold fold index），默认空字符串。
    multivariate : bool, optional
        是否使用多元 DC-SIS 模式（多表型拼接联合筛选），默认 ``False``。
    missing_strategy : str, optional
        显式缺失策略标识，参与缓存键；编码/算法版本也参与缓存键。
    train_indices, train_genotype, train_phenotype : np.ndarray, optional
        普通训练必须同时传入训练原始行号及训练子矩阵；只哈希这些内容。

    Returns
    -------
    str
        32 位十六进制 MD5 哈希字符串。
    """
    hasher = hashlib.md5()
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
            "screen_k": screen_k, "gwas_k": gwas_k, "seed": seed,
            "multivariate": multivariate, "split_info": split_info,
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
    hasher.update(f"{screen_k}_{gwas_k}_{seed}_{split_info}".encode())

    # 筛选方法标识（多元 vs 逐表型并集）
    hasher.update(
        f"method={'mv_gwas_sis_v2' if multivariate else 'per_phenotype'}".encode()
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
    y_2d: np.ndarray, use_rint: bool = True
) -> np.ndarray:
    """从单表型时序数据提取 3 个动态性状 (Mean / AUC / Slope)。

    Parameters
    ----------
    y_2d : np.ndarray
        单表型数据, 形状为 ``(n_sample, n_time)``。
    use_rint : bool, optional
        是否对每个性状做秩次逆正态变换（RINT），默认 ``True``。

    Returns
    -------
    np.ndarray
        动态性状矩阵, 形状为 ``(n_sample, 3)``，列顺序为 Mean, AUC, Slope。
    """
    time_points = np.arange(1, y_2d.shape[1] + 1, dtype=np.float64)

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


def gwas_screen(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    k: int,
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
            _extract_dynamic_traits(pheno_train[:, i, :], use_rint)
            for i in range(n_pheno)
        ]
        Y = np.concatenate(trait_cols, axis=1)  # [N, n_pheno*3]
        trait_names = [
            f"P{i + 1}_{t}" for i in range(n_pheno) for t in ("Mean", "AUC", "Slope")
        ]
    else:
        y_2d = pheno_train.reshape(N, -1)  # [N, T]
        Y = _extract_dynamic_traits(y_2d, use_rint)  # [N, 3]
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


# ---------------------------------------------------------------------------
# 统一筛选入口（多表型 + 两阶段 + 缓存）
# ---------------------------------------------------------------------------


def screen_snps(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    screen_k: int = 0,
    gwas_k: int = 0,
    multivariate: bool = False,
    cache_key: str | None = None,
    cache_dir: str = "dataset/screen_cache",
    verbose: bool = True,
    logger=None,
) -> np.ndarray | None:
    """统一 SNP 筛选入口：多表型 + 两阶段串联 + 磁盘缓存。

    默认对多表型数据 ``(N, n_pheno, T)`` 逐表型独立筛选后取并集；
    ``multivariate=True`` 时改为多元联合筛选：GWAS 将各表型的动态性状拼接为
    ``(N, n_pheno*3)`` 一次回归取 min-P，DC-SIS 将各表型 z-score 标准化后
    拼接为 ``(N, n_pheno * T)`` 一次筛选，即 SNP 与 Y=(Y1,Y2,...,Yk) 联合关联。
    两阶段策略：GWAS 先在全 P 维向量化粗筛，DC-SIS 再在 GWAS 筛后的
    子集上精筛（避免对 2M SNP 跑 Python 循环）。

    Parameters
    ----------
    snp_train : np.ndarray
        完整的离散剂量0/1/2，形状为 ``(n_sample, n_snp)``；拒绝缺失与模型类别码。
    pheno_train : np.ndarray
        训练集表型数据，形状为 ``(n_sample, n_pheno, n_time)``。
    screen_k : int, optional
        DC-SIS 保留的 SNP 数目，``0`` 表示跳过，默认 ``0``。
    gwas_k : int, optional
        GWAS 保留的 SNP 数目，``0`` 表示跳过，默认 ``0``。
    multivariate : bool, optional
        是否使用多元联合筛选（GWAS 与 DC-SIS 均多表型拼接联合筛选）。
        ``False`` 时逐表型筛选取并集，默认 ``False``。
    cache_key : str | None, optional
        缓存哈希键。传入时启用磁盘缓存；``None`` 表示不缓存，默认 ``None``。
    cache_dir : str, optional
        缓存文件存放目录，默认 ``'outputs/screen_cache'``。
    verbose : bool, optional
        是否打印筛选信息，默认 ``True``。
    logger : optional
        Logger 实例，传入时日志写入训练日志文件；``None`` 时输出到 stdout。

    Returns
    -------
    np.ndarray | None
        排序后的 SNP 列索引数组；若 screen_k 和 gwas_k 均为 0 则返回 ``None``。
    """
    # 即使缓存命中或筛选关闭，也不能让非法类别码/缺失绕过剂量契约。
    snp_train = validate_dosage(snp_train)
    if screen_k <= 0 and gwas_k <= 0:
        return None

    N, P = snp_train.shape
    n_pheno = pheno_train.shape[1]

    # ---- 尝试加载缓存 ----
    if cache_key is not None:
        cached = _load_cache(cache_key, cache_dir, logger=logger)
        if cached is not None:
            return cached

    if verbose:
        _log(logger, "=" * 60)
        _log(logger, "SNP 特征筛选")
        _log(logger, f"  编码版本: {GENOTYPE_ENCODING_VERSION}")
        _log(logger, f"  筛选算法版本: {SCREENING_ALGORITHM_VERSION}")
        _log(logger, f"  原始特征数: {P}")
        _log(logger, f"  表型数量:   {n_pheno}")
        _log(logger, f"  screen_k:   {screen_k}")
        _log(logger, f"  gwas_k:     {gwas_k}")
        _log(logger, f"  筛选模式: {'多元联合 (GWAS+DC-SIS)' if multivariate else '逐表型并集'}")
        _log(logger, "=" * 60)

    # ---- 两阶段串联 ----
    # 阶段 1: GWAS 在全 P 维向量化粗筛（快，几分钟内完成 2M）
    gwas_union = None
    if gwas_k > 0 and gwas_k < P:
        if multivariate:
            # 多元模式: 多表型拼接动态性状一次回归, 直接取全局 top-k
            if verbose:
                _log(logger, f"[阶段1] 多元 GWAS 粗筛: {P} -> top-{gwas_k}")
            idx_gwas, _ = gwas_screen(
                snp_train,
                pheno_train,
                k=gwas_k,
                multivariate=True,
                verbose=verbose,
                logger=logger,
            )
            gwas_union = np.asarray(idx_gwas)
            if verbose:
                _log(logger, f"  多元 GWAS 联合筛选: {len(gwas_union)} SNPs")
        else:
            if verbose:
                _log(logger, f"[阶段1] GWAS 粗筛: {P} -> 每表型 top-{gwas_k}")
            gwas_sets = []
            for i in range(n_pheno):
                pheno_i = pheno_train[:, i, :]  # (N, T)
                if verbose:
                    _log(logger, f"  表型 {i + 1}/{n_pheno}:")
                idx_i, _ = gwas_screen(
                    snp_train, pheno_i, k=gwas_k, verbose=verbose, logger=logger
                )
                gwas_sets.append(idx_i)
            gwas_union = np.unique(np.concatenate(gwas_sets))
            if verbose:
                _log(logger, f"  GWAS 跨表型并集: {len(gwas_union)} SNPs")

    # 阶段 2: DC-SIS 筛选
    # P 较小时直接在全维度独立跑（并行策略，能发现 GWAS 遗漏的位点）
    # P 很大时（>5万）在 GWAS 子集上跑（串联策略，避免 Python 循环数小时）
    DC_SIS_FULL_THRESHOLD = 50000
    dc_union = None
    use_gwas_subset = gwas_union is not None and P > DC_SIS_FULL_THRESHOLD
    if screen_k > 0 and screen_k < P:
        if use_gwas_subset:
            subset = np.sort(np.asarray(gwas_union))
            snp_subset = snp_train[:, subset]
            if verbose:
                _log(
                    logger,
                    f"[阶段2] DC-SIS 精筛 (串联): 在 GWAS 候选 {len(subset)} SNPs 上, "
                    f"每表型 top-{screen_k}",
                )
        else:
            subset = np.arange(P)
            snp_subset = snp_train
            if verbose:
                _log(logger, f"[DC-SIS] 全维度独立筛选: P={P}, 每表型 top-{screen_k}")

        dc_sets = []
        if multivariate:
            # 多元模式: 各表型 z-score 标准化后拼接为 (N, n_pheno*T), 一次筛选
            if verbose:
                _log(
                    logger,
                    f"  多元 DC-SIS: 多表型拼接联合筛选 top-{screen_k}",
                )
            y_flat = _multivariate_pheno_matrix(pheno_train, logger=logger)
            idx_local, _ = dc_sis_screen(
                snp_subset, y_flat, k=screen_k, verbose=verbose, logger=logger
            )
            dc_sets.append(subset[idx_local])
        else:
            for i in range(n_pheno):
                pheno_i = pheno_train[:, i, :]
                if verbose:
                    _log(logger, f"  表型 {i + 1}/{n_pheno}:")
                idx_local, _ = dc_sis_screen(
                    snp_subset, pheno_i, k=screen_k, verbose=verbose, logger=logger
                )
                # 将子集局部索引映射回原始列索引
                dc_sets.append(subset[idx_local])
        dc_union = np.unique(np.concatenate(dc_sets))
        if verbose:
            _log(logger, f"  DC-SIS 保留 SNPs: {len(dc_union)}")

    # ---- 合并最终结果 ----
    if dc_union is not None and gwas_union is not None:
        if use_gwas_subset:
            # 串联模式: GWAS 仅做预过滤, 最终结果取 DC-SIS 精选
            screened_idx = dc_union
            if verbose:
                _log(
                    logger,
                    f"串联模式: GWAS 预筛 {len(gwas_union)} -> "
                    f"DC-SIS 精选 {len(dc_union)} SNPs (最终结果)",
                )
        else:
            # 并行模式: 两者独立筛选, 取并集
            screened_idx = np.union1d(dc_union, gwas_union)
            if verbose:
                overlap = len(dc_union) + len(gwas_union) - len(screened_idx)
                _log(
                    logger,
                    f"最终并集: DC-SIS {len(dc_union)} ∪ GWAS {len(gwas_union)} "
                    f"= {len(screened_idx)} SNPs (交集 {overlap})",
                )
    elif dc_union is not None:
        screened_idx = dc_union
    else:
        screened_idx = gwas_union

    if screened_idx is None:
        return None  # k >= 原始维度时保持全列，不尝试把 None 转成索引。
    screened_idx = np.sort(np.asarray(screened_idx)).astype(np.int64)

    if verbose:
        _log(logger, f"筛选完成: {P} -> {len(screened_idx)} SNPs")
        _log(logger, "=" * 60)

    # ---- 写入缓存 ----
    if cache_key is not None:
        _save_cache(cache_key, cache_dir, screened_idx, logger=logger)

    return screened_idx
