"""SNP 特征筛选共享模块。

提供 DC-SIS（距离相关系数）和 GWAS（OLS + PCA 协变量校正 min-P）两种筛选方法，
支持多表型（逐表型筛选后取并集）、两阶段串联（GWAS 粗筛 → DC-SIS 精筛）、
以及基于文件哈希的磁盘缓存（相同数据+参数直接加载索引，跳过计算）。
"""

import hashlib
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

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
) -> str:
    """根据数据文件路径+mtime 和筛选参数计算 MD5 缓存键。

    数据文件被修改（mtime 变化）时哈希自动改变，缓存失效。

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

    Returns
    -------
    str
        32 位十六进制 MD5 哈希字符串。
    """
    hasher = hashlib.md5()

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

    return hasher.hexdigest()


def _load_cache(cache_key: str, cache_dir: str) -> np.ndarray | None:
    """尝试从磁盘加载缓存的筛选索引。"""
    cache_path = Path(cache_dir) / f"{cache_key}.npy"
    if cache_path.exists():
        indices = np.load(cache_path)
        print(f"[缓存命中] 从 {cache_path} 加载 {len(indices)} 个 SNP 索引")
        return indices
    return None


def _save_cache(cache_key: str, cache_dir: str, indices: np.ndarray) -> None:
    """将筛选索引保存到磁盘。"""
    cache_path = Path(cache_dir) / f"{cache_key}.npy"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, indices)
    print(f"[缓存写入] 已保存 {len(indices)} 个 SNP 索引到 {cache_path}")


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


def dc_sis_screen(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    k: int,
    verbose: bool = True,
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
        print(f"DC-SIS 筛选完成: 按 dCor 保留 {k}/{P} SNPs")
        print(
            f"  dCor 分布: min={dcor[idx[-1]]:.4f}  max={dcor[idx[0]]:.4f}  "
            f"mean={dcor.mean():.4f}  median={np.median(dcor):.4f}"
        )

    return idx, dcor


def gwas_screen(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    k: int,
    n_pcs: int = 5,
    use_rint: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化 GWAS 特征筛选（动态性状 + PCA 协变量 + min-P 策略）。

    使用 Frisch-Waugh-Lovell 定理将逐 SNP 独立 OLS 回归转换成全 P 维
    向量化线性代数运算。对每个 SNP x 3 个性状（Mean / AUC / Slope）计算
    OLS P 值，对每个 SNP 取性状间最小 P 值，选出 min-P 最小的 top-k SNPs。

    Parameters
    ----------
    snp_train : np.ndarray
        训练集基因型矩阵，形状为 ``(n_sample, n_snp)``，取值 ``{0, 1, 2}``。
    pheno_train : np.ndarray
        训练集表型数据，形状为 ``(n_sample, n_time)`` 或 ``(n_sample,)``。
    k : int
        保留的 SNP 数目。
    n_pcs : int, optional
        PCA 主成分个数，作为群体结构校正协变量，默认 ``5``。
    use_rint : bool, optional
        是否对性状做秩次逆正态变换（RINT），默认 ``True``。
    verbose : bool, optional
        是否打印筛选进度与统计信息，默认 ``True``。

    Returns
    -------
    indices : np.ndarray
        按 min-P 从小到大排序的列索引，形状为 ``(k,)``，dtype ``int64``。
    min_p : np.ndarray
        每个 SNP 的最小 P 值，形状为 ``(n_snp,)``，dtype ``float64``。
    """
    N, P = snp_train.shape
    y_2d = pheno_train.reshape(N, -1)  # [N, T]

    # ---- 1. PCA 群体结构校正 ----
    snp_f64 = snp_train.astype(np.float64)
    n_pc = min(n_pcs, min(N, P) - 2)
    pca = PCA(n_components=n_pc, random_state=42)
    pcs = pca.fit_transform(snp_f64)  # [N, n_pc]

    # ---- 2. 协变量矩阵 A = [1, pcs] ----
    A = np.column_stack([np.ones(N), pcs])
    k_full = A.shape[1] + 1

    # ---- 3. QR 分解 ----
    Q, _ = np.linalg.qr(A)

    # ---- 4. 残差化所有 SNP ----
    Qt_snp = Q.T @ snp_f64
    snp_res = snp_f64 - Q @ Qt_snp
    snp_res_sq = (snp_res**2).sum(axis=0)
    snp_res_sq = np.maximum(snp_res_sq, 1e-20)

    # ---- 5. 提取 3 个动态性状: Mean / AUC / Slope ----
    T = y_2d.shape[1]
    time_points = np.arange(1, T + 1, dtype=np.float64)

    feat_mean = y_2d.mean(axis=1)
    feat_auc = np.trapz(y_2d, x=time_points, axis=1)

    t_bar = time_points.mean()
    t_dev = time_points - t_bar
    y_bar = y_2d.mean(axis=1, keepdims=True)
    feat_slope = ((y_2d - y_bar) * t_dev[None, :]).sum(axis=1) / (t_dev**2).sum()

    traits = {"Mean": feat_mean, "AUC": feat_auc, "Slope": feat_slope}
    if use_rint:
        traits = {name: _rint_transform(y) for name, y in traits.items()}

    # ---- 6. 对每个性状: 残差化 y, 向量化回归 ----
    df = N - k_full
    min_p = np.full(P, 1.0, dtype=np.float64)

    for trait_name, y in traits.items():
        y_r = y.astype(np.float64)
        y_res = y_r - Q @ (Q.T @ y_r)

        num = snp_res.T @ y_res
        beta = num / snp_res_sq

        y_res_sq = y_res @ y_res
        rss = y_res_sq - beta * num
        rss = np.maximum(rss, 1e-20)

        sigma2 = rss / df
        se = np.sqrt(np.maximum(sigma2 / snp_res_sq, 1e-30))

        t_stat = beta / np.maximum(se, 1e-30)
        p_vals = 2 * stats.t.sf(np.abs(t_stat), df=df)
        p_vals = np.nan_to_num(p_vals, nan=1.0, posinf=1.0, neginf=1.0)

        min_p = np.minimum(min_p, p_vals)
        if verbose:
            print(
                f"  GWAS 性状 {trait_name}: "
                f"min p={p_vals.min():.2e}  median p={np.median(p_vals):.2e}"
            )

    # ---- 7. Top-k by 最小 min-P ----
    k = min(k, P)
    idx = np.argsort(min_p)[:k].astype(np.int64)
    if verbose:
        print(f"GWAS 筛选完成: 按 min-P 保留 {k}/{P} SNPs")
        print(f"  min-P 分布: min={min_p[idx[0]]:.2e}  max={min_p[idx[-1]]:.2e}")

    return idx, min_p


# ---------------------------------------------------------------------------
# 统一筛选入口（多表型 + 两阶段 + 缓存）
# ---------------------------------------------------------------------------


def screen_snps(
    snp_train: np.ndarray,
    pheno_train: np.ndarray,
    screen_k: int = 0,
    gwas_k: int = 0,
    cache_key: str | None = None,
    cache_dir: str = "dataset/screen_cache",
    verbose: bool = True,
) -> np.ndarray | None:
    """统一 SNP 筛选入口：多表型 + 两阶段串联 + 磁盘缓存。

    对多表型数据 ``(N, n_pheno, T)``，逐表型独立筛选后取并集。
    两阶段策略：GWAS 先在全 P 维向量化粗筛，DC-SIS 再在 GWAS 筛后的
    子集上精筛（避免对 2M SNP 跑 Python 循环）。

    Parameters
    ----------
    snp_train : np.ndarray
        训练集基因型矩阵，形状为 ``(n_sample, n_snp)``。
    pheno_train : np.ndarray
        训练集表型数据，形状为 ``(n_sample, n_pheno, n_time)``。
    screen_k : int, optional
        DC-SIS 每个表型保留的 SNP 数目，``0`` 表示跳过，默认 ``0``。
    gwas_k : int, optional
        GWAS 每个表型保留的 SNP 数目，``0`` 表示跳过，默认 ``0``。
    cache_key : str | None, optional
        缓存哈希键。传入时启用磁盘缓存；``None`` 表示不缓存，默认 ``None``。
    cache_dir : str, optional
        缓存文件存放目录，默认 ``'outputs/screen_cache'``。
    verbose : bool, optional
        是否打印筛选信息，默认 ``True``。

    Returns
    -------
    np.ndarray | None
        排序后的 SNP 列索引数组；若 screen_k 和 gwas_k 均为 0 则返回 ``None``。
    """
    if screen_k <= 0 and gwas_k <= 0:
        return None

    N, P = snp_train.shape
    n_pheno = pheno_train.shape[1]

    # ---- 尝试加载缓存 ----
    if cache_key is not None:
        cached = _load_cache(cache_key, cache_dir)
        if cached is not None:
            return cached

    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"SNP 筛选: P={P}, n_pheno={n_pheno}, screen_k={screen_k}, gwas_k={gwas_k}"
        )
        print(f"{'=' * 60}")

    # ---- 两阶段串联 ----
    # 阶段 1: GWAS 在全 P 维向量化粗筛（快，几分钟内完成 2M）
    gwas_union = None
    if gwas_k > 0 and gwas_k < P:
        if verbose:
            print(f"\n[阶段1] GWAS 粗筛: {P} -> 每表型 top-{gwas_k}")
        gwas_sets = []
        for i in range(n_pheno):
            pheno_i = pheno_train[:, i, :]  # (N, T)
            if verbose:
                print(f"  表型 {i + 1}/{n_pheno}:")
            idx_i, _ = gwas_screen(snp_train, pheno_i, k=gwas_k, verbose=verbose)
            gwas_sets.append(idx_i)
        gwas_union = np.unique(np.concatenate(gwas_sets))
        if verbose:
            print(f"  GWAS 跨表型并集: {len(gwas_union)} SNPs")

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
                print(
                    f"\n[阶段2] DC-SIS 精筛 (串联): 在 GWAS 候选 {len(subset)} SNPs 上, "
                    f"每表型 top-{screen_k}"
                )
        else:
            subset = np.arange(P)
            snp_subset = snp_train
            if verbose:
                print(f"\n[DC-SIS] 全维度独立筛选: P={P}, 每表型 top-{screen_k}")

        dc_sets = []
        for i in range(n_pheno):
            pheno_i = pheno_train[:, i, :]
            if verbose:
                print(f"  表型 {i + 1}/{n_pheno}:")
            idx_local, _ = dc_sis_screen(
                snp_subset, pheno_i, k=screen_k, verbose=verbose
            )
            # 将子集局部索引映射回原始列索引
            dc_sets.append(subset[idx_local])
        dc_union = np.unique(np.concatenate(dc_sets))
        if verbose:
            print(f"  DC-SIS 跨表型并集: {len(dc_union)} SNPs")

    # ---- 合并最终结果 ----
    if dc_union is not None and gwas_union is not None:
        if use_gwas_subset:
            # 串联模式: GWAS 仅做预过滤, 最终结果取 DC-SIS 精选
            screened_idx = dc_union
            if verbose:
                print(
                    f"\n串联模式: GWAS 预筛 {len(gwas_union)} -> "
                    f"DC-SIS 精选 {len(dc_union)} SNPs (最终结果)"
                )
        else:
            # 并行模式: 两者独立筛选, 取并集
            screened_idx = np.union1d(dc_union, gwas_union)
            if verbose:
                overlap = len(dc_union) + len(gwas_union) - len(screened_idx)
                print(
                    f"\n最终并集: DC-SIS {len(dc_union)} ∪ GWAS {len(gwas_union)} "
                    f"= {len(screened_idx)} SNPs (交集 {overlap})"
                )
    elif dc_union is not None:
        screened_idx = dc_union
    else:
        screened_idx = gwas_union

    screened_idx = np.sort(np.asarray(screened_idx)).astype(np.int64)

    if verbose:
        print(f"筛选完成: {P} -> {len(screened_idx)} SNPs")

    # ---- 写入缓存 ----
    if cache_key is not None:
        _save_cache(cache_key, cache_dir, screened_idx)

    return screened_idx
