"""模拟数据专用 DataLoader

与 TFPP_DataLoader 的关键差异:
- 数据已经是 Train/Val/Test 预划分,无需 random_split / KFold
- 数据格式为 .csv (DynaGS_S*_P*_H*_N*_Rep*_{Train|Val|Test}_{Genotype|Phenotype|TrueGV}.csv)
- V1..V20 视为同一表型 sim_pheno 的 20 个时间点
- 复用 TFPP_DataLoader 中的 _apply_normalization 与 _build_dataloaders_and_scalers 公共逻辑
- 内置 DC-SIS 筛选 (screen_k) 和 GWAS 筛选 (gwas_k), 仅用训练集计算, 取并集应用到 val/test
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

from .TFPP_DataLoader import _apply_normalization, _build_dataloaders_and_scalers


def _load_snp_csv(csv_path: str) -> np.ndarray:
    """读取 Genotype csv, 丢弃首列样本ID"""
    df = pd.read_csv(csv_path)
    snp = df.drop(columns=[df.columns[0]]).values.astype(np.float32)
    return snp


def _load_pheno_csv(csv_path: str) -> np.ndarray:
    """读取 Phenotype csv, 丢弃首列样本ID, reshape 为 [N, 1, T] (单表型多时间点)"""
    df = pd.read_csv(csv_path)
    pheno = df.drop(columns=[df.columns[0]]).values.astype(np.float32)
    return pheno[:, np.newaxis, :]  # [N, 1, T]


def _dc_sis_screen(snp_train, pheno_train, k, verbose=True):
    """DC-SIS 特征筛选 (距离相关系数)

    仅用训练集计算每个 SNP 与表型 Y 的距离相关系数, 返回 top-k SNP 列索引。
    利用 SNP 取值仅为 {0,1,2} 的特性, 用矩阵-向量积 + 计数闭式代替 N×N 矩阵分配,
    O(N) per SNP, 适合 P>>N 场景。

    Args:
        snp_train:    [N, P] float32 训练集基因型 (0/1/2)
        pheno_train:  [N, 1, T] float32 训练集表型
        k:            int, 要保留的 SNP 数目
        verbose:      bool, 打印进度

    Returns:
        indices: [k] int64, 排序后的列索引 (dcor 从大到小)
        dcor:    [P] float32, 每个 SNP 的距离相关系数
    """
    N, P = snp_train.shape
    y_2d = pheno_train.reshape(N, -1)  # [N, T]

    # ---- 1. Y 的距离矩阵 D_Y [N, N] (只算一次) ----
    diff = y_2d[:, None, :] - y_2d[None, :, :]
    D_Y = np.sqrt((diff**2).sum(axis=-1)).astype(np.float32)

    n = N
    nn2 = float(n * n)  # n²  注意: 不要与循环内的 cnt2 重名
    nn3 = float(nn2 * n)  # n³
    _eps = 1e-20

    # ---- dVar²(Y) / S2_Y / rY (只算一次) ----
    S1_YY = (D_Y * D_Y).sum() / nn2
    S2_Y = D_Y.mean()
    rY = D_Y.sum(axis=1).astype(np.float64)  # [N] 用于后续 dot
    dvar2_Y = S1_YY + S2_Y * S2_Y - 2.0 * (rY * rY).sum() / nn3

    # ---- 2. 逐 SNP 扫描 (O(N) per SNP, 无 N×N 分配) ----
    dcor = np.zeros(P, dtype=np.float32)
    sqrt_dvar2_Y = np.sqrt(max(dvar2_Y, 0.0))

    for p in range(P):
        # SNP 列 ∈ {0, 1, 2}
        x = snp_train[:, p]  # [N]
        is0 = np.asarray(x == 0, dtype=np.float32)
        is1 = np.asarray(x == 1, dtype=np.float32)
        is2 = np.asarray(x == 2, dtype=np.float32)

        cnt0 = is0.sum()
        cnt1 = is1.sum()
        cnt2 = is2.sum()

        # 单态 SNP: 距离方差为 0, dcor=0
        if cnt0 == n or cnt1 == n or cnt2 == n:
            dcor[p] = 0.0
            continue

        # D_Y @ v_k  每个 O(N²), 三个 = 3×N²
        # u0 = D_Y @ is0  # [N]
        u1 = D_Y @ is1
        u2 = D_Y @ is2

        # 各基因型组之间的 D_Y 子矩阵和: v_a^T D_Y v_b = dot(v_a, u_b)
        sm01 = np.dot(is0, u1)
        sm02 = np.dot(is0, u2)
        sm12 = np.dot(is1, u2)

        # sum(D_Y ⊙ D_X): D_X 按 {0,1,2} 分解
        # D_X[i,j] = 1*(v1_i*v0_j+v0_i*v1_j+v2_i*v1_j+v1_i*v2_j) + 2*(v2_i*v0_j+v0_i*v2_j)
        # => sum(D_Y*D_X) = 2*(sm01 + sm12 + 2*sm02)
        sum_DY_DX = 2.0 * (sm01 + sm12 + 2.0 * sm02)

        # ---- dCov²(Y, X) ----
        S1_YX = sum_DY_DX / nn2

        # mean(D_X): 用计数闭式, sum|xi-xj| = 2*(cnt0*cnt1 + cnt1*cnt2 + 2*cnt0*cnt2)
        sum_DX = 2.0 * (cnt0 * cnt1 + cnt1 * cnt2 + 2.0 * cnt0 * cnt2)
        S2_YX = S2_Y * sum_DX / nn2

        # rowSums_X[i] = sum_j |xi - xj|
        rX = is0 * (cnt1 + 2.0 * cnt2) + is1 * (cnt0 + cnt2) + is2 * (2.0 * cnt0 + cnt1)
        S3_YX = np.dot(rY, rX.astype(np.float64)) / nn3

        dcov2_YX = S1_YX + S2_YX - 2.0 * S3_YX

        # ---- dVar²(X) (全部用计数闭式, 无矩阵分配) ----
        S1_XX = 2.0 * (cnt0 * cnt1 + cnt1 * cnt2 + 4.0 * cnt0 * cnt2) / nn2

        mean_DX = sum_DX / nn2
        S2_XX = mean_DX * mean_DX

        # S3_XX = [cnt0*(cnt1+2*cnt2)² + cnt1*(cnt0+cnt2)² + cnt2*(2*cnt0+cnt1)²] / n³
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

        # if verbose and (p + 1) % 1000 == 0:
        #     print(f"  DC-SIS: {p + 1}/{P} SNPs 已扫描...")

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


def _rint_transform(x):
    """基于秩次的逆正态变换 (Rank-Based Inverse Normal Transformation)。

    将任意分布映射到标准正态, GWAS 中常用于减小偏离正态的影响。
    """
    ranks = stats.rankdata(x, method="average")
    probs = (ranks - 0.5) / len(x)
    return stats.norm.ppf(probs).astype(np.float64)


def _gwas_screen(snp_train, pheno_train, k, n_pcs=5, use_rint=True, verbose=True):
    """向量化 GWAS 特征筛选 (动态性状 + PCA 协变量 + min-P 策略)

    移植自 dyn_screen_info.R 的 GWAS 逻辑, 使用 Frisch-Waugh-Lovell 定理
    将逐 SNP 独立 OLS 回归转换成全 P 维向量化线性代数运算, 大幅提速。

    对每个 SNP × 3 个性状 (Mean / AUC / Slope) 计算 OLS P 值
    (校正前 5 个 PCA 主成分协变量), 对每个 SNP 取性状间最小 P 值,
    选出 min-P 最小的 top-k SNPs。

    Args:
        snp_train:    [N, P] float32 训练集基因型 (0/1/2)
        pheno_train:  [N, 1, T] float32 训练集表型
        k:            int, 保留 SNP 数
        n_pcs:        PCA 主成分个数 (作为协变量)
        use_rint:     是否对性状做 RINT 变换
        verbose:      打印进度

    Returns:
        indices: [k] int64, 按 min-P 从小到大排序的列索引
        min_p:   [P] float64, 每个 SNP 的最小 P 值
    """
    N, P = snp_train.shape
    y_2d = pheno_train.reshape(N, -1)  # [N, T]

    # ---- 1. PCA 群体结构校正: 提取前 n_pcs 个主成分作为协变量 ----
    snp_f64 = snp_train.astype(np.float64)
    n_pc = min(n_pcs, min(N, P) - 2)
    pca = PCA(n_components=n_pc, random_state=42)
    pcs = pca.fit_transform(snp_f64)  # [N, n_pc]

    # ---- 2. 协变量矩阵 A = [1, pcs] ----
    A = np.column_stack([np.ones(N), pcs])  # [N, n_pc + 1]
    k_full = A.shape[1] + 1  # 协变量数 + SNP 系数, 用于残差 dof

    # ---- 3. QR 分解: Q 是 A 的正交基 ----
    # M_A = I - Q Q^T  => y_res = M_A y, x_res = M_A x (列向量独立成立)
    Q, _ = np.linalg.qr(A)  # Q: [N, n_pc+1]

    # ---- 4. 残差化所有 SNP: x_res = M_A @ snp = snp - Q @ (Q^T @ snp) (一次 matmul, 全 P 维) ----
    Qt_snp = Q.T @ snp_f64  # [n_pc+1, P]
    snp_res = snp_f64 - Q @ Qt_snp  # [N, P]
    snp_res_sq = (snp_res**2).sum(axis=0)  # [P]  x_res^T x_res
    snp_res_sq = np.maximum(snp_res_sq, 1e-20)

    # ---- 5. 提取 3 个动态性状: Mean / AUC / Slope ----
    T = y_2d.shape[1]
    time_points = np.arange(1, T + 1, dtype=np.float64)

    feat_mean = y_2d.mean(axis=1)
    feat_auc = np.trapz(y_2d, x=time_points, axis=1)

    t_bar = time_points.mean()
    t_dev = time_points - t_bar  # [T]
    y_bar = y_2d.mean(axis=1, keepdims=True)  # [N, 1]
    feat_slope = ((y_2d - y_bar) * t_dev[None, :]).sum(axis=1) / (t_dev**2).sum()

    traits = {"Mean": feat_mean, "AUC": feat_auc, "Slope": feat_slope}
    if use_rint:
        traits = {name: _rint_transform(y) for name, y in traits.items()}

    # ---- 6. 对每个性状: 残差化 y, 向量化简单回归拿到 t-stat / p-value ----
    df = N - k_full
    min_p = np.full(P, 1.0, dtype=np.float64)  # 初始化为最大 p

    for trait_name, y in traits.items():
        y_r = y.astype(np.float64)
        y_res = y_r - Q @ (Q.T @ y_r)  # [N]

        # β = (x_res^T y_res) / (x_res^T x_res)  -- 向量化, 一次得到所有 P 个 β
        num = snp_res.T @ y_res  # [P]
        beta = num / snp_res_sq

        # RSS = ||y_res||² - β * (x_res^T y_res) = ||y_res||² - β * num
        y_res_sq = y_res @ y_res
        rss = y_res_sq - beta * num
        rss = np.maximum(rss, 1e-20)

        # SE(β) = sqrt(σ² / x_res^T x_res)
        sigma2 = rss / df
        se = np.sqrt(np.maximum(sigma2 / snp_res_sq, 1e-30))

        # t -> 两尾 P 值
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


class SimTFPPDataset:
    """
    模拟数据集 (预划分 Train/Val/Test)

    训练实际使用的是已划分好的三组数组: snp_train/val/test 与 pheno_train/val/test。

    双重特征筛选 (仅在训练集计算, 列索引应用到 val/test, 避免数据泄露):
    - DC-SIS 筛选 (screen_k): 用距离相关系数筛 top-k 个 SNP
    - GWAS 筛选 (gwas_k): 用 OLS+PCA 协变量校正的 min-P 筛 top-k 个 SNP
    - 最终取两者并集作为输入特征
    """

    def __init__(
        self,
        train_snp_csv: str,
        train_pheno_csv: str,
        val_snp_csv: str,
        val_pheno_csv: str,
        test_snp_csv: str,
        test_pheno_csv: str,
        phenotype_names: list[str],
        screen_k: int = 100,
        gwas_k: int = 100,
    ):
        if phenotype_names is None:
            phenotype_names = ["sim_pheno"]

        for p in [
            train_snp_csv,
            train_pheno_csv,
            val_snp_csv,
            val_pheno_csv,
            test_snp_csv,
            test_pheno_csv,
        ]:
            if not Path(p).exists():
                raise FileNotFoundError(f"模拟数据文件不存在: {p}")

        self.phenotype_names = phenotype_names

        # 读取各 split 的 snp 与 pheno
        self.snp_train = _load_snp_csv(train_snp_csv)
        self.snp_val = _load_snp_csv(val_snp_csv)
        self.snp_test = _load_snp_csv(test_snp_csv)

        self.pheno_train = _load_pheno_csv(train_pheno_csv)
        self.pheno_val = _load_pheno_csv(val_pheno_csv)
        self.pheno_test = _load_pheno_csv(test_pheno_csv)

        # 校验 snp_dim 与表型维度一致 (筛选前)
        if not (
            self.snp_train.shape[1] == self.snp_val.shape[1] == self.snp_test.shape[1]
        ):
            raise ValueError(
                f"各 split snp_dim 不一致: "
                f"{self.snp_train.shape[1]} / {self.snp_val.shape[1]} / {self.snp_test.shape[1]}"
            )

        if not (
            self.pheno_train.shape[2]
            == self.pheno_val.shape[2]
            == self.pheno_test.shape[2]
        ):
            raise ValueError(
                f"各 split 时间点数不一致: "
                f"{self.pheno_train.shape[2]} / {self.pheno_val.shape[2]} / {self.pheno_test.shape[2]}"
            )

        # ---- 双重特征筛选 (仅用训练集计算, 应用到 val/test, 避免数据泄露) ----
        original_p = self.snp_train.shape[1]
        dc_idx = None
        gw_idx = None

        if screen_k > 0 and screen_k < original_p:
            print(
                f"\n[1/2] DC-SIS 筛选: {original_p} -> {screen_k} SNPs "
                f"(仅训练集计算距离相关系数)"
            )
            dc_idx, _ = _dc_sis_screen(
                self.snp_train, self.pheno_train, k=screen_k, verbose=True
            )
        elif screen_k > 0:
            print(f"跳过 DC-SIS 筛选: screen_k={screen_k} >= SNP 总数 {original_p}")

        if gwas_k > 0 and gwas_k < original_p:
            print(
                f"\n[2/2] GWAS 筛选: {original_p} -> {gwas_k} SNPs "
                f"(PCA 协变量校正 + Mean/AUC/Slope 三性状 min-P)"
            )
            gw_idx, _ = _gwas_screen(
                self.snp_train, self.pheno_train, k=gwas_k, verbose=True
            )
        elif gwas_k > 0:
            print(f"跳过 GWAS 筛选: gwas_k={gwas_k} >= SNP 总数 {original_p}")

        if dc_idx is not None and gw_idx is not None:
            screened_idx = np.union1d(dc_idx, gw_idx)
            overlap = len(dc_idx) + len(gw_idx) - len(screened_idx)
            print(
                f"\n并集: DC-SIS {len(dc_idx)} ∪ GWAS {len(gw_idx)} "
                f"= {len(screened_idx)} SNPs (交集 {overlap})"
            )
        elif dc_idx is not None:
            screened_idx = dc_idx
        elif gw_idx is not None:
            screened_idx = gw_idx
        else:
            screened_idx = None

        if screened_idx is not None:
            screened_idx = np.sort(screened_idx)
            self.snp_train = self.snp_train[:, screened_idx].copy()
            self.snp_val = self.snp_val[:, screened_idx].copy()
            self.snp_test = self.snp_test[:, screened_idx].copy()
            self._screened_indices = screened_idx
            print(
                f"  最终特征维度: train={self.snp_train.shape}  "
                f"val={self.snp_val.shape}  test={self.snp_test.shape}"
            )
        else:
            self._screened_indices = None

    def __len__(self):
        return self.snp_train.shape[0]


def get_sim_dataloader(
    dataset: SimTFPPDataset,
    batch_size: int,
    norm_method: str,
    scaler_type: str,
    num_workers: int = 4,
    device: str = "cpu",
):
    """基于预划分的 train/val/test 直接构建 DataLoader

    Args:
        dataset: SimTFPPDataset 实例
        batch_size: 批次大小
        norm_method: 归一化策略 (global / timepoint / residual_global)
        scaler_type: 缩放器 (zscore / minmax / robust)
        num_workers: DataLoader 工作进程数
        device: 训练设备
    """
    # 1. 直接使用 dataset 中已划分好的三组数组
    train_snp, train_phenotype = dataset.snp_train, dataset.pheno_train
    val_snp, val_phenotype = dataset.snp_val, dataset.pheno_val
    test_snp, test_phenotype = dataset.snp_test, dataset.pheno_test

    # 2. 归一化 (仅用 train 拟合, 应用到 val/test)
    train_phe_norm, val_phe_norm, test_phe_norm, pheno_meta = _apply_normalization(
        train_phenotype,
        val_phenotype,
        test_phenotype,
        dataset.phenotype_names,
        norm_method,
        scaler_type,
    )

    # 3. 复用公共构建逻辑
    return _build_dataloaders_and_scalers(
        train_snp,
        train_phe_norm,
        val_snp,
        val_phe_norm,
        test_snp,
        test_phe_norm,
        dataset.phenotype_names,
        norm_method,
        pheno_meta,
        batch_size,
        num_workers,
        device,
    )
