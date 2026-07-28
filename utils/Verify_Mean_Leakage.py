import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_maize_phenotypes(phenotype_names: list[str]) -> np.ndarray:
    """
    加载并合并训练集和测试集的表型数据。
    """
    train_phenos = []
    test_phenos = []

    pheno_train_dir = Path("../dataset/maize/phenotype/train")
    pheno_test_dir = Path("../dataset/maize/phenotype/test")

    for phenotype_name in phenotype_names:
        train_path = pheno_train_dir / f"{phenotype_name}.csv"
        test_path = pheno_test_dir / f"{phenotype_name}.csv"

        if not train_path.exists() or not test_path.exists():
            print(f"错误: 找不到表型 {phenotype_name} 的CSV文件。")
            sys.exit(1)

        # 加载训练集和测试集
        train_pheno = pd.read_csv(train_path)
        train_pheno_np = train_pheno.values[:, np.newaxis, :].astype(np.float32)
        test_pheno = pd.read_csv(test_path)
        test_pheno_np = test_pheno.values[:, np.newaxis, :].astype(np.float32)

        train_phenos.append(train_pheno_np)
        test_phenos.append(test_pheno_np)

    # 合并所有表型数据
    train_phenotype_all = np.concatenate(train_phenos, axis=1)
    test_phenotype_all = np.concatenate(test_phenos, axis=1)
    all_phenotypes = np.concatenate((test_phenotype_all, train_phenotype_all), axis=0)

    return all_phenotypes


def load_microbe_phenotypes(phenotype_names: list[str]) -> np.ndarray:
    """
    加载并合并训练集和测试集的表型数据。
    """
    phenos = []

    pheno_dir = Path("../dataset/microbe/pheno")

    for phenotype_name in phenotype_names:
        pheno_path = pheno_dir / f"phenotype_{phenotype_name}.csv"
        if not pheno_path.exists():
            print(f"错误: 找不到表型 {phenotype_name} 的CSV文件。")
            sys.exit(1)

        # 加载训练集和测试集
        pheno_data = pd.read_csv(pheno_path).values[:, np.newaxis, :].astype(np.float32)

        phenos.append(pheno_data)

    # 合并所有表型数据
    all_phenotypes = np.concatenate(phenos, axis=1)

    return all_phenotypes


def verify_column_mean_leakage(PHENOTYPE_NAMES, phenotype_data):
    """检测列均值预测是否存在信息泄露。

    对每个表型的每个时间点列，使用训练集列均值作为所有样本的预测值，
    计算验证集和测试集上的 R2/PCC/MSE 指标。若测试集上出现高 R2 或高 PCC，
    则说明该时间点列方差极小，仅用列均值即可获得高性能，存在潜在泄露风险。

    Parameters
    ----------
    PHENOTYPE_NAMES : list[str]
        表型名称列表。
    phenotype_data : np.ndarray
        合并后的表型数据, shape ``(n_samples, n_pheno, n_time)``。
    """
    seed = 73
    print("=" * 80)
    print(f"检测列均值预测的性能 (Seed = {seed})")
    print("每个时间点列使用该列的均值作为所有样本的预测值")
    print("=" * 80)
    num_samples, num_phenotypes, num_timepoints = phenotype_data.shape
    # 数据划分
    indices = np.arange(num_samples)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.1, random_state=seed, shuffle=True
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.1, random_state=seed, shuffle=True
    )

    print(
        f"数据划分: {len(train_idx)} 训练 / {len(val_idx)} 验证 / {len(test_idx)} 测试\n"
    )

    # 提取数据
    train_phenotype = phenotype_data[train_idx]
    val_phenotype = phenotype_data[val_idx]
    test_phenotype = phenotype_data[test_idx]

    # 归一化
    train_phenotype_norm = np.zeros_like(train_phenotype)
    val_phenotype_norm = np.zeros_like(val_phenotype)
    test_phenotype_norm = np.zeros_like(test_phenotype)

    for pheno_idx in range(num_phenotypes):
        train_pheno = train_phenotype[:, pheno_idx, :]
        val_pheno = val_phenotype[:, pheno_idx, :]
        test_pheno = test_phenotype[:, pheno_idx, :]

        train_pheno_flat = train_pheno.flatten().reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(train_pheno_flat)

        train_phenotype_norm[:, pheno_idx, :] = scaler.transform(
            train_pheno.reshape(-1, 1)
        ).reshape(train_pheno.shape)

        val_phenotype_norm[:, pheno_idx, :] = scaler.transform(
            val_pheno.reshape(-1, 1)
        ).reshape(val_pheno.shape)

        test_phenotype_norm[:, pheno_idx, :] = scaler.transform(
            test_pheno.reshape(-1, 1)
        ).reshape(test_pheno.shape)

    # 逐表型检测
    results_summary = []

    for pheno_idx, pheno_name in enumerate(PHENOTYPE_NAMES):
        # 提取归一化后的数据 [n_samples, n_timepoints]
        train_pheno_norm = train_phenotype_norm[:, pheno_idx, :]  # [324, 14]
        val_pheno_norm = val_phenotype_norm[:, pheno_idx, :]  # [36, 14]
        test_pheno_norm = test_phenotype_norm[:, pheno_idx, :]  # [40, 14]

        # 计算训练集每个时间点列的均值
        train_column_means = train_pheno_norm.mean(axis=0)  # [14]

        # 创建预测值：每个时间点列都用该列的训练集均值填充
        # val_pred: [36, 14]，每一列都是该列的均值
        val_pred = np.tile(train_column_means, (val_pheno_norm.shape[0], 1))
        test_pred = np.tile(train_column_means, (test_pheno_norm.shape[0], 1))

        # 展平为1D用于计算整体指标
        val_true_flat = val_pheno_norm.flatten()
        val_pred_flat = val_pred.flatten()
        test_true_flat = test_pheno_norm.flatten()
        test_pred_flat = test_pred.flatten()

        # 计算整体指标
        val_r2 = r2_score(val_true_flat, val_pred_flat)
        val_pcc, _ = pearsonr(val_true_flat, val_pred_flat)
        val_mse = mean_squared_error(val_true_flat, val_pred_flat)

        test_r2 = r2_score(test_true_flat, test_pred_flat)
        test_pcc, _ = pearsonr(test_true_flat, test_pred_flat)
        test_mse = mean_squared_error(test_true_flat, test_pred_flat)

        # 计算每个时间点列的方差（用于判断是否存在低方差列）
        train_column_stds = train_pheno_norm.std(axis=0)  # [14]
        val_column_stds = val_pheno_norm.std(axis=0)
        test_column_stds = test_pheno_norm.std(axis=0)

        results_summary.append(
            {
                "Phenotype": pheno_name,
                "Val_R2": val_r2,
                "Val_PCC": val_pcc,
                "Val_MSE": val_mse,
                "Test_R2": test_r2,
                "Test_PCC": test_pcc,
                "Test_MSE": test_mse,
                "Train_Std_Min": train_column_stds.min(),
                "Train_Std_Max": train_column_stds.max(),
                "Train_Std_Mean": train_column_stds.mean(),
                "Val_Std_Mean": val_column_stds.mean(),
                "Test_Std_Mean": test_column_stds.mean(),
            }
        )

        # 打印详细信息（前3个表型）
        if pheno_idx < 3:
            print(f"\n表型: {pheno_name}")
            print(f"  训练集各时间点标准差: {train_column_stds}")
            print(
                f"  验证集 - R²: {val_r2:.4f}, PCC: {val_pcc:.4f}, MSE: {val_mse:.4f}"
            )
            print(
                f"  测试集 - R²: {test_r2:.4f}, PCC: {test_pcc:.4f}, MSE: {test_mse:.4f}"
            )

    # 打印结果表格
    df_results = pd.DataFrame(results_summary)

    print("\n" + "=" * 80)
    print("结果：使用每个时间点列的训练集均值作为预测值")
    print("=" * 80)
    print(
        df_results[["Phenotype", "Val_R2", "Val_PCC", "Test_R2", "Test_PCC"]].to_string(
            index=False
        )
    )

    print("\n各时间点列的标准差统计（训练集）:")
    print(
        df_results[
            ["Phenotype", "Train_Std_Min", "Train_Std_Max", "Train_Std_Mean"]
        ].to_string(index=False)
    )

    # 统计分析
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    print(
        f"验证集 - R²: {df_results['Val_R2'].mean():.4f} ± {df_results['Val_R2'].std():.4f} "
        f"[{df_results['Val_R2'].min():.4f}, {df_results['Val_R2'].max():.4f}]"
    )
    print(
        f"验证集 - PCC: {df_results['Val_PCC'].mean():.4f} ± {df_results['Val_PCC'].std():.4f} "
        f"[{df_results['Val_PCC'].min():.4f}, {df_results['Val_PCC'].max():.4f}]"
    )
    print(
        f"测试集 - R²: {df_results['Test_R2'].mean():.4f} ± {df_results['Test_R2'].std():.4f} "
        f"[{df_results['Test_R2'].min():.4f}, {df_results['Test_R2'].max():.4f}]"
    )
    print(
        f"测试集 - PCC: {df_results['Test_PCC'].mean():.4f} ± {df_results['Test_PCC'].std():.4f} "
        f"[{df_results['Test_PCC'].min():.4f}, {df_results['Test_PCC'].max():.4f}]"
    )

    # 泄露判断
    print("\n" + "=" * 80)
    print("泄露判断")
    print("=" * 80)

    high_r2_val = (df_results["Val_R2"] > 0.5).sum()
    high_pcc_val = (df_results["Val_PCC"].abs() > 0.7).sum()
    high_r2_test = (df_results["Test_R2"] > 0.5).sum()
    high_pcc_test = (df_results["Test_PCC"].abs() > 0.7).sum()

    if high_r2_test > 0 or high_pcc_test > 0:
        print("⚠️  警告：检测到列均值泄露！")
        print(
            f"   验证集: {high_r2_val}/{len(PHENOTYPE_NAMES)} 个表型 R²>0.5, "
            f"{high_pcc_val}/{len(PHENOTYPE_NAMES)} 个表型 |PCC|>0.7"
        )
        print(
            f"   测试集: {high_r2_test}/{len(PHENOTYPE_NAMES)} 个表型 R²>0.5, "
            f"{high_pcc_test}/{len(PHENOTYPE_NAMES)} 个表型 |PCC|>0.7"
        )
        print("   这说明某些时间点列的方差很小，仅用列均值就能获得高性能。")
    else:
        print("✓ 未检测到明显的列均值泄露")

    # 检查低方差列
    low_var_phenotypes = df_results[df_results["Train_Std_Min"] < 0.5]
    if len(low_var_phenotypes) > 0:
        print(
            f"\n⚠️  发现 {len(low_var_phenotypes)} 个表型存在低方差时间点列（Std < 0.5）:"
        )
        print(
            low_var_phenotypes[
                ["Phenotype", "Train_Std_Min", "Test_R2", "Test_PCC"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    # MAIZE_PHENOTYPE_NAMES = ['RGBVI', 'WI', 'ExB', 'CIVE', 'COM', 'GLI', 'ExG', 'gn',
    #                          'RGB_B', 'GreenCoverage', 'ints', 'veg', 'bn', 'RGB_G',
    #                          'RGB_R', 'IPCA', 'ExGR', 'GRVI', 'NDI', 'MGRVI',
    #                          'ExR', 'IKAW', 'GLI2', 'rn']
    MICROBE_PHENOTYPE_NAMES = ["E", "P", "S"]

    all_phenotypes = load_microbe_phenotypes(MICROBE_PHENOTYPE_NAMES)

    verify_column_mean_leakage(MICROBE_PHENOTYPE_NAMES, all_phenotypes)
