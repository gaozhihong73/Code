import numpy as np
import pandas as pd


def raw_to_npy(raw_file, npy_file):
    print(f"正在读取 PLINK .raw 文件: {raw_file}")

    # 读取 PLINK 生成的矩阵文本
    df = pd.read_csv(raw_file, delim_whitespace=True)

    # 提取第 7 列及之后的所有纯基因型数据 (去掉 FID, IID 等前 6 列)
    genotype_data = df.iloc[:, 6:].values

    # 将缺失值转换为 -1
    genotype_data = np.nan_to_num(genotype_data, nan=-1)

    print(f"提取成功！最终矩阵形状为 : {genotype_data.shape}")

    # 保存为高度压缩的 NumPy 格式
    np.save(npy_file, genotype_data)
    print(f"已成功保存为: {npy_file}")


# 运行转换
raw_to_npy(
    "../temp/raw/data_gwas_snp_3_50000.raw",
    "../dataset/maize/snp_3/data_gwas_snp_3_50000.npy",
)
