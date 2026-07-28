"""将 SNP CSV 文件批量转换为 PLINK2 所需的制表符分隔文本格式。

读取 ``../temp/csv/`` 下的所有 ``.csv`` 文件，从 SNP ID 中解析染色体和位置信息，
生成包含 CHR、START、END、LABEL 四列的无表头 ``.txt`` 文件。
"""

import glob
import os

import pandas as pd

# ==========================================
# 分别提取每个 CSV 文件中的染色体和位置信息
# ==========================================

# 1. 获取当前目录下所有的 .csv 文件
csv_files = glob.glob("../temp/csv/*.csv")

if not csv_files:
    print("未找到任何 CSV 文件，请检查路径。")

# 2. 循环处理每一个 CSV 文件
for file_path in csv_files:
    print(f"正在生成 PLINK2 格式文件: {file_path}")
    df = pd.read_csv(file_path)

    snp_ids = df.iloc[:, 0].astype("string")
    regions_df = snp_ids.str.split("_", expand=True)[[0, 1]]

    plink2_df = pd.DataFrame()
    plink2_df["CHR"] = regions_df[0]
    plink2_df["START"] = regions_df[1]  # PLINK2 bed1 格式起点
    plink2_df["END"] = regions_df[1]  # 终点与起点相同
    plink2_df["LABEL"] = snp_ids

    plink2_df = plink2_df.drop_duplicates()

    output_file = f"{os.path.splitext(file_path)[0]}.txt"
    # 必须以制表符分隔，不带表头
    plink2_df.to_csv(output_file, sep="\t", index=False, header=False)
    print(f"  -> 生成完毕: {output_file}")

print("所有文件处理完毕！")
