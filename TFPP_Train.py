"""
表型预测Transformer模型训练脚本
支持单次训练和批量训练所有实验组合
"""

from config.TFPP_Config import TFPPConfig
from trainer.TFPP_Trainer import TFPPTrainer


# 论文 4.1 的模拟数据
def train_sim_4_1():
    """
    训练单个实验组合
    """
    # ============ 配置实验参数 ============
    config = TFPPConfig(
        exp_name="result_4.1",
        use_kfold=False,  # 使用K折交叉验证
        snp_file="dataset/maize/snp/sim_snp.npy",
        pheno_dir="dataset/maize/cleaned_pheno",
        epochs=100,
        lr=0.0005,
        weight_decay=1e-4,
        batch_size=16,
        dropout=0.3,
        time_coordinates=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        phenotype_names=["sim_pheno"],
        norm_method="global",
    )

    # ============ 创建训练器 ============
    trainer = TFPPTrainer(config=config)

    # ============ 开始训练 ============
    results = trainer.train()
    return results


# 姜老师的模拟数据
def train_sim_j():
    """
    训练单个实验组合
    """
    # ============ 配置实验参数 ============
    config = TFPPConfig(
        exp_name="sim_snp",
        use_kfold=False,  # 使用K折交叉验证
        snp_file="dataset/sim_data/snp/sim_snp.npy",
        pheno_dir="dataset/sim_data/cleaned_pheno",
        epochs=100,
        lr=0.0005,
        weight_decay=1e-4,
        batch_size=32,
        dropout=0.3,
        time_coordinates=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        phenotype_names=["sim_pheno"],
        norm_method="global",
    )

    # ============ 创建训练器 ============
    trainer = TFPPTrainer(config=config)

    # ============ 开始训练 ============
    results = trainer.train()
    return results


# 玉米数据
def train_maize():
    """
    训练单个实验组合
    """

    # ============ 配置实验参数 ============
    config = TFPPConfig(
        exp_name="test_temp",
        use_kfold=False,  # 使用K折交叉验证
        snp_file="dataset/maize/snp/snp_50000.npy",
        pheno_dir="dataset/maize/cleaned_pheno",
        epochs=100,
        lr=0.0001,
        weight_decay=0.0001,
        batch_size=16,
        dropout=0.3,
        # phenotype_names = ['bn', 'CIVE', 'COM', 'ExB', 'ExG', 'ExGR', 'ExR', 'GLI', 'GLI2', 'gn', 'GreenCoverage', 'GRVI', 'IKAW', 'ints', 'IPCA', 'MGRVI', 'NDI', 'RGB_B', 'RGB_G', 'RGB_R', 'RGBVI', 'rn', 'veg', 'WI'],
        # phenotype_names = ['RGBVI', 'WI', 'ExB', 'COM', 'GreenCoverage', 'bn', 'IPCA', 'NDI'],
        # phenotype_names=['ExG', 'veg', 'GLI2', 'NDI'],
        # phenotype_names=["RGBVI"],
        time_coordinates=[
            10,
            12,
            14,
            15,
            17,
            19,
            24,
            26,
            37,
            44,
            51,
            65,
            80,
            101,
        ],  # 旧表型
        # time_coordinates=[10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 73, 80, 88, 94, 101, 108, 115],  # 新表型
        norm_method="global",  # 归一化策略：global, timepoint, residual_global
        scaler_type="zscore",  # 归一化函数：zscore，minmax，robust
    )

    # ============ 创建训练器 ============
    trainer = TFPPTrainer(config=config)

    # ============ 开始训练 ============
    results = trainer.train()

    return results


# 自然缺水数据
def train_nrwl():
    """
    训练单个实验组合
    """
    # ============ 配置实验参数 ============
    config = TFPPConfig(
        exp_name="nrwl",
        use_kfold=False,  # 使用K折交叉验证
        snp_file="dataset/nrwl/snp/snp_nrwl.npy",
        pheno_dir="dataset/nrwl/pheno",
        epochs=100,
        lr=0.0001,
        weight_decay=1e-4,
        batch_size=32,
        dropout=0.3,
        time_coordinates=[0, 2, 4, 6, 8, 10],
        phenotype_names=["pheno_nrwl"],
        norm_method="global",
    )

    # ============ 创建训练器 ============
    trainer = TFPPTrainer(config=config)

    # ============ 开始训练 ============
    results = trainer.train()

    return results


if __name__ == "__main__":
    # train_sim_4_1()
    # train_sim_j()
    train_maize()
    # train_nrwl()
