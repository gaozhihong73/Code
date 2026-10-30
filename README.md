# DynaGS

基于 Wide-Deep Transformer 的动态基因组选择框架。当前项目使用 **TFPP**，根据 SNP 基因型预测一个或多个表型在多个时间点上的动态轨迹。

## TFPP 简介

TFPP 结合两个互补通路：

- **Wide 通路**：以 SNP 剂量和 Legendre 时间多项式建模全局、可解释的加性时间趋势。
- **Deep 通路**：将 SNP 编码为 token，使用 Transformer 学习位点间的非线性交互。
- **门控融合**：自适应融合两条通路的输出，得到各表型、各时间点的预测值。

模型支持多表型联合预测、训练集内 SNP 筛选、早停和 K 折交叉验证。`--ablation_mode` 可切换至仅 Wide、仅 Deep 或无门控的直接相加模式。

## 项目结构

```text
DynaGS/
├── train.py                  # TFPP 训练入口
├── infer.py                  # TFPP 推理入口
├── config/TFPP_Config.py     # 配置定义
├── dataloader/TFPP_DataLoader.py
├── model/TFPP.py             # Wide-Deep Transformer
├── trainer/TFPP_Trainer.py   # 训练、验证与测试
├── utils/                    # 划分、筛选、归一化、指标等工具
├── dataset/                  # 本地数据目录
└── outputs/                  # 训练输出目录
```

## 安装

项目使用 Python 3.10 与 `uv` 管理依赖：

```bash
git clone https://github.com/gaozhihong73/DynaGP.git
cd DynaGS
uv sync
```

查看训练参数：

```bash
uv run python train.py --help
```

## 数据准备

默认数据目录：

```text
dataset/maize/
├── snp/snp_50000.npy
└── cleaned_pheno/
    ├── RGBVI.csv
    └── GreenCoverage.csv
```

- SNP 数据是形状为 `[样本数, SNP 数]` 的 `.npy` 矩阵；观测基因型采用 `0/1/2` 剂量编码，缺失值可为 `-1` 或 `NaN`。
- 每个表型对应一个 CSV，文件名即表型名称。CSV 的行顺序必须与 SNP 矩阵一致，列对应时间点。
- `--pheno_names` 指定读取的表型；`--time_coordinates` 的数量和顺序必须与 CSV 时间列一致。
- 如有稳定样本 ID，可用 `--sample_ids_file` 提供仅含 `SampleID` 列的 CSV，以校验复用的划分文件。

## 训练（`train.py`）

### 训练流程

1. **加载数据**：读取 SNP 矩阵及指定表型 CSV，校验样本数和时间点。
2. **划分个体**：普通模式按 `train/val/test` 比例划分；可用 `--split_file` 固定或复用。K 折模式为每一折分别确定训练、验证和测试个体。
3. **训练集内预处理**：仅使用训练个体拟合 SNP 缺失填补、GWAS/DC-SIS 筛选及表型归一化器；验证和测试数据只应用已拟合的结果。
4. **构建模型**：按筛选后的 SNP 数初始化 TFPP，Wide 通路建模时间趋势，Deep 通路学习 SNP 交互，再经门控融合。
5. **训练与选择**：采用 AdamW、Warmup + Cosine 学习率调度和梯度裁剪；按验证损失保存最佳权重，并使用 `patience` 早停。
6. **测试与保存**：恢复最佳验证权重后对测试集评估一次，保存模型、划分、筛选索引、归一化器、日志和训练历史。

### 快速开始

训练设置由 JSON 配置文件统一管理。使用默认真实数据配置：

```bash
uv run python train.py --config config/profiles/real_data.json --exp_name maize_rgbvi
```

要修改表型、时间点、模型结构、筛选数量或训练超参数，请编辑
`config/profiles/real_data.json`；模拟数据使用 `config/profiles/simulation.json`。

模拟数据批量入口默认遍历正式清单中的全部300套数据。只查看任务而不训练：

```bash
uv run python experiments/simulation/train_experiment2_screened.py \
  --config config/profiles/simulation.json --plan-only
```

通过`--condition`和`--rep`可限定单个任务；完整用法见
`experiments/simulation/README.md`。

配置文件字段与 `TFPPConfig` 一一对应，启动时会拒绝未知字段或不符合约束的值。

### 划分与可复现性

默认随机种子为 `73`。普通训练可保存并复用同一份划分，以保证不同实验使用相同个体：

```bash
uv run python train.py \
  --exp_name maize_seed73 \
  --split_file splits/maize_seed73.npz \
  --seed 73
```

指定的 `split_file` 不存在时会自动创建，存在时则直接复用。`--split_file` 不能与 `--use_kfold true` 同时使用。

K 折交叉验证示例：

```bash
uv run python train.py \
  --exp_name maize_cv \
  --use_kfold true \
  --n_folds 5 \
  --kfold_val_ratio 0.1 \
  --num_workers 0
```

### SNP 筛选与缺失值

`--gwas_k` 指定 GWAS 保留位点数，`--screen_k` 指定 DC-SIS 保留位点数；设为 `0` 即跳过对应步骤。筛选仅使用训练个体及其标签，不读取验证集或测试集标签。

```bash
uv run python train.py \
  --exp_name maize_screened \
  --gwas_k 10000 \
  --screen_k 10000 \
  --screen_multivariate true \
  --screen_missing_strategy mode \
  --wide_missing_strategy mode
```

`screen_missing_strategy=mode` 只依据训练个体，按位点以众数填补筛选阶段的缺失值；Deep 通路仍保留自身缺失类别。使用 `error` 时，检测到缺失值会直接报错。

### 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--exp_name` | 自动时间戳 | 实验名称，输出至 `outputs/tfpp/<exp_name>` |
| `--snp_file` | `dataset/maize/snp/snp_50000.npy` | SNP `.npy` 文件 |
| `--pheno_dir` | `dataset/maize/cleaned_pheno` | 表型 CSV 目录 |
| `--pheno_names` | `RGBVI` | 待预测表型，可传多个名称 |
| `--epochs` / `--batch_size` | `200` / `32` | 最大训练轮数和批大小 |
| `--lr` / `--weight_decay` | `1e-4` / `1e-4` | AdamW 学习率和权重衰减 |
| `--patience` | `20` | 验证损失无改善时的早停轮数 |
| `--norm_method` | `residual_global` | `global`、`timepoint` 或 `residual_global` |
| `--scaler_type` | `zscore` | `zscore`、`minmax` 或 `robust` |
| `--gwas_k` / `--screen_k` | `10000` / `10000` | GWAS 与 DC-SIS 保留数，`0` 为关闭 |
| `--degree` | `5` | Wide 通路 Legendre 多项式阶数 |
| `--ablation_mode` | `all` | `all`、`wide`、`deep` 或 `add` |
| `--use_kfold` / `--n_folds` | `false` / `10` | 是否使用 K 折及折数 |

布尔参数需显式使用 `true` 或 `false`，例如 `--use_kfold false`。

### 训练产物

普通训练的主要产物位于 `outputs/tfpp/<exp_name>/`：

```text
outputs/tfpp/<exp_name>/
├── models/
│   ├── best_model.pt
│   ├── model_config.json
│   └── checkpoint_integrity.json
├── preprocessing/
│   ├── split.npz
│   ├── screened_indices.npy
│   └── preprocessing.json
├── scalers/scaler.pkl
├── training_history/
│   ├── training_history_metrics.csv
│   └── training_summary.json
├── test_evaluation.json
└── <exp_name>.log
```

请保留整个实验目录。推理会读取其中的模型配置、筛选索引和归一化器，不应重新筛选 SNP 或拟合 scaler。

## 推理（`infer.py`）

使用保存的实验目录进行推理：

```bash
uv run python infer.py --model_path outputs/tfpp/maize_rgbvi
```

输入 SNP 的列顺序和训练时的筛选设置必须一致。更多参数请查看：

```bash
uv run python infer.py --help
```

## 开发检查

代码修改后可进行语法检查：

```bash
uv run python -m compileall config dataloader model trainer utils train.py infer.py
```

`dataset/` 与 `outputs/` 在当前环境中可能是本地软链接；迁移到其他机器时，请重新创建它们或在命令中指定实际路径。
