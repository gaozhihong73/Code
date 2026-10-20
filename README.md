<p align="center">
    <a href="https://github.com/gaozhihong73/DynaGP/releases">
        <img src="https://img.shields.io/github/v/release/gaozhihong73/DynaGP?style=flat-plastic&color=007acc&include_prereleases" alt="Release">
    </a>
    <a href="https://github.com/gaozhihong73/DynaGP/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/gaozhihong73/DynaGP?style=flat-plastic&color=42b883" alt="License">
    </a>
    <a href="https://github.com/gaozhihong73/DynaGP/commits/main">
        <img src="https://img.shields.io/github/last-commit/gaozhihong73/DynaGP?style=flat-plastic&color=fb8c00" alt="Last Commit">
    </a>
    <img src="https://img.shields.io/github/languages/top/gaozhihong73/DynaGP?style=flat-plastic&color=f1e05a" alt="Languages">
    <a href="https://github.com/gaozhihong73/DynaGP/stargazers">
        <img src="https://img.shields.io/github/stars/gaozhihong73/DynaGP?style=flat-plastic&color=dfb317" alt="Stars">
    </a>
</p>

# DynaGS

**基于 Wide-Deep Transformer 的动态基因组选择框架**

DynaGS 提出了一种融合 Legendre 多项式宽通路 (Wide) 与 Transformer 深通路 (Deep) 的基因组选择模型 (TFPP)，通过门控网络自适应融合两条通路的输出，实现对多时间点、多表型动态生长轨迹的高精度预测。

<!-- TODO: 插入模型架构图 -->
<!-- ![Architecture](docs/architecture.png) -->

## 核心亮点

- **Wide-Deep 双通路架构**: 宽通路使用 Legendre 多项式基函数捕捉全局生长趋势，深通路使用 Transformer 编码器建模 SNP 间的非线性交互效应，门控网络动态融合两者。
- **两阶段 SNP 特征筛选**: GWAS (F 检验) 粗筛 + DC-SIS (距离相关确定独立性筛选) 精筛，将 50000+ 维 SNP 压缩至 informative 子集，大幅降低计算开销。
- **多表型联合预测**: 单次训练同时预测多个表型 (如 RGBVI、GreenCoverage 等) 在全部时间点上的动态曲线。
- **灵活的消融实验**: 通过 `--ablation_mode` 一键切换完整模型 / 仅 Wide / 仅 Deep / 无门控相加，便于对比分析。
- **自动模型加载**: 推理时自动从 checkpoint 推测模型结构参数，无需手动指定网络配置。

## 项目目录结构

```
DynaGS/
├── train.py                  # 训练入口
├── infer.py                  # 推理入口
├── config/                   # Pydantic 配置定义
│   ├── TFPP_Config.py        #   主模型配置
│   ├── Sim_TFPP_Config.py    #   模拟数据配置
│   └── ConvAE_Config.py      #   ConvAE 配置
├── model/                    # 模型定义
│   ├── TFPP.py               #   Wide-Deep Transformer 主模型
│   └── ConvAE.py             #   卷积自编码器
├── trainer/                  # 训练器
│   ├── TFPP_Trainer.py       #   主模型训练器
│   ├── Sim_TFPP_Trainer.py   #   模拟数据训练器
│   ├── ConvAE_Trainer.py     #   ConvAE 训练器
│   └── _scheduler.py         #   学习率调度器工具
├── dataloader/               # 数据加载与预处理
│   ├── TFPP_DataLoader.py    #   真实数据加载器
│   ├── Sim_DataLoader.py     #   模拟数据加载器
│   └── ConvAE_DataLoader.py  #   ConvAE 数据加载器
├── utils/                    # 工具模块
│   ├── Screening.py          #   SNP 特征筛选 (GWAS + DC-SIS)
│   ├── Normalization.py      #   归一化 (zscore/minmax/robust)
│   ├── Loss.py               #   损失函数 (MSE, MSE+PCC, MSE+Offset)
│   ├── Calculate_Metrics.py  #   评估指标 (MSE, PCC, r²)
│   ├── model_loader.py       #   通用模型加载器 (自动推测参数)
│   └── Logger.py             #   日志工具
├── dataset/                  # 数据目录
├── outputs/                  # 训练输出目录
└── pyproject.toml            # 依赖管理 (uv)
```

## 环境准备与安装

### 系统要求

| 项目 | 要求 |
|:----:|:----:|
| 操作系统 | Linux (推荐 Ubuntu 20.04+) |
| Python | 3.10 |
| CUDA | 12.4 |
| GPU 显存 | >= 8 GB |
| 包管理器 | [uv](https://docs.astral.sh/uv/) |

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/gaozhihong73/DynaGP.git
cd DynaGS

# 安装 uv (如尚未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装全部依赖
uv sync
```

### 核心依赖

```bash
python >= 3.10, < 3.11
torch == 2.5.1
numpy == 1.23.5
scipy == 1.10.1
pandas == 2.0.3
scikit-learn == 1.3.2
pydantic >= 2.13
matplotlib == 3.8.2
seaborn == 0.13.2
```

## 数据集准备

### 数据目录结构

```bash
dataset/maize/
├── snp/
│   └── snp_50000.npy         # SNP 基因型矩阵 [样本数, SNP数], 编码 0/1/2
├── cleaned_pheno/            # 清洗后的表型数据
│   ├── RGBVI.csv             #   每个 CSV 对应一个表型
│   ├── GreenCoverage.csv     #   列名 = 时间点 (天), 行 = 样本
│   ├── CIVE.csv
│   └── ...                   #   共 20+ 个植被指数表型
└── pheno/                    # 原始表型数据 (未清洗)
```

### 数据格式说明

**SNP 数据** (`.npy`):

- 形状: `[n_samples, n_snps]`
- 编码: 加性编码 0/1/2 (对应 AA/Aa/aa)
- 示例: `snp_50000.npy` 包含 50000 个经过质控的 SNP 位点

**表型数据** (`.csv`):
- 文件名即表型名 (如 `RGBVI.csv`)
- 列名为采样时间点 (天)，如 `10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 80, 101`
- 每行为一个样本在各时间点的表型观测值

## 训练 (`train.py`)

### 基因型编码与筛选缺失边界

筛选与模型输入采用不同表示，转换集中在 `utils/Genotype.py`，调用方不得凭数值猜测表示：

| 含义 | 原始/筛选剂量 | 模型类别码 |
|------|---------------|------------|
| 观测基因型 | `0 / 1 / 2` | `1 / 2 / 3` |
| 原始缺失 | `-1` 或 `NaN`（不能直接进入筛选） | `0` |
| ConvAE 边界填充 | 不参与筛选 | `0`，与现有缺失类别共用 |

`dosage_to_model_codes()` 显式编码；`model_codes_to_dosage()` 将类别0恢复为缺失标记 `-1`。
`validate_dosage()` 与 `prepare_screening_dosage()` 保证 GWAS/DC-SIS 只接收完整的离散剂量。
三个筛选入口在计算或读取缓存前均校验输入，拒绝类别3、缺失、小数和无穷值。
若数组恰好只含1/2，数值本身不能区分两种表示，因此必须遵循调用方的显式转换契约。
模型通过 `model_code_tensor()` 校验后转换整数索引，禁止用截断连续表征来冒充类别码。

仓库原先只记录了模型缺失类别，没有训练集拟合的筛选填补规则。新增
`--screen_missing_strategy error|mode`：默认 `error` 明确拒绝缺失；`mode` 使用
`TrainingDosageImputer.fit(dosage, train_idx=...)` 在指定训练个体上拟合逐位点众数，
并列选择较小的观测剂量，训练中整列缺失则报错。`metadata_` 记录训练行号、缺失计数、
填充值和编码版本，可随实验保存为 JSON。缺失可以经显式众数策略填为0，但不会自动当作观测剂量0。
第1A步仅提供接口与合成测试，生产流程不执行 fit；遇缺失必须停止，待第2步在 train_idx 上接入。

真实管线在筛选前显式解码模型类别；模拟管线使用相同剂量筛选，进入模型时统一转为四类别码，
`SimTFPPConfig.num_embeddings` 默认改为4。旧三类别模拟 checkpoint 仍依赖旧编码，
不能与新编码管线混用。TFPP/ConvAE 网络结构及筛选时机没有改变。

`screen_multivariate` 的单一默认来源为 `TFPPConfig`，当前为 `true`；训练、推理 CLI
仍可显式传 `--screen_multivariate false`。有效值与缺失策略写入日志及 `model_config.json`。
旧实验若使用逐表型并集，推理时必须显式指定 `false`。
缓存键新增编码版本 `dosage012_missing-1_categorical0123_v1`、筛选算法版本
`gwas_pca_rint_minp_dcsis_v3` 和缺失策略，使旧文件不再命中；不删除旧缓存，训练索引哈希留待第2步加入。

### 共享个体划分

可用同一份 `.npz` 文件固定不同模型、不同表型实验的个体划分：

```bash
uv run python train.py --split_file splits/maize_seed73.npz --seed 73
```

文件不存在时，按当前比例和 seed 生成并保存；文件存在时，以文件中的索引为准，保留其划分 seed 和比例元数据，
不会重新划分或覆盖。`--split_file auto` 根据样本数、seed、比例和个体顺序哈希在 `splits/` 下命名。
单份划分不能与 `--use_kfold true` 同时使用；未指定 `split_file` 时保持原来的随机划分行为。

当前真实数据加载器没有显式个体 ID，因此文件明确记录 `sample_id_source=row_index`，
使用原始数据行号作为 sample_id。复用时须保持 SNP 与表型行顺序一致；仅凭行号无法识别
同样行数的数据被重新排序或替换。若上游有真实 ID，可通过公共 API 提供并校验顺序：

```python
from utils.Split import generate_split, save_split, load_split

split = generate_split(sample_ids=["maize-A", "maize-B", "maize-C"],
                       ratios=(1 / 3, 1 / 3, 1 / 3), seed=73)
save_split(split, "splits/example.npz")
split = load_split("splits/example.npz", sample_ids=["maize-A", "maize-B", "maize-C"])
```

ConvAE 通过 `ConvAEConfig(split_file="splits/maize_seed73.npz")` 复用文件，
只将其中 train/val 个体传给训练/验证 loader，test 个体保持留出。
两个 loader 也支持 `indices=split`，K 折 loader 继续接收三组显式索引并校验覆盖和互斥。
每行个体的全部时间点始终一起取出。本阶段仅统一划分，筛选和归一化原有调用顺序保持不变。

快速 CPU 测试（使用合成数据与临时目录，不需要玉米数据或 GPU）：

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

### 快速开始

```bash
# 使用默认参数训练 (自动按时间戳命名实验)
python train.py

# 自定义训练
python train.py \
  --exp_name maize_rgbvi \
  --pheno_names RGBVI \
  --epochs 200 \
  --degree 5 \
  --screen_k 10000 \
  --gwas_k 10000 \
  --loss_type mse_pcc \
  --use_kfold false
```

### 输出目录结构

训练完成后，输出保存在 `outputs/tfpp/{exp_name}/`:

```
outputs/tfpp/{exp_name}/
├── models/
│   ├── best_model.pt         # 最优模型权重
│   └── model_config.json     # 模型结构配置 (供推理自动加载)
├── scalers/
│   └── scaler.pkl            # 归一化器 (推理时反归一化用)
├── training_history/
│   ├── train_history.json    # 训练/验证损失曲线
│   └── training_summary.json # 训练摘要 (最优轮次、指标等)
└── logs/
    └── {exp_name}.log        # 完整训练日志
```

### 全部参数说明

#### 实验标识与数据路径

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--exp_name` | 实验名称，不指定则自动按时间戳生成 | `exp_YYYYMMDD_HHMMSS` |
| `--snp_file` | SNP 基因型数据路径 (.npy, 编码 0/1/2) | `dataset/maize/snp/snp_50000.npy` |
| `--pheno_dir` | 表型数据目录 (每个 CSV 对应一个表型) | `dataset/maize/cleaned_pheno` |

#### 数据集划分与预处理

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train_ratio` | 训练集比例 | 0.8 |
| `--val_ratio` | 验证集比例 (早停与模型选择) | 0.1 |
| `--test_ratio` | 测试集比例 (最终评估) | 0.1 |
| `--pheno_names` | 预测的表型名称列表 (需与 CSV 文件名对应) | `RGBVI` |
| `--time_coordinates` | 采样时间点坐标 (天) | `10 12 14 15 17 19 24 26 37 44 51 65 80 101` |
| `--norm_method` | 归一化策略: `global` / `timepoint` / `residual_global` | `residual_global` |
| `--scaler_type` | 缩放器类型: `zscore` / `minmax` / `robust` | `zscore` |

#### 训练超参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练总轮数上限 (配合早停) | 200 |
| `--batch_size` | 批次大小 (必须为 2 的倍数) | 32 |
| `--lr` | Deep 通路学习率 (Wide 通路自动 ×0.1) | 0.0001 |
| `--min_lr` | 学习率下限 (Cosine 衰减终点) | 1e-7 |
| `--weight_decay` | 权重衰减 (AdamW L2 正则) | 0.0001 |
| `--dropout` | Dropout 比例 | 0.25 |
| `--grad_clip_norm` | 梯度裁剪范数 (0=不裁剪) | 1.0 |
| `--patience` | 早停耐心值 | 20 |
| `--seed` | 随机种子 (训练与推理必须一致) | 73 |
| `--is_save` | 是否保存模型与训练历史 | true |
| `--num_workers` | DataLoader 工作线程数 | 8 |

#### SNP 特征筛选

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--screen_k` | DC-SIS 精筛保留 SNP 数 (0=跳过) | 10000 |
| `--gwas_k` | GWAS 粗筛保留 SNP 数 (0=跳过) | 10000 |

> **筛选策略** (取决于 SNP 总数 P):
> - **P > 50000 (串联模式)**: GWAS (F 检验) 先在全维度粗筛 `gwas_k` 个，DC-SIS 再在 GWAS 候选子集上精筛至 `screen_k` 个，最终结果取 DC-SIS 精选。
> - **P <= 50000 (并行模式)**: GWAS 和 DC-SIS 各自独立在全维度筛选，最终取两者并集 (互补覆盖)。
>
> 多表型时逐表型独立筛选后取并集；筛选结果基于文件哈希自动磁盘缓存。

#### 损失函数与正则化

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--loss_type` | 损失类型: `mse` / `mse_pcc` / `mse_offset` | `mse` |
| `--pcc_weight` | PCC 损失权重 (mse_pcc/mse_offset 生效) | 0.2 |
| `--offset_weight` | 个体偏移损失权重 (mse_offset 生效) | 0.5 |
| `--l1_wide_weight` | Wide 通路 L1 稀疏正则系数 (0=关闭) | 1e-5 |

#### 学习率调度 (Warmup + Cosine)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_warmup` | 是否启用学习率预热 | true |
| `--warmup_epochs` | 预热轮数 | 5 |
| `--warmup_start_factor` | 起始系数 (初始 lr = base_lr × 此值) | 0.1 |
| `--warmup_end_factor` | 结束系数 (预热结束 lr = base_lr × 此值) | 1.0 |

#### 模型结构

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--d_model` | Transformer 隐藏维度 (需能被 num_heads 整除) | 32 |
| `--num_heads` | 多头注意力头数 | 4 |
| `--num_layers` | 编码器堆叠层数 | 1 |
| `--d_ff` | FFN 隐藏层维度 | 128 |
| `--num_embeddings` | SNP 嵌入类别数 (-1/0/1/2) | 4 |
| `--degree` | Legendre 多项式阶数 (0=纯加性模型) | 5 |
| `--ablation_mode` | 消融模式: `all` / `wide` / `deep` / `add` | `all` |

#### K 折交叉验证

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_kfold` | 是否启用 K 折交叉验证 | false |
| `--n_folds` | 折数 | 10 |

> **Note:** 布尔参数 (`--is_save`, `--use_warmup`, `--use_kfold`) 接受 `true`/`false` 字符串。

## 推理 (`infer.py`)

### 快速开始

```bash
# 最简用法: 只需指定实验目录, 自动定位模型和归一化器
python infer.py --model_path outputs/tfpp/maize_rgbvi

# 对外部 SNP 数据进行预测
python infer.py \
  --model_path outputs/tfpp/maize_rgbvi \
  --input_snp dataset/maize/snp/snp_50000.npy \
  --output outputs/infer/prediction.csv

# 复现测试集预测 (需与训练时参数一致)
python infer.py \
  --model_path outputs/tfpp/maize_rgbvi \
  --seed 73 \
  --screen_k 10000 \
  --gwas_k 10000
```

### 两种预测模式

| 模式 | 触发条件 | 说明 |
|------|----------|------|
| 外部文件模式 | 传入 `--input_snp` | 对指定的 SNP 数据直接预测 |
| 测试集模式 | 不传 `--input_snp` | 自动复现训练时的数据划分，在测试集上预测并输出真实值对比 |

### 全部参数说明

#### 核心路径

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | **[必填]** 实验目录或 .pt 文件路径 | - |
| `--scaler_path` | 归一化器路径 (不指定则自动推导: `{exp_dir}/scalers/scaler.pkl`) | 自动推导 |
| `--input_snp` | 外部 SNP 数据路径 (.npy) | None (使用测试集) |
| `--output` | 预测结果 CSV 保存路径 | `outputs/infer/inference_{timestamp}.csv` |

#### 测试集复现参数 (需与训练时一致)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--snp_file` | 训练时的 SNP 文件路径 | `dataset/maize/snp/snp_50000.npy` |
| `--pheno_dir` | 训练时的表型目录 | `dataset/maize/cleaned_pheno` |
| `--train_ratio` | 训练集比例 | 0.8 |
| `--val_ratio` | 验证集比例 | 0.1 |
| `--test_ratio` | 测试集比例 | 0.1 |
| `--seed` | 随机种子 | 73 |
| `--scaler_type` | 缩放器类型 | `zscore` |
| `--screen_k` | DC-SIS 筛选数 | 10000 |
| `--gwas_k` | GWAS 筛选数 | 10000 |

#### 运行时参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch_size` | 推理批次大小 | 32 |
| `--num_workers` | DataLoader 工作线程数 | 8 |

> **Note:** 模型结构参数 (d_model, num_layers, degree 等) 无需手动指定，`infer.py` 通过 `utils/model_loader.py` 自动从 checkpoint 同级的 `model_config.json` 读取或从权重张量形状推测。
