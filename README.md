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
- **训练产物复用**: 推理严格加载保存的模型配置、划分、筛选索引和 scaler，不重新筛选或拟合。

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
│   └── ConvAE.py             #   带表型辅助任务的 masked autoencoder
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
| ConvAE 边界填充 | 不参与筛选 | `0`，由独立 padding mask 区分真实缺失 |

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
普通 TFPP 与每个 K 折实验均在固定划分后接入 mode，仅用当前 train 个体拟合；模拟筛选仍不自动 fit。

真实管线在筛选前显式解码模型类别；模拟管线使用相同剂量筛选，进入模型时统一转为四类别码，
`SimTFPPConfig.num_embeddings` 默认改为4。旧三类别模拟 checkpoint 仍依赖旧编码，
不能与新编码管线混用。普通/折内 TFPP 均在划分后筛选；模拟训练暂保持旧流程。

`screen_multivariate` 的单一默认来源为 `TFPPConfig`，当前为 `true`；训练可显式传
`--screen_multivariate false`。有效值与缺失策略写入日志及 `model_config.json`。
推理仅使用训练记录，保留的同名 CLI 参数不重新计算或覆盖筛选。
缓存键新增编码版本 `dosage012_missing-1_categorical0123_v1`、筛选算法版本
`gwas_pca_rint_minp_dcsis_v3` 和缺失策略。普通训练还加入稳定训练索引哈希、训练 SNP/表型内容指纹、
表型列表、筛选参数及 split 标识；不读取全数据文件 mtime，所以修改验证/测试行不改变缓存键。
旧缓存保留，但普通训练不会命中原先全量筛选缓存。

### 普通 TFPP 的训练个体隔离与双输入视图

普通模式的顺序是：加载类别 SNP/原始表型 → 固定 split → 仅训练行转换剂量并筛选 →
用原始列索引切片三组类别 SNP → 仅训练表型拟合 scaler → 按筛选后维度创建模型。
多表型联合筛选能力与 `screen_multivariate` 显式配置保持可用。

Deep 只接收类别 `0/1/2/3`，保留缺失类别0。Wide 独立恢复剂量，再除以2得到 `0/0.5/1`；
`--wide_missing_strategy mode`（默认）只用训练个体的逐位点众数填补 Wide 缺失，并将填充值
保存为 checkpoint buffer；`error` 拒绝 Wide 缺失。不将筛选的填补结果写回 Deep 输入。
线性注意力、时间预测头和门控未改变；普通/折内模型统一使用 `dosage_v1`，模拟模型暂保留旧兼容路径。

每次普通训练在实验目录保存：

- `preprocessing/split.npz`：实际划分副本；JSON 同时记录源路径及文件 SHA-256。
- `preprocessing/screened_indices.npy`：相对于原始 SNP 矩阵的列索引，关闭筛选时为全部列。
- `preprocessing/preprocessing.json`：维度、筛选配置、训练行/数据指纹、缓存键、编码版本及填补元数据。
- `scalers/scaler.pkl`：只由训练表型拟合的 scaler。

`load_tfpp_model()` 和 `load_model_from_checkpoint()` 默认严格校验训练产物，并恢复 Wide
填补 buffer。测试集推理复用保存的 test_idx 和原始列索引，不再通过 seed 重建或读取表型筛选。
K折每折独立保存相同 schema 3 产物，可直接把 fold 目录传给新推理；旧 K 折产物仅允许显式 legacy 兼容。

### 共享个体划分

可用同一份 `.npz` 文件固定不同模型、不同表型实验的个体划分：

```bash
uv run python train.py --split_file splits/maize_seed73.npz --seed 73
```

文件不存在时，按当前比例和 seed 生成并保存；文件存在时，以文件中的索引为准，保留其划分 seed 和比例元数据，
不会重新划分或覆盖。`--split_file auto` 根据样本数、seed、比例和个体顺序哈希在 `splits/` 下命名。
单份划分不能与 `--use_kfold true` 同时使用；普通 TFPP 未指定 `split_file` 时也自动保存并复用划分，
索引生成规则与原先相同 seed/比例的 `random_split` 一致。

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
每行个体的全部时间点始终一起取出。普通/折内 TFPP 筛选和 ConvAE 表型归一化均只拟合 train 个体。

快速 CPU 测试（使用合成数据与临时目录，不需要玉米数据或 GPU）：

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

### ConvAE 的共享划分与表型归一化

ConvAE 沿用现有名称，实际为带表型辅助任务的 masked autoencoder；本步骤不增加卷积层或 TFPP 连续输入头。
`ConvAEDataset` 只加载原始表型、类别 SNP、稳定行索引 sample_id 和分块信息，不计算全体均值/标准差。
数据流为：原始数据 → 已确定的 IndividualSplit → train 表型拟合逐表型全局 StandardScaler →
仅 transform train/val/test → 独立归一化视图。原始 Dataset 不被归一化结果覆盖，test 不进入训练或早停。
表型含 NaN/Inf 时采用 `phenotype_missing_strategy=error`，不从 test 或全体样本估算填充值。

```bash
# 只准备数据，不初始化模型或运行训练；使用 TFPP 保存的普通 split
.venv/bin/python -B train_convae.py --prepare_only --exp_name maize_mae \
  --split_file outputs/tfpp/EXP/preprocessing/split.npz \
  --snp_file dataset/maize/snp/snp_50000.npy --device cpu --num_workers 0

# 与 TFPP K 折比较时，原样使用指定 fold 的三组索引
.venv/bin/python -B train_convae.py --prepare_only --exp_name maize_mae_cv \
  --split_file outputs/tfpp/EXP/folds/fold_01/preprocessing/split.npz \
  --snp_file dataset/maize/snp/snp_50000.npy --device cpu --num_workers 0
```

API 可直接传内存划分：`ConvAETrainer(config, indices=split, initialize_model=False)`；
此时不要同时设置 `config.split_file`。`get_mae_dataloaders(dataset, split, ...)` 返回
train/val/test loader、scalers 和 split 的 `MAEDataLoaders`；旧 `get_mae_dataloader()` 仍返回 train/val 两项，
但改为只用 train 拟合。公共数据接口默认保留 `(chunks, normalized_phenotype)`；trainer 内部
启用 include_sample_ids=True，使用三项 `(chunks, normalized_phenotype, sample_id)` 固定验证掩码。
直接访问原始 Dataset 则返回 `(chunks, raw_phenotype)`；旧 `phenotype_norm` 不再存在，
调用 Dataset 的旧归一化方法会明确报错，应改用固定 split 后的 loader。

`padding_mask` 为长度 target_length 的 bool 向量，True 只表示结构性 padding；
`chunk_padding_mask` 为 `[num_chunks, chunk_size]`，`chunk_valid_mask` 为其反值。
`original_length` 记录原始 SNP 有效长度，`chunk_starts`、`chunk_valid_lengths` 可恢复每块的原始位置。
`sample_with_metadata(i)` 提供真实缺失 mask、padding mask 和 sample_id：真实缺失类别0仍是有效位置，
不能把 `SNP == 0` 当作 padding。模型现已把这些 mask 用于特征投影、注意力、池化和重建目标。

数据产物保存于 `outputs/convae/<exp_name>/<split或fold标识>_<索引内容哈希>/`：

```text
config.json
preprocessing/preprocessing.json   # schema 3，编码/缺失策略、个体身份、时间列、chunk 契约与哈希
preprocessing/split.npz            # 实际使用的完整 manifest 副本，保留 TFPP fold 信息
preprocessing/padding_mask.npy
preprocessing/chunk_padding_mask.npy
scalers/scaler.pkl                 # 与 TFPP 同格式，支持 transform 和反归一化
```

`load_convae_preprocessing(path, dataset=..., config=...)` 校验 split/fold、sample_id 顺序、
表型和时间点、编码版本、原始有效长度、chunk 参数及文件 SHA-256；缺失/篡改时失败，不重新 fit。
当前 sample_id 是稳定行索引，不是生物学 ID。默认未显式提供时间坐标时保留 CSV 列名和顺序，不假定真实天数；
可用 `--time_coordinates` 声明逐列坐标。

兼容性变化：ConvAEConfig 默认比例现与 TFPP 一致为 0.8/0.1/0.1；None/auto 使用共享 splits/ 命名规则，
显式 split_file 必须已存在，不再为缺失的 fold 文件静默生成随机划分。旧80/20实验仍可显式设置
train_ratio=0.8、val_ratio=0.2、test_ratio=0，test 保持为空。
输出不再直接写入共享的 outputs/convae 根目录；同目录下不同数据契约/配置会拒绝覆盖，应使用新 exp_name。
旧本地 ConvAE_Train.py 被仓库忽略；可版本控制的命令入口为 train_convae.py。
自动潜在特征导出仍暂停；ConvAE v2 已支持动态表型维度和真正的辅助任务开关，见下节。

### ConvAE v2 模型、目标和 checkpoint

模型显式接收 `n_phenotypes`、`n_time_points`，表型输出为 `[batch, n_phenotypes, n_time_points]`。
trainer 根据 phenotype_names 和 CSV 时间列确定维度，并与显式 n_time_points/time_coordinates 校验，
不再固定输出14；不提供时间坐标时仅使用 CSV 列顺序。单表型14时间点继续受支持。
`use_pheno_help=false` 不构建/调用表型头，对应参数不进入优化器，表型损失严格为零。

模型选择口径在配置中固定，不允许根据运行结果切换：

- 开启辅助：`val_total_loss = recon_weight * val_recon_loss + pheno_weight * val_pheno_loss`。
- 关闭辅助：使用 `val_recon_loss`；API 配置必须显式设置 recon_weight=1、pheno_weight=0。
  CLI 仅在未显式给权重时按开关选择默认值。权重非负、有限且和为1。
- train/val 共用同一损失函数；重建 CE 除以可用 SNP 目标数，表型 MSE 除以个体×表型×时间点数。
  epoch 指标按监督元素总数汇总，不能简单平均各 batch 均值。
- 验证也执行掩码重建，mask 由 seed、完整 split 指纹和 sample_id 独立生成并保存；
  不随 epoch、batch 顺序或 batch_size 改变。test 个体不能进入验证选择入口。

padding 的 embedding 贡献先清零，完整 padding patch 不作为注意力 key/value，也不参与 mask 抽样或池化；
全 padding chunk 跳过注意力。最后的部分 patch 自动补内部 padding，保留其有效 SNP。
池化按可见的有效 SNP 数加权，真实缺失类别0保留可学习 embedding；类别0不提供重建监督。
重建只监督“选中 patch ∩ 非结构性 padding ∩ 观测类别1/2/3”的 SNP。
若某行仅有一个有效 patch 且全部被遮挡，解码使用非 SNP 的零上下文，不把 padding 当上下文。
无可用重建目标时损失与梯度为零；纯重建 batch 没有有效目标时跳过优化器更新，避免 weight decay 改动权重。
验证也记录可用目标数和无目标 batch 数；无目标时的零值不代表有可用的重建评分信息。

```bash
# 仅重建，不使用表型辅助；显式启用新的固定验证重建口径
.venv/bin/python -B train_convae.py --exp_name maize_mae_recon \
  --split_file outputs/tfpp/EXP/preprocessing/split.npz \
  --use_pheno_help false --recon_weight 1 --pheno_weight 0
```

`ConvAE.forward()` 仍返回四项：chunk 表征、三维表型预测（关闭时为 None）、重建 logits、bool patch mask。
新增 padding_mask、reconstruction_mask、generator 关键字参数；eval 传固定 mask 时也执行解码，
纯表征推理应显式使用 mask_ratio=0。省略 padding_mask 明确表示所有输入位置有效，不能从类别0推测 padding。
patch_size、d_model、d_ext、层数、dropout 和 train_mask_ratio/val_mask_ratio 均在配置与 CLI 中记录。

新增产物包括 `preprocessing/validation_masks.npz`、`models/model_config.json`、
`models/checkpoint_metadata.json`；history 同时记录 train/val total、recon、pheno、选择指标及目标数。
`best_model.pt` 采用 format_version=2 的结构化包，包含 state_dict、模型配置、最佳验证分项、
选择定义，以及预处理/固定 mask 的哈希。训练结束恢复最佳权重，不使用最后一轮权重代替。

```python
from utils.convae_model_loader import load_convae_model
model, info = load_convae_model("outputs/convae/EXP/SPLIT_TAG", device="cpu")
```

加载器使用 weights_only=True，校验 schema 3 数据契约、辅助开关、维度、选择口径、mask 和权重形状。
旧裸 state_dict、固定14维旧实验不会静默迁移：旧归一化/未屏蔽 padding 的语义不能等价恢复，
请保留旧代码用于诊断或在新 exp_name 下从头训练。第5步数据产物可独立读取，但新模型实验应使用新目录。
此步不增加 TFPP 连续输入头，也不导出最终潜在特征。

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
| `--n_folds` | 外层折数，至少2 | 10 |
| `--kfold_val_ratio` | 外层 train 内再分给早停 val 的比例 | 0.1 |

外层使用 `KFold(shuffle=True, random_state=seed)`，每个个体恰好进入 test 一次。
内层在剩余个体中按 `kfold_val_ratio` 划出 val，随机种子为 `seed + fold - 1`。
例如2折、400个体、默认比例时，每折约180个 train、20个 val、200个 test。
普通模式的 train/val/test_ratio 不控制此划分；单次 `split_file` 不能用于外层 K 折。

每折先固定三组原始行号，仅训练行拟合 GWAS/DC-SIS、筛选众数、Wide 众数和表型 scaler；
原始列索引应用于三组类别 SNP 后，按该折真实维度初始化全新 TFPP、优化器和调度器。
只用 val 早停和选模型，恢复最佳权重后仅评估 test 一次，预测导出复用该次前向结果。
模型初始化种子为 `seed + fold`（fold 从1开始），前一折训练轮数不影响下一折初态。
多表型联合/逐表型并集及 screen_k/gwas_k 仍按显式配置执行；缓存包含 fold 编号和训练行/内容指纹。

```bash
.venv/bin/python -B train.py --exp_name maize_cv --use_kfold true --n_folds 2 \
  --kfold_val_ratio 0.1 --screen_missing_strategy mode --num_workers 0

.venv/bin/python -B infer.py --model_path outputs/tfpp/maize_cv/folds/fold_01 \
  --snp_file dataset/maize/snp/snp_50000.npy --device cpu
```

产物布局（K折始终保存每折可复现产物，即使传入 `--is_save false`）：

```text
outputs/tfpp/maize_cv/
├── folds/fold_01/            # fold_02、fold_03... 同结构
│   ├── preprocessing/       # schema 3 JSON、split.npz、screened_indices.npy
│   ├── scalers/scaler.pkl
│   ├── models/              # best_model.pt（含 Wide buffer）、model_config.json
│   ├── training_history/
│   ├── config.json
│   ├── metrics.json
│   ├── test_predictions.npz # 归一化/原尺度预测及标签、sample_id、sample_idx、fold
│   └── test_predictions.csv
└── training_history/
    ├── kfold_summary.json   # 各折明细、总体及每表型指标均值/标准差
    ├── kfold_fold_details.csv
    ├── kfold_overall_stats.csv
    ├── kfold_phenotype_stats.csv
    └── kfold_predictions.csv # 按原始 sample_idx 排序，携带 sample_id、fold、phenotype
```

标准差使用 `ddof=0`，保留每折 PCC，不能以混合全部预测的单个 PCC 替代折间统计。
当前 sample_id 明确为稳定行索引，不代表已有生物学 ID。每折目录可独立通过严格推理哈希校验；
已有折 split 与新配置冲突时拒绝覆盖，应使用新实验目录。

> **Note:** 布尔参数 (`--is_save`, `--use_warmup`, `--use_kfold`) 接受 `true`/`false` 字符串。

## 推理 (`infer.py`)

推理默认要求实验目录中有 `models/best_model.pt`、`models/model_config.json`、
`preprocessing/{preprocessing.json,split.npz,screened_indices.npy}` 和 `scalers/scaler.pkl`。
缺少产物时直接报错，不重新筛选、不重新拟合 scaler 或缺失填充值。

```bash
# 保存的 test_idx + 保存的原始列索引；不需要读取表型
.venv/bin/python -B infer.py --model_path outputs/tfpp/maize_rgbvi \
  --snp_file dataset/maize/snp/snp_50000.npy --device cpu

# 可选读取测试标签：仅使用训练 scaler.transform，CSV 增加 True 列
.venv/bin/python -B infer.py --model_path outputs/tfpp/maize_rgbvi \
  --snp_file dataset/maize/snp/snp_50000.npy --pheno_dir dataset/maize/cleaned_pheno

# 原始全维剂量输入：校验原始维度，然后按训练索引切片
.venv/bin/python -B infer.py --model_path outputs/tfpp/maize_rgbvi \
  --input_snp external_raw.npy --input_stage raw --output outputs/infer/raw.csv

# 已筛选剂量输入：列序必须与 screened_indices.npy 一致，不再切片
.venv/bin/python -B infer.py --model_path outputs/tfpp/maize_rgbvi \
  --input_snp external_screened.npy --input_stage screened
```

两种外部输入均为二维 dosage `0/1/2`，缺失只能标为 `-1` 或 `NaN`，不是模型类别码。
必须显式指定 `--input_stage`，即使原始维度恰好等于筛选后维度。Deep 保留缺失类别0；
Wide 使用 checkpoint 内的训练填充值恢复剂量后除以2，推理期间禁止重新拟合。
`wide_missing_strategy=error` 的实验持久化空填补 buffer，拒绝任何筛选后缺失输入。

测试集原始数据必须保持训练时的个体行序；当前 sample_id 是行索引，不是生物学个体 ID。
可选表型 CSV 同样必须保持行序和训练时间点顺序。输出 `sample_idx` 是保存的原始个体索引。
`--batch_size`（默认32）控制批量大小，`--device cpu` 可强制 CPU。
不再接受推理期 seed、划分比例、screen_k/gwas_k 或 scaler_type 来重建预处理。

新训练保存 schema 3 预处理记录，包含 split、筛选索引、scaler、Wide 填充值的哈希及时间坐标；
完整的第2步 schema 2 产物仍可读取，Wide 填充值必须与 checkpoint buffer 一致。
每次推理输出 CSV 和同名 `.metadata.json`，记录 checkpoint/产物哈希及时间坐标。

旧 checkpoint 默认拒绝；仅诊断性复现可显式使用：

```bash
.venv/bin/python -B infer.py --model_path outputs/tfpp/old_experiment --legacy \
  --input_snp old_screened_dosage.npy --input_stage screened
```

legacy 会打印醒目警告，只接受外部 screened 数据，不猜测旧测试划分或重做监督筛选。
保留旧 categorical Wide 语义，CSV 与 JSON 均标记 `legacy=true, paper_eligible=false`，
不得进入论文正式实验汇总；旧三类别模拟 checkpoint 不可与新的四类别输入编码混用。
新格式产物损坏或缺失时，`--legacy` 不能绕过校验。`--scaler_path` 仅可用于 legacy。

CPU 合成测试（无需真实数据或 GPU）：

```bash
.venv/bin/python -B -m unittest discover -s tests -v
git diff --check
```

## ConvAE 潜在特征导出（第7步）

`export_convae.py` 只通过严格 `load_convae_model()` 加载当前 split/fold 上由 train
训练、由 val 选出的 `models/best_model.pt`；第6步的 schema 3 产物和 v2 checkpoint
继续兼容。旧 `_compress_data()` 入口明确报错，不能绕过最佳模型/产物校验。
训练不自动导出；TFPP 模型及输入接口保持原样。

```bash
.venv/bin/python -B export_convae.py \
  --model_path outputs/convae/<experiment>/<split_or_fold> \
  --split_file splits/individuals_n400_seed73.npz \
  --snp_file dataset/maize/snp/snp_50000.npy --batch_size 32 --device cpu
```

`--model_path` 使用训练日志中的实际实验目录；K折传入训练时使用的同一 fold manifest。
`--split_file` 必须显式指定，校验文件 SHA256 与保存的三组索引；不按 seed 重建。
`--snp_file` 可省略，默认保存配置中的原始 dosage 文件（0/1/2，缺失 -1/NaN）。
当前 sample_id 是原始行索引，SNP 文件必须保持原个体行序，无法鉴别无 ID 文件的生物学身份。
可用 `--checkpoint_sha256 <hash>` 和 `--model_config <models/model_config.json>`
进一步绑定任务预期。缺少产物、哈希/配置不符或非法 SNP 编码均失败。

`ConvAE.encode_features(x, padding_mask=...) -> Tensor[B,D]` 使用 `eval()` 和
`inference_mode()`；只走编码器，不调用表型头/decoder、不抽随机掩码、不读表型 CSV。
严格加载器仍校验训练 scaler 的完整性，但它不传入编码函数或参与特征计算。
Deep 输入为模型类别码，真实缺失保留0；结构性 padding 由独立 mask 排除。
表示固定为 chunk-major C-order 展平：`D = num_chunks * d_ext`，
`column = chunk_index * d_ext + feature_index`，按 SNP 起点递增保留最后不完整 chunk。

默认产物：

```text
<experiment>/exports/<split_fingerprint>/<checkpoint_sha256>/<snp_file_sha256>/
├── metadata.json
└── features/
    ├── train.npy             # float32 [N_train, D]
    ├── train_sample_ids.npy  # int64 [N_train]，manifest 顺序
    ├── val.npy
    ├── val_sample_ids.npy
    ├── test.npy
    └── test_sample_ids.npy
```

`--output_root` 可改导出根目录；不同 split/fold、checkpoint、SNP 内容分别寻址。
已存在目录禁止覆盖，重复独立复现应使用另一输出根目录。
`encode_feature_batches()` 将批次按 sample_id 恢复 manifest 原序，拒绝重复、缺失或额外 ID。
写入后 `load_exported_features()` 重新校验每个文件 SHA256、shape、dtype、有限值及 ID 顺序；
写入/校验失败清理本次新目录。

metadata 摘要示例（40个合成个体、13 SNP、chunk_size=8、overlap=2、d_ext=4）：

```json
{
  "export_schema_version": 1,
  "preprocessing_schema_version": 3,
  "latent_layout": "chunk_major_flatten_C_order",
  "original_snp_dim": 13,
  "num_chunks": 2,
  "d_ext": 4,
  "latent_dim": 8,
  "reconstruction_mask_ratio": 0,
  "encoder_only": true
}
```

完整 JSON 另含 git commit/dirty、split/fold/样本顺序、checkpoint 和数据文件路径/哈希、
配置及模型选择口径、表型/时间点、编码与 padding 规则、每组数组 shape/dtype/文件内容哈希。
同 checkpoint/输入重复导出相同，CPU 不同 batch size 的测试容差为 `rtol=1e-5, atol=1e-6`。

CPU 合成冒烟测试自动在临时目录生成40个体、13 SNP、真实缺失与结构性 padding，
单轮训练最小模型后通过上述 CLI 导出三集合（24/8/8行，每行8维），结束清理临时产物：

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_convae_export.py -v
.venv/bin/python -B -m unittest discover -s tests -v
git diff --check
```

## ConvAE 独立 Ridge 下游基线（第8步）

`ridge_convae.py` 仅用于评价“ConvAE 表征的独立线性探针/下游基线”，
不等同于 TFPP 的同构模型比较，也不能据此声称 ConvAE 优于监督式 SNP 筛选。
本步骤不增加 MLP、不改变 TFPP 或 ConvAE，不自动执行正式实验。

模型固定为多输出 `Ridge(fit_intercept=True, solver="lsqr", tol=1e-6, max_iter=10000)`；
目标按 phenotype-major、time-minor 顺序从 `[N,P,T]` 展平为 `[N,P*T]`。
alpha 固定为 `[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]`，不提供覆盖选项。
每个候选只在 train 上 fit；按归一化尺度整体 val MSE 选择，数值并列
（`rtol=1e-12, atol=1e-12`）取较大 alpha。保留选中的 train-only 模型，不合并 val 重拟合。
连续特征逐列 Z-score、表型按各表型全时间点 global Z-score，均只 fit train。
表型 scaler 独立拟合并保存，不复用 ConvAE scaler；缺失表型明确报错。

标签采用显式 ID 的 NPZ，不猜测或自动重排；字段为：

```text
phenotype_names, time_labels                 # 必须与 export metadata 同序
train_phenotype, train_sample_ids             # 原始量纲 [N_train,P,T]、[N_train]
val_phenotype, val_sample_ids
test_phenotype, test_sample_ids              # 同样为 manifest 原序
```

可用 `utils.ConvAE_Ridge.save_probe_targets()` 建立此文件。以下仅展示已有数据如何打包，
CSV 必须已确认与原始 SNP/sample_id 行序一致；行索引不代表生物学身份验证：

```python
import numpy as np
import pandas as pd
from utils.ConvAE_Export import load_exported_features
from utils.ConvAE_Ridge import save_probe_targets
from utils.Split import load_split

arrays, meta = load_exported_features(export_dir)
split = load_split(meta["split_file"])
raw = np.stack([pd.read_csv(f"{pheno_dir}/{name}.csv").to_numpy()
                for name in meta["phenotype_names"]], axis=1)
targets = {role: (raw[getattr(split, role + "_idx")], arrays[role][1])
           for role in ("train", "val", "test")}
save_probe_targets("ridge_targets.npz", targets,
                   phenotype_names=meta["phenotype_names"], time_labels=meta["time_labels"])
```

运行示例（不自动执行）：

```bash
.venv/bin/python -B ridge_convae.py \
  --export_dir <第7步导出目录> --phenotype_file ridge_targets.npz \
  --output_root outputs/baselines/convae_ridge --seed 73
```

先校验所有特征/ID 文件哈希，再与独立 split/fold manifest 逐行比较。
`--split_file` 可提供迁移后的 manifest 路径，内容哈希必须保持一致。
train/val 标签分别读取；九个候选搜索结束后才读取 test 标签并评估一次，
状态保存为 `test_evaluation_count=1`，重复评估入口明确失败。
重新加载模型仅做无标签预测复现，不增加测试指标评估次数。

产物目录为 `<output_root>/<export_metadata_sha256>/<baseline_config_sha256>/`，
已存在则报错，禁止覆盖。配置哈希包含固定模型/归一化/并列规则、seed 和标签文件指纹；
对标签做回归测试时可使用另一输出根目录。目录保存：

```text
config.json, alpha_search.json, feature_scaler.pkl, phenotype_scaler.pkl,
ridge_model.pkl, metrics.json, test_predictions.npz, test_predictions.csv,
summary.json, environment.json, run.log, artifacts.json
```

`metrics.json` 同时记录 normalized/denormalized 下的整体、逐表型、逐时间点、
逐表型逐时间点五项指标。沿用 TFPP 的 `compute_metrics_vectorized()`：
先时间列均值再按表型等权平均；R² 为时间校正全局 R²，保留负值，不截断。
仅 PCC 的未定义值处理按本步骤要求调整：常数/近常数列保留 `NaN`，
汇总只对有限 PCC 取均值并记录有效/总列数及有效聚合数；全无效时仍为 `NaN`。
JSON 使用 Python 可读取的 `NaN`，不会把它替换成0。TFPP 本身的行为没有改变。
预测 CSV 以 manifest 的 test sample_id 为第一排序层，其后为 phenotype/time 原序，
含 sample_id、split/fold 指纹、表型、时间点、原始量纲真实值和预测值。

summary/environment 记录 commit/dirty、seed、split/fold、checkpoint/export 哈希、
特征维度、训练/选择时间、预测时间和包含导入及先前工作的 Linux 进程峰值 RSS。
所有产物带文件哈希；`load_ridge_probe()` 加载本项目产生的可信本地 pickle，
`predict_probe()` 只变换和预测。运行结束会加载保存模型，校验 test 预测复现。

CPU 合成冒烟测试生成第7步的最小 checkpoint/export，运行上述 CLI，临时产物自动清理：

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_convae_ridge.py -v
.venv/bin/python -B -m unittest discover -s tests -v
git diff --check
```

## 配置驱动实验与结果追溯（第10步，spec/result schema 1）

`run_experiments.py` 负责计划、校验、状态、运行和严格汇总；不实现第9步的连续 TFPP。
`experiments/example.spec.json` 是需要填写真实文件指纹和时间列的示例，零哈希不是有效数据。
`experiments/spec.py` 定义严格 Pydantic spec，`results.py` 定义状态/结果 schema。
每条 `experiments` 记录是明确的一个实验协议；只按给定外层 folds 展开，不自动做全轴组合。

核心 spec 结构如下（路径相对于 spec 文件；所有 SHA256 必须填入真实值）：

```json
{
  "schema_version": 1,
  "data": {
    "snp": {"path": "snp.npy", "sha256": "<文件SHA256>"},
    "phenotypes": {"RGBVI": {"path": "RGBVI.csv", "sha256": "<文件SHA256>"}},
    "n_samples": 400, "n_snps": 50000,
    "sample_id_source": "row_index", "sample_ids_sha256": "<manifest中的ID顺序哈希>",
    "time_labels": ["10", "12"], "time_coordinates": [10, 12]
  },
  "splits": {"mode": "regular", "manifests": [{"path": "split.npz", "sha256": "<文件SHA256>"}]},
  "experiments": [{
    "route": "GWAS->DC-SIS", "traits": ["RGBVI"], "seed": 73,
    "train_size": {"fraction": 0.5}, "gwas_k": 10000, "k": 1000,
    "screening_mode": "joint", "ablation": "full",
    "tfpp": {"epochs": 200, "screen_missing_strategy": "mode", "device": "cpu"}
  }],
  "output_root": "outputs/experiments"
}
```

时间列必须覆盖实际 CSV 的完整列和顺序；示例仅展示两个时间点，不隐式截取原始 CSV。
`file_digest(path)` 提供分块内容 SHA256。大数据的声明指纹在实际运行前核实。
普通模式只能有一个普通 manifest；K折须提供完整 fold 1..K，且每个样本恰好 test 一次。
所有路线读取同一个外层 split/fold 集合，seed 只控制训练、随机位点和嵌套子集，不重新分外层集合。
当前数据没有生物学 ID，明确使用行索引及顺序哈希，不能据此鉴别未记录的生物学行重排。

已支持路线（名称区分大小写）：

| route | 输入与筛选 |
| --- | --- |
| `fixed_indices` | 必须给 `fixed_indices: {path, sha256}`，校验非空/整数/范围/唯一性，保留文件原列序 |
| `random_k` | 只由训练基因型列数、k、seed 生成并保存有序索引，不接触表型 |
| `GWAS` | 训练行拟合 GWAS；k 使用 `gwas_k` |
| `DC-SIS` | 训练行拟合 DC-SIS；k 使用 `k` |
| `GWAS->DC-SIS` | 显式在 GWAS 候选列内做 DC-SIS，包含小 SNP 矩阵，不使用旧并行并集启发式 |
| `ConvAE->Ridge` | 按 manifest 顺序提供 `exports: [{path: ".../metadata.json", sha256: "..."}]`，复用第7/8步 |

监督筛选可选 `screening_mode=joint` 或 `per_trait_union`，method id 分开。
保存各阶段原始列索引、逐性状入选及独占数量、去重前后数量、最终有序索引哈希；
联合关联没有独立性状归因，不伪造贡献数。实际维度从保存的 `screened_indices.npy` 读取。
筛选缺失策略由 `tfpp.screen_missing_strategy` 控制，所有 fit 只用本子集训练行。
runner 禁用内部筛选缓存，TFPPTrainer 通过 `preselected_snp_file` 复用外部已保存索引。

TFPP 消融明确映射 `full→all`、`wide_only→wide`、`deep_only→deep`、
`no_gate_add→add`；最后一项是已有的直接相加路径，本步骤有真实训练/反向测试。
未知消融明确拒绝，不回退 full；任何 `continuous_feature` 路线当前明确失败。
ConvAE->Ridge 不接受 TFPP 消融或配置，模型仍为第8步固定 Ridge。

`train_size` 恰好提供 `count` 或 `fraction`（向下取整，至少2个训练个体）。
只在原 train 中按 `SHA256(seed,sample_id)` 稳定排序取前缀，随后恢复原 manifest 行序；
较小集合严格包含于较大集合，val/test 完全不变。保存原 train、排除行、子集规则及哈希。
`IndividualSplit` 新增可选 excluded_train_idx/parent_train_idx 约定：
train+val+test+明确排除的原训练行必须互斥并完整覆盖个体；普通旧 manifest 行为不变。
每个子集重新拟合筛选、填补、表型 scaler 和新模型。
ConvAE export 必须由同一子集/fold 的编码器产生，不能把全 train 编码器用于较小训练集对比。

命令：

```bash
# 只读计划：校验小型 split/JSON/固定索引及依赖，不载入 SNP/表型/潜在特征大数组，不写结果
.venv/bin/python -B run_experiments.py dry-run experiments/example.spec.json

# 显式正式运行；须先完善 spec，并提交代码使工作区 clean
.venv/bin/python -B run_experiments.py run <真实spec.json>

# 仅相同配置、代码版本/dirty tree、数据指纹和 split 的安全恢复
.venv/bin/python -B run_experiments.py run <真实spec.json> --resume

# 本步骤允许的实际执行：40个体、13 SNP、单轮CPU，六条路线全部走通
.venv/bin/python -B run_experiments.py smoke --directory /tmp/dynags_experiment_smoke

# smoke 生成的可用 spec 也可执行 dry-run；目录已存在不会写伪结果
.venv/bin/python -B run_experiments.py dry-run /tmp/dynags_experiment_smoke/spec.json --development-smoke
.venv/bin/python -B run_experiments.py summarize <run_hash目录> <另一个run_hash目录>
```

正式运行默认拒绝 dirty 工作区；`--development-smoke` 仅用于开发，显式记录到配置和汇总轴。
运行 hash 覆盖 resolved 数据/模型/损失/早停/指标口径、所有轴、split、commit/dirty tree 指纹。
每次执行重验数据内容及实际 shape。默认禁止覆盖相同 hash。
resume 的成功实验只校验并返回，不重训/重测；test 前失败允许新 attempt 从头重启，保留旧错误日志。
不支持中途优化器状态续训。test 预约持久化后，无论成功与否均禁止失败重试中重测。
数据、split、代码或配置变化必须作为新实验；`--resume` 不会自动寻找相似旧目录。

产物示例：

```text
outputs/experiments/<canonical_config_sha256>/
├── resolved_config.json
├── status.json                  # running / failed / success，attempt、test_evaluation_count
├── manifest.json                # ResultManifest schema 1，实验轴、样本身份、维度、参数量、耗时、资源峰值
├── artifacts.json               # 配置、manifest、split、模型、scaler、history、特征/索引、预测、指标哈希
└── attempt_001/
    ├── split.npz                # 原始行索引，子集保留 excluded train 元数据
    ├── selection.json           # TFPP：各阶段/性状贡献、训练指纹及筛选填补
    ├── screened_indices.npy
    ├── backend_config.json
    ├── backend/                 # 复用原 TFPPTrainer 的 schema 3 预处理、checkpoint、scaler、训练历史与日志
    ├── predictions.npz          # manifest 顺序的 sample_ids、预测、标签
    ├── predictions.csv          # sample_id、fold、split、trait、time、真实值、预测
    ├── metrics.json             # 归一化/原尺度，各四层指标；保留负R²及未定义PCC
    └── environment.json
```

Ridge attempt 另存 `ridge/` 的完整第8步产物、export metadata 副本和特征数组哈希引用。
参数量及实际 D 来自已运行模型/已保存产物；耗时区分筛选、训练、验证和推理。
ConvAE->Ridge 的训练/验证时间指 Ridge，不冒充重新训练编码器的时间。
CPU RSS 是注明范围的 Linux 进程高水位；GPU 记录峰值 allocated 显存；CPU smoke 为0。
失败保留 resolved config、status、attempt/error.log，不以空值或旧结果冒充成功。

汇总只接受版本/schema、success、配置/commit、所有产物哈希及预测身份校验通过的运行。
保留 mode、split family、seed、train size、trait set、route/method id、ablation、fold 和完整协议 hash。
不同 k、模型/优化配置、数据、代码或开发/正式状态不会混合；重复 fold 不重复计权。
fold 未齐全时标记 incomplete，不计算均值/std；齐全后输出每折明细及各层指标均值/std（ddof=0）、有效数。
普通 split 与K折始终分组。CLI run 在 `summaries/<计划hash>.json` 保存汇总，不覆盖变化内容。
smoke 将真实合成结果保存在指定目录和 `smoke_summary.json`，不会进入正式数据目录。

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_experiments.py -v
.venv/bin/python -B -m unittest discover -s tests -v
git diff --check
```
