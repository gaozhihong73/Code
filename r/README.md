# 动态表型模拟数据生成

本目录保存动态遗传模拟器。正式实验通过Python调度器调用R脚本，生成的数据统一放在 `dataset/experiments/`。

## 环境

使用现有Conda环境 `dynags-r`：

```bash
conda run -n dynags-r Rscript -e \
  "cat(R.version.string, '\n'); for (p in c('data.table','jsonlite','mvtnorm')) cat(p, as.character(packageVersion(p)), '\n')"
```

当前生成流程需要：

- R；
- `data.table`；
- `jsonlite`；
- `mvtnorm`；
- 项目的Python虚拟环境 `.venv`。

## 脚本职责

| 脚本 | 作用 |
| --- | --- |
| `r/dyn_curve.R` | Logistic等曲线和Legendre基函数 |
| `r/dyn_sim.R` | 生成基因型、动态遗传值、潜在曲线和观测表型 |
| `r/generate_simulation.R` | R命令行入口，导出CSV、QTL真值和模拟配置 |
| `prepare_simulation.py` | 校验、生成split、可选CSV转NPY、发布manifest和溯源文件 |
| `experiments/simulation/generate_dataset.py` | 按正式实验条件调度以上步骤 |
| `experiments/simulation/generate_all_datasets.py` | 生成、复验完整300份正式数据并写总清单 |

生成器会把本次实际使用的脚本保存到共享版本包
`dataset/experiments/provenance/generator_<hash>/`。相同脚本版本只保存一次，
每份数据的 `data_manifest.json` 记录版本包路径及脚本哈希。

## 样本量定义

条件名中的 `NTR` 是实际训练样本数，不是生成总数。调度器按现有8:1:1划分及取整规则反推完整数据行数：

| NTR | 生成总数 | Train | Validation | Test |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 124 | 100 | 12 | 12 |
| 200 | 250 | 200 | 25 | 25 |
| 400 | 500 | 400 | 50 | 50 |
| 800 | 1,000 | 800 | 100 | 100 |
| 1,600 | 2,000 | 1,600 | 200 | 200 |

正式因素为：

```text
P   = 5000, 10000
NTR = 100, 200, 400, 800, 1600
h²  = 0.1, 0.4, 0.8
QTL = 20
rep = 1..10
```

时间坐标固定为1–20，`n_epistasis=0`，20个QTL均限制在前5,000个位点中。

## 生成一个数据集

推荐保存为NPY，以加快重复训练：

```bash
conda run -n dynags-r .venv/bin/python -B \
  experiments/simulation/generate_dataset.py \
  --output_root dataset/experiments \
  --n_train 400 \
  --n_snps 5000 \
  --heritability 0.4 \
  --rep 1 \
  --snp_format npy
```

如需直接保留并训练CSV：

```bash
conda run -n dynags-r .venv/bin/python -B \
  experiments/simulation/generate_dataset.py \
  --output_root dataset/experiments \
  --n_train 400 \
  --n_snps 5000 \
  --heritability 0.4 \
  --rep 1 \
  --snp_format csv
```

输出目录示例：

```text
dataset/experiments/
├── provenance/generator_<hash>/
│   ├── bundle_manifest.json
│   └── scripts/
└── P5000_NTR400_H0.4/rep_01/
    ├── sim_snp.npy
    ├── sim_pheno.csv
    ├── sample_ids.csv
    ├── simulation_config.json
    ├── generation_request.json
    ├── data_manifest.json
    ├── splits/split_seed73.npz
    └── truth/
        ├── causal_effects.csv
        ├── true_curve.csv
        └── true_gv.csv
```

生成器拒绝覆盖已有目录。若命令失败，应检查并修复原因，不要手工拼接半成品目录。

## 验证生成数据

```bash
.venv/bin/python -B validate_simulation.py \
  --data_dir dataset/experiments/P5000_NTR400_H0.4/rep_01
```

验证必须通过以下检查：

- 基因型为0/1/2剂量；
- SNP和表型行数一致；
- SampleID顺序一致；
- train/validation/test互斥且完整；
- scaler只在训练个体上拟合；
- QTL真值不作为模型输入。

## 实验一、二批量入口

两个独立脚本都直接读取正式总清单，默认各自处理全部300套数据：

```bash
# 只核对任务，不训练
.venv/bin/python -B experiments/simulation/train_experiment1_oracle.py --plan-only
.venv/bin/python -B experiments/simulation/train_experiment2_screened.py --plan-only

# 单卡整批运行；两个命令可同时使用cuda:0
.venv/bin/python -B experiments/simulation/train_experiment1_oracle.py \
  --workers 1 --devices cuda:0
.venv/bin/python -B experiments/simulation/train_experiment2_screened.py \
  --workers 1 --devices cuda:0
```

实验一输出到`outputs/simulation/experiment1_oracle/`，强制使用20个真实QTL并关闭
内部筛选；实验二输出到`outputs/simulation/experiment2_screened/`，筛选配置来自
`config/profiles/simulation.json`且只拟合训练个体。二者输出和批量状态互不覆盖。

默认精简产物并断点续跑。限定单项可使用：

```bash
.venv/bin/python -B experiments/simulation/train_experiment2_screened.py \
  --condition P10000_NTR800_H0.4 --rep 3
```

内部同卡并发可显式使用`--workers 2 --devices cuda:0`；显存不足时保持默认
`--workers 1`。完整的并行、归档和产物保留规则见`experiments/simulation/README.md`。
