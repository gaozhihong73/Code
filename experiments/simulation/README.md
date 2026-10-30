# 模拟数据生成

正式条件中的 `NTR` 表示实际训练样本数。生成器按现有8:1:1取整规则反推完整数据行数：

| NTR | 总样本 | Train | Validation | Test |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 124 | 100 | 12 | 12 |
| 200 | 250 | 200 | 25 | 25 |
| 400 | 500 | 400 | 50 | 50 |
| 800 | 1,000 | 800 | 100 | 100 |
| 1,600 | 2,000 | 1,600 | 200 | 200 |

## 依赖

- Python 3.10及项目环境；
- `Rscript`；
- R包：`data.table`、`jsonlite`、`mvtnorm`。

## 单数据集生成

```bash
conda run -n dynags-r .venv/bin/python -B experiments/simulation/generate_dataset.py \
  --output_root dataset/experiments \
  --n_train 400 \
  --n_snps 5000 \
  --heritability 0.4 \
  --rep 1 \
  --snp_format npy
```

生成器拒绝覆盖已有目录。每个数据目录保存生成请求、模拟配置、manifest和split；本次实际使用的R/Python脚本按内容哈希保存在共享的 `dataset/experiments/provenance/` 中，manifest记录对应版本。

`--snp_format csv` 可保留可直接训练的 `sim_snp.csv`；正式重复训练推荐默认的 `sim_snp.npy`。

## 完整网格生成

正式数据统一使用批量入口生成：

```bash
conda run -n dynags-r .venv/bin/python -B \
  experiments/simulation/generate_all_datasets.py \
  --output_root dataset/experiments
```

脚本按顺序生成300份数据，每份生成后立即校验文件哈希、参数、20个QTL、
固定split、实现遗传力和TFPP加载流程，并持续更新
`dataset_manifest.json` 与 `dataset_manifest.csv`。任务中断后可原命令续跑；
已有目录必须完整通过复验才会跳过，不会覆盖或默默接受半成品。

## 正式主实验范围

正式主实验固定使用无上位性互作的加性、显性遗传机制，即 `n_epistasis=0`。参数网格为
`P={5,000, 10,000}`、`NTR={100, 200, 400, 800, 1,600}`、
`h²={0.1, 0.4, 0.8}`，每个条件 10 个独立重复。每一份数据成对运行 Oracle 20-QTL 与未知 QTL 筛选实验，并使用同一份固定个体划分。

该设计用于评估样本量、特征维度和遗传力对预测与筛选的影响。Ridge-A+D 是该机制下的强基线；不预设 TFPP 必然优于所有基线模型。

## 批量训练入口

实验一和实验二是两个完全独立的入口，均直接读取
`dataset/experiments/dataset_manifest.json`，默认覆盖全部300套数据，支持断点续跑。
二者不引用仓库根目录的`train_sim.py`或`train_sim_oracle_qtl20.py`，输出目录也互不重叠：

- 实验一：`outputs/simulation/experiment1_oracle/`
- 实验二：`outputs/simulation/experiment2_screened/`

先只查看任务，不启动训练：

```bash
.venv/bin/python -B experiments/simulation/train_experiment1_oracle.py --plan-only
.venv/bin/python -B experiments/simulation/train_experiment2_screened.py --plan-only
```

单GPU分别启动整批任务：

```bash
.venv/bin/python -B experiments/simulation/train_experiment1_oracle.py \
  --workers 1 --devices cuda:0

.venv/bin/python -B experiments/simulation/train_experiment2_screened.py \
  --workers 1 --devices cuda:0
```

两个命令可以同时运行并争用同一块GPU。若显存不足或速度下降，应保留两个进程，
但错开启动时间；脚本不会自动终止另一实验。实验内部也允许同卡并发，例如
`--workers 2 --devices cuda:0`，但这会同时驻留两个模型，只有显存和吞吐测试证明
有收益时才建议使用。多GPU可使用`--workers 2 --devices cuda:0,cuda:1`。

默认启用`--resume`：完整任务跳过；不能原地续训的半成品先移动到
`_batch/incomplete_archives/`后整项重试，不会直接删除。可用`--condition`、
`--rep`和`--limit`限制任务范围。每个实验的`_batch/batch_manifest.json`及CSV会持续
记录进度，并汇总反归一化测试集的MSE、RMSE、MAE、PCC和R²；成功任务不保留重复
的调度日志。

默认`--artifact-policy compact`会在训练成功后删除以下重复产物：

- `test_predictions.csv`（保留信息等价且更紧凑的`test_predictions.npz`）；
- 5份`phenotype_*_history.csv`（模拟实验只有一个表型，信息已包含在综合历史中）；
- 每次成功训练的文本日志（失败任务日志保存在`_batch/failures/`）。

精简后会重新封存checkpoint完整性清单。模型权重、模型配置、scaler、固定split、
筛选索引、预处理元数据、测试指标摘要、综合训练曲线和NPZ测试预测都会保留，足以
复现推理、汇总论文指标并绘制训练曲线及真实值/预测值曲线。若调试时需要全部原始
产物，可显式传入`--artifact-policy full`。

## Oracle 20-QTL诊断

Oracle实验复用同一份表型和固定划分，只把模型输入替换为真值文件中记录的20个QTL：

```bash
.venv/bin/python -B experiments/simulation/train_experiment1_oracle.py \
  --condition P5000_NTR400_H0.4 --rep 1
```

Oracle与未知QTL实验共同读取`config/profiles/simulation.json`。Oracle入口强制关闭
GWAS/DC-SIS筛选；实验二按profile启用筛选。Oracle是可学习性上界诊断；只有当全部
比较模型均输入同一组真实20个QTL时，Oracle模型比较才公平。不能将Oracle结果与
未知QTL筛选结果混合作为方法排名。

Oracle 输入下的传统基线模型使用同一份数据、固定划分和 20 个真值 QTL：

```bash
.venv/bin/python -B train_baseline.py \
  --data_dir dataset/experiments/P5000_NTR400_H0.4/rep_01
```

该脚本比较 Ridge、GBLUP、RF 和 XGBoost；输出保存在 `outputs/baseline_comparison/oracle_qtl20/`。基线模型不共享 TFPP 的优化器超参数，但严格共享样本划分、表型归一化流程、测试集和 Oracle SNP 输入。

## 改进筛选协议

未知QTL条件通过专用 runner 启用固定的条件残差筛选：

```bash
.venv/bin/python -B experiments/simulation/train_experiment2_screened.py \
  --condition P5000_NTR400_H0.4 --rep 1
```

该 runner 自动读取数据 manifest、固定划分和时间坐标。筛选参数统一由 `config/profiles/simulation.json` 管理：第一轮多元 GWAS 与 DC-SIS，选取最强15个位点回归其加性和显性效应，再对残差进行第二轮相同筛选；合并候选后以固定多任务 Elastic Net 保留20个 SNP。筛选只使用训练集；开发数据不得作为独立正式重复。

## 探索性互作敏感性分析

带互作的数据仅用于补充敏感性分析，不替代上述正式主实验，也不得因某个模型表现较高而被选为主实验。生成器支持显式设置互作数量与强度，例如：

```bash
PATH=/root/miniconda3/envs/dynags-r/bin:$PATH .venv/bin/python -B \
  experiments/simulation/generate_dataset.py \
  --output_root dataset/experiments \
  --n_train 400 --n_snps 5000 --heritability 0.4 --rep 1 --seed 73 \
  --n_epistasis 10 --epistasis_strength 0.5
```

当前 `EPI10_S0.5` 数据只证明模拟器能够生成互作，互作对总遗传方差的比例尚未受控，不能据此声称某种模型因互作而更优。正式互作分析需要固定总遗传力，并预先设定互作方差贡献比例（如0%、10%、30%、50%），再在每个水平运行独立重复和所有比较模型。
