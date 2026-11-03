# 实验配置文件

`simulation.json` 管理正式模拟数据训练的全部超参数；`simulation_debug.json` 仅用于小数据链路检查；`real_data.json` 管理真实数据训练的全部数据路径、表型定义和超参数。

修改实验设置时只编辑对应 JSON，不要在训练入口脚本中修改学习率、模型结构、筛选数量或训练轮数。

正式筛选固定为“训练集质量控制 → PCA校正的加性+显性联合GWAS → 候选内
DC-SIS → final-K”。质量控制固定为 MAC≥5、训练集缺失率≤20%；`gwas_k`
指定候选池大小，`screen_final_k` 指定最终 SNP 数。GWAS 的 AUC 和斜率使用
profile 中的 `time_coordinates`。

常用命令：

```bash
# 真实数据
.venv/bin/python -B train.py --config config/profiles/real_data.json

# 模拟数据
.venv/bin/python -B experiments/e1_e2/simulation/runners/train_experiment2_screened.py \
  --config config/profiles/simulation.json \
  --data_dir dataset/experiments/P5000_NTR400_H0.4/rep_01
```

`--exp_name` 和 `--output_root` 是单次运行的输出覆盖项；模拟数据的 SNP、表型、样本 ID、时间坐标和固定划分由数据目录的 manifest 自动读取。

## maize_gdb2026

`maize_gdb2026_hn.json` 和 `maize_gdb2026_ln.json` 分别管理高氮、低氮训练。
profile 默认预测株高 `PH`，三个有序时期 `V9/VT/R1` 映射为时间坐标
`0/1/2`。未传 `--traits` 时使用 profile 的 `phenotype_names`；命令行显式传入
`--traits` 或 `--all-traits` 时覆盖 profile：

```bash
# 只检查输入和最终配置，不启动训练
bash run_maize_gdb2026.sh --condition HN --traits PH --plan-only

# 只准备模型输入（正式训练时也会自动执行或复用）
bash run_maize_gdb2026.sh --prepare-only

# 单性状正式训练（推荐先从此设置开始）
bash run_maize_gdb2026.sh --condition HN --traits PH

# 多性状联合训练
bash run_maize_gdb2026.sh --condition LN --traits PH PW PSD

# selected_traits.csv 中全部 10 个性状
bash run_maize_gdb2026.sh --condition LN --all-traits
```

输出默认写入 `outputs/tfpp/maize_gdb2026/<HN|LN>/`。筛选、归一化和缺失
处理仍由现有 `TFPPTrainer` 完成，并且只在训练个体上拟合。需要调整训练轮数、
筛选数量或模型结构时，请复制或编辑对应 profile，再用 `--config` 指定。

原始 `genotype.npy` 有全样本均值填补形成的小数剂量，现有离散剂量模型不能直接
读取。专用入口会保留原文件，并原子生成 `genotype_tfpp_missing.npy`：把可识别的
非整数填充值恢复为 `-1`，使后续众数填补只在训练个体上拟合；转换计数和文件
SHA-256 记录于同目录 JSON。若旧填充值恰好为整数 `0/1/2`，仅凭当前文件无法与
真实剂量区分；要完全消除此历史处理边界，需要从填补前的 PGEN/VCF 重新导出。
