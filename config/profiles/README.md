# 实验配置文件

`simulation.json` 管理正式模拟数据训练的全部超参数；`simulation_debug.json` 仅用于小数据链路检查；`real_data.json` 管理真实数据训练的全部数据路径、表型定义和超参数。

修改实验设置时只编辑对应 JSON，不要在训练入口脚本中修改学习率、模型结构、筛选数量或训练轮数。

常用命令：

```bash
# 真实数据
.venv/bin/python -B train.py --config config/profiles/real_data.json

# 模拟数据
.venv/bin/python -B experiments/simulation/train_experiment2_screened.py \
  --config config/profiles/simulation.json \
  --data_dir dataset/experiments/P5000_NTR400_H0.4/rep_01
```

`--exp_name` 和 `--output_root` 是单次运行的输出覆盖项；模拟数据的 SNP、表型、样本 ID、时间坐标和固定划分由数据目录的 manifest 自动读取。
