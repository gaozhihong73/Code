# DynaGS paper experiments

E1/E2 已完成，入口位于 `experiments/e1_e2/simulation/`；不要从 E3 runner
修改或重跑这些产物。E3 冻结协议见 `experiment_plan.md`，机器可读配置为
`experiments/e3/configs/e3.json`；正式训练使用冻结的
`experiments/e3/configs/e3_formal_v1.json`。

## E3 safety and status

调度器默认只列计划，不执行训练：

```bash
.venv/bin/python -B -m experiments.e3.runners.run_e3 \
  --config experiments/e3/configs/e3.json --model Mean --rep 1
```

当前配置标记为 `development_smoke_only`，即使传入正式执行开关也会拒绝
10 个正式重复。非正式数据 smoke 必须显式传 `--execute --smoke --data-dir`
并且一次只运行一个模型。待搜索空间、资源预算和代码审查全部冻结后，才可把
配置状态改为 `formal_frozen` 并显式使用 `--allow-formal`。

当前12个模型均已有可运行接口。RR-GBLUP 使用基因组核与 Legendre 时间核；
DNNGP-SNP、DLGWAS、WheatGP 和 Cropformer 均按20个时间点独立训练并拼接
预测。DynaGS 接口只读校验并导入既有 E2 Screened 预测，不重新训练。深度
模型来源、许可证及适配变更见 `baselines/THIRD_PARTY.md`。

每个 run 独立保存配置、哈希、完整 split 索引、validation 历史、反归一化
预测与指标、资源和失败状态。汇总入口只读取通过 schema 校验的 manifest：

```bash
.venv/bin/python -B -m experiments.e3.analysis.summarize_e3
```

本目录不包含 E3 绘图代码；后续绘图必须按 `dynags-figure` skill 执行。

## One-command batches

一键 pilot（默认 `rep_01`–`rep_03`、全部12个模型、串行执行）：

```bash
bash experiments/e3/run_e3.sh pilot
```

一键正式阶段（默认10个重复、全部12个模型，必须明确确认）：

```bash
bash experiments/e3/run_e3.sh formal \
  --config experiments/e3/configs/e3_formal_v1.json \
  --confirm-formal
```

批量入口不修改配置文件；`formal` 阶段只接受 `protocol_status=formal_frozen` 的
配置。pilot 写入 `outputs/paper_experiments/pilots/E3/`，formal 写入
`outputs/paper_experiments/runs/E3/`。默认 `--resume` 会跳过成功 run；失败或半成品
run 不覆盖，必须使用新的 `--output-root` 以保留重试轨迹。可先预览计划：

```bash
.venv/bin/python -B -m experiments.e3.runners.run_e3_batch --stage pilot
```
