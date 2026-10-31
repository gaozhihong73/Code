# DynaGS paper experiments

实验代码按论文实验边界组织：

```text
experiments/
├── experiment_plan.md            # 冻结实验协议
├── e1_e2/                        # 已完成的 Oracle / Screened 模拟实验
│   └── simulation/
│       ├── data/                 # 数据生成与清单校验
│       └── runners/              # E1/E2 批量训练入口
└── e3/                           # 固定参考条件的端到端模型比较
    ├── configs/ schemas/ baselines/ metrics/
    ├── runners/ analysis/ tests/
    └── README.md
```

E1/E2 已完成；不要修改或重跑其既有输出。E3 的执行、安全门禁和复现命令见
[`e3/README.md`](e3/README.md)。
