# AISHPerf 评测工作区（OmniKernel 对接）

本目录按 `aishperf_openness`（https://gitee.com/aishperf-caict/aishperf_openness）
开放评测工作区的约定组织，用于将 OmniKernel 生成的算子接入 AISHPerf 3.0
「算子生成智能体评测套件」做标准化验证。

## 目录结构

```
aishperf/
├── suites/          # 评测套件定义：本项目的算子集 + 指标口径
│   └── omnikernel_operator_gen.yaml
├── entities/        # 被测实体：芯片、框架、模型环境
│   ├── ascend-910b.yaml
│   └── nvidia-a100.yaml
├── rules/           # 评测规则：编译/精度/性能判据与阈值
│   └── operator_gen_rule.yaml
├── metrics/         # 指标定义与提取映射（对接 harness JSON 字段）
│   └── metric_map.yaml
└── run/             # 执行入口
    └── run_eval.sh
```

## 与上游工作区的对齐方式

1. 将本目录整体拷贝进 `aishperf_openness` 的 `suites/` 下作为自定义套件；
2. 或在评测机上设置 `AISHPERF_WORKSPACE=$(pwd)/aishperf` 独立运行；
3. 真实阈值与实体字段以官方仓库 `rules/`、`entities/` 的最新 schema 为准，
   本目录提供的是可直接解析的草案，接入前用官方 `validate` 子命令校验一次。

> 注：本目录不含任何实测结果。所有数字需在你自己的评测机跑 `run/run_eval.sh`
> 生成，再归档到 `results/`。
