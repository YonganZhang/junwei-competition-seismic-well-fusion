# P31 六赛道智能决策 Pipeline 验收证据

## 验收范围

- 集成分支：`p31-agentic-pipeline-integration`
- 六 Pipeline 实现提交：`3c0e105`
- 六 Pipeline 机器印记提交：`7075511`
- 共享入口：`_pipelines/02_task_datasets/track_lifecycle.py`
- 生命周期：`validate → prepare → baseline → optimize → promote → refit → verify`

## 晋级结论

| 赛道 | 保留/默认 | 明确拒绝 |
|---|---|---|
| 断层 | 局部逻辑回归 + A2D 治理，P30 连续三维体 | CIG-Bench guard F1 低于 baseline；A2L 直接执行 |
| 地震相 | 数据集条件化确定性 hybrid，mean mIoU 相对 A0 `+0.030481` | 直接 LLM endpoint |
| 储层物性 | A2D `reservoir_linear` | A2L 不单独保留；不可用的 CIG-Bench PropertyPredictor |
| 岩相 | XGBoost `depth=3, eta=0.1, rounds=60`，Macro-F1 `0.213349` | P29 LLM 策略；MOMENT/大模型因果归因 |
| 甜点 | 冻结 A0 XGBoost | 候选无正改善后的 LLM 策略 |
| 三维重建 | P21 固定三核集成 | P29 LLM 数值策略 |

## 回归证据

| 门禁 | 结果 |
|---|---|
| 断层 P29 + P30 | 12 passed |
| 地震相 P29 | 13 passed |
| 储层物性 P29 | 7 passed |
| 岩相 P29 + default baseline | 16 passed |
| 甜点 P29 | 4 passed |
| 三维重建 P29 | 6 passed |
| 六赛道 lifecycle | 3 passed |
| 合计 | 61 passed |

额外可移植性检查：断层 `audited_v2` 逻辑回归的五维系数、截距、原 joblib SHA-256 和 metrics SHA-256 已归档到 P30 可移植检查点。不依赖 ignored joblib 重算完整 P30 体，fit/guard 的 precision、recall、F1、IoU 和 threshold 与归档 comparison 在 12 位小数内相等。

## Pipeline 信任

`sixone-cli verify-pipeline` 分别为六条 pipeline 生成 `verified_steps_hash`。`sixone-cli doctor . --no-links` 返回 `verdict=ok`，项目共 8 条 pipeline，`stale=0`、`broken=0`，六条新 pipeline 全部 `fresh`。

## 边界

此验收证明六赛道的入口、动作效应、晋级结论和反伪完成门禁已可发现、可复现。它不证明直接 LLM 数值决策已超过确定性优化器。下一研究阶段必须使用匹配预算的 LLM candidate proposal + BO/ASHA，并在②③小范围验证后再扩展。
