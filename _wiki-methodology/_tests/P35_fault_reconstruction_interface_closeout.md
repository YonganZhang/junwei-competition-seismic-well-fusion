# P35 断层与重建接口收口验收

## 验收范围

- 断层 ST10010 连续三维开发体、CIG-Bench lift/tolerance v2 和原子 Pipeline 入口；
- 重建 P29 v2 显式 feature-cache/查询侧模态接口及真实五折重算；
- 重建 P30 v2 地质统计复验、P21 门禁和井震跨模态 I/O 合同；
- 六赛道 manifest、adapter、共享 lifecycle 与统一运行时回归。

## 核心事实

| 项目 | 结果 |
|---|---|
| 断层冻结测试访问 | `false` |
| CIG precision/AP lift | `1.179665x / 1.423894x` |
| CIG 半径 2 容差 F1 | `0.020237746385` |
| 断层结论 | `do_not_advance`，仅高召回诊断候选 |
| P29 v2 provider | 10/10 `success` |
| P29 v2 A0 RMSE | `0.027734374378` |
| P29 v2 晋级 | 2/5 折改善，保留 P21 |
| P30 v2 行数/折数 | `10240 / 5` |
| 普通克里金 RMSE | `0.030569516403` |
| 回归克里金 RMSE | `0.030093884156` |
| P30 v2 决策 | `FEASIBLE_NO_PROMOTION` |
| 跨模态合同 | `fusion_io_contract.json` 已生成并纳入 manifest |

## 复验命令

```bash
python3 -m unittest discover -s _pipelines/02_task_datasets/fault/tests -p 'test_fault_p30_finalize.py' -v
python3 -m unittest discover -s _pipelines/02_task_datasets/reconstruction/_tests -p 'test_p29_agent_action_effect_repair.py' -v
python3 -m unittest discover -s _pipelines/02_task_datasets/reconstruction/_tests -p 'test_p30_bounded_geostatistics_feasibility.py' -v
python3 -m unittest discover -s _code/six_track_pipeline/tests -v
python3 -m unittest discover -s _pipelines/02_task_datasets/tests -p 'test_track_lifecycle.py' -v
python3 _code/six_track_pipeline/cli.py verify --track all --through verify
```

## 实际复验结果

- 断层 P30 回归：`16/16` 通过；便携系数重放与 joblib 路径分别验收，避免把不同浮点执行路径误作同一回放。
- 重建 P29/P30：`9/9 + 7/7` 通过；P30 v2 的 10 项 artifact manifest 全部复算一致。
- 六赛道共享层：runtime `14/14`、lifecycle `3/3` 通过，统一 `verify --track all` 完成 6 条 Pipeline、42 个阶段并返回 `PASS`。
- 断层 ST10010 资产 manifest、最终 CIG-Bench manifest、重建 P29 v2 manifest 均逐项复算哈希，无缺失或漂移。

冻结测试集未读取，旧 P29 v1 未重新进入晋级证据。
