# P9 基础模型效果验收证据

日期：2026-07-28

## 结论先行

六条赛道的大模型都已接到真实权重和任务专用接口，但“接通”不等于“替换现有模型”。
本轮在冻结的 development split 上用同指标、强基线和负控制验收：

| 赛道 | 基础模型 | 最关键结果 | 判定 | 默认 |
|---|---|---|---|---|
| 储层属性 | TabICLv2 | PHIF/KLOGH/SW RMSE 分别比强基线低 4.89%/13.57%/25.19%，且胜过 target-shuffle | 有提升，尚缺同架构随机初始化控制 | 关闭 |
| 甜点预测 | Chronos-2 | MAE 172.316；比历史均值低 6.69%，比同日历 ExtraTrees 低 22.34%，打乱历史顺序后变差 | 有提升，尚缺同架构随机初始化控制 | 关闭 |
| F3 岩相分割 | SAM 2.1 | mIoU 0.0820，强基线 0.1313 | 未打赢 | 关闭 |
| Penobscot 岩相分割 | SAM 2.1 | mIoU 0.0768，强基线 0.1320 | 未打赢 | 关闭 |
| 井筒岩性 | MOMENT-1-base | F1 0.0559，高于随机初始化 0.0322，但远低于 XGBoost 0.1949 | 预训练有信号，方案未打赢 | 关闭 |
| 三维重建 | OpenMind MAE | RMSE 0.5415，优于随机初始化 1.0524，但远差于 PyKrige 0.02121 | 预训练有信号，方案未打赢 | 关闭 |
| 断层 3D 分割 | SAM-Med3D | 真实 93.9M 参数权重与 3D forward 已通过；现有 256 个样本实际是 `[1,33,65]` 独立 2D 切片 | 数据门禁阻塞，不能伪造 3D 分数 | 关闭 |

## 公平性合同

预注册文件：`_models/gaia_dagt/foundation_effect_protocol.v1.json`。

- 同一 sample universe、同一 fold、同一 metric。
- 预处理只在 fold-train 拟合。
- 必须比较当前最强合法基线和至少一个负控制。
- 默认替换还必须通过“同架构随机初始化”控制。
- frozen test 与 known holdout 均未打开。

## 分赛道机读证据

- 属性：`_pipelines/02_task_datasets/reservoir/_outputs/p9_tabicl_effect/summary.json`
- 甜点：`_pipelines/02_task_datasets/sweetspot/p8/_outputs/t3_chronos2_calendar_cv/summary.json`
- F3/Penobscot：`_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*/summary.json`
- 岩性：`_pipelines/02_task_datasets/lithofacies/_outputs/p9_moment_effect/summary.json`
- 重建：`_pipelines/02_task_datasets/reconstruction/_outputs/p9_openmind_effect/summary.json`
- 断层：`_pipelines/02_task_datasets/fault/_outputs/p9_sammed3d_gate/summary.json`

## 断层数据门说明

开发 H5 有 128 个 fault-centred 和 128 个 non-fault 样本，也确实包含 6,215 个正类 voxel；
问题不是“没有负样本文件”，而是 128 个 non-fault 样本把未标注区域假设成负类，审计证明的
verified-negative voxel 仍为 0。数据又是 crossline×time 的 2D 切片，无法构成 SAM-Med3D
所需的连续 3D development block，因此不运行虚假的 3D IoU。

## 科学边界

TabICLv2 和 Chronos-2 的 improvement 是 development 证据，不是最终盲测结论；由于预注册
promotion gate 尚未完整，两者继续走 fallback，不默认替换。SAM2、MOMENT 和 OpenMind 已完成
真实预训练权重与随机初始化/强基线比较；未打赢的路线明确记为 `VERIFIED_NO_GAIN`。

## 自动化验收

- P9/route/Chronos 专项：`34 passed, 24 subtests passed`。
- framework、foundation 与 sweetspot 宽回归：`204 passed, 7 skipped, 24 subtests passed`；
  skip 为缺少可选重依赖时的既有 fail-closed 路径。
