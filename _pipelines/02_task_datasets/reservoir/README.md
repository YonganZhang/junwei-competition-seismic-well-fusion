# ③ 储层物性预测 — Volve真实多模态baseline

目标固定为 `PHIF`、`log1p(KLOGH)`、`SW`。输入固定为真实 ST0202 地震patch与原始测井序列；不使用合成标签，不把 `KLOGH_NEW`、`LFP_PHIE` 混入目标，也不允许任何目标或派生解释曲线进入输入。

## 泄漏边界

- 井眼先归并到母井家族，再按家族SHA-256顺序确定性划分train/guard/test。
- 划分发生在LAS插值、深度窗口、地震patch提取和统计拟合之前。
- guard只用于val loss；test在best checkpoint选定后才加载。
- 归一化统计只fit训练井；缺失输入值保持0并配套显式mask。
- 使用官方井拾取建立弱MD→TWT/XY映射，再以官方三维Hugin Top/Base TWT面筛选储层段。

## 测试门与复跑顺序

默认测试不依赖任何被Git忽略的HDF5、guard或checkpoint，因此干净checkout可直接运行；真实产物检查是显式integration gate。正式复跑顺序必须是 **build → train → integration/stats**：

```bash
# 干净checkout默认门：7项无产物测试；2项integration会明确skip
python3 -m pytest -q -p no:cacheprovider _pipelines/02_task_datasets/reservoir/tests

# 正式真实数据顺序
python3 _pipelines/02_task_datasets/reservoir/build_dataset.py
python3 _pipelines/02_task_datasets/reservoir/train_baseline.py --epochs 60
python3 -m pytest -q -p no:cacheprovider \
  _pipelines/02_task_datasets/reservoir/tests --run-integration
python3 _code/dataset_io.py stats reservoir/train
python3 _code/dataset_io.py stats reservoir/test
```

若带`--run-integration`但HDF5、guard或训练审计产物缺失，测试会列出缺失文件并明确skip，不会把缺数据伪装成通过。

构建后的完整机器可读证据位于 `_outputs/build_report.json` 和 `_outputs/split_manifest.json`。

## 可切换简单模型

- `tiny_mlp`：已有的24单元单隐层MLP，是下文正式指标的唯一来源。
- `reservoir_linear`：简单线性SGD回归；兼容旧三输出和P4独立单输出。
- `reservoir_ridge`：同接口的L2 ridge SGD回归；兼容旧三输出和P4独立单输出。

两个线性替代模型仅通过默认便携契约测试，未进行正式重训；它们不改变下文`tiny_mlp`的既有指标与科学结论。

## 真实构建结果

默认2 m中心点间隔、9点测井序列、`3×3×9`地震patch，实测得到train 1,135、guard 81、test 344个样本。确定性井族为：

- train：`15/9-19`、`15/9-F-1`、`15/9-F-11`；10个井眼。
- guard：`15/9-F-12`；1个井眼。
- test：`15/9-F-15`；井眼`15/9-F-15 D`。

| 井眼 | split | 三目标均有效行 | Hugin内有效行 | 最终样本 | 无效目标丢弃 | 储层外丢弃 |
|---|---:|---:|---:|---:|---:|---:|
| 15/9-19 A | train | 1,965 | 825 | 59 | 1,043 | 1,140 |
| 15/9-19 BT2 | train | 2,926 | 1,030 | 74 | 236 | 1,896 |
| 15/9-19 SR | train | 6,782 | 193 | 14 | 231 | 6,589 |
| 15/9-F-1 | train | 2,872 | 117 | 7 | 738 | 2,755 |
| 15/9-F-1 A | train | 2,281 | 511 | 26 | 959 | 1,770 |
| 15/9-F-1 B | train | 2,105 | 780 | 40 | 505 | 1,325 |
| 15/9-F-1 C | train | 8,422 | 7,527 | 378 | 467 | 895 |
| 15/9-F-11 A | train | 1,533 | 965 | 50 | 336 | 568 |
| 15/9-F-11 B | train | 13,933 | 8,588 | 430 | 144 | 5,345 |
| 15/9-F-11 T2 | train | 2,017 | 1,138 | 57 | 254 | 879 |
| 15/9-F-12 | guard | 2,647 | 1,123 | 81 | 3 | 1,524 |
| 15/9-F-15 D | test | 11,936 | 6,838 | 344 | 262 | 5,098 |

所有最终窗口至少有真实输入；本次四通道观测mask均为1，因此“无输入”丢弃数为0。mask仍被持久化并由测试覆盖，供其他缺失模式使用。Hugin面最近点最大距离为8.82 m。

## 真实baseline结果

纯NumPy单隐层24单元MLP，60 epochs，best epoch=26，best guard loss=0.210252。测试指标在固定建模空间`PHIF/log1p(KLOGH)/SW`计算：

| 目标 | RMSE | MAE | R² | Pearson |
|---|---:|---:|---:|---:|
| PHIF | 0.020901 | 0.015648 | 0.902977 | 0.959015 |
| log1p(KLOGH) | 1.128879 | 0.893296 | 0.776217 | 0.933053 |
| SW | 0.232633 | 0.185150 | 0.192233 | 0.718125 |

综合训练标准差归一化RMSE为0.458635。完整指标、训练审计、逐深度预测、loss曲线、GT-vs-pred图和真实地震输入图位于 `_outputs/`。

## 已知限制

- MD→TWT/XY是官方离散井拾取间的分段线性弱标定，不是VSP或速度模型标定。
- 只有5个母井家族；本次test只有一个`F-15`家族，指标不是跨区块泛化结论。
- 三维Hugin面用于逐样本储层筛选；水平井多次穿层会造成同井样本高度相关，但不会跨split泄漏。
- 这是管道验证用简单baseline，不代表最终模型或调优结论。

## P4与甜点目标6/7插件

`p4_pipeline.py`是在上述真实数据层之上的独立P4入口，不改旧三目标baseline指标。它提供：

- 目标6主版本`PHIF`和目标7`KLOGH`各自独立的TaskSpec、label availability、母井split manifest、4折LOGO OOF、fold-train统计、完整checkpoint、冻结test指标与四类专属图。
- requested5会如实降级为effective4，因为冻结`15/9-F-15`后只有4个development母井家族。
- KLOGH仅在`log1p(KLOGH)`域训练，评估时通过`expm1`回到mD；同时保留log域诊断。
- PHIE是目标6下完全独立的label version。当前实扫精确`PHIE`为0；只出现一个井族的`LFP_PHIE`，不会作为别名或混入PHIF，因此状态为`not_feasible`。
- HPO目标固定为development OOF物理单位MAE、方向`minimize`；本轮只跑简单ridge基线，没有运行Optuna或长HPO。

实现与测试仍在reservoir赛道目录；两个面向甜点目标的规范入口位于`_pipelines/02_task_datasets/sweetspot/targets/porosity/`与`_pipelines/02_task_datasets/sweetspot/targets/permeability/`。真实复跑命令见各目标README和`P4_SOP.md`。
