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

## P5 首批模型 Stage-1

P5 只做 development-only contract smoke，不加载冻结test、不计算正式指标，也不运行
HPO。三个输出固定为独立mask的 `PHIF`、`log1p(KLOGH_mD)`、`SW`；
模型域原始输出会保留，物理视图才执行PHIF/SW边界裁剪和KLOGH的可逆
`expm1`。

首批动态模型入口位于 `_models/property/`：
`catboost_regressor`、`lightgbm_regressor`、`tabm_regressor`、
`xgboost_regressor`、`extra_trees_regressor`、
`hist_gradient_boosting_regressor`、`realmlp_regressor`、
`ft_transformer_regressor`、`tabiclv2_regressor` 和
`monai_densenet3d_regressor`。精确上游revision、许可证、依赖版本和权重gate
在 `_models/property/source_lock.json`。runner从项目根执行：

```bash
python3 _pipelines/02_task_datasets/reservoir/p5_stage1.py --help
python3 _pipelines/02_task_datasets/reservoir/p5_stage1.py prepare \
  --train-h5 <development-only-train.h5> \
  --guard-npz <development-only-guard.npz> \
  --output _pipelines/02_task_datasets/reservoir/_outputs/p5_stage1/development.npz
${TABULAR_PYTHON:-python3} _pipelines/02_task_datasets/reservoir/p5_stage1.py run \
  --development-batch _pipelines/02_task_datasets/reservoir/_outputs/p5_stage1/development.npz \
  --output-dir _pipelines/02_task_datasets/reservoir/_outputs/p5_stage1 \
  --models catboost_regressor,lightgbm_regressor,tabm_regressor,xgboost_regressor,extra_trees_regressor,hist_gradient_boosting_regressor,realmlp_regressor,ft_transformer_regressor,tabiclv2_regressor
${TORCH_PYTHON:-python3} _pipelines/02_task_datasets/reservoir/p5_stage1.py run \
  --development-batch _pipelines/02_task_datasets/reservoir/_outputs/p5_stage1/development.npz \
  --output-dir _pipelines/02_task_datasets/reservoir/_outputs/p5_stage1 \
  --models monai_densenet3d_regressor
```

TabICLv2必须先在source lock中确认checkpoint许可证与SHA-256，并由负责人显式
提供本地权重；否则结构化 `SKIP`，绝不自动下载。

MONAI DenseNet3D使用scratch-only权重。为满足CUDA同seed严格replay，PyTorch 2.12
中没有确定性反向实现的3D pooling会替换为固定depthwise下采样和直接空间均值；
替换计数随checkpoint config记录，replay不通过仍判为失败。

## P5 Stage-2 固定预算 pilot

Stage-2 入口使用赛道唯一模块名 `reservoir_p5_stage2.py`，只重用P4冻结的首个
development fold，不接收或加载冻结test。所有候选固定seed 2693、相同的192个
fold-train样本和81个validation样本；预处理仅fit fold-train。PHIF、KLOGH、SW
始终使用独立mask、独立物理指标和独立最差母井家族证据，KLOGH另保留
`log1p(KLOGH_mD)`域诊断。

先准备一次私有runtime fold，再分别运行CPU与GPU lane：

```bash
python3 _pipelines/02_task_datasets/reservoir/reservoir_p5_stage2.py prepare \
  --train-h5 <development-only-train.h5> \
  --guard-npz <development-only-guard.npz>

${TABULAR_PYTHON:-python3} \
  _pipelines/02_task_datasets/reservoir/reservoir_p5_stage2.py run \
  --models catboost_regressor,lightgbm_regressor,tabm_regressor,xgboost_regressor,extra_trees_regressor,hist_gradient_boosting_regressor,realmlp_regressor,ft_transformer_regressor,tabiclv2_regressor \
  --seed 2693 --device cpu

${TORCH_PYTHON:-python3} \
  _pipelines/02_task_datasets/reservoir/reservoir_p5_stage2.py run \
  --models monai_densenet3d_regressor --seed 2693 --device cuda \
  --gpu-lock "${GPU0_LOCK:?set GPU0_LOCK to the shared gpu0.lock path}"
```

GPU入口对上述唯一lock文件执行阻塞式`flock`；锁等待不计入模型预算墙钟。
便携证据只写入本赛道 `_outputs/p5_stage2/`，runtime fold与checkpoint不纳入Git。
三个target leaderboard均严格分成`tabular_cpu`与`seismic_3d_gpu` lane，禁止跨输入
模态排序；当前tabular lane有8个合法候选可排名，MONAI 3D lane只有1个真实pilot，
因此按合同标记为`not_rankable`。TabICLv2仍因权重许可证未确认而结构化skip。

## P5 Stage-3 多seed LOGO4确认

Stage-3入口为`reservoir_p5_stage3.py`。它冻结复用Stage-2的模型配置、32步更新预算、
输入预处理和P4 development母井LOGO4，不运行HPO，也没有frozen-test参数或loader。
固定repeat seeds为`1867973658/2137841944/3902865753`。PHIF、KLOGH和SW分别保留
独立label mask、目标变换、指标和排行榜；每个fold的预处理只fit另外三个母井家族。
KLOGH继续在`log1p(KLOGH_mD)`域训练，并在物理mD域排名与绘图。

固定矩阵共108个cell：

| 目标 | tabular候选 | LOGO folds | seeds | cells |
|---|---|---:|---:|---:|
| PHIF | Extra Trees、LightGBM、HistGradientBoosting | 4 | 3 | 36 |
| KLOGH | LightGBM、Extra Trees、XGBoost | 4 | 3 | 36 |
| SW | LightGBM、HistGradientBoosting、XGBoost | 4 | 3 | 36 |

从项目根复跑。`prepare`只读取development train/guard产物，并生成被Git忽略的私有
LOGO4 archive；`run`仅用tabular CPU环境消费该archive：

```bash
${PREP_PYTHON:-python3} \
  _pipelines/02_task_datasets/reservoir/reservoir_p5_stage3.py prepare \
  --train-h5 <development-only-train.h5> \
  --guard-npz <development-only-guard.npz>

${TABULAR_PYTHON:-python3} \
  _pipelines/02_task_datasets/reservoir/reservoir_p5_stage3.py run

${TABULAR_PYTHON:-python3} -m pytest -q \
  _pipelines/02_task_datasets/reservoir/tests/test_reservoir_p5_stage3.py
```

本次固定运行108/108个cell完成，合法完成率100%，三个tabular目标均`rankable`。
按跨fold、跨seed物理RMSE均值排序，PHIF最佳为Extra Trees（0.027641 fraction），
KLOGH最佳为Extra Trees（542.902564 mD），SW最佳为XGBoost（0.170427 fraction）。
这些是development OOF结果，不是frozen-test指标。MONAI 3D只有一个候选且不在本轮
108-cell矩阵内，其`seismic_3d_gpu` lane继续明确标为`not_rankable`，不与tabular排序。

便携证据位于`_outputs/p5_stage3/`：逐cell JSONL、预算/seed/split/OOF/可视化
manifest、每目标每lane排行榜，以及每目标的分井深度真值-预测、散点、残差、
worst-family和fold×seed分布图。完整OOF数组、checkpoint和视觉质检contact sheet留在
被忽略的`runtime/`。请求Times New Roman但当前环境未安装，图件manifest如实记录为
Liberation Serif回退；不影响数据或指标。

## P5 Stage-4 已知持有集确认

Stage-4不是新的盲测。`15/9-F-15`已被历史三输出baseline及P4 PHIF/KLOGH
消费，因此全部状态、指标和图件固定标注
`previously_seen_reusable_holdout`、`prior_test_consumed=true`、
`fresh_blind=false`。Stage-4不会覆盖P4或历史输出。

冻结胜者保持Stage-3结论：PHIF与KLOGH使用`extra_trees_regressor`，SW使用
`xgboost_regressor`；seed固定2693，每个target保持32 estimator/update预算。
在任何fit前，runner会验证Stage-3三份leaderboard哈希、P4/Stage-3 split身份、
输入白名单、真实数据哈希和既有F-15暴露证据。预处理只fit四个development母井
家族的全部1,216行；single-use状态推进后才允许模型访问344行F-15。

tabular环境不需要安装h5py。先用已有含h5py的解释器只做私有runtime prepare，
再用冻结tabular环境执行唯一一次confirmation：

```bash
${H5PY_PYTHON:-python3} \
  _pipelines/02_task_datasets/reservoir/reservoir_p5_stage4.py prepare \
  --train-h5 /path/to/read-only/reservoir/train.h5 \
  --test-h5 /path/to/read-only/reservoir/test.h5 \
  --guard-npz /path/to/read-only/guard.npz

${TABULAR_PYTHON:-python3} \
  _pipelines/02_task_datasets/reservoir/reservoir_p5_stage4.py confirm

${TABULAR_PYTHON:-python3} \
  _pipelines/02_task_datasets/reservoir/reservoir_p5_stage4.py audit
```

`confirm`是单次命令；一旦`confirmation_state.json`存在就fail-closed，不允许重复
refit或重复确认。当前真实结果均为物理域：PHIF MAE/RMSE/R² =
0.006067/0.009319/0.980711；KLOGH = 145.640 mD/278.839 mD/0.889133；
SW = 0.064412/0.080550/0.903155。完整target config、refit证据、344行预测、
物理及模型域指标、OOF-q90区间诊断、9张图和artifact哈希位于
`_outputs/p5_stage4_confirmation/`；私有NPZ、checkpoint和临时视觉复核继续忽略。
