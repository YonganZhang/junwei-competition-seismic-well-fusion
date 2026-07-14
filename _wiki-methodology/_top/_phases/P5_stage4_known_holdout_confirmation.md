# P5 Stage-4 全开发集重训与已见留出集确认

> 冻结日期：2026-07-14  
> 基线：`p5-model-benchmark-integration@3df7620`  
> 根随机种子：`2693`  
> 状态：已完成并集成验收
> 验收证据：`../../_tests/P5_stage4_acceptance_evidence.md`

## 1. 目的与证据等级

Stage-4只使用Stage-3已经冻结的唯一胜者、预处理、target transform、训练预算和指标口径：先在全部合法
development数据上refit，再对项目历史已经看过的空间/井族留出集做一次P5胜者确认评估。由于同一holdout
已被P4或更早基线消费，本阶段结果统一标记为：

- `evidence_class=previously_seen_reusable_holdout`；
- `prior_test_consumed=true`；
- `fresh_blind=false`。

这些结果可以比较P5胜者与历史基线，但不能称为首次盲测、独立外部验证或竞赛私榜成绩。真正的fresh-blind
证据仍需新的未触碰外部留出集或赛事方隐藏测试集。

## 2. 冻结执行矩阵

| 赛道/任务 | Stage-3冻结胜者 | Stage-4动作 |
|---|---|---|
| ①断层 | 无 | 继续`blocked/not_rankable`；不refit、不读holdout |
| ②F3 | `smp_fpn_r18` | 全development refit + 已见空间holdout确认 |
| ②Penobscot | `smp_deeplabv3plus_r18` | 全development refit + 已见空间holdout确认 |
| ③PHIF | `extra_trees_regressor` | 全development refit + 已见F-15确认 |
| ③KLOGH | `extra_trees_regressor` | 全development refit + 已见F-15确认 |
| ③SW | `xgboost_regressor` | 全development refit + 已见F-15确认 |
| ④GM09 P | `xgboost_multisoftprob_window` | 全development refit + 已见F-5确认 |
| ⑤T1 | `lightgbm` | 全development refit + 已见holdout确认 |
| ⑤T2 | `catboost` | 全development refit + 已见holdout确认 |
| ⑤T3 | `xgboost` | 全development refit + 已见holdout确认 |
| ⑤T4 | `catboost` | 全development refit + 已见holdout确认 |
| ⑤T5 | 无 | `not_feasible`；缺批准标签/模拟状态合同 |
| ⑤T6/T7 | 无 | `blocked`；缺development-only特征源，不借用test回填 |
| ⑥strict | `pykrige_ok3d` | 全development refit + 已见空间holdout确认 |
| ⑥conditional | `pykrige_ok3d` | 全development refit + 已见条件式holdout确认；不宣称strict泛化 |

## 3. 不可变红线

1. 不重新排名、不调参、不因确认集指标改变模型、loss、特征、阈值、训练步数或seed。
2. 每个赛道先校验Stage-3榜单、split和source hash，再允许refit；refit结束并落盘证据后才允许读取holdout。
3. 新结果写入独立`p5_stage4_confirmation`目录；不得重置、覆盖或伪造P4的`TEST_CONSUMED`状态。
4. 预测、指标和图件必须绑定到冻结配置、refit证据、split与源文件hash；大数组可留在赛道私有忽略目录，
   但便携manifest必须记录路径、hash、shape和存储边界。
5. 断层、T5、T6、T7只输出结构化阻塞证据；不得用随机负例、代理test特征或空标签制造分数。
6. 失败或超时保留原配置和原始状态；不得通过缩数据、换模型或增加预算把失败改写成通过。

## 4. 赛道专属验收

- 地震相：Accuracy、mIoU、Macro-F1、逐类指标、混淆、可靠性和代表剖面。
- 物性：PHIF/KLOGH/SW物理域误差、R²、相关/残差及适用的不确定性诊断。
- 岩相：固定九类Macro-F1为主，supported-class指标仅诊断；无真实`center_md_m`时深度轨继续`not_feasible`。
- 甜点：T1/T3保持回归口径，T2/T4保持AP/Brier/F1及原有任务诊断；代理目标不得升级为统一甜点真值。
- 重建：strict/conditional分开报告RMSE、MAE、R²、频谱等；CDF仅为诊断；conditional保留测试区域约束注记。
- 每轨必须有自动测试、真实执行命令、紧凑结果、可视化manifest、artifact hash和干净提交。

## 5. 集成与后续

六个赛道在各自Stage-3隔离工作树内实现并执行，负责人逐一复核后按①至⑥固定顺序选择性集成到本分支。
集成后统一运行共享回归、赛道Stage-3/Stage-4测试、hash审计和TOP doctor。Stage-4结果只能用于当前已见
holdout上的确认；若需要新的模型选择循环，必须回到development CV并登记为下一阶段，不能回读本阶段指标。

2026-07-14完成状态：六轨提交已按固定顺序集成到`p5-model-benchmark-integration@5af968c`；五个可运行
赛道均完成真实refit与已见holdout确认，断层继续`blocked/not_rankable`。集成后回归为`178 passed`、
`6 subtests passed`、`1 skipped`；完整指标、图件目检、facies单次访问恢复记录和fresh-blind边界见验收证据。
