# P5 Stage-3 top-3 × 3-seed × 有效fold确认

> 冻结日期：2026-07-14
> 基线：`p5-model-benchmark-integration@b1a2cf4`
> 状态：✅ 已执行并独立验收；证据见 `../../_tests/P5_stage3_acceptance_evidence.md`
> 根随机种子：`2693`
> 重复模型seed：`1867973658`、`2137841944`、`3902865753`

## 1. 目的与红线

Stage-3只确认Stage-2同任务、同lane top-3在全部科学有效development folds和三个稳定派生seed上的
平均、最差fold、稳定性与资源。不得重新选择Stage-2榜外候选，不得因单一模型改变样本预算、预处理、
loss或训练步数，不做frozen-test，不把不同数据集、任务、模态或strict/conditional混榜。

三个重复seed由`derive_seed(2693, "model", "p5-stage3", repeat_id)`冻结；split seed仍由P4 manifest决定。
每个fold的预处理、类别权重、target transform和校准只fit该fold-train。Stage-2分数不得与Stage-3 OOF
拼接，也不得用于缺失fold补数。

## 2. 冻结准入矩阵

| 任务/lane | 有效fold | 冻结top-3 |
|---|---:|---|
| ①断层data gate | 0 | 无；继续blocked，不训练 |
| ②F3 scratch | 5 | `smp_fpn_r18`、`smp_deeplabv3plus_r18`、`hf_segformer_b0` |
| ②Penobscot scratch | 5 | `smp_deeplabv3plus_r18`、`smp_fpn_r18`、`smp_unet_r18` |
| ③PHIF tabular | 4 | `extra_trees_regressor`、`lightgbm_regressor`、`hist_gradient_boosting_regressor` |
| ③KLOGH tabular | 4 | `lightgbm_regressor`、`extra_trees_regressor`、`xgboost_regressor` |
| ③SW tabular | 4 | `lightgbm_regressor`、`hist_gradient_boosting_regressor`、`xgboost_regressor` |
| ④GM09 P lane | 4 | `xgboost_multisoftprob_window`、`catboost_multiclass_window`、`inceptiontime_window` |
| ⑤T1 | 3 | `lightgbm`、`catboost`、`xgboost` |
| ⑤T2 | 3 | `catboost`、`xgboost`、`lightgbm` |
| ⑤T3 | 4 | `lightgbm`、`inceptiontime`、`xgboost` |
| ⑤T4 | 3 | `catboost`、`lightgbm`、`inceptiontime` |
| ⑥strict | 5 | `pykrige_ok3d`、`gpytorch_svgp`、`gstools_krige_condsrf` |
| ⑥conditional | 5 | `pykrige_ok3d`、`gpytorch_svgp`、`scipy_rbf_neighbors` |

③的4折为四个development母井家族LOGO4；④为F-5冻结测试之外的四个development母井家族LOGO4。
⑤各目标只使用自身P4 manifest声明的有效fold。物性MONAI 3D单候选lane、甜点T5/T6/T7和断层均不强凑top-3。

## 3. 统一预算与资源合同

- 延续Stage-2已冻结配置和每fold输入上限；Stage-3只改变fold与repeat seed，不调参。
- CPU模型每cell最多300秒；1D/2D神经模型最多200次更新/600秒；3D神经/算子最多80次更新/900秒。
- GPU cell必须持有`VOLVE_P5_GPU_LOCK`指向的统一排他锁并记录`cuda:0`、机制、等待和峰值VRAM。
- checkpoint、完整预测和缓存保存在赛道私有忽略目录；Git仅提交便携JSON/图件/manifest。
- 单cell失败不补跑成不同预算；允许同配置最多一次技术性重试，并保留首次原因与重试关系。

## 4. 输出与排名

每个赛道至少提交：

1. `p5_stage3_results.jsonl`：每个`task/lane/model/fold/repeat`一行；
2. `p5_stage3_summary.json`：预期cell、完成/skip/fail/timeout、source/split/result哈希；
3. 每任务/lane一个leaderboard：主指标均值、95% bootstrap CI、worst-fold、seed标准差、资源；
4. OOF预测manifest和可视化manifest；原始大预测不进入Git；
5. budget/split/seed/test-firewall、重复cell和跨lane污染的fail-closed测试；
6. 一个干净赛道提交，由负责人独立复跑后再进入集成分支。

排名先按预注册主指标均值，再按worst-fold，再按seed标准差和资源；并列不靠运行先后打破。若少于80%的
预期cell合法完成，该任务`not_rankable`。传统确定性模型仍运行三个seed，用于证明重复稳定性，不省略cell。

## 5. OOF可视化验收

- ①断层：仅data-readiness/负例/unknown覆盖图，不生成伪性能图。
- ②地震相：F3/Penobscot各自OOF剖面、error/entropy、逐类IoU/F1、混淆、fold×seed分布。
- ③物性：PHIF/KLOGH/SW各自逐井深度曲线、真值-预测、残差、worst-family与fold×seed分布。
- ④岩相：连续井深轨、固定九类混淆/PR/F1、校准、fold×seed与缺模态诊断。
- ⑤甜点：T1–T4各自回归散点/PR-校准/时间或井组误差；不得只画综合甜点图。T5–T7画状态与数据门。
- ⑥重建：strict/conditional各自三视图、CDF、频谱、变差函数、距井误差与fold×seed分布。

所有图必须从归档OOF预测或便携聚合重建，测试需证明图件不读取frozen test和历史test指标。

## 6. 完成后的下一步

Stage-3只冻结每任务/lane一个候选与配置。若确有必要，Optuna只能在development CV内围绕该胜出候选做
小规模预注册搜索；否则直接用全部development refit。最终frozen test仍由单独命令一次性消费，结果不得回流。

Stage-3已于2026-07-14按本合同执行：五个可运行赛道共441个cell，437 pass、3 fail、1 timeout；
断层因零个合法fold保持`not_rankable`。各任务胜出模型、精确指标、测试分组与科学边界见上述验收证据。
