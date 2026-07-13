# P5 六赛道开源模型基准协议

> 冻结日期：2026-07-14
> 基线：`p4-training-integration@2d128b009bd1d943918f948785bf5f4e19ce4b7b`
> 分支：`p5-model-benchmark-integration`
> 状态：执行合同 v1；候选来源调研、Stage-1与Stage-2均已验收，待Stage-3准入冻结
> 全局随机种子：`2693`

## 1. 目的与边界

P5 的目标不是把开源仓库原样搬进项目，而是在 P4 已冻结的 `ModelBatch -> ModelOutput`、
split、test firewall、artifact 和可视化合同上，为六个赛道各接入并筛查至少 10 个实质不同的模型。

本协议区分四种证据状态：

1. `scouted`：只完成主源、精确 revision、许可证和接口调研。
2. `contract_smoked`：在隔离环境中完成 import、build、真实小批次 forward/loss/backward、
   checkpoint round-trip、shape/finite 和确定性检查。
3. `development_piloted`：固定 development fold 和固定小预算运行，产生可比较的 validation 证据。
4. `cv_confirmed`：完成全部有效 folds 和预注册 seed；只有最终冻结配置可进入单次 frozen test。

调研报告中的 `L2 可实测` 只对应第 1 档后的准入判断，绝不等于已跑通或已有分数。

## 2. 已验收调研

| 赛道 | 去重候选 | L2 候选 | 首批实测数 | 科学硬门 |
|---|---:|---:|---:|---|
| ①断层预测 | 19 | 16 | 10 | 正例、审核负例和 unknown mask 必须分开；无合法负例时只做工程 smoke，不发布模型排名 |
| ②地震相分类 | 22 | 17 | 10 | F3 与 Penobscot 独立标签空间、独立 head、独立 split 和独立榜单 |
| ③储层物性 | 20 | 16 | 10 | PHIF/KLOGH/SW 独立 mask；母井家族隔离；预训练权重另过许可证门 |
| ④岩相预测 | 22 | 15 | 10 | GM09 固定九类；F-5 frozen test；development 仅四个母井家族，因此诚实使用 LOGO4 |
| ⑤甜点预测 | 19 | 16 | 10 | 七个独立目标必须各有已批准 `label_spec`；未批准目标 fail-closed，不制造代理标签 |
| ⑥三维重建 | 25 | 12 | 10 | strict 与 conditional 两种任务分开训练、评价和展示，禁止混报 |
| **合计** | **127** | **92** | **60** | 每个赛道 frozen test 都不能参与模型选择 |

逐项来源、commit、许可证、资源估算和失败降级保存在主工作区
`_tmp/model_scout_20260714/{fault,facies,property,lithofacies,sweetspot,reconstruction}.md`。

## 3. 首批 60 个候选

### ①断层预测

`monai_segresnet`、`monai_dynunet`、`nnunet_v2_3d_fullres`、
`pytorch3dunet_unet3d`、`faultnet_md`、`faultseg3d_keras`、`monai_vnet`、
`mednext_v1_s_k3`、`uxnet3d`、`monai_swinunetr`。

`faultnet_md` 是推理锚点而非可训练架构；`faultseg3d_keras` 受非商业研究许可证约束。
没有经审计负例时，这 10 个候选只过 contract smoke，停止在正式 HPO/CV 之前。

### ②地震相分类

`smp_unet_r18`、`smp_deeplabv3plus_r18`、`smp_unetpp_r18`、`smp_fpn_r18`、
`torchvision_lraspp_mbv3`、`deepseismic_patch_skip`、`deepseismic_seresnet_unet`、
`hf_segformer_b0`、`sfm_base_facies`、`monai_unet3d`。

自然图像/领域预训练与随机初始化分成两条 lane；权重许可证或哈希未冻结时只允许 scratch lane。

### ③储层物性

`catboost_regressor`、`lightgbm_regressor`、`tabm_regressor`、`xgboost_regressor`、
`extra_trees_regressor`、`hist_gradient_boosting_regressor`、`realmlp_regressor`、
`ft_transformer_regressor`、`tabiclv2_regressor`、`monai_densenet3d_regressor`。

前三个物性目标分别输出并分别计分；任何 target transform、反变换和范围检查均由 `TaskSpec` 冻结。

### ④岩相预测

`xgboost_multisoftprob_window`、`catboost_multiclass_window`、`minirocket_ridge_window`、
`inceptiontime_window`、`tcn_center_head`、`balanced_softmax_tcn`、`moderntcn_window`、
`ms_tcn2_dense`、`embracenet_missing_modal`、`multibench_lowrank_tensor_fusion`。

中心窗口分类 P 榜与逐序列位置标注 S 榜分开；不能用不同上下文预算混成一个排行榜。

### ⑤甜点预测

`xgboost`、`catboost`、`lightgbm`、`autogluon_limited`、`inceptiontime`、`patchtst`、
`temporal_fusion_transformer`、`seg_spatial_tcn`、`graphsage`、`monai_unet3d`。

七目标分别训练 estimator/head、分别生成 manifest 和榜单。T6 孔隙度、T7 渗透率是独立任务，
不能藏在综合甜点分数里；T5 无已批准模拟时刻/标签时维持 `not_feasible`。

### ⑥三维重建

`scipy_rbf_neighbors`、`pykrige_ok3d`、`gstools_krige_condsrf`、`mpslib_snesim3d`、
`gpytorch_svgp`、`monai_basicunet3d`、`monai_segresnet3d`、`neuralop_fno3d`、
`tcnn_hashgrid_inr`、`siren_inr`。

MPS 无合法 training image 时必须 skip；所有方法分别运行 strict/conditional 合同，不得把条件井值
带来的增益写成 strict 空间泛化。

## 4. 执行阶梯

### Stage 0：静态准入

- 锁定主源 URL、tag/commit、代码许可证、权重许可证和 SHA-256。
- 不复制整个上游仓库；只在 `_models/<track>/` 写薄 adapter，第三方版本进入 source lock。
- 每个候选必须声明输入模态、输出 shape、任务类型、mask/uncertainty 能力和依赖组。
- 许可证不清、revision 漂移或必须使用不明镜像时，记录 `skipped`，不得换同名第三方实现冒充。

### Stage 1：60 个 contract smoke

每个候选至少完成：

- 动态发现和 `build_model(task_spec, **config)`；
- 一个合成 batch 和一个真实 development 小 batch；
- finite raw output、shape、target mask、单步 loss/backward（可训练模型）；
- checkpoint save/load 后预测一致；同环境相同 seed 的允许误差内复跑；
- 峰值 CPU/RAM/VRAM、wall time、下载字节、环境锁和失败原因归档；
- 不接收 test loader，不读取 test 路径，不产生 test 指标。

Stage 1 只回答“能否按统一接口运行”，不做优劣排名。

### Stage 2：固定 development pilot

本轮冻结参数和六赛道执行矩阵见 `P5_stage2_fixed_budget_pilot.md`。

- 每赛道 10 个候选使用相同的预注册 fold、输入预算、更新步数或 wall-clock 上限。
- 预处理、类别权重、target transform、采样器和阈值只在 fold-train 拟合。
- 树/插值模型可用 CPU；神经模型使用单 GPU，单卡同一时刻一个可比 run。
- 输出 validation 主指标、worst-group、稳定性、资源和 guardrail；仍不打开 frozen test。

### Stage 3：Top-3 确认

- 每赛道按主指标、worst fold、稳定性和资源 Pareto 选择最多 3 个候选。
- `top 3 × 3 seeds × 全部有效 folds`；折数由独立 group/空间/时间支持决定，不能强凑 5 折。
- 可选 Optuna 只在 development CV 内运行；默认 20–30 trials，搜索空间和方向预注册。
- loss、最后一层参数化、学习率、scheduler 和 early stop 都只能由 validation 选择。

### Stage 4：refit 与 frozen test

- 冻结一个胜出配置后，用全部 development 重训。
- frozen test 由单独命令只消费一次，产物立刻归档；任何模型选择、阈值、校准或绘图调参不得回流。
- 科学硬门失败的赛道输出 `not_feasible.json`，不以 proxy、训练集拟合或 regression evidence 冒充 test。

## 5. 统一随机性、HPO 和比较预算

- `root_seed=2693`，由稳定哈希派生 split/model/loader/sampler/augmentation/HPO/diagnostic seed。
- Stage 1 固定最小配置；Stage 2 不因某模型擅自增加数据、epoch、外部权重或 trial 数。
- 神经模型默认以 AdamW 为起点，学习率用对数空间；候选 scheduler 由赛道限定。
- 分类/分割输出 raw logits；sigmoid/softmax 只在推理、校准和可视化层使用。
- 回归同时保留物理空间和变换空间指标；不静默 clip 越界输出。
- 自动调参是可审计的搜索，不是“自动得到最优真理”；test 永远不进入 objective。

## 6. 环境和磁盘策略

当前数据盘使用率高，禁止为每个模型复制仓库、CUDA wheel 和缓存：

- 复用少量依赖组：`tabular-cpu`、`torch-common`、`monai-3d`、`legacy-isolated`、
  `geostat-cpu`、`operator-inr`。
- 下载缓存共享但只读；每个 run 记录来源、哈希和实际字节。
- 旧 TensorFlow/旧 CUDA 或需要编译的候选放独立环境；失败不得污染公共环境。
- 默认先跑无权重/scratch smoke；大权重、gated 或非商业权重单独审批。
- 同一时刻最多四个安装/训练任务，避免 HDF5/磁盘和 GPU 争抢；每 GPU 最多一个可比 run。

## 7. 可视化和自测验收

所有赛道都要生成通用的训练曲线、fold/seed 分布、OOF 误差和资源图；除此之外必须交付：

- 断层：三正交概率/GT/error、PR/阈值、连通性和 3D 表面。
- 地震相：F3/Penobscot 各自剖面真值/预测/error/entropy、逐类 IoU/F1、混淆和校准。
- 储层物性：逐井深度曲线、真值-预测、残差/空间误差、物理范围和 uncertainty coverage。
- 岩相：连续井深轨、逐类 PR/F1、混淆、校准、缺模态和融合消融。
- 甜点：T1–T7 各自图件；允许共享空间叠图，但不允许只画一个综合甜点热图。
- 三维重建：strict/conditional 各自三视图、CDF、频谱、变差函数、等值面和距井误差。

最低自动化测试包括 unit、contract、tiny-overfit、real-data smoke、split 零交叉、test firewall、
checkpoint/resume、artifact manifest、可视化从保存预测重建以及相同 seed 回归。

## 8. 工作树与合并策略

1. 主仓当前有未归属修改和与 P4 重叠的未跟踪文件，今晚不直接 merge。
2. P5 集成工作树从干净的 P4 HEAD 创建，所有 P5 赛道分支均以它为祖先。
3. 六个旧 track worktree 只保留作历史证据；后续实现应使用新的 P5 worktree，避免继续堆叠在旧提交上。
4. 主仓先归属/保存脏改动，再受控 fast-forward/merge P4；P5 之后按独立验收 commit 顺序集成。
5. 未通过独立测试的分支不得合并；不 push、不发布未经用户确认的外部资产。

## 9. P5 完成条件

- 六赛道各至少 10 个候选有独立 Stage-1 结果；失败也必须有可审计失败证据。
- 每赛道所有科学可行候选完成同预算 Stage-2；不可行赛道明确停止线。
- 每赛道至多 3 个胜出候选完成有效 folds × 3 seeds；简单 P4 baseline 始终作为 control。
- frozen test 仅由最终冻结配置消费一次；全部指标、checkpoint、环境、源码/数据哈希和专属可视化齐全。
- 所有变更在 clean P5 worktree 中形成可审计 commit，并经负责人独立复跑后再进入合并决策。

## 10. 2026-07-14 Stage-1 验收证据

完整命令门、实时数据旅程与Trace/SSDO降级证据见 `../../_tests/P5_stage1_acceptance_evidence.md`。

Stage-1 的完成定义是“每个候选都有合同检查结果或可审计的硬门停止原因”，不是“60个候选全部训练成功”，
更不是性能排名。所有运行都禁止读取 frozen test。

| 赛道 | 首批10个的Stage-1结果 | 科学停止线 |
|---|---|---|
| ①断层 | 可用依赖完成工程forward/contract检查；未形成正式比较榜 | 缺经审核负例，停止在HPO/CV前 |
| ②地震相 | F3与Penobscot各6个`contract_smoked`、4个结构化skip | 两数据集独立榜单；未访问测试归档 |
| ③储层物性 | 9个`contract_smoked`、1个许可证门skip | PHIF/KLOGH/SW独立mask；TabICLv2权重未批准 |
| ④岩相 | P通道9个通过、S通道1个结构化skip | 固定小批次缺连续同井MD样本，不跨通道凑结果 |
| ⑤甜点 | 70个“模型×目标”真实格全部label-gated skip；adapter fixture合同测试通过 | 七目标均无已批准真实`label_spec`，不生成代理标签 |
| ⑥三维重建 | strict与conditional各8个通过、2个结构化skip | 两任务独立；条件井信息不流入strict |

六个赛道提交以固定顺序进入P5集成分支：

1. `53daaa7` fault
2. `26c4250` facies
3. `6ebea6f` property
4. `0ce3bf1` lithofacies
5. `2559b7b` sweetspot
6. `fabe99b` reconstruction

集成后发现多个赛道使用相同测试文件名和裸模块名，pytest联合收集会发生模块碰撞；现已在集成层使用
赛道唯一文件名和显式文件路径加载进行隔离。两套共享环境联合复跑结果：

- `torch-common`全量套件：53 passed、6 skipped、77 subtests passed；
- `tabular-cpu`：31 passed、2 skipped、20 subtests passed。

因此，Stage-2只能纳入已有真实development标签、依赖和许可证边界清楚的候选。①与⑤应先完成数据/标签
硬门，不得为了满足“10个模型”而制造负例、代理标签或训练分数。

## 11. 2026-07-14 Stage-2验收结论

完整逐轨命令门、结果矩阵、集成提交和科学停止线见
`../../_tests/P5_stage2_acceptance_evidence.md`。Stage-2在140个预注册cell上得到53个真实development pilot、
87个结构化skip/blocked、0个failed/timeout；所有结果都保持`root_seed=2693`、P4锁定fold和frozen-test防火墙。

只有以下同lane、同任务候选可进入Stage-3 Pareto准入：地震相两数据集各6个、物性tabular lane 8个、
岩相P lane 9个、甜点T1–T4各4个、重建strict/conditional各8个。断层、甜点T5、T6、T7继续保持
科学阻塞；物性单候选3D lane为`not_rankable`。Stage-2开发折榜单不得改写为最终模型排名。
