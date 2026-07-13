---
phase_id: P4
reviewer: codex-leader-plus-five-workers
status: accepted
review_date: 2026-07-13
scope: training-validation-hpo-reproducibility-visualization
---

# P4 五赛道训练体系调研合并报告

> 日期：2026-07-13  
> 调研负责人：`junwei-p4-training-research-20260713`  
> 状态：五个窗口均已独立验收并关闭；本轮只读，零安装、零长训练、零代码修改、零提交。  
> 范围：①断层、②地震相、③储层物性、④岩相、⑥三维重建；⑤甜点另按七目标合同收敛。

## 总结结论

1. **五条已有赛道都有真实留出推理证据，不是只在训练集自测。** 但现有测试多是同一地震体内空间外推或同一油田内母井家族外推，不能升级表述为跨地震体、跨油田泛化。
2. **现有 test 已被看过。** 它仍可作为版本化回归测试集，但后续不能一边看其结果一边选 loss、epoch、阈值或超参数。若没有新增独立数据，必须把新的选择过程锁在 development CV，并把最终 test 作为一次性验收活动记录。
3. **“全部五折”不能机械执行。** `GroupKFold` 要求独立 group 数至少等于折数；储层物性和岩相冻结母井测试后都只剩 4 个可用母井家族，因此当前只能做 4-fold leave-one-family-out。硬拆同一母井凑五折会制造泄漏。
4. **共享训练循环存在样本加权聚合缺口。** 当前按 batch mean 等权平均时，最后一个较小 batch 会被过度加权；P4 必须改为按样本数或有效标签数聚合。
5. **调参和复现尚未形成全局合同。** 各赛道已有不同程度的 seed、checkpoint 和 manifest，但没有统一 seed tree、逐折预处理、HPO study、OOF 预测、frozen-test 防火墙与统一 artifact envelope。
6. **现有图件是真的，但不完整。** 每条赛道已有 loss、预测或混淆/切片图中的一部分；缺的是固定抽样规则、置信度/误差、逐 fold/逐 group、完整空间/井深/三维结构和失败案例图。

## 分赛道审计

| 赛道 | 当前真实测试证据 | 主要划分风险 | P4 CV 决策 | 现有图件 | 必补图件 |
|---|---|---|---|---|---|
| ①断层 | 同一 Volve 体内 8-inline 缓冲的 held-out patch test；256 train、96 test | patch 不是完整体；命名断层实体可跨区；已看 test | frozen 空间块之外做 buffered spatial 5-fold；有效正例块不足时降 3-fold/LOBO，禁止随机 patch | 3 个 test patch 的输入/真值/概率 + loss | 完整块滑窗、三视图、TP/FP/FN、PR/阈值、边界/连通性、三维面 |
| ②地震相 | F3 与 Penobscot 各有空间隔离 test 和 best-checkpoint inference | 两数据集标签空间不同；已有 HDF5 用单次 train 统计归一化，直接 CV 会泄漏 | 每数据集独立 buffered 5-fold；每折从原始 development 数据重拟合归一化和 class weight | test patch 输入/真值/预测 | 稠密整剖面、逐类 IoU/F1、混淆、置信度/熵、校准、空间误差 |
| ③储层物性 | 母井家族隔离 test，PHIF/log1p(KLOGH)/SW 有 best checkpoint 与真实图 | test 仅一个母井家族；当前三个标签被共同 finite 条件捆绑；验证 loss 聚合有 batch 权重问题 | 当前 4 个非 test 家族，执行 4-fold LOGO；新增至少 1 个非 test 家族后才升 5-fold | loss、深度预测、地震输入 | 真值-预测、残差、逐井深度、空间误差、范围违例、不确定性/覆盖率 |
| ④岩相 | GM09 九类，F-5 母井家族独立 test | 冻结 test 后仅 4 个 family；guard/test 缺部分类别；折内归一化必须重拟合 | 4-fold leave-one-family-out；同时报告固定 9 类和 observed-support 两套指标 | loss、混淆、4 个 test 样本 | 连续井深轨、逐类 PR/F1、置信度/校准、embedding、模态缺失与消融 |
| ⑥三维重建 | conditional 与 strict 两套同体空间 held-out inference | conditional test 含 test 区井约束，不能冒充纯外推；strict 仍非跨体 | strict frozen test 之外按 K/I 空间块做 buffered 5-fold；每折重建 IDW/井约束白名单 | 单 K 切片、loss、caveat | 三视图、signed/absolute difference、CDF、频谱、变差函数、等值面、距井误差 |

## 各赛道损失、输出和主指标建议

| 任务 | 首选稳定基线 | 必比候选 | 输出/推理 | 主指标与守门指标 |
|---|---|---|---|---|
| 断层二分类分割 | `BCEWithLogits + soft Dice` | BCE、Focal、Tversky | 模型返回 raw logits；指标/图时 sigmoid | AP/PR-AUC；Dice/IoU；boundary F1；连通性；worst block |
| 地震相多分类 | weighted CrossEntropy | Focal、CE+Dice、Lovasz-Softmax | raw logits；指标/图时 softmax/argmax | mIoU、macro-F1、逐类 IoU/PR、校准、worst block |
| 物性多输出回归 | Huber 或 MSE 基线比较 | MAE、Gaussian NLL、quantile | PHIF/SW 比较 identity 与 bounded 参数化；K 在 log1p 空间 identity/非负参数化 | 逐目标 MAE/RMSE/R²、逐井偏差、范围违例、log/raw 双空间 |
| 岩相九分类 | weighted CrossEntropy | Focal、class-balanced CE | raw logits；softmax 只在推理/图件 | macro-F1、balanced accuracy、逐类 AP/F1、ECE/Brier、逐井/逐家族 |
| 三维重建 | Huber + identity | MSE、MAE；结构/频谱项只做二阶段消融 | 默认 identity；同时报告 raw 与物理范围违例 | MAE/RMSE、bias、3D SSIM、梯度/频谱/变差函数、worst block |

## 关键本地发现

### 断层

- 当前输入是 `[1, 33, 65]` 的二维 crossline/time patch，现有三个模型仍是 voxel-wise 2D 简单模型，不应在文档中称为三维深度网络。
- 当前 test precision、AP 很低且正例极稀疏；accuracy/ROC-AUC 不能作为主要选择目标。
- 阈值必须来自 pooled OOF 预测，不能来自 test；最终完整 test block 要滑窗融合到物理 voxel 后再计分。

### 地震相

- F3 和 Penobscot 必须继续作为两个任务 schema，不能把 F3 十类与 Penobscot `0..7` 八类合并。
- Penobscot 发布页“7 类”与数据日志/标签值域“8 类”存在元数据冲突；工程合同以实际 `0..7` 为准，并保留冲突说明。
- 类别分布存在显著空间漂移，宏平均必须伴随逐类和逐空间块指标。

### 储层物性

- 孔隙度与渗透率可直接成为⑤的第 6、7 个独立任务，但必须拆开 label mask；不能因为 SW 缺失而丢弃本可用于 PHIF/KLOGH 的样本。
- 当前 test 上曾出现 PHIF/SW 越界和反变换后负渗透率；P4 必须同时报告 raw 预测、约束预测、越界率，禁止只静默 clip 后报分。
- 推理 API 应允许无标签输入，避免为了复用标准化函数而把 test label 带入推理路径。

### 岩相

- 当前图件显示 epoch 2 后明显过拟合，且预测集中到少数类别；这正说明需要按母井家族 CV、逐类指标、校准和早停，而不是只加 epoch。
- guard/test 缺类时要同时报告固定九类口径和测试实际支持类口径，不能删除困难类别美化分数。

### 三维重建

- `conditional` 协议允许测试区井约束进入 IDW，回答的是条件重建；`strict` 才更接近空间外推。两者必须分开命名、分开目录、分开指标。
- 当前 train-range clip 实际未改变 Ridge 预测，但未来模型仍须报告裁剪前指标和越界率。
- SSIM、频谱、变差函数和等值面是结构诊断，不应在未经 development CV 消融前取代体素误差主目标。

## 公共能力与赛道私有边界

### 进入共享框架

- `RunConfig`、`TaskSpec`、`ModelBatch`、`ModelOutput` 外层合同。
- seed tree、确定性报告、环境与 Git/data/config 哈希。
- split/fold manifest schema、test firewall、CV/HPO orchestration。
- sample/valid-label weighted loss reducer、checkpoint/resume、early-stop/scheduler 事件。
- OOF/final/test 产物 envelope、逐 fold/seed 聚合和通用训练/HPO 图。

### 留在赛道插件

- group/spatial/time splitter 的领域规则与 buffer。
- target transform、label mask、loss/metric/search space。
- fault surface、facies section、well-depth、modality ablation、3D spectrum/isosurface 等专属可视化。
- 物理范围、单位、类别 schema、阈值/事件操作定义。

## 调研验收记录

- `volve-worker-fault`：已验收，工作树 clean。
- `volve-worker-facies`：已验收，工作树 clean。
- `volve-worker-property`：已验收，工作树 clean。
- `volve-worker-lithofacies`：supervisor 状态曾因 pane busy 标记失败，但任务实际完整到达原 pane；人工核对报告、HEAD `86281c6...` 与 clean worktree 后验收。
- `volve-worker-reconstruction`：已验收，工作树 clean。

负责人最终关闭 leader 的证据为：五赛道均完成只读本地审计和官方/原始来源研究，且未产生工作树变更。

## 主要权威依据

- [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [Optuna efficient optimization algorithms](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
- [Roberts et al. 2017：空间/时间/层级结构数据交叉验证](https://doi.org/10.1111/ecog.02881)
- [Valavi et al. 2019：blockCV](https://doi.org/10.1111/2041-210X.13107)
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch HuberLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)
- [Equinor Open Data / Volve](https://www.equinor.com/energy/data-sharing)
