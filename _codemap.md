# Code Map — 军伟的比赛（地震+测井多模态融合识别有利油气目标）

更新: 2026-08-01

本文只做 COL4 代码地图 / 注册路由。当前 phase、下一步、运行态、最近测试结果只写 `_wiki-methodology/_top/_task_plan.md`。

## 冷启动读法

1. 当前状态: `_wiki-methodology/_top/_task_plan.md`
2. 六合一职责: `_wiki-methodology/_top/_six_dim_contract.md`（若项目已创建）
3. 代码注册: `_meta/_registry.yml`
4. 数据注册: `_meta/_data_registry.yml`
5. 测试地图: `_wiki-methodology/_tests/`

## 项目身份

| 字段 | 值 |
|---|---|
| 项目 | 军伟的比赛（地震+测井多模态融合识别有利油气目标） |
| Mode | research |
| 目标 | 输入地震数据+测井数据+井位轨迹+专家标签，输出断层/储层/有利目标概率图或分割结果；核心证明"地震+测井融合"优于"地震单模态" |
| 方法 | 5阶段pipeline骨架（地质属性预测→测井编码→井震融合→弱对齐→空间编码解码），具体算法待补，见 `_wiki-methodology/_wiki/_methods/pipeline-skeleton.md` |

## COL4 真源关系

| 层 | 文件/目录 | 职责 |
|---|---|---|
| 人读代码地图 | `_codemap.md` | 代码域、入口、调用关系、测试指针 |
| 机读代码注册 | `_meta/_registry.yml` | 代码域、工具、pipeline、脚本、可复用入口 |
| 数据注册 | `_meta/_data_registry.yml` | 数据集、模型、外部资产、生成物血缘 |
| 查询/同步工具 | `_code/` | registry 查询、lint、同步 wiki 投影 |

## 主要代码域

| 代码域 | 入口 | 注册/测试 |
|---|---|---|
| 主流程 | `_pipelines/` | `_meta/_registry.yml` + `_wiki-methodology/_tests/` |
| 可复用代码 | `_code/` | `_meta/_registry.yml` |
| 数据接口 | `_data/` | `_meta/_data_registry.yml` |
| 输出/图/论文 | `_outputs/` / `_figures/` / `_paper/` | `_meta/_runs.yml` / `_meta/_data_registry.yml` |
| 实验代码 | `_sandbox/` | `_sandbox/_index.yml`; 优于 baseline 才进入主流程 |
| 归档代码 | `_legacy/` | 每个归档目录需 README/TOMBSTONE |

## P4 训练与评价系统

| 代码域 | 真源 / 入口 | 测试指针 |
|---|---|---|
| 公共训练合同 | `_code/ml_framework/` | `_code/ml_framework/tests/` |
| 规范模型真源 | `_models/<track>/<model_id>.py` | `_code/ml_framework/tests/test_canonical_track_models.py` |
| ①断层 | `_pipelines/02_task_datasets/fault/p4_workflow.py` | `fault/test_fault_p4.py` |
| ②地震相 | `_pipelines/02_task_datasets/facies/p4_training.py` | `facies/tests/test_p4_*.py` |
| ③储层物性 | `_pipelines/02_task_datasets/reservoir/p4_pipeline.py` | `reservoir/tests/test_p4_*.py` |
| ④岩相 | `_pipelines/02_task_datasets/lithofacies/p4_runner.py` | `lithofacies/tests/test_p4_contract.py` |
| ⑤甜点七目标 | `_pipelines/02_task_datasets/sweetspot/targets/` | `sweetspot/tests/test_p4_*.py` |
| ⑥三维重建 | `_pipelines/02_task_datasets/reconstruction/p4_reconstruction.py` | `reconstruction/_tests/test_p4_*.py` |

公共外层是 `ModelBatch -> ModelOutput`，任务内部张量 shape、loss、activation、mask 和 metric 按赛道保留差异。
⑤七目标的机读状态与产物哈希汇总在
`_pipelines/02_task_datasets/sweetspot/targets/_outputs/registry_targets_1_to_7.json`。
最新真实运行证据和科学性边界在 `_wiki-methodology/_tests/P4_acceptance_evidence.md`。

## P8 多模态基础模型路由

| 代码域 | 真源 / 入口 | 测试指针 |
|---|---|---|
| 统一条件/晋级合同 | `_models/gaia_dagt/foundation.py` | `_models/gaia_dagt/tests/test_foundation_contract.py` |
| source/weights 运行门 | `_models/gaia_dagt/foundation_runtime.py`、`foundation_routes.v1.json` | `test_foundation_contract.py` |
| 监督 LLM 模板/调用边界 | `_models/gaia_dagt/foundation_prompts.py` | `test_foundation_contract.py` |
| 四条新非时序 adapter | `_models/{fault,facies,lithofacies,reconstruction}/` | `test_foundation_contract.py` |
| TabICLv2 / Chronos-2 | `_models/property/tabiclv2_regressor.py`、`_models/sweetspot/p7_chronos2.py` | property Stage-1 + sweetspot P8 tests |
| Chronos 日历 runner | `_pipelines/02_task_datasets/sweetspot/p8/runner.py` | `test_sweetspot_p8_calendar.py` |

六条路线统一可发现，但保持任务专属张量、head、loss 与 metric；route 接通不自动成为默认。

## P17–P23 赛道⑥基础模型与井震校准

| 代码域 | 真源 / 入口 | 测试指针 |
|---|---|---|
| GFM 非平稳邻域 runner | `_pipelines/02_task_datasets/reconstruction/p17_foundation_geostatistics.py` | `reconstruction/_tests/test_p17_foundation_geostatistics.py` |
| 便携结果与独立复算 | `_pipelines/02_task_datasets/reconstruction/_outputs/p17_foundation_geostatistics/` | `_wiki-methodology/_tests/P17_reconstruction_foundation_acceptance_evidence.md` |
| 各向异性 + 嵌套选型 runner | `_pipelines/02_task_datasets/reconstruction/p18_anisotropic_foundation_geostatistics.py` | `reconstruction/_tests/test_p18_anisotropic_foundation_geostatistics.py` |
| P18 便携结果与独立复算 | `_pipelines/02_task_datasets/reconstruction/_outputs/p18_anisotropic_foundation_geostatistics/` | `_wiki-methodology/_tests/P18_reconstruction_anisotropic_acceptance_evidence.md` |
| 元拟合坐标去重 runner | `_pipelines/02_task_datasets/reconstruction/p19_meta_purged_geostatistics.py` | `reconstruction/_tests/test_p19_training_diagnostics_artifacts.py` |
| P19 诊断结果与独立复算 | `_pipelines/02_task_datasets/reconstruction/_outputs/p19_training_diagnostics/` | `_wiki-methodology/_tests/P19_reconstruction_training_diagnostics_acceptance_evidence.md` |
| P23 Checkshot 独立校验 runner | `_pipelines/02_task_datasets/reconstruction/p23_checkshot_calibration.py` | `reconstruction/_tests/test_p23_checkshot_calibration.py` |
| P23 标定结果与声明边界 | `_pipelines/02_task_datasets/reconstruction/_outputs/p23_checkshot_calibration/` | `_wiki-methodology/_tests/P23_reconstruction_checkshot_calibration_evidence.md` |

P19 取代 P18：每个报告折不仅排除自身指标行，还会先从其余折的元选择训练
子集中删除当前验证坐标，再重新拟合全部候选。严格结果保持 5/5 折改善；每折
最终预测仍只使用 512 条合法训练标签，CLI 无 test/holdout 入口，候选默认关闭。

P23 使用 19A/19BT2/19SR 拟合时深关系，将 F11T2/F15A 保留为
独立标定井。该入口只证明 checkshot 校准精度，不读孔隙度标签，
也不把标定改善报告为下游模型改善。P21 仍是赛道⑥的默认模型。

## 注册维护

- 新增主流程代码:先查 `_meta/_registry.yml`,复用已有条目;确实新增再注册。
- 新增数据/模型/外部资产:写 `_meta/_data_registry.yml`,不要只写 README。
- 新增测试:测试地图写 `_wiki-methodology/_tests/`,不要只在提交说明里写。
- 临时探查:留 `_tmp/` 或 `_sandbox/`,不进 registry。

## 不放在本文的内容

- 当前 phase / next action / service online-offline: `_wiki-methodology/_top/_task_plan.md`
- 长期路线和阶段边界: `_wiki-methodology/_top/_roadmap.md`
- 历史交接: `_wiki-methodology/_top/_handoff*.md`
- 架构图: `_wiki-methodology/_top/_SYSTEM-ARCH.md`
- 文件生命周期缺口: `_meta/_known_issues.md`
