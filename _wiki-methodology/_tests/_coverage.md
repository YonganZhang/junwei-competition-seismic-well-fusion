# Test Coverage - 军伟的比赛

> COL3 测试地图。长运行证据见 `_run_ledger.md`，P4与P5 Stage-1结果及科学性边界分别见
> `P4_acceptance_evidence.md`、`P5_stage1_acceptance_evidence.md`、
> `P5_stage2_acceptance_evidence.md`、`P5_stage3_acceptance_evidence.md`、
> `P8_multimodal_foundation_acceptance_evidence.md`、
> `P17_reconstruction_foundation_acceptance_evidence.md`、
> `P18_reconstruction_anisotropic_acceptance_evidence.md`、
> `P19_reconstruction_training_diagnostics_acceptance_evidence.md`、
> `P23_reconstruction_checkshot_calibration_evidence.md`、
> `P24_reconstruction_historical_transfer_evidence.md`。

## Audit-First / SSDO

| Gate | Entrypoint | Coverage Purpose |
|---|---|---|
| domain visualization audit | `python3 _pipelines/03_domain_visualization_delivery/step_00_discover.py --check` | 固定发现 P12 赛道1/3/5的渲染器、测试、manifest、输入输出哈希和 paused 边界；失败时不得进入人工验收 |
| legacy six-track audit | `python3 _pipelines/03_domain_visualization_delivery/step_01_validate_manifest.py --check-only` | 读取旧六赛道白名单的HEAD、哈希、脚本、证据、尺寸与人工复核状态；失败时不生成交付清单 |
| P4 验收报告 | `sed -n '1,260p' _wiki-methodology/_tests/P4_acceptance_evidence.md` | 先区分“工程流程通过”、“模型精度达标”和 `not_feasible` |
| P5 Stage-1 验收报告 | `sed -n '1,260p' _wiki-methodology/_tests/P5_stage1_acceptance_evidence.md` | 区分候选尝试、contract smoke、结构化skip与禁止排名的科学硬门 |
| P5 Stage-3 验收报告 | `sed -n '1,300p' _wiki-methodology/_tests/P5_stage3_acceptance_evidence.md` | 核对top-3多seed全fold结果、worst-fold、OOF图、真实失败/超时和test firewall |
| P8 多模态基础模型验收 | `sed -n '1,320p' _wiki-methodology/_tests/P8_multimodal_foundation_acceptance_evidence.md` | 核对六赛道真实权重/源码锁、typed conditioning、运行时证据、泄漏防火墙和“连接不等于晋级” |
| P17 重建基础模型验收 | `sed -n '1,300p' _wiki-methodology/_tests/P17_reconstruction_foundation_acceptance_evidence.md` | 核对 512 标签/折、5 空间折、holdout 防火墙、整折不确定性与独立复算 |
| P18 各向异性重建验收 | `sed -n '1,300p' _wiki-methodology/_tests/P18_reconstruction_anisotropic_acceptance_evidence.md` | 核对 P17 选择偏差修正、嵌套 LOFO、各向异性、5/5 折改善、整折区间与独立复算 |
| P19 重建训练诊断验收 | `sed -n '1,320p' _wiki-methodology/_tests/P19_reconstruction_training_diagnostics_acceptance_evidence.md` | 核对元选择训练坐标去重、逐层张量/激活/梯度、五条替代路线和 holdout 防火墙 |
| P23 Checkshot 标定验收 | `sed -n '1,240p' _wiki-methodology/_tests/P23_reconstruction_checkshot_calibration_evidence.md` | 核对三口拟合井、两口独立校验井、标定改进和下游模型不晋级边界 |
| P24 历史版本迁移验收 | `sed -n '1,240p' _wiki-methodology/_tests/P24_reconstruction_historical_transfer_evidence.md` | 核对预注册、冻结 P21、一次性目标开放、4/5 折改善和非跨场区声明 |
| 甜点七目标注册 | `python3 -m _pipelines.02_task_datasets.sweetspot.targets.registry --output _tmp/p4-target-registry-audit.json` | 重建独立目标注册并校验必需产物存在 |
| TOP 结题检查 | `python3 /mnt/data/yongan-admin-2/.codex/skills/share-top/scripts/topic-closeout.py .` | 发现计划/codemap/registry/test map 漂移 |

## Trunk

| Gate | Entrypoint | Coverage Purpose |
|---|---|---|
| `domain-visualization-delivery` | `python3 -m unittest discover -s _pipelines/03_domain_visualization_delivery/tests -p 'test_*.py' -v` | 证明旧状态图被拒绝、六赛道白名单继续有效、P12赛道1/3/5可发现、人工验收必须显式接受且复制后哈希保持一致 |
| 公共合同 | `_code/ml_framework/tests/` | envelope、seed、split、reducer、checkpoint/resume、HPO、artifact、test firewall、规范模型动态发现 |
| 六赛道便携回归 | `_wiki-methodology/_tests/_gates.yml` 中 `p4-*` gates | 无需私有数据时检查合同；PyTorch gates 要求显式设置 `P4_TORCH_PYTHON`；缺真实产物的用例必须明确 skip |
| 真实数据验收 | 按 `_run_ledger.md` 中赛道命令执行 | 真实 smoke、CV、freeze、refit、single-use test 与归档可视化 |

## Branch

| Test | Covers |
|---|---|
| `_pipelines/03_domain_visualization_delivery/tests/test_delivery_pipeline.py` | 路径分类、真实六赛道清单、P12合同、PNG尺寸、独立人工验收与复制哈希 |
| `_pipelines/02_task_datasets/fault/test_fault_p4.py` | 断层标签审计、buffered spatial split、缺审核负例时 fail closed |
| `_pipelines/02_task_datasets/facies/tests/test_p4_*.py` | F3/Penobscot 独立任务、像素 mask、CV/lifecycle、real smoke |
| `_pipelines/02_task_datasets/reservoir/tests/test_p4_*.py` | PHIF/KLOGH/SW 分离 mask、物理空间指标、⑤目标6/7 adapter |
| `_pipelines/02_task_datasets/lithofacies/tests/test_p4_contract.py` | 母井家族划分、固定九类/支持类双口径、一次性 F-5 test |
| `_pipelines/02_task_datasets/sweetspot/tests/test_p4_*.py` | 七目标独立性、代理标签泄漏防火墙、可行/`not_feasible` 分流 |
| `_pipelines/02_task_datasets/reconstruction/_tests/test_p4_*.py` | strict/conditional 拆分、buffered block CV、约束审计、冻结测试 |
| 六赛道 `*p5*` 测试 | 开源adapter/source lock、真实development小批次、确定性/shape/finite、结构化skip与test firewall |
| `_models/gaia_dagt/tests/test_foundation_contract.py` | 六路基础模型注册、源码/权重锁、typed conditioning、晋级状态机、监督LLM响应schema及真实Torch adapter回归 |
| `_pipelines/02_task_datasets/sweetspot/tests/test_sweetspot_p8_calendar.py` | Chronos-2严格日历时轴、30日因果窗口、同井隔离、缺日mask及未来标签防火墙 |
| `_pipelines/02_task_datasets/reconstruction/_tests/test_p17_foundation_geostatistics.py` | GFM 非平稳邻域、座标反映射、训练内标准化/PCA、无 test CLI、真实证据与独立复算 |
| `_pipelines/02_task_datasets/reconstruction/_tests/test_p18_anisotropic_foundation_geostatistics.py` | 垂向各向异性、有界 1,215 候选、嵌套选型防泄漏、P17 偏差修正、真实证据与独立复算 |
| `_pipelines/02_task_datasets/reconstruction/_tests/test_p19_training_diagnostics_artifacts.py` | 元选择坐标重叠修正、训练动力学量级、严格失败路线、持久化 verifier 与冻结 holdout 边界 |
| `_pipelines/02_task_datasets/reconstruction/_tests/test_p21_fixed_foundation_ensemble.py` | 固定三基础核平均、产物哈希、拒绝残差路线和默认模型边界 |
| `_pipelines/02_task_datasets/reconstruction/_tests/test_p24_historical_transfer.py` | 历史 RMS 映射、冻结成功门、目标开放防火墙和声明边界 |

## Leaves

| Unit / Sample | Purpose |
|---|---|
| `_outputs/domain_visualization_delivery/v1/validation_report.json` | 结构化 anti-fake-completion 断言 |
| `_outputs/domain_visualization_delivery/v1/published_manifest.json` | 六个永久URL及HTTP 200发布证据 |
| `_outputs/domain_visualization_delivery/p12/review_attestation.json` | 赛道1/3/5的负责人逐图验收、来源/稳定副本路径与39个文件哈希 |
| `_code/ml_framework/tests/test_canonical_track_models.py` | 六赛道 `_models/` 真源均能动态发现、训练一批、存取 checkpoint |
| `_pipelines/02_task_datasets/reconstruction/_tests/test_contract.py` | 从项目根目录直接执行 task-spec/tiny-smoke/visualization CLI |
| 各赛道 tiny-overfit/smoke | 在长训练前检查有限输出、loss 方向和检查点恢复 |

## Current Gaps

- 卡片公网内容本身没有视觉像素回归；当前以源图SHA-256、确定性双渲染测试与发布前人工逐张预览锁定内容。
- 赛道⑥的全尾块微调、LoRA、Adapter、分阶段解冻和残差路线均已完成严格复核；最终默认模型为 P21 固定三基础核平均。P24 在未使用的同场区历史版本上通过预注册门，用户已接受该证据等级并停止跨场区扩展；不再把 PEFT 或外部盲测列为当前下一步。其他赛道的同 split 基础模型控制门仍需分别按数据条件验证。
- 断层只有 3998 个审核正例点，没有覆盖已核验负例与连续三维development块；SAM-Med3D只能作为合成体运行证据，正式 blind test/CV 不可行，也不允许用随机非断层patch或验证/测试真值提示点伪造结果。
- 监督LLM已具备统一提示模板、provider-neutral调用边界和严格JSON响应校验；因尚未批准具体provider/model/revision，本轮没有真实外部API调用，也没有让LLM接触标签、原始预测或冻结测试指标。
- 甜点目标5缺已验证 Eclipse cell-state parser 与冻结的时间/候选井/经济约束；只能是 simulation case，不是 field truth。
- 精确 PHIE 缺独立真值；不用 LFP_PHIE 替代。
- 岩相冻结预测未持久化真实 `center_md_m`，因此深度轨迹图 `not_feasible`；其他分类/校准图已可用。
- Stage-3多seed全有效fold已执行；长预算HPO仍未执行且不是默认必做，必须先写预注册理由并仅在development CV内搜索。
- 最终全development refit和一次性frozen-test尚未执行；断层审核负例、甜点T5模拟约束和T6/T7 development-only特征源仍是硬阻塞。
- 岩相CatBoost在一个fold的三个seed均遇到NaN/Inf；重建PyKrige一个conditional cell达到300秒上限。两者均保留为真实Stage-3证据，未换预算补数。
