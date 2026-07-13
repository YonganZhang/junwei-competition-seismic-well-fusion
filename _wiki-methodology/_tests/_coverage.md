# Test Coverage - 军伟的比赛

> COL3 测试地图。长运行证据见 `_run_ledger.md`，P4与P5 Stage-1结果及科学性边界分别见
> `P4_acceptance_evidence.md`、`P5_stage1_acceptance_evidence.md`、
> `P5_stage2_acceptance_evidence.md`、`P5_stage3_acceptance_evidence.md`。

## Audit-First / SSDO

| Gate | Entrypoint | Coverage Purpose |
|---|---|---|
| P4 验收报告 | `sed -n '1,260p' _wiki-methodology/_tests/P4_acceptance_evidence.md` | 先区分“工程流程通过”、“模型精度达标”和 `not_feasible` |
| P5 Stage-1 验收报告 | `sed -n '1,260p' _wiki-methodology/_tests/P5_stage1_acceptance_evidence.md` | 区分候选尝试、contract smoke、结构化skip与禁止排名的科学硬门 |
| P5 Stage-3 验收报告 | `sed -n '1,300p' _wiki-methodology/_tests/P5_stage3_acceptance_evidence.md` | 核对top-3多seed全fold结果、worst-fold、OOF图、真实失败/超时和test firewall |
| 甜点七目标注册 | `python3 -m _pipelines.02_task_datasets.sweetspot.targets.registry --output _tmp/p4-target-registry-audit.json` | 重建独立目标注册并校验必需产物存在 |
| TOP 结题检查 | `python3 /mnt/data/yongan-admin-2/.codex/skills/share-top/scripts/topic-closeout.py .` | 发现计划/codemap/registry/test map 漂移 |

## Trunk

| Gate | Entrypoint | Coverage Purpose |
|---|---|---|
| 公共合同 | `_code/ml_framework/tests/` | envelope、seed、split、reducer、checkpoint/resume、HPO、artifact、test firewall、规范模型动态发现 |
| 六赛道便携回归 | `_wiki-methodology/_tests/_gates.yml` 中 `p4-*` gates | 无需私有数据时检查合同；PyTorch gates 要求显式设置 `P4_TORCH_PYTHON`；缺真实产物的用例必须明确 skip |
| 真实数据验收 | 按 `_run_ledger.md` 中赛道命令执行 | 真实 smoke、CV、freeze、refit、single-use test 与归档可视化 |

## Branch

| Test | Covers |
|---|---|
| `_pipelines/02_task_datasets/fault/test_fault_p4.py` | 断层标签审计、buffered spatial split、缺审核负例时 fail closed |
| `_pipelines/02_task_datasets/facies/tests/test_p4_*.py` | F3/Penobscot 独立任务、像素 mask、CV/lifecycle、real smoke |
| `_pipelines/02_task_datasets/reservoir/tests/test_p4_*.py` | PHIF/KLOGH/SW 分离 mask、物理空间指标、⑤目标6/7 adapter |
| `_pipelines/02_task_datasets/lithofacies/tests/test_p4_contract.py` | 母井家族划分、固定九类/支持类双口径、一次性 F-5 test |
| `_pipelines/02_task_datasets/sweetspot/tests/test_p4_*.py` | 七目标独立性、代理标签泄漏防火墙、可行/`not_feasible` 分流 |
| `_pipelines/02_task_datasets/reconstruction/_tests/test_p4_*.py` | strict/conditional 拆分、buffered block CV、约束审计、冻结测试 |
| 六赛道 `*p5*` 测试 | 开源adapter/source lock、真实development小批次、确定性/shape/finite、结构化skip与test firewall |

## Leaves

| Unit / Sample | Purpose |
|---|---|
| `_code/ml_framework/tests/test_canonical_track_models.py` | 六赛道 `_models/` 真源均能动态发现、训练一批、存取 checkpoint |
| `_pipelines/02_task_datasets/reconstruction/_tests/test_contract.py` | 从项目根目录直接执行 task-spec/tiny-smoke/visualization CLI |
| 各赛道 tiny-overfit/smoke | 在长训练前检查有限输出、loss 方向和检查点恢复 |

## Current Gaps

- 断层只有 3998 个审核正例点，没有覆盖已核验负例；正式 blind test/CV 不可行，不允许用随机非断层 patch 伪造负例。
- 甜点目标5缺已验证 Eclipse cell-state parser 与冻结的时间/候选井/经济约束；只能是 simulation case，不是 field truth。
- 精确 PHIE 缺独立真值；不用 LFP_PHIE 替代。
- 岩相冻结预测未持久化真实 `center_md_m`，因此深度轨迹图 `not_feasible`；其他分类/校准图已可用。
- Stage-3多seed全有效fold已执行；长预算HPO仍未执行且不是默认必做，必须先写预注册理由并仅在development CV内搜索。
- 最终全development refit和一次性frozen-test尚未执行；断层审核负例、甜点T5模拟约束和T6/T7 development-only特征源仍是硬阻塞。
- 岩相CatBoost在一个fold的三个seed均遇到NaN/Inf；重建PyKrige一个conditional cell达到300秒上限。两者均保留为真实Stage-3证据，未换预算补数。
