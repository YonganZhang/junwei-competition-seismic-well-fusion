---
phase_id: P43
status: accepted
severity: major
owner_col: COL4
source: audit
created_at: 2026-08-07
---

# 甜点赛道七目标门限根因：T6/T7 卡在样本身份，T2 是 proxy 而非含油气

## Local Case

⑤甜点赛道的统一 Pipeline 七阶段 `verify` 全部 `exit_code=0`，`status=PASS`，但 P5 Stage-3
的七个目标只有四个 `rankable`。本轮对两处门限做了只读根因审计，结论与门限的字面表述不完全一致。

### T6/T7：不是"没有特征源"，是样本身份不可逆

Stage-3 给 T6/T7 的 `reason` 是 `no development-only feature source exists; materialized
test fallback is forbidden`，`load_development_pilot_data` 对这两个目标是**硬编码 raise**
（`sweetspot_p5_stage2_data.py:683-687`），不做任何探测。

实测表明标签源本身完全可得：

- `_load_development_petrophysical_tables` 已经在读 development 井族的 CPI LAS，并且
  第 277 行显式要求 `{"PHIF","KLOGH"} <= set(output.curves)`，不满足就跳过。
- 也就是说 T6 的标签 `PHIF` 和 T7 的标签 `KLOGH` **现在就在被读取**——T1 的 RQI
  （`rqi-0.0314-sqrt-klogh-over-phif-v1`）正是由这两条曲线算出来的。
- 在 development 井族上统计，`PHIF` 与 `KLOGH` 各有 **35810** 个有限值样本。

真正的阻塞是样本身份：

| | T1 | T6 / T7 |
|---|---|---|
| `development_sample_ids` 条数 | 35810 | 1216 |
| ID 形态 | `rqi:15-9-F-1-A:3429.4000`（井+深度）| `target6_porosity_phif-fb42abff337e7ed765fb`（20 位 hex）|
| `development_rebuild` | `raw_source_development_groups_only` | `requires_existing_development_only_feature_source` |
| 可从原始源重建 ID | 是 | **否** |

P4 时代 T6/T7 的样本身份是内容哈希，随物化的 `train.h5`/`test.h5` 一同冻结；那些文件属于
frozen test，禁止打开。因此重建出的 `{target}:{well}:{depth}` 与 manifest 里的哈希 ID
永远无法对齐，`rebuilt_ids != expected_ids` 必然触发，硬编码 raise 只是把这个必然结果提前。

另有一处数据一致性问题：T6/T7 的 `development_groups` 声明了 4 口井，比 T1 多一口
`15/9-19`。该井在归档中确有 3 张含 `PHIF`+`KLOGH` 的 CPI 表，但其授权成员只有 4 个
（3 张 CPI + 1 张 `KLOGH_NEW`），**没有配套的原始测井曲线**。即使解除闸门，它也无法进入
以 `RAW_LOG_FEATURES` 构造的特征矩阵——实测传入 4 口井与传入 3 口井得到的表集合完全相同。

### T2：AP 0.985 不是含油气预测的成绩

Stage-3 的 T2 三个模型 average precision 落在 `0.9813`–`0.9847`，worst fold 仍有 `0.9584`。
这不是标签泄漏，输入白名单里没有标签字段；但它也不是"含油气/有效厚度"的成绩：

- `label_mapping` 记 `targets.T2.is_proxy = True`，
  `proxy_semantics = "SAND_FLAG net-reservoir/sand proxy, not direct hydrocarbon-pay truth"`。
- R0 合同 `T2.v1.json` 记 `truth_class: "proxy"`、`field_truth: false`、
  `semantic_name: "net_reservoir_sand_proxy"`，并带两条 warning 明说它不是 hydrocarbon pay。
- 标签是 `near_binary(SAND_FLAG)`，输入是 16 条原始曲线（GR/RHOB/NPHI/RT/DT/…）。
  `SAND_FLAG` 本身就是解释人员依据这些曲线按截断规则给出的产物，所以模型实际在**逆向拟合
  一条确定性解释规则**，高 AP 是任务性质决定的，不构成地质预测能力证据。

系统对此已如实标注，问题只会出现在引用环节。

## Class Pattern

门限的 `reason` 字符串是给人读的摘要，不等于机器可核验的根因。当摘要说"某某源不存在"而代码是
无条件 raise 时，二者都不能证明源真的不存在——需要独立跑一次数据侧探测才能区分"源缺失"、
"身份不可对齐"和"下游拒绝"。

同类还有：代理标签一旦进入排行榜，其 `is_proxy` 标注留在元数据里，而指标数值会被单独摘出来引用。
标注与数值分离到不同文件时，误引用只是时间问题。

## Evidence

- `_pipelines/02_task_datasets/sweetspot/p5/sweetspot_p5_stage2_data.py:250-290, 349-380, 675-690`
- `_pipelines/02_task_datasets/sweetspot/p5/sweetspot_p5_stage3.py:765-775`
- `_pipelines/02_task_datasets/sweetspot/p5/sweetspot_p5_label_mapping.v1.json`（`targets.T1/T2/T6/T7`）
- `_pipelines/02_task_datasets/sweetspot/p5/r01/contracts/T2.v1.json`
- `_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage3_cv/leaderboards/T1..T7.json`
- 数据侧探测（只读，未落盘）：development 井族 `PHIF`/`KLOGH` 各 35810 个有限值；
  `15/9-19` 授权成员 4 个，无原始曲线；传入 3 口井与 4 口井得到相同表集合。

## Impact

- 解锁 T6/T7 不是"接上一个特征源"的工程动作。可重建的只有标签和特征，**样本身份必须重新定义**，
  即以 `{target}:{well}:{depth}` 形态重做 split manifest 并重新冻结。这会切断与
  `target6-phif-cpi-v1` / `target7-klogh-cpi-v1` 当前 split 绑定关系，属于科学决策，不是实现细节。
- 重做后可用样本量级从 1216 变为 35810，与 T1 同源同粒度，这也意味着 T6/T7 将与 T1 高度相关
  （RQI 由 PHIF 和 KLOGH 定义），独立性需要在合同层重新论证。
- `15/9-19` 需要单独决策：补原始曲线、或从 T6/T7 的 `development_groups` 中移除。保留现状会让
  声明的井数与实际参与建模的井数长期不一致。
- T2 的 `0.9847` 若被写进任何对外材料，必须同时写明它是 `SAND_FLAG` 代理任务，
  否则等同于宣称含油气识别接近完美。

## Prevention Rule (candidate)

排行榜条目在渲染时应带出 `is_proxy` 与 `truth_class`，让代理标签的指标无法脱离其语义被单独引用。
门限 `reason` 若声称某类数据源不存在，应由一次真实探测产生，而不是写死在分支里。

## Links

- task_plan: ../_task_plan.md
- 前序: P42_six_track_progress_and_claude_handoff.md（把⑤列为第二优先级，指出瓶颈在数据与标签门）
- 七目标合同: ../_phases/P4_sweetspot_seven_target_contract.md
