# reservoir / track-property 交接说明（2026-08-06）

> **主线定位（2026-08-08 补记）**：本文件原为 `track-property` 工作树内一个**从未提交**的未跟踪文件，
> 现取入主线归档，与 `_handoff_20260806_sweetspot.md` 并列。它描述的是 2026-08-06 的状态，
> 此后主线已合入 P47--P53 的大量内容，**当前状态以 `_task_plan.md` 与各赛道真源为准**。
> 同工作树的其余未提交项经比对均无需集成：`_models/` 12 个文件是 08-01 旧版（主线已含更新的
> loss/activation 机制），`p5_contract.py` 与主线相同，`p18` evidence 差异仅 request_id。


这份文档写于 2026-08-06，记录 `track-property` 工作树在当时已确认的记录、已完成产物、已知阻塞与下一步入口。它是历史快照；实时进度见 `_task_plan.md`。

## 1. 2026-08-06 当时的仓库状态（历史追溯用，当前状态见 `_task_plan.md`）

- 当前分支：`track-property`
- 当前 HEAD：`54cae62ee4bf92b02deffab517b5f2c55f65802c`
- 当前工作树仍是脏的，`git status --short` 看到的未提交项：
  - 已修改：`_pipelines/02_task_datasets/reservoir/_outputs/p18_cigbench_property/evidence.md`
  - 未跟踪：`_models/`
  - 未跟踪：`_pipelines/02_task_datasets/reservoir/p5_contract.py`
  - 未跟踪：`_pipelines/02_task_datasets/reservoir/p5_stage1.py`

如果要继续开发，先决定这些未跟踪项是要保留为本地草稿、补测试后纳入提交，还是由接手者重新整理后再提交。

## 2. 我检查过的主要记录

我已经实际阅读/核对过下面这些记录，不是只看文件名：

- 仓库状态：`git status`、`git log --oneline`
- 赛道说明：`_pipelines/02_task_datasets/reservoir/README.md`
- P18 证据：`_pipelines/02_task_datasets/reservoir/_outputs/p18_cigbench_property/evidence.md`
- P5 赛道合同草案：`_pipelines/02_task_datasets/reservoir/p5_contract.py`
- P5 Stage1 入口：`_pipelines/02_task_datasets/reservoir/p5_stage1.py`
- P5 共享约束：`_models/property/_p5_common.py`
- P5 模型锁：`_models/property/source_lock.json`
- 近几个已完成产物目录：
  - `_pipelines/02_task_datasets/reservoir/_outputs/3d_sci_v1/`
  - `_pipelines/02_task_datasets/reservoir/_outputs/agent_chapter/`
  - `_pipelines/02_task_datasets/reservoir/_outputs/p28_agentic_optimization/`
  - `_pipelines/02_task_datasets/reservoir/_outputs/p29_agent_action_effect/`
  - `_pipelines/02_task_datasets/reservoir/_outputs/p18_cigbench_property/`

## 3. 2026-08-06 当时已确认的进度

### 3.1 3D 科学可视化（已完成）

目录：`_pipelines/02_task_datasets/reservoir/_outputs/3d_sci_v1/`

已确认的事实：

- 使用了真实 sample 点坐标，不是 trajectory/volume。
- 记录里明确写了：
  - `inline`
  - `crossline`
  - `time_ms`
- `three_d_feasibility.json` 标明：
  - `uses_real_sample_points = true`
  - `trajectory_used = false`
  - `volume_used = false`
- 产物包括：
  - `phif_spatial_context.png/.pdf`
  - `klogh_spatial_context.png/.pdf`
  - `sw_spatial_context.png/.pdf`
  - `spatial_context.html`
  - `caption.md`
  - `provenance.json`

如果后续要继续可视化，这一块已经不是“是否可做”的问题，而是“是否要继续优化版式/补充展示”的问题。

### 3.2 智能体分析章节（已完成）

目录：`_pipelines/02_task_datasets/reservoir/_outputs/agent_chapter/`

证据文件：`evidence.md`

已记录内容：

- 实际定位了当前真实产物文件
- 抽取了当前模型和评测结果
- 生成了 DeepSeek 常识分析的结构化结论

对后续接手者最重要的是：这里已经形成了“可解释建议”证据，不要把它当成单纯的文本草稿。

### 3.3 P28 / P29 结果修正线（已完成到当前记录）

目录：

- `_pipelines/02_task_datasets/reservoir/_outputs/p28_agentic_optimization/`
- `_pipelines/02_task_datasets/reservoir/_outputs/p29_agent_action_effect/`

我已经核对到的关键事实：

- P28 里明确保留了 `CIG-Bench` 的阻塞结果，`gate.status = blocked`
- P28 / P29 的协议里都保留了：
  - A0 / A1 相关记录
  - matched budget / identity replay 语义
  - 以真实 dev-only 结果为准
- P29 的协议明确写了：
  - `historical_reference` 只是非因果参考
  - `primary_metric` 的因果比较状态是 `insufficient_for_primary`
  - `implemented_causal_metric` 是 `physical_MAE_macro`

如果后续要继续 P28/P29，不要再把历史 checkpoint 当作因果比较主对象。

### 3.4 P18 CIG-Bench 可行性检查（当前仍是阻塞态）

证据文件：`_pipelines/02_task_datasets/reservoir/_outputs/p18_cigbench_property/evidence.md`

当前结论是：

- `Verdict: BLOCKED_DATA_OR_API`
- 已确认的 API / 包信息：
  - `cig_bench 0.2.0`
  - `modelscope 1.39.0`
  - `torch 2.13.0`
  - `PropertyPredictor` 的签名和 registry 条目都已查到
- 但实际阻塞原因仍然成立：
  - ModelScope 默认 checkpoint 下载失败
  - 开发集的 `seismic_patch + well_log_seq` 不是 `PropertyPredictor` 需要的合法输入契约

因此，P18 目前不能被当成“已经完成的可比较基线”，只能当成“失败原因已被证实”的记录。

### 3.5 P5 赛道草案与 Stage1 框架（当前为未跟踪草稿）

未跟踪文件：

- `_pipelines/02_task_datasets/reservoir/p5_contract.py`
- `_pipelines/02_task_datasets/reservoir/p5_stage1.py`
- `_models/property/_p5_common.py`
- `_models/property/source_lock.json`
- 以及同目录下的 10 个模型适配器

我已经读到的关键信息：

- `p5_contract.py` 已经把三目标固定为：
  - `PHIF`
  - `KLOGH`
  - `SW`
- 目标变换为：
  - `PHIF`: identity
  - `KLOGH`: `log1p / expm1`
  - `SW`: identity
- 输入白名单/禁用输入已经写在合同里
- `p5_stage1.py` 是一个真实的 Stage1 contract-smoke runner：
  - 只吃显式 `train.h5` 和 `guard.npz`
  - 不允许读取 frozen test
  - 会做 checkpoint roundtrip
  - 会做 same-seed replay deterministic check

`source_lock.json` 当前的事实也很重要：

- TabICLv2 的代码来源已经锁定到 `soda-inria/tabicl`
- 但 `weights.license_status` 仍然是 `unconfirmed`
- checkpoint 名称是 `tabicl-regressor-v2-20260212.ckpt`

如果后续 Claude 要继续 P5，先不要假设 TabICLv2 权重已经可直接用；它在锁文件里仍然是“未确认”状态。

## 4. 当前最值得接着做的事

我建议后续 AI 按这个顺序处理：

1. 先决定当前工作树里这 4 个未提交项的去留：
   - 修改的 `p18` evidence
   - 新增的 `_models/`
   - 新增的 `p5_contract.py`
   - 新增的 `p5_stage1.py`
2. 如果继续 P5：
   - 先补测试，再决定是否纳入提交
   - 严格保持 `train.h5 / guard.npz` 之外不碰 frozen test
3. 如果继续 P18：
   - 先接受它当前就是 `BLOCKED_DATA_OR_API`
   - 不要把它写成“只是没跑完”
4. 如果只需要交接：
   - 直接从这个文档 + 最近 commit log 开始，不必再重扫整个仓库

## 5. 适合接手者直接执行的起点命令

```bash
git status --short
git log --oneline --decorate -n 12
sed -n '1,220p' _pipelines/02_task_datasets/reservoir/README.md
sed -n '1,220p' _pipelines/02_task_datasets/reservoir/_outputs/p18_cigbench_property/evidence.md
```

如果要继续 P5，建议先看：

```bash
sed -n '1,220p' _pipelines/02_task_datasets/reservoir/p5_contract.py
sed -n '1,260p' _pipelines/02_task_datasets/reservoir/p5_stage1.py
sed -n '1,220p' _models/property/source_lock.json
```

## 6. 一句话状态（截至 2026-08-06）

当前仓库不是“空白待开工”，而是已经有一批真实产物、阻塞证据和未跟踪草稿；接手时优先处理未提交项和 P18/P5 的边界，不要重复造证据。

