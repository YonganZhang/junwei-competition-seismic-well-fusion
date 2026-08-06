# ⑤ 甜点预测：validate-only 数据就绪 pipeline

## 当前状态

本目录只实现标签决策合同和真实字段审计。Volve 和现有 Layer1 产物没有经批准的 sweetspot 真值，因此当前不允许生成标签、`train.h5`/`test.h5`、模型、checkpoint 或指标。

P5 的十候选模型接入骨架位于 [`p5/`](p5/README.md)。它只增加 source lock、七目标独立 TaskSpec/adapter 和 development-only Stage-1 gate；当前所有适配单元仍会在未批准标签合同处结构化 `SKIP`，不会把 P4 代理实现自动升级为 P5 真值。

`build_dataset.py` 的命名是为了与六赛道未来公共接口保持一致；当前文件刻意只有 `audit` 和 `validate-only` 两种模式，不导入 `_code/dataset_io.py`，也没有数据写入路径。

## 文件

| 路径 | 职责 |
|---|---|
| `label_spec.schema.v1.json` | 版本化 JSON Schema；未知属性直接拒绝 |
| `label_spec.template.v1.yml` | `approved=false` 的决策模板，不是标签定义 |
| `build_dataset.py` | 真实数据审计和 fail-closed 合同闸门 |
| `candidate_label_schemes.md` | 三种候选方案的证据与风险，不给阈值/权重 |
| `audit/data_availability.json` | 机器可读的真实字段、覆盖率、缺失和坐标对齐证据 |
| `audit/data_readiness.md` | 人工审阅摘要 |
| `audit/contract_validation.json` | 最近一次合同闸门结果；当前记录草案被 fail closed |
| `_outputs/agent_chapter/evidence.md` | 赛道末尾的智能体分析章节；仅汇总真实结果与 DeepSeek 常识性建议 |
| `tests/test_validate_only.py` | 合同闸门和“绝不创建 sweetspot 数据集”回归测试 |

## 重跑真实数据审计

从项目根目录执行：

```bash
python3 _pipelines/02_task_datasets/sweetspot/build_dataset.py --mode audit
```

脚本会在 worktree 中自动解析 Git 主仓，从主仓读取共享 `_sandbox` 和被 Git 忽略的 `well_logs_clean.h5`。它只覆写本目录下的两份审计报告。

审计范围：

- Layer1 地震索引、断层点、BCU 层位点、三井弱标定和清洗后测井 HDF5。
- 三份 LFP LAS 的真实曲线头和跨井字段覆盖。
- Volve 日/月生产表的列名、非空率。
- 断层、BCU 和弱井震标定在当前 inline/crossline/TWT 地震网格中的对齐覆盖。

## 合同验证

直接验证模板必然失败（退出码 `2`），因为它保持 `approved=false` 且包含待决策项：

```bash
python3 _pipelines/02_task_datasets/sweetspot/build_dataset.py \
  --mode validate-only \
  --spec _pipelines/02_task_datasets/sweetspot/label_spec.template.v1.yml
```

军伟/领域专家完成并签批一份独立合同后，准确的下一步命令是：

```bash
python3 _pipelines/02_task_datasets/sweetspot/build_dataset.py \
  --mode validate-only \
  --spec <APPROVED_LABEL_SPEC.yml>
```

只有这条命令退出 `0` 才表示“合同与真实数据字段对得上”；它仍然不会生成标签或数据集。

## 最小必批字段

完成的 `label_spec` 必须明确：

1. 目标语义：地质、工程、生产或联合。
2. 输出类型、类别/物理意义。
3. 允许源字段的审计 source id 和精确字段名。
4. 公式及其字段引用。
5. 阈值、权重及其拟合域；若不适用必须写明理由。
6. 时间窗口与泄漏截止点。
7. 空间支撑尺度、坐标系、垂向域、分辨率与对齐容差。
8. 正样本、负样本和未标注样本规则。
9. 井/空间 split 规则、train-only 统计域和泄漏防护。
10. 推理期允许输入和评估指标。
11. 批准人、角色、时间、决策记录和 spec 版本。

以下任一情况都会 fail closed：缺 spec、`approved=false`、未知 source/字段、使用 test 统计、缺负样本规则、缺空间尺度、缺完整 split，或合同仍含占位符。

## 批准后的未来接口

这一部分只定义边界，尚未实现：

1. 未来 builder 必须先重跑 `audit`，再读入已通过闸门的 spec。
2. builder 只能使用 `allowed_source_fields`，严格按 `class_rules`、`time_window`、`spatial_scale` 和 `split_strategy` 构造样本。
3. 任何拟合阈值/权重只能使用 train split；未标注样本不得默认当负样本。
4. 通过独立授权的实现任务后，builder 才可调用 `_code/dataset_io.py` 的公共写入接口。
5. 未来模型的数据输入必须是 `inference_allowed_inputs` 子集，不得偷看 `label_only` 字段。

## 测试

合同闸门类测试只依赖标准科学栈，系统解释器即可：

```bash
python3 -m pytest _pipelines/02_task_datasets/sweetspot/tests -q
```

测试中的 `approved` spec 是纯合同闸门 fixture，明确标注为非真实批准；它不产生样本、标签或 HDF5。

### P28 / P29 的运行前提（2026-08-07 实测）

`p28_agentic_optimization.py` / `p29_agent_action_effect.py` 只有一个运行前提。
报错前请先核对本节，多数失败不是代码回归。

**前提：解释器必须带 `xgboost`。**
系统 `/usr/bin/python3` 没有装，会报 `ModuleNotFoundError: No module named 'xgboost'`
（P28 挂 2/5，P29 挂 3/4）。本机已验证可用（py310，`xgboost 3.2.0` + `sklearn 1.7.2`，
与 canonical 产物生成环境一致）：

```bash
PY=/mnt/data/yongan-admin-2/envs/geocfc-train/bin/python
```

本机另有 `envs/gaia-v2-8a4915-py312`、`envs/gaia-petro-inference-mcp-py312` 带 xgboost，
但 sklearn 是 1.8.0；用它们复现 canonical 产物哈希前需自行验证版本影响。

### 运行位置（2026-08-07 起不再受限）

早期版本的路径常量按目录深度硬推，只在 `.claude/worktrees/<name>/` 布局下成立：在主仓
checkout 上 `PROJECT_ROOT` 会解析成 `/mnt/data`、`REFERENCE_ROOT` 指向不存在的目录，
`_reference_inputs()` 直接以 `CalledProcessError` 中止。现在三个根一律由 git 反查：

| 常量 | 来源 | 含义 |
|---|---|---|
| `WORKTREE_ROOT` | `rev-parse --show-toplevel` | 当前 checkout（主仓或任意 worktree） |
| `PROJECT_ROOT` | `rev-parse --git-common-dir` 的父目录 | 主仓，共享数据 `_sandbox/volve_data` 所在 |
| `REFERENCE_ROOT` | 见下 | P5/P7/P8 参考产物所在 |

`REFERENCE_ROOT` 按顺序解析，命中即止：

1. 环境变量 `SWEETSPOT_REFERENCE_ROOT` — 显式覆盖
2. 兄弟 worktree `p10-results-sweetspot`（若仍存在）— 排在前面是为了让 provenance 与归档
   manifest 保持连续，那些 manifest 把该 worktree 的 commit 记作每个输入的 `source_commit`
3. 主仓 — 被引用的参考产物全部在版本控制里，且与兄弟 worktree 的副本逐字节相同

所以 `p10-results-sweetspot` 现在是可选的：删掉它 P28/P29 照常运行，只是 `source_commit`
会记成主仓 commit 而不是 `1c81a03`。已实测：切换 `REFERENCE_ROOT` 时全部科学产物逐字节不变，
差异只出现在 `source_commit` 及由它级联的 manifest 哈希。

已验证的跑法（**任意 checkout 位置均可**，含主仓）：

```bash
$PY -m unittest _pipelines.02_task_datasets.sweetspot.tests.test_p28_path_resolution      # 9 tests OK
$PY -m unittest _pipelines.02_task_datasets.sweetspot.tests.test_p28_agentic_optimization # 5 tests OK, ~95s
$PY -m unittest _pipelines.02_task_datasets.sweetspot.tests.test_p29_agent_action_effect  # 4 tests OK, ~245s
```

⚠️ 剩余已知缺口：`_outputs/*/manifest.json` 记录了输入哈希和 `source_commit`，
但没有记录解释器与依赖版本；换环境重跑若哈希变化，目前无法归因。
