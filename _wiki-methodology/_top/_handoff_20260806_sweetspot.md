# Sweetspot 交接说明（2026-08-06）

本文档用于把 sweetspot P28 / P29 的真实状态交给后续 Claude / 其他 AI 接手。这里只记录已验证事实，不写计划性猜测。

> **主线定位（2026-08-07 补记）**：本文档原本写在 `track-sweetspot` 工作树内，现已随内容一并进入 `master`。
> 经逐文件哈希比对确认，P28 / P29 的**代码、测试和全部 canonical 产物在 `master` 上与 `track-sweetspot` 逐字节相同**
> （master 侧对应 cherry-pick 提交 `86f9e8c` / `d7432d2` / `eef7d92` / `8f9e058` / `e08211c`）。
> 因此下面记录的所有哈希与结论对 `master` 同样成立，直接在主仓核对即可，不需要切回那个工作树。
> 该工作树此时已落后 `master` 约 230 个提交，除本文档外无独有资产。

## 1. 源工作树状态（历史追溯用）

- 分支：`track-sweetspot`
- HEAD：`eadf82bae63901f9ee79aaefbab0e59a9686e01e`（原始文档提交；代码最后一次改动是 `3654066`）
- 工作树状态：clean
- 最近提交：
  - `eadf82b docs(sweetspot): add handoff summary`
  - `3654066 fix(p29): baseline candidate feedback on same-fold a0`
  - `b355302 fix(p28): use relative scientific source ids`
  - `002afed fix(p28): remove absolute provenance locators`

## 2. 已完成的关键工作

### P28

P28 的核心修复已完成，重点是把 canonical 产物中的源身份、路径和 manifest 变成可移植、可审计的相对表示。

当前 P28 canonical 结果：

- `verdict = reject`
- `retain_llm = false`
- `a2l.status = STOPPED`
- `A0 / A1` 预测 hash 一致

已确认的 P28 输出哈希：

- `protocol.json`
  - `be6f88487c27d661d57ca7a2cc75d1337f1d36bf1fd1b85455e29e36125fd0e6`
- `summary.json`
  - `938c76d73b5c5b002bc67bb8bbbd7c04c7e8c0b4c616c43f5eb1fdc9403e9ef4`
- `manifest.json`
  - `b51690c25b3b5ac8bf90f77889dd1e20b93160b88f8373a4dfd953e24b684e6c`

### P29

P29 的反馈基线 bug 已修复并重新验证。

修复要点：

- candidate selection MAE 现在比较的是 `same_fold_same_executor_a0`
- prompt 只暴露安全的 `signed_normalized_delta` 和 `remaining_budget_trials`
- selection / promotion 仍然保持 disjoint
- A2D 与 A3 是独立控制
- A0 / A1 identity replay hash 仍然一致

当前 P29 canonical 结果：

- `verdict = REJECT_AGENT`
- `retain_llm = false`
- `a2l.status = STOPPED`
- `a2l.stop_requested = true`
- `baseline_kind = same_fold_same_executor_a0`
- `A0 / A1` 预测 hash 一致

已确认的 P29 输出哈希：

- `protocol.json`
  - `d763dc728631c7c9519c6066e01c14b1d71153c3cb6549c0842495af8dda3770`
- `protocol.jsonl`
  - `7b44562ec8d5bc18b2f4a8af2814d2e254c52b7193bb58753c0e5cb8f081de7b`
- `action_effects.json`
  - `0a3008fa87497c9e77b94993822b7c7c9b288409effda97132df2487b1536e77`
- `root_cause.md`
  - `43750126c6c7cf6cd26696d5163c3d22270f7af03a94e012905ae2986cd564cb`
- `summary.json`
  - `d4f45ea9b276639c000d6a38c802c1304052c78f8957059ffaf418e976642270`
- `evidence.md`
  - `851489576daf828ba5d01e517c7bff089d084a52f09fe80c6be23951c95129f1`
- `manifest.json`
  - `0db94cb04790a313f5d495c6cb5d9fa300a379c5298ac41783f6b843612d5c41`

## 3. 现有可直接接手的文件

### 代码

- `_pipelines/02_task_datasets/sweetspot/p28_agentic_optimization.py`
- `_pipelines/02_task_datasets/sweetspot/p29_agent_action_effect.py`

### 测试

- `_pipelines/02_task_datasets/sweetspot/tests/test_p28_agentic_optimization.py`
- `_pipelines/02_task_datasets/sweetspot/tests/test_p29_agent_action_effect.py`

### 产物

- `_pipelines/02_task_datasets/sweetspot/_outputs/p28_agentic_optimization/`
- `_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/`

## 4. 已验证的运行状态

### 🔴 运行环境（必读，否则测试必挂）

P28 / P29 依赖 `xgboost`，**系统 `/usr/bin/python3` 没有装**。直接用 `python3 -m unittest`
会得到 `ModuleNotFoundError: No module named 'xgboost'`，P28 挂 2/5、P29 挂 3/4，
这是环境问题，不是代码回归。

本机确认可用的解释器（py310，`sklearn 1.7.2` / `xgboost 3.2.0`，与产物生成环境一致）：

```
/mnt/data/yongan-admin-2/envs/geocfc-train/bin/python
```

（`gaia-v2-8a4915-py312`、`gaia-petro-inference-mcp-py312` 也有 xgboost，但 sklearn 是 1.8.0，
版本不同，复现 canonical 产物前需自行验证。）

### ✅ 运行位置限制已修复（2026-08-07）

原先即使解释器带 xgboost，**在主仓 checkout 上跑这两套测试仍然失败**：路径常量按 worktree
目录深度硬推，主仓上 `PROJECT_ROOT` 解析成 `/mnt/data`、`REFERENCE_ROOT` 指向不存在的
`projects/p10-results-sweetspot`，`_reference_inputs()` 在 `git rev-parse HEAD` 抛
`CalledProcessError`（实测 P28 errors=2、P29 errors=3）。之前几轮 "remove absolute
provenance locators" 只清理了产物侧，源码侧的推导没动，而测试一直在 worktree 里跑，缺陷被布局掩盖。

已在分支 `fix-sweetspot-path-derivation` 修复：三个根改为从 git 反查
（`rev-parse --show-toplevel` / `--git-common-dir`），`REFERENCE_ROOT` 按
`$SWEETSPOT_REFERENCE_ROOT` → 兄弟 worktree `p10-results-sweetspot` → 主仓 依次回落。
兄弟 worktree 因此从硬依赖变成可选，删掉它 P28/P29 仍可运行。

行为等价性已实测两轮：

- legacy worktree 存在的现状下，产物与修复前逐字节一致，唯一差异是脚本自身的 `renderer.sha256`
- 强制切到主仓回落时，全部科学产物仍逐字节一致，差异只有 8 处 `inputs[*].source_commit`
  及由它级联的 manifest / protocol 哈希；`inputs[*].sha256` 不变，即两处参考数据字节相同

### 已跑过且通过的项目级检查

- `python3 ~/.codex/skills/share-top/scripts/topic-brief.py .`
- `<geocfc-train-python> -m unittest _pipelines.02_task_datasets.sweetspot.tests.test_p28_path_resolution` → Ran 9 tests, OK
- `<geocfc-train-python> -m unittest _pipelines.02_task_datasets.sweetspot.tests.test_p28_agentic_optimization` → Ran 5 tests, OK（约 95s）
- `<geocfc-train-python> -m unittest _pipelines.02_task_datasets.sweetspot.tests.test_p29_agent_action_effect` → Ran 4 tests, OK（约 245s）
- `python3 -m pytest _pipelines/02_task_datasets/sweetspot/tests/test_validate_only.py -q` → 11 passed
- `git diff --check`

以上三套 unittest 在 `.claude/worktrees/fix-sweetspot-paths`（worktree 布局）和
`~/tmp/20260807-sweetspot-mainlayout`（**非 worktree 布局**，即修复前完全跑不了的场景）
各跑一遍，结果相同。

当前检查结果：

- canonical 产物里没有机器绝对路径或 `.claude/worktrees/...` 兄弟工作树定位
- 源码侧的根推导也不再绑定布局，并有 `test_p28_path_resolution.py` 兜底
  （含一条防止退回 `parents[N]` 写法的守卫）
- 工作树 clean

## 5. 接手建议

如果后续 Claude 要继续这个工作，建议顺序是：

1. 先读本文件，确认当前不是“待修复状态”，而是“已完成可交接状态”。
2. 如果要继续 sweetspot，先确认新目标是新 phase / 新 goal，而不是重复修 P28/P29。
3. 不要回滚 P28/P29 的 canonical 输出，它们现在是已验证的历史证据。
4. 如果要改新的 pipeline 或新阶段，先看 `_wiki-methodology/_top/_task_plan.md` 和对应 phase 规范，再动 owner path。

## 6. 备注

- 本文档只总结已验证事实，不包含未执行计划。
- 当前没有运行中的修复任务。
- 如果接手方需要继续，可直接基于现有 clean HEAD 开新 goal。
- 已知缺口：`manifest.json` 记录了输入哈希和 `source_commit`，但**没有记录 python 解释器
  与 xgboost / sklearn 版本**。如果后续换环境重跑，产物哈希可能变化却无法归因。
  建议在下一次改 P28/P29 时把 runtime 版本写进 manifest。
- 已修复（2026-08-07，分支 `fix-sweetspot-path-derivation`）：源码路径推导原先按 worktree
  目录深度硬推，模块在主线 checkout 上等同不可运行，且隐式依赖兄弟 worktree
  `p10-results-sweetspot` 存在。现已改为从 git 反查、参考数据位置可配置可回落，
  行为等价性经两轮逐字节比对确认，细节见第 4 节。
- 归档的 canonical 产物**未**重新生成，仍是原始证据；本次修复不触碰它们。

