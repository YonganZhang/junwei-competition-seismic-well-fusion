---
phase_id: P54
status: accepted
severity: major
owner_col: COL4
source: audit
created_at: 2026-08-08
---

# 79 个 worktree 里有 220 个文件从未进过主线，其中 P5.2 R2 是「有结果无代码」

## Local Case

项目有 79 个 worktree、占 45GB。P42 曾判断旧 track 分支「有效内容已由 P31–P34 选择性进入主仓，
再次合并只会引入重复历史或覆盖新接口」，此后这些工作区一直搁置。

本轮对全部分支做了逐文件核对：取每个分支自分叉点以来的贡献（`git diff master...<branch>`），
再逐个比较 blob 哈希，判断该文件是「未进主线」「已集成」还是「同路径内容分歧」。

结果与「内容都已进主线」的印象不符：

| | 数量 |
|---|---|
| 未进主线的文件 | **220** |
| 同路径内容分歧 | 134 |
| 涉及分支 | 52 |

其中 46 个是 `.py`，不是可再生的图件。类型分布：`.py` 46、`.png` 24、`.json` 23、
`.md` 11、`.csv` 8、`.xlsx` 4。

P42 的判断对**共享文件**成立——那些分支落后 165–247 个提交，合并会把 pre-P31 版本盖回当前
adapter/registry/lifecycle 接口。但它被当成了「这些分支没东西」，于是 220 个纯新增文件一并被搁置。
两件事不同：**不该 merge 分支** ≠ **分支里没有该拿的东西**。

### 最严重的一处：P5.2 R2 有结果无代码

主线有 26 项 R2 产物——`_outputs/p5_r2_visualization/figure_02_r2_scientific_results.pdf`、
`LARGE_MODEL_ROADMAP.md`、六赛道 `source_snapshots/*.json`——以及可视化脚本
`_code/visualization/p5_r2_six_track.py`。但产生这些结果的六赛道实现**一行都不在主线**：

| 实现 | 体量 | 来源分支 |
|---|---|---|
| `facies/facies_p5_r2.py` | 91KB | `p5-r2-facies` |
| `reconstruction/reconstruction_p5_r2.py` | 65KB | `p5-r2-reconstruction-v2` |
| `reservoir/reservoir_p5_r2.py` | 49KB | `p5-r2-property` |
| `lithofacies/lithofacies_p5_r2.py` | 37KB | `p5-r2-lithofacies` |
| `fault/fault_p5_r2.py` | 25KB | `p5-r2-fault-v2` |

R2 的科学结果在主线上可被引用，却无法从主线复现。

### 其余未集成的主要内容

- `p6-foundation-reprogramming-pilot`（23 个未进）：`_code/foundation/timellm_reprogrammer.py`、
  `_models/{lithofacies,property}/gaia_timellm_gpt2.py`、整个 `_pipelines/04_foundation_adaptation/`
  目录。主线**零产物零代码**，是一条 2026-07-26 的完整探索线；7-28 后项目转向 SAM-Med3D /
  TabICLv2 / MOMENT / Chronos-2 等基础模型，该线未被取代也未被集成。
- `p10-results-integration`（30 个未进）：`_outputs/model_results_summary/` 的六赛道汇总
  xlsx、`human_summary.md`、`build_six_track_summary.py` 及测试。
- `p6-gaia-*`、`p10-results-*`、`p12-viz-*`：各 5–25 个，多为图件与 manifest。

## Class Pattern

「分支落后很多、不该合并」和「分支里没有未集成内容」是两个独立判断，把前者当后者会让工作静默丢失。
可靠判据不是 `ahead` 提交数（内容可能已被 cherry-pick，计数仍然很大），而是**逐文件比较 blob 哈希**。

同类风险：产物与其生成代码分处不同 checkout 时，产物先被集成、代码留在原地，结果是主线拥有
可引用但不可复现的科学结论。

## Evidence

- 逐分支逐文件核对（只读）：52 个分支有未集成内容，明细见本 finding 上表
- 主线 R2 产物：`git ls-tree -r master --name-only | grep -i p5_r2` → 26 项
- 主线 R2 实现：仅 `_code/visualization/p5_r2_six_track.py`
- 集成后在 master 基线上的验证：12 个文件全部编译通过；5 个模块全部可 import
  （facies、lithofacies 需 torch）；无跨 worktree 路径引用

## Impact

本轮已集成 P5.2 R2 的六赛道实现与测试（按文件取，不 merge 分支，全部路径主线原本不存在，
无覆盖风险）。三处已知适配缺口如实保留，未通过改测试掩盖：

1. `reservoir` 测试 8/10 通过。`test_r2_contract...historical_source_lock_mismatch` 期望 `True`
   实得 `False`——它比较的历史源在主线上已存在，该断言编码了分支局部环境。
2. 同文件 `test_torch_adapter_supports_mae_bounded_roundtrip` 抛 `KeyError: 'loss_name'`——
   主线 property adapter 不再返回该键。**这是真实的接口漂移，需要决策而不是改测试。**
3. `lithofacies` 的 InceptionTime 需可选依赖 `tsai`；模块本身已通过
   `OptionalDependencyUnavailable` 优雅降级，只有测试是强依赖。

`fault` 4/4 通过。

顺带修复：`facies_p5_r2.py` 的 `EXPECTED_GPU_LOCK` 原为写死的本机绝对路径，且对 CUDA 运行强制
校验，导致模块只能在本机跑。该 flock 护栏在 8 卡共享机上是正确设计，因此保留强制校验，
只把路径改为 `VOLVE_P5_GPU_LOCK` 可覆盖，默认值逐字符不变。

尚未集成、待后续决策：`p6-foundation-reprogramming-pilot` 的 TimeLLM 线、
`p10-results-integration` 的汇总产出、以及 134 处同路径内容分歧。
3 个工作区有未提交改动（`p5-r2-reconstruction-v2` 19 处、`track-property` 5 处、
`p12-viz-property` 1 处），属其他会话在做的工作，未触碰。

## Prevention Rule (candidate)

判断一个分支是否还有未集成工作，用逐文件 blob 哈希比较，不用 `ahead` 计数，也不用
`git diff` 的 two-dot 形式（后者把主线自身的更新混进结果）。
产物进入主线时，其生成代码应同批进入；只有产物没有代码的目录应被视为可复现性缺口。

## Links

- task_plan: ../_task_plan.md
- 前序: P42_six_track_progress_and_claude_handoff.md（判断旧 track 分支不必再合并）
- registry: `_meta/_registry.yml` 的 `six_track_p5_r2_implementations`
