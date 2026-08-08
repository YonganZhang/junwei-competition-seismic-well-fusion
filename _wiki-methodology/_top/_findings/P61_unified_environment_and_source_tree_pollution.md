---
phase_id: P61
status: accepted
severity: major
owner_col: COL3
source: audit
created_at: 2026-08-09
---

# 统一运行环境建立；测试写源码树导致「通过率」长期不可比

## Local Case

本机原有 6 个候选解释器，**没有任何一个能跑全六赛道测试**，依赖散在不同 env：
torch 在 `volve-chronos2`、openpyxl 在 `gaia-v2`、lasio/tsai/plotly 各有缺失。
后果是不同会话报出的 "984 passed" / "863 passed" 之类数字**互不可比**——解释器不同、
错误类型不同、还叠加了下面的源码树污染。

### 统一环境 `envs/geo-sixtrack-py311`

`uv venv --python 3.11`，不继承 system site-packages、不链式继承任何 `.cache` 目录。
体积 6.1G。清单落在项目根 `requirements-sixtrack.txt`。

硬钉版本及其**约束来源**（不是随手选的）：

| 包 | 版本 | 为什么是这个值 |
|---|---|---|
| `scikit-learn` | `1.7.2` | 三个约束的唯一交点：① `1.9.0` 会让 `fault/_outputs/runs/audited_v2/baseline_model.joblib` 反序列化失败（`No module named '_loss'`）；② `tsai 1.0.1` 硬性要求 `<1.8`；③ 实测 1.7.2 与 1.8.0 数值 hash 一致，故取低者 |
| `torch` | `2.12.1+cu130` | 由下面四个发行版的闸门反推 |
| `torchvision` / `monai` / `segmentation-models-pytorch` / `transformers` | `0.27.1+cu130` / `1.6.0` / `0.5.0` / `4.50.0` | `_models/facies/_p5_common.py` 的 `allowed_versions` 闸门校验这几个发行版，不符即 `P5AdapterSkip(runtime_version_mismatch)` |
| `lightgbm` | `4.6.0` | sweetspot p5 `source_lock.v1.json` 的 `accepted_version_prefixes=['4.6.']`；`4.7.0` 会让 2 个断言红 |

验证：6 个硬钉版本全部精确匹配，`torch.cuda.is_available() = True`。

### 源码树污染：读取路径带写副作用

用统一环境跑 sweetspot 测试得到 **193 passed / 2 skipped / 0 failed**，但跑完
`git status` 显示 `_outputs/p10_model_results/before_after_primary_metric.png` 被改写
（`140210 → 143812` 字节）。

根因：`p10/results.py` 的 `_figure_path()` 无参数、恒返回 `OUTPUT_DIR / ...`，
而 `build(output_dir)` 的调用方——包括测试——传的是临时目录。
**签名支持导向、内部却硬编码回源码树**，于是每跑一次测试就覆盖一次归档图。

顺带暴露：归档图渲染于 matplotlib `3.10.9`，统一环境是 `3.11.1`，同一份数据渲染字节不同。
项目要求「确定性渲染」，该要求实际上**依赖 matplotlib 版本未被记录**。

同类问题在 `targets/registry.py:72-76` 也存在：`_remaining_oil_case()` 无条件
`atomic_write_json` 写 git 跟踪的 `not_feasible.json`，而写入内容由
`importlib.util.find_spec('resdata')` 决定——实测 `/usr/bin/python3` 探测 False、
`gaia-v2` 探测 True，**同一份仓库文件的内容随解释器变**。在统一环境下探测结果
恰与归档一致故未触发，但根因未除。

## Class Pattern

「函数签名接受 output_dir，内部某处却直接引用模块级默认路径」——调用方以为自己隔离了副作用，
实际没有。这类缺陷在测试里最隐蔽：测试通过、结果正确，只是顺手改了仓库。

判据：任何声称可导向输出的函数，其内部所有落盘点都必须由参数派生；
模块级 `OUTPUT_DIR` 只应作为默认实参出现一次。

## Evidence

- `requirements-sixtrack.txt`（项目根）
- 硬钉版本核验：6/6 精确匹配，`torch 2.12.1+cu130`、`cuda True`
- 污染实测：跑 sweetspot 测试前后 `git status --short` 对比，PNG 由
  `ff88895f610f4feb…`（140210B）变为 `368d488db07eca9c…`（143812B）
- 修复后复跑 `test_p10_model_results.py`：2 passed，`git status` 仅剩本次源码改动
- `p10/results.py:950-951, 1215`（修复点）
- `targets/registry.py:72-76`、`remaining_oil_infill/contract.py:38-40`（同类未修）

## Impact

- 六赛道测试从此有了**唯一可信解释器**；此前任何跨会话的「通过率」对比都不成立，
  不应写入报告或验收结论。
- 归档 PNG 已还原为 `ff88895f610f4feb`。
- `registry.py` 的同类污染仍在，且它的写入内容随解释器变，属更严重的一类
  （污染 + 不可复现），建议按同样方式修：`_remaining_oil_case()` 接受落点参数，
  或改为只读已归档 evidence、由显式 build 入口刷新。
- 「确定性渲染」的前提需要把 matplotlib 版本纳入 provenance，否则换环境重渲即失效。

## Prevention Rule (candidate)

跑测试后应检查 `git status`；测试改动了版本控制中的文件即视为缺陷，不论测试是否通过。
可导向输出的函数，内部落盘点必须全部由参数派生。

## Links

- task_plan: ../_task_plan.md
- 前序: P54_worktree_contribution_audit_and_p5r2_integration.md
- 清单: `requirements-sixtrack.txt`
