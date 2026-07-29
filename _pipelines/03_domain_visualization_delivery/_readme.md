# 03_domain_visualization_delivery

这是六赛道“领域可视化 → 卡片渲染”的唯一项目级发布入口。

## 为什么存在

旧的 `_code/visualization/p5_r2_six_track.py` 读取 R2 汇总 JSON，画的是协议覆盖率、
预算曲线和模型性能。它生成的 `track_01_fault.png` 等文件名容易被误认成领域图。
本 Pipeline 不接受模糊文件名或窗口口头结论，只接受经过白名单和哈希审核的真实图片。

## 三道门

1. `step_01_validate_manifest.py`
   - 必须恰好包含 fault、facies、property、lithofacies、sweetspot、reconstruction。
   - 图片、来源脚本和证据文件必须实际存在。
   - 清单中的来源 commit 必须是当前工作树 HEAD 的祖先；图片 SHA-256、来源脚本 blob、
     PNG 尺寸和 Git 跟踪策略必须与该来源一致。这样允许同一工作树继续提交不相关工作，
     但图片或生成脚本一旦漂移仍会 fail closed。
   - 路径含 `status`、`readiness`、`protocol`、`gate`、`placeholder`、
     `not_feasible`、`summary` 或旧 `p5_r2_visualization` 时直接拒绝。
   - 人工复核哈希必须与待发布文件当前哈希完全一致。
2. `step_02_stage_delivery.py`
   - 只复制通过第一步的图片。
   - 复制后重新计算 SHA-256；不一致则不生成暂存清单。
3. `step_03_publish_cards.py`
   - 默认不公开；必须显式传入 `--yes-public`。
   - 只发布第二步清单中的副本，并要求每个永久 URL 返回 HTTP 200。

## 运行

```bash
python3 _pipelines/03_domain_visualization_delivery/step_01_validate_manifest.py
python3 _pipelines/03_domain_visualization_delivery/step_02_stage_delivery.py
python3 _pipelines/03_domain_visualization_delivery/step_03_publish_cards.py --yes-public
```

验证但不写交付物：

```bash
python3 _pipelines/03_domain_visualization_delivery/step_01_validate_manifest.py --check-only
```

## P12：赛道 1 / 3 / 5 出版级图组

六赛道 v1 白名单发布合同继续保留，不因本轮调整而改写。P12 只处理用户指定的三条线：

| 赛道 | 任务 | 统一渲染入口 | 统一测试入口 | 统一产物目录 |
|---|---|---|---|---|
| 1 | fault | `_pipelines/02_task_datasets/fault/p12_visualization.py` | `_pipelines/02_task_datasets/fault/test_p12_visualization.py` | `_pipelines/02_task_datasets/fault/_outputs/p12_visualization/` |
| 3 | property | `_pipelines/02_task_datasets/reservoir/p12_visualization.py` | `_pipelines/02_task_datasets/reservoir/test_p12_visualization.py` | `_pipelines/02_task_datasets/reservoir/_outputs/p12_visualization/` |
| 5 | sweetspot | `_pipelines/02_task_datasets/sweetspot/p12_visualization.py` | `_pipelines/02_task_datasets/sweetspot/tests/test_p12_visualization.py` | `_pipelines/02_task_datasets/sweetspot/_outputs/p12_visualization/` |

赛道 2、4、6 当前为 paused，不得被新窗口顺手重绘或更改。统一字体、配色、panel label、
输出格式、血缘字段和人工视觉复核要求以 `_meta/_visual_style_guide.yml` 为准；各赛道只统一
这些“合同”，不强制使用相同子图，因为分割、连续物性回归和七目标评价需要不同的科学诊断。

新窗口必须按以下顺序工作：

1. 读本文件与 `_meta/_visual_style_guide.yml`。
2. 在对应 `p12-viz-*` 工作树运行赛道渲染器和测试。
3. 打开并逐张检查 PNG，同时确认 SVG/PDF 可渲染。
4. 检查 `manifest.json` 的输入/输出哈希、split scope、科学 caveat 与人工复核哈希。
5. 只有新的 manifest 通过共享校验后，才允许进入稳定暂存和卡片发布。

共享发现和负责人验收命令：

```bash
python3 _pipelines/03_domain_visualization_delivery/step_00_discover.py --check
python3 _pipelines/03_domain_visualization_delivery/step_04_stage_p12_review.py \
  --reviewer codex-leader --accept-visual-qa
```

第二条命令不会相信 renderer 自己写的“已复核”状态。三个 track-local manifest 必须保持
`manual_review.reviewed=false`；负责人真正逐图检查后，独立的
`_outputs/domain_visualization_delivery/p12/review_attestation.json` 才会记录验收结论，
并把每张 PNG/PDF/SVG 按哈希复制到稳定项目路径。这避免新窗口覆盖掉人工复核证据。

### P12 完成证据（2026-07-30）

- fault：来源 HEAD `5d22c9a`，2 张 PNG + PDF/SVG，5 项赛道测试通过。
- property：来源 HEAD `c914bd6`，3 张 PNG + PDF/SVG，2 项赛道测试通过。
- sweetspot：来源 HEAD `39e0e97`，8 张 PNG + PDF/SVG，5 项赛道测试通过。
- 三条线共 13 张 PNG、39 个稳定文件；负责人逐张原分辨率检查后，暂存文件 SHA-256
  与来源完全一致。
- 为避免集成后只能在旧窗口复跑，property 的 9 份、sweetspot 的 17 份最小来源证据也按
  manifest 原路径和原 SHA-256 固化到当前分支；fault 继续由统一发现入口指向
  `.claude/worktrees/track-fault` 的 13 份审计证据。三条线在当前项目根均已独立复跑通过。
- 共享发现为 `ready`，中央门禁 7/7 通过；正式人工验收记录见
  `_outputs/domain_visualization_delivery/p12/review_attestation.json`。
- 当前分支对应集成提交末端分别为 fault `c77d003`、property `2f243c6`、
  sweetspot `aa426b4`。来源 HEAD 与集成提交不同是 cherry-pick 的正常结果，科学内容未重写。

## 科学边界

- “真实领域图”不等于“模型成绩好”。图中不理想的精度、异常范围或负 R² 必须保留。
- 断层图是当前真实测试 baseline，精度很低。
- 岩相图是 known F-5 确认，不是 fresh-blind 结论。
- 甜点图是 development OOF 的 T2 PR/校准。
- 重构图是 conditional development-only，不代表 strict holdout 泛化。
