---
phase_id: P25
status: accepted
owner_col: COL2
source: manual
created_at: 2026-08-01
---

# CIG-Bench(伍老师组开源，douyimin/CIG-bench)增量接入尝试：①③被阻塞（非权限问题），⑥有真实development对比但未晋升

## Local Case

用户提供CIG-Bench（`douyimin/CIG-bench`，MIT许可，pip install cig_bench，权重经ModelScope下载，原生3D体输入`(tline,iline,xline)`）作为候选增量信号源，任务定义直接对应①断层(`FaultPredictor`)、③物性(`PropertyPredictor`)，⑥重建则用`RGT`(相对地质年代)作为带外部漂移克里金(Kriging with External Drift)的结构漂移项做增量设计（不替换PyKrige baseline本身）。三条赛道并行评估，结果：

1. **①断层：`DATA_GATE_BLOCKED`**。`FaultPredictor`权重真实下载成功（138,135,635 bytes，sha256 `9d6e8668f0fd27cf0f0131d2b600d79e26bd5cd8f483c3ca2d1614d448351a36`，独立核实过），但评测本身跑不起来——①赛道当前development数据是2D patch形式(`sample_shape=[1,33,65]`)，CIG-Bench要求的是带"已审计verified背景mask + 显式unknown-mask provenance + group隔离切分"的连续3D体，我们没有这个资产。**这不是权限/网络问题，是数据资产缺口**，且与更早的SAM-Med3D评测(P8/P9)撞上的是同一个门禁要求，从未被补齐过。
2. **③物性：`BLOCKED_DATA_OR_API`**。直接查询ModelScope `douyimin/CIG-Bench`仓库的真实文件清单（`GET /api/v1/models/douyimin/CIG-Bench/repo/files?Revision=master`），确认该仓库**只上传了`CIG-Bench-{Channel,Fault,Karst,RGT}.pth`四个权重文件，从未上传`CIG-Bench-Property.pth`**——CIG-Bench自己的代码里注册了`PropertyPredictor`类和指向该文件的指针，但权重文件本身没有发布。这是**CIG-Bench上游项目自身的发布缺口**，不是网络重试或授权能解决的问题；已用两种独立方式核实（预测器初始化时的HTTP 404 + 直接查仓库文件列表，两者一致）。
3. **⑥重建：有真实development对比数字，但未通过晋升**。用RGT体作为KED结构漂移项，跟纯PyKrige baseline同口径development对比：整体RMSE KED=0.028632 vs baseline=0.028450（差+0.64%），5个独立空间折2赢3输，未通过预注册晋升门槛（需≥1%整体提升且≥4/5折获胜）。**准确表述**：这是"PyKrige baseline优于PyKrige+CIG-Bench RGT结构先验这一融合尝试"，**不是**"我们的模型优于CIG-Bench的模型"——CIG-Bench自己的端到端`PropertyPredictor`从未在本赛道被真正测试过（见上条③的阻塞），本次只用了它的RGT中间产物做辅助信号，结论范围仅限于"这一种融合方式没有帮上忙"。

## Class Pattern

第三方工具库"技术上能装、能连通"（pip装得上、ModelScope网络可达）不等于"能用"。真正决定能不能用的是两类边界：(a) 我方数据资产是否满足对方接口的结构性前提（本例①的3D体+mask前提）；(b) 对方项目自身发布是否完整（本例③的Property权重缺失）。这两类阻塞都不是"权限"问题，混为一谈会让人误以为"点一下登录/授权"就能解决，从而浪费排查方向。同时，用第三方工具的**衍生/中间产物**（如RGT）做增量融合时，得到的否定结论只能约束到"这个中间产物+这种融合方式"，不能外推为对该工具"整体模型能力"的评判——这是本项目P16反复强调过的归因边界规则的又一次实例。

## Evidence

- ①: `.claude/worktrees/track-fault` commit `76c4464`（及更早的`9cfac42`），`_pipelines/02_task_datasets/fault/_outputs/p18_cigbench_fault/evidence.md`，含权重真实sha256核对、门禁逐项判定原因、"最低解锁条件"清单。
- ③: `.claude/worktrees/track-property` commit `9696c9b`，`_pipelines/02_task_datasets/reservoir/_outputs/p18_cigbench_property/evidence.md`，含真实HTTP 404报错原文；本finding额外用`curl`直接核实了ModelScope仓库文件清单，确认Property权重确实不存在（非本地/网络原因）。
- ⑥: `.claude/worktrees/p10-results-reconstruction` commit `f2ea5bc`，`_pipelines/02_task_datasets/reconstruction/_outputs/p18_cigbench_ked/summary.json`，含逐折RMSE、bootstrap区间、`decision.state=NO_ROBUST_DEVELOPMENT_GAIN`、`promotion_rule_passed=false`。该赛道后续P28 agentic optimization框架（Codex自主推进）也把这次RGT-KED结果重新纳入其route gate复核，独立得出一致的拒绝结论。

## Impact

- ①③两条的CIG-Bench接入暂时无法推进，不是我们这边能单方面解决的：①需要真的构建3D development体资产（工程量，非权限问题，是否投入待拍板）；③需要等CIG-Bench上游补发Property权重（或联系作者），不建议在权重不存在的情况下等待或重试。
- ⑥的development对比结果已写入报告，措辞已按"PyKrige baseline优于该融合尝试"的精确范围表述，不作"我方模型整体优于CIG-Bench"这类过宽外推。
- CIG-Bench对①③赛道仍是一个有价值的候选（任务定义直接对应，比之前试过的通用大模型更贴切），只是当前被资产/上游发布缺口挡住，不是被证伪，值得记录以便条件满足后重新评估。

## Prevention Rule (candidate)

第三方工具"跑不起来"时，先分类原因属于哪一类（我方数据资产缺口 / 对方发布缺口 / 真实权限缺口），三类处理路径完全不同（工程补齐 / 联系上游或换资源 / 走授权流程），不要笼统归为"权限问题"就默认可以靠用户点击网页解决。

## Links

- task_plan: ../_task_plan.md
- 相关finding: `P16_reconstruction_foundation_model_bridging_exhausted.md`（同一条归因边界规则的更早实例）
