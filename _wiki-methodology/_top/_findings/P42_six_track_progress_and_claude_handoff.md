---
phase_id: P42
status: accepted
severity: major
owner_col: COL4
source: integration
created_at: 2026-08-06
closed_at: 2026-08-06
closure_evidence: _wiki-methodology/_tests/P42_six_track_handoff_acceptance.md
---

# 六赛道最新进度与 Claude 接手说明

## 总体结论

六个赛道已经形成独立 Pipeline，并共用固定的 `validate → prepare → baseline → optimize → promote → refit → verify` 生命周期。主仓已保留 P31–P35 的统一接口、智能体门禁和可移植证据。本轮又集成了⑥重建 P36–P39、④岩相 P40 和③物性 P41。新增实验均保留真实结果，没有因为未晋级而改弱基线或降低门槛。

当前可以确认三项正向进展。②地震相和③物性的混合智能体优化器在匹配候选预算下超过确定性对照；④岩相的纯 XGBoost 默认配置已经稳定提高 Macro-F1；⑥重建的 P21 固定三核方案仍是 Eclipse-PORO 任务的默认模型。当前不能确认的是冻结双基础模型融合已经稳定提高③、④或⑥的最终指标。P38–P41 均完成真实前向、对齐和消融，但没有通过各自的晋级门。

## 六赛道状态

| 赛道 | 统一 Pipeline | 当前保留方案 | 最新实验结论 | 下一步 |
|---|---|---|---|---|
| ① 断层识别 | `fault_agentic_optimization` | 审计后的局部逻辑回归 baseline 与确定性治理 | CIG-Bench 的 guard precision lift 为 `1.179665` 倍，AP lift 为 `1.423894` 倍，但半径 2 容差 F1 仅 `0.020238`，不替代默认模型 | 扩充可信连续三维标注与硬负例，再比较结构感知损失和表面级指标 |
| ② 地震相识别 | `facies_agentic_optimization` | P32 混合智能体优化器 | 等均 mIoU 从 `0.301022749` 提高到 `0.325721341`；F3 提高 `0.049397183`，Penobscot 保持不降 | 保留混合架构，扩大独立数据集和种子验证，不恢复直接 LLM 数值裁决 |
| ③ 储层物性预测 | `property_agentic_optimization` | P32 候选生成加确定性调度；跨模态资格门仍以 P5 Stage-3 强基线为 B0 | P32 相对确定性对照将复合 RMSE 改善 `4.269645%`。P41 双基础融合只改善 `0.1703%`，`2/4` 外层胜出且置信区间跨零，不晋级 | 先补独立井族与更强配对监督；未出现稳定内层信号前不进入 LoRA/Adapter |
| ④ 岩相识别 | `lithofacies_agentic_optimization` | XGBoost `depth=3, eta=0.1, rounds=60` | 默认 Macro-F1 从 `0.194938` 提高到 `0.213349`。P40 双基础融合为 `0.161312`，低于同环境 B0 `0.213349`，不晋级 | 保留新 XGBoost 默认；优先改善类别支持和配对数据，不继续增加融合头容量 |
| ⑤ 甜点评价 | `sweetspot_agentic_optimization` | 冻结 A0 与七目标 fail-closed 合同 | P29 的 LLM 路线返回 `STOPPED`，最终 `REJECT_AGENT`。部分目标仍没有获批的统一标签与可训练数据 | 先完成七目标标签和评价合同；目标未通过 `prepare` 前禁止自动训练或晋级 |
| ⑥ 三维重建 | `reconstruction_agentic_optimization` | P21 固定三核集成，Eclipse-PORO 开发 OOF RMSE `0.027734374378` | P38 的真实井 PHIF 双基础融合为 `0.079781229`，弱于同任务 well-only `0.075314433`。P39 固定共同基线后双预训练为 `0.075908484`，仍不晋级 | 保留 P21；后续井震跨模态必须在同一 PHIF 目标、同一锁定 well-only base 和独立井族上继续 |

## 智能体的有效范围

智能体已经从“直接决定最终数值动作”改为“提出受约束候选”。确定性调度器负责候选合法性、真实训练和排序，独立 promotion 再决定是否替换 incumbent。这个组合在②和③取得了 development 指标提升。直接 LLM 数值裁决在六赛道仍没有稳定优势，因此不得把 P32 的组合收益写成大模型单独完成了优化。

①、④、⑤、⑥继续保留拒绝结果。拒绝表示候选真实执行但没有超过当前默认，不表示 Pipeline 失败。任何新候选都必须同时超过匹配预算的确定性对照和当前 incumbent，不能只挑较弱对照制造正增益。

## 本轮主仓集成

本轮从 `master@4dd6a7a` 建立隔离集成分支，纳入以下增量：

1. ⑥重建 `4110454 → e926801 → eb4789b → c6056a0 → d3851e9`，依次修复 P30 方差、关闭真实 PHIE 监督门、完成真实 PHIF 井震小试、增加查询侧融合并统一 P39 的锁定基线。
2. ④岩相 `b4b37cb`，集成 P40 双基础模型资格门及完整可移植证据。其主仓 cherry-pick 为 `a665ec5`。
3. ③物性 `d232f9d`，集成 P41 runner、基础特征适配器和测试。其主仓 cherry-pick 为 `788429b`；本轮进一步移除了旧 worktree 的硬编码依赖，并归档轻量证据。
4. 主工作树中三处未提交但有明确语义的修订被吸收：六 worker 窗口说明、P16 结论边界收窄和 P2.1 数据规模 finding 的 superseded 标记。

①断层、②地震相和⑤甜点的旧 track 分支没有再次合并。树级比较显示，这三条线的有效内容已由 P31–P34 选择性进入主仓，而主仓还包含更新的 adapter、可视化和测试。再次合并只会引入重复历史或覆盖新接口。

主工作树中的技术报告、图片和临时审阅目录约 `1.2 GB`，本轮没有提交。它们是本地派生产物，不应与六赛道代码集成混在同一次 Git 交付中。原主工作树的未提交文件也没有被删除或重置。

## 代码入口

Claude 接手后先从统一入口查看状态：

```bash
python3 _code/six_track_pipeline/cli.py list
python3 _code/six_track_pipeline/cli.py plan --track all --through verify
python3 _code/six_track_pipeline/cli.py verify --track all --through verify
sixone-cli doctor .
```

各赛道仍通过 adapter 把统一阶段翻译为具体科学入口。P40、P41 和 P39 属于研究资格门或非晋级实验，不会自动替换 P33、P32 和 P21 的权威默认。需要继续实验时，先阅读对应 finding 和 `rerun_commands.json`，再运行脚本；不要从文件名猜测最新模型。

## 接手优先级

第一优先级是保持六条 Pipeline 的验签和默认模型不漂移。任何 adapter、manifest、registry 或证据路径变化后，都要重新执行六条 pipeline 的 verify 和 TOP doctor。

第二优先级是处理⑤甜点的数据与标签门。该赛道没有通过 `prepare` 时，不应把模型结构或智能体提示词当作主要瓶颈。

第三优先级是继续③、④、⑥的井震跨模态研究。下一轮必须先增加可归因信号，再考虑 LoRA 或分阶段解冻。③需要更多独立井族或更强配对监督；④需要改善长尾类别支持；⑥应沿 P39 的共同锁定 well-only base 和查询侧对齐合同继续，不能再混用 P21 Eclipse-PORO 与真实井 PHIF 指标。

## 结论边界

本文件区分“Pipeline 已接通”“候选真实执行”“development 指标提升”和“默认模型晋级”。②、③已证明混合智能体优化在当前 development 协议下有效。③、④、⑥的冻结双基础融合尚未证明稳定提升。①的 CIG-Bench 有高召回和相对先验 lift，但精度不足。⑤仍受标签和数据门限制。以上结论均不使用 frozen test，也不外推为跨场区泛化。
