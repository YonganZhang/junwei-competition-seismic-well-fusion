# P42 六赛道主仓交接验收

- 日期：2026-08-06
- 范围：六赛道统一 Pipeline、P36--P41 研究增量、TOP 与 Claude 接手说明
- 结论：通过

## 集成核对

1. ⑥重建已纳入 P36--P39；④岩相已纳入 P40；③物性已纳入 P41。
2. ①断层、②地震相与⑤甜点的旧分支经树级比较没有未吸收的赛道文件，因此不重复合并。
3. P40、P41 与 P39 均保持研究资格门结论，不覆盖已晋级的默认模型。
4. 原主工作树约 1.2 GB 的报告、图片和临时产物未进入本次提交；其他窗口的脏工作树未被重置或修改。

## Command gate

| 检查 | 结果 |
|---|---|
| P37--P41 Python 编译 | PASS |
| P39 聚焦测试 | 26 passed |
| P40 聚焦测试 | 6 passed |
| P41 聚焦测试 | 9 passed |
| 六赛道统一运行时与生命周期测试 | 17 passed，6 subtests |
| `cli.py list` | 六个 adapter 全部可发现 |
| `cli.py plan --track all --through verify` | PASS |
| `cli.py verify --track all --through verify` | 六赛道 42 个阶段全部 PASS |
| P39 `--verify-only` | `FEASIBLE_NO_PROMOTION`，证据验签通过 |
| P40 `--verify-only` | `R0_STOP_NO_ATTRIBUTABLE_SIGNAL`，证据验签通过 |
| P41 `--verify-only` | `R0_STOP_NO_ATTRIBUTABLE_SIGNAL`，11 项轻量证据验签通过 |

## Live/user journey

本轮交付是代码、证据与文档集成，不包含需要浏览器操作的用户界面，因此没有伪造网页验收。以 Claude 接手者的真实命令路径代替交互式 journey：从仓库根目录依次执行 `list`、`plan` 和 `verify`，能够发现六个赛道、展开完整依赖链并核验 42 个阶段。该路径已在隔离集成工作树实际执行通过。

## Trace/SSDO audit

本轮没有启动新的长训练，SSDO 降级为可复跑证据审计：P39、P40、P41 均通过各自的 `--verify-only`，并逐项核对 artifact manifest、摘要、预测哈希与独立指标重算。六赛道统一 CLI 的 `verify` 未指定 `--output`，因此没有额外持久化运行 trace；已有研究 trace 继续由各实验输出目录中的 manifest、summary、verification 与 rerun commands 承担。

## 科学结论核对

- 只把②、③的 P32 结果表述为匹配候选预算下的混合智能体 development 改善，不归因成 LLM 单独提升。
- ④的新 XGBoost 默认配置与 P40 双基础模型失败结论分开表述。
- ⑥的 Eclipse-PORO 与真实井 PHIF 属于不同目标，报告中不作跨任务优劣比较。
- ①的相对先验 lift 与低精度同时报告；⑤保持数据和标签门关闭。
- 本轮没有使用 frozen test，也没有形成跨场区泛化声明。

## 接手入口

六赛道状态、默认模型、非晋级实验、下一步顺序与复跑入口统一记录在
`_wiki-methodology/_top/_findings/P42_six_track_progress_and_claude_handoff.md`。
