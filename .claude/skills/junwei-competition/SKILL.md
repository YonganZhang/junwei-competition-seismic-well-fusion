---
name: junwei-competition
description: 军伟的比赛项目专属入口：新会话或新AI需要了解项目数据、六条赛道、算法源头、当前边界和权威文档时触发。不用于替代各赛道自己的实现说明。
---

# 军伟的比赛

本项目研究地震与测井多模态融合识别有利油气目标，数据包含挪威北海 Volve、荷兰 F3 和加拿大 Penobscot。这个 skill 只做项目级路由，不重复权威文档正文。

## 触发时机

- 新会话或新 AI 第一次处理本项目数据、训练集或六条赛道之前
- 需要选择某条赛道的 baseline 或参考算法之前
- 有人询问项目数据是否下载完整、数据位置、井名匹配或已知边界时

## 权威文档

1. **数据档案**：`_wiki-methodology/_wiki/_entities/volve-dataset.md`
   - Volve 14 个 ZIP 与官方类别的映射、24 口井覆盖矩阵、命名陷阱及 LAS/DLIS/PDF/WITSML 内容级验证结果。
2. **六赛道算法源头**：`_wiki-methodology/_wiki/_entities/algorithm-baselines-6tracks.md`
   - ①断层预测、②地震相分类、③储层物性预测、④岩相预测、⑤甜点预测、⑥三维模型重建的权威代码/论文、可用性和文档质量。
3. **联合审查报告**：`_wiki-methodology/_top/_external_reviews/codex_data_algorithm_audit_20260713.md`
   - 本地文件系统、Databricks Marketplace、Zenodo MD5 和算法仓库的审计证据。

## 数据资产登记

`_meta/_data_registry.yml` 是三个数据资产路径、大小、状态和来源的机器可读真源。脚本读取 registry；人工理解优先从上述实体文档和审查报告进入。

## 已确认事实

- Volve、F3、Penobscot 三批数据的官方下载包完整，无遗漏类别。
- Volve 14 个 ZIP 合计 4.566 TB；Equinor 官方为 11 个文件夹类别，本项目可归并成 9 个业务组。
- F3 是 9 个解释层位分隔 10 个地震相类别。
- Penobscot 页面文字写 7 类，但发布数据 `dataset-log.txt` 为 `num_classes=8`，HDF5 标签值域为 0–7；建模以数据文件为准。
- ⑤甜点预测没有现成真值，代理标签定义属于军伟的决策权，任何 worker 不得自行猜测。

## 六窗口组织

AI Session Cards 统一分类名为 **`军伟的比赛`**。六条赛道各有一个独立 Codex 窗口和 worktree；项目 Claude/Codex 总窗口可以同组，但不能算进“六任务六窗口”的数量。

## 找不到答案时

- 先检查上述三个入口，再查看 `_wiki-methodology/_top/_findings/P2.1~P2.5` 和 `_sandbox/volve_data/_full_inventory/` 的原始证据。
- 数据可能变化时必须重新运行真实的 `ls`、`du`、`find`、校验和或远端 API，不凭旧文档猜测。
- 模型架构、标签定义、赛道范围或资源分配需要人类判断时，停止实现并向军伟报告。
