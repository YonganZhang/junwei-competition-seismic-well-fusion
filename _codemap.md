# Code Map — 军伟的比赛（地震+测井多模态融合识别有利油气目标）

更新: 2026-07-08

本文只做 COL4 代码地图 / 注册路由。当前 phase、下一步、运行态、最近测试结果只写 `_wiki-methodology/_top/_task_plan.md`。

## 冷启动读法

1. 当前状态: `_wiki-methodology/_top/_task_plan.md`
2. 六合一职责: `_wiki-methodology/_top/_six_dim_contract.md`（若项目已创建）
3. 代码注册: `_meta/_registry.yml`
4. 数据注册: `_meta/_data_registry.yml`
5. 测试地图: `_wiki-methodology/_tests/`

## 项目身份

| 字段 | 值 |
|---|---|
| 项目 | 军伟的比赛（地震+测井多模态融合识别有利油气目标） |
| Mode | research |
| 目标 | 输入地震数据+测井数据+井位轨迹+专家标签，输出断层/储层/有利目标概率图或分割结果；核心证明"地震+测井融合"优于"地震单模态" |
| 方法 | 5阶段pipeline骨架（地质属性预测→测井编码→井震融合→弱对齐→空间编码解码），具体算法待补，见 `_wiki-methodology/_wiki/_methods/pipeline-skeleton.md` |

## COL4 真源关系

| 层 | 文件/目录 | 职责 |
|---|---|---|
| 人读代码地图 | `_codemap.md` | 代码域、入口、调用关系、测试指针 |
| 机读代码注册 | `_meta/_registry.yml` | 代码域、工具、pipeline、脚本、可复用入口 |
| 数据注册 | `_meta/_data_registry.yml` | 数据集、模型、外部资产、生成物血缘 |
| 查询/同步工具 | `_code/` | registry 查询、lint、同步 wiki 投影 |

## 主要代码域

| 代码域 | 入口 | 注册/测试 |
|---|---|---|
| 主流程 | `_pipelines/` | `_meta/_registry.yml` + `_wiki-methodology/_tests/` |
| 可复用代码 | `_code/` | `_meta/_registry.yml` |
| 数据接口 | `_data/` | `_meta/_data_registry.yml` |
| 输出/图/论文 | `_outputs/` / `_figures/` / `_paper/` | `_meta/_runs.yml` / `_meta/_data_registry.yml` |
| 实验代码 | `_sandbox/` | `_sandbox/_index.yml`; 优于 baseline 才进入主流程 |
| 归档代码 | `_legacy/` | 每个归档目录需 README/TOMBSTONE |

## 注册维护

- 新增主流程代码:先查 `_meta/_registry.yml`,复用已有条目;确实新增再注册。
- 新增数据/模型/外部资产:写 `_meta/_data_registry.yml`,不要只写 README。
- 新增测试:测试地图写 `_wiki-methodology/_tests/`,不要只在提交说明里写。
- 临时探查:留 `_tmp/` 或 `_sandbox/`,不进 registry。

## 不放在本文的内容

- 当前 phase / next action / service online-offline: `_wiki-methodology/_top/_task_plan.md`
- 长期路线和阶段边界: `_wiki-methodology/_top/_roadmap.md`
- 历史交接: `_wiki-methodology/_top/_handoff*.md`
- 架构图: `_wiki-methodology/_top/_SYSTEM-ARCH.md`
- 文件生命周期缺口: `_meta/_known_issues.md`
