# ADR: P8 多模态基础模型路由与晋级门

日期：2026-07-28
状态：accepted

## Decision

六赛道分别使用与物理轴匹配的基础模型，不把通用聊天 LLM 作为统一预测器。所有路线通过
`FoundationTaskEnvelope` 描述输入、目标、可见性、条件、fallback 与晋级门；监督 LLM 只做
schema/QC，不产生样本级预测特征。

连接状态与默认状态分离。只有同 split 强基线、random-init 同架构、shuffle/causal 控制、泄漏检查
和预注册最低胜出折全部通过，路线才能从 `CONNECTED_UNVERIFIED` 晋级到 `PROMOTED_DEV`。

## Consequences

- 六条公开权重与 source commit 被固定和真实加载；运行时不联网自动补权重。
- 分割、分类、表格回归、时序预测和体回归保留不同 head、loss、metric 与坐标轴。
- 验证/推理的空间提示不得由 GT、fault stick 或标签采样器生成。
- strict 三维重建不得读取目标派生稀疏值。
- frozen test、known holdout 和 fresh blind 不用于选模型或调 prompt。

## Rejected alternatives

- 同一个语言模型预测六种张量：缺少物理归纳偏置，且接口会掩盖任务差异。
- 原始 SAM mask 直接当岩相类别：SAM 是 class-agnostic，必须增加闭集语义头。
- 用验证标签点击 SAM-Med3D：会形成提示泄漏。
- 用 SAM 3D Objects 做地震体重建：任务定义不匹配。
- 连接即默认：无法证明预训练权重贡献，也会掩盖负迁移。

## Evidence

- `code:foundation_model_routes`
- `_wiki-methodology/_tests/P8_multimodal_foundation_acceptance_evidence.md`
- `data:p8-foundation-runtime-smoke`
- `data:p8-chronos-calendar-development`
