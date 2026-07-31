# ADR: P9 基础模型连接、证明与默认晋级

日期：2026-07-28
状态：accepted

## Decision

六条基础模型 route 全部保留，但默认路径只由冻结 development 证据决定。

- 属性 TabICLv2、甜点 Chronos-2：记为 `CONNECTED_UNVERIFIED`。它们已经打赢强基线和现有
  负控制，但按预注册规则仍缺同架构随机初始化证据，不默认启用。
- F3/Penobscot SAM2、井筒岩性 MOMENT、三维重建 OpenMind：记为 `VERIFIED_NO_GAIN`，
  继续使用既有强基线。
- 断层 SAM-Med3D：模型接口与真实权重已接通，数据门仍阻塞；不得把孤立 2D 切片当成合法
  3D development evidence。

## Why

基础模型的价值必须拆成三层：

1. 权重和 forward 是否真实可用；
2. 预训练是否优于同架构随机初始化；
3. 完整方案是否打赢同 split 的现有强基线。

只过第一层叫“接通”，过第二层叫“预训练有贡献”，三层都过且满足预注册门槛才允许改默认。
这样既不会漏接大模型，也不会因为模型名气大就把更有效的小模型替换掉。

## Prompt 与接口

统一 prompt 仅用于监督/QC，不把聊天 LLM 当数值预测器。任务预测仍分别使用 support set、
time window、depth window、spatial prompt 和 masked volume。提示模板继续禁止标签值、验证
GT 点击、冻结测试指标、路径和凭证；所有模型保留任务专用 head、loss、metric 与 fallback。

## Evidence

- `_models/gaia_dagt/foundation_effect_protocol.v1.json`
- `_models/gaia_dagt/foundation_routes.v1.json`
- `_wiki-methodology/_tests/P9_foundation_effect_acceptance_evidence.md`
