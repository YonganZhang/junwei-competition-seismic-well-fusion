# P33 ④岩相混合智能体验收证据

- 协议测试：`7 passed`。
- 候选预算：LLM/确定性策略各 `2160` selection boosting rounds、`180` promotion boosting rounds。
- 数据边界：选择折 `0--2`，promotion 折 `3`；无交集；frozen test/known holdout 未读取。
- 独立调用：两个不同 DeepSeek response id；4/4 候选池一致。
- 最终配置：两次均为 `depth=3, eta=0.2, rounds=60, subsample=1, colsample=1`。
- 端点复现：逐 seed 指标与预测 SHA-256 一致。
- 智能体相对确定性端点：Macro-F1 `+0.027372974127`，3/3 seed 获胜。
- 智能体相对当前 A0：Macro-F1 `-0.013749663974`，0/3 seed 获胜。
- 最终决定：`KEEP_CURRENT_DEFAULT`；默认仍为 `depth=3, eta=0.1, rounds=60`。

验收重点不是“智能体必须晋级”，而是共同 A0 护栏能够拒绝由确定性选择过拟合造成的假胜利。
