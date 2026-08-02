# P33：④岩相现默认护栏阻止一次错误晋级

status: accepted
owner_col: col3

## 结论

④岩相完成两次独立 DeepSeek 候选生成和完整 matched-budget 重训。两次调用均选择 `depth=3, eta=0.2, rounds=60, subsample=1, colsample=1`。该配置在独立 promotion 折上的 fixed-schema Macro-F1 为 `0.217824736469`，高于确定性调度器选出的 `depth=5, eta=0.05` 端点 `0.190451762342`，差值 `+0.027372974127`，3/3 配对 seed 获胜。

但真正的当前默认 A0 `depth=3, eta=0.1, rounds=60` 在同一 promotion 折达到 `0.231574400442`。智能体候选相对 A0 下降 `-0.013749663974`，0/3 seed 获胜，因此最终结论为 `KEEP_CURRENT_DEFAULT`。

## 漏洞与修复

第一版门禁只要求智能体超过确定性搜索端点，曾给出 `RETAIN_HYBRID`。这遗漏了“新候选必须超过已部署默认”的基本护栏。确定性候选池在选择折上挑中了一个 promotion 表现较差的配置，使智能体看似获胜。修复后，晋级必须同时满足：

1. 相对确定性搜索端点 Macro-F1 至少提高 `0.005`；
2. 相对当前 A0 Macro-F1 至少提高 `0.005`；
3. 对两者均至少 2/3 配对 seed 获胜。

修复后从头执行两次独立 provider 调用与全量训练，候选池、最终配置、预测哈希、端点指标和拒绝结论完全稳定。frozen test 与 known holdout 均未读取。

## 科学边界

本结果不能解释为“智能体无用”。智能体确实提出了优于一个发生选择过拟合的确定性端点的配置；但它没有超过已经较强的真实默认。④继续保留 XGBoost `depth=3, eta=0.1, rounds=60`，P33 候选只作为失败诊断，不进入默认模型。
