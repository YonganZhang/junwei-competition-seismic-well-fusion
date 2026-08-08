# P10 property model results audit

结论：
- P9 的 TabICLv2 证据不是“没结果”：它在 development LOGO4、same-fold、未访问 holdout 的条件下，对 PHIF / KLOGH / SW 三目标都优于各自强 baseline，并且都显著优于 target-shuffle control。
- 本次 P10 复现最初确实遇到共享环境缺包，但本地已有 tabicl 源码缓存与纯 Python 依赖缓存，通过最小 runtime 修复后已完成 development-only 重跑；最终交付不再把它写成“没有结果”。
- known-holdout F-15 仍只作为 Stage-4 的生产参考，不参与调参；PHIF / KLOGH / SW 的最终 holdout 参考分别保持 extra_trees_regressor、extra_trees_regressor、xgboost_regressor。
- 本地 TabICLv2 checkpoint blob SHA-256 = 0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a；仓内 checkpoint_path 统一记为 artifact_unavailable，避免把大权重写进提交。

before/after（development LOGO4，primary metric = physical RMSE；baseline / TabICLv2 / target-shuffle control）：
- PHIF: baseline=0.027641 -> TabICLv2=0.026288; target_shuffle_control=0.072793
- KLOGH: baseline=542.902564 -> TabICLv2=469.254724; target_shuffle_control=797.881181
- SW: baseline=0.170427 -> TabICLv2=0.127505; target_shuffle_control=0.399831

接口与流程审计：

| 检查项 | 证据与结论 |
|---|---|
| 数据划分 | P9 与 P10 都使用 mother-family LOGO4；`split_hash=2334f3cc301fc66d6b98c6edf3a4f9c920776469531003d62f5370e119426a18`；训练井族与验证井族不重叠。 |
| 测试集防火墙 | P9 summary 明确记录 `frozen_test_accessed=false`、`known_holdout_accessed=false`；F-15 只保留生产确认参考，不参与本轮选择。 |
| 输入接口 | baseline 与 TabICLv2 读取同一份有限值 `tabular` 特征矩阵，固定 153 维；不为大模型另开特征或标签通道。 |
| 预处理 | 地震与测井统计量只在每个 fold 的训练样本上拟合；验证样本只做变换，`target_statistics_fitted=false`，无全局归一化泄漏。 |
| 目标与掩码 | PHIF、KLOGH、SW 三目标独立拟合并各用自身有效标签掩码；KLOGH 在模型域使用 log1p，PHIF/SW 为 identity。 |
| 反变换与物理指标 | KLOGH 用 `expm1(max(output,0))` 回到 mD，PHIF/SW 仅在物理视图裁剪到 [0,1]；RMSE/MAE 在相同验证样本上计算。 |
| 大模型权重与下载 | 本地 checkpoint 哈希固定，`allow_auto_download=false`；三目标使用独立 TabICL regressor，不做跨目标标签融合。 |
| 对照与公平性 | P9 含 target-shuffle control；P10 重跑使用与强基线相同的 fold、repeat seed、预处理和指标方向。逐折历史行仅作 evidence_only，只有同口径 macro 行才计算 P9 的提升百分比。 |

根因/修复：
- 目前没有证据显示需要改分裂、归一化、特征白名单、冻结/PEFT 或融合逻辑；实际修复是把本地可用的 tabicl 源码缓存、torch-common 纯 Python 包，以及 openpyxl/et_xmlfile/defusedxml 的本地缓存拼到 runtime。

文件/测试/commit：
- 输出目录：`_pipelines/02_task_datasets/reservoir/_outputs/p10_model_results`
- 当前 commit：`ba68969dce674c154fc28c349c2389290c0a5a18`
- openpyxl 已重新打开并验证单 Sheet。
- evidence_path 均指向真实存在的仓内文件。

残余风险：
- TabICLv2 仍读取仓外 HuggingFace cache blob；本交付只提交可复跑证据和哈希，不提交大 checkpoint。
- holdout 只是生产参考，不用于模型选择。
