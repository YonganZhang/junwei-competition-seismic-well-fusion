---
phase_id: P32
status: accepted
severity: major
owner_col: COL4
source: experiment
created_at: 2026-08-03
closed_at: 2026-08-03
closure_evidence: _wiki-methodology/_tests/P32_hybrid_agent_optimizer_acceptance_evidence.md
---

# 混合智能体优化结果

## 结论

将大模型从“最终数值动作裁决者”改为“有界候选生成器”，再由确定性调度器负责真实训练、排序和 promotion，解决了 P29 中直接 LLM 端点没有优势的问题。②地震相和③物性均在匹配候选预算下超过确定性对照，并通过第二次独立 provider 调用和完整训练复跑。

该结论不能简化为“大模型单独提升了指标”。真正有效的是组合机制：语言模型扩大或重组候选空间，严格校验器拒绝越界配置，确定性执行器按 selection-development 排序，最后只在未暴露给提示词的 promotion-development 上晋级。

## ③物性

两种策略均评估 4 个候选，每个候选运行 8 个更新步；各自选出一个候选后，以 3 个固定种子各复训 32 步。每个策略的总预算均为 128 个更新步。

智能体两次都提出并最终选中 `reservoir_linear`、`learning_rate=0.01`、`l2_strength=0`。确定性候选池最终选中 `learning_rate=0.008`。promotion 的 train-std normalized RMSE 中位数从 `0.662219999` 降至 `0.633945555`，相对改善 `4.269645%`；三个配对种子全部获胜，PHIF、KLOGH、SW 的最差组 RMSE 均没有超过 2% 非劣界限。

两次 provider 返回的四候选集合不同，但最终可执行配置和全部端点指标一致。这说明结果对候选表措辞和部分备选项不敏感，但尚不能推出任意提示词或任意 provider 都会产生同样结果。

## ②地震相

两种策略各执行 4 个 selection 配置和 1 个 promotion 条件化配置。智能体提出联合配置，确定性对照沿用 P29 单因素候选；两者候选数和实际配置包数完全一致。

智能体两次都选择：F3 使用 `fusion_scale_initial=0.8`、`fusion_lr=5e-4`、`dice_weight=0.75`、SAM2 解冻；Penobscot 保持 A0。promotion 等均 mIoU 从确定性对照的 `0.301022749` 提升至 `0.325721341`，绝对增益 `0.024698592`。F3 从 `0.367532093` 提升至 `0.416929276`，Penobscot 保持 `0.234513406`。

第二次 provider 调用与第一次有 3/4 个可执行候选重合，最终条件化配置和端点指标完全一致。PyTorch 对 CUDA NLL loss 和 memory-efficient attention 给出非位级确定性警告；尽管两次结果完全复现，该边界仍保留在独立验证证据中。

## 科学边界

1. 两条结果都属于嵌套 development 证据，frozen test 没有读取。
2. 物性证明了候选提出可以找到当前确定性池之外的更好学习率，不证明确定性密集网格永远找不到同一点。
3. 地震相证明联合配置优于当前单因素池，不证明语言模型独立完成了数值优化。
4. 完整候选池不完全稳定；稳定的是通过确定性调度后得到的最终配置与指标。
5. 直接 LLM 选择器仍不晋级，保留的是混合优化器。

## 后续建议

下一阶段扩展到其余四赛道时，必须先建立有科学意义的连续或结构化搜索空间。④以新 XGBoost `depth=3/eta=0.1/rounds=60` 为 A0；⑥围绕 P21 三核权重、各向异性和非平稳结构参数；①围绕合法连续 3D 体的特征与阈值；⑤只在目标本身可训练时开展。仍使用匹配候选预算、独立 promotion 和 fail-closed 门禁。

## Prevention Rule

不得把“大模型提出候选、确定性调度器选择并晋级”的整体增益写成直接 LLM 数值决策增益；必须同时报告候选预算、执行预算、promotion 边界、独立复跑和完整候选池稳定性。

## Links

- phase: ../_phases/P32_hybrid_agent_optimizer_pilot.md
- acceptance: ../../_tests/P32_hybrid_agent_optimizer_acceptance_evidence.md
- property runner: ../../../_pipelines/02_task_datasets/reservoir/p32_hybrid_agent_optimizer.py
- facies runner: ../../../_pipelines/02_task_datasets/facies/p32_hybrid_agent_optimizer.py
