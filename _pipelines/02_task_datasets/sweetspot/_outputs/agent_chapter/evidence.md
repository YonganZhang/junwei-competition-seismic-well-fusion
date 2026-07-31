# ⑤ 甜点预测赛道：智能体分析章节

## 作用边界

本章节只做“分析建议归档”，不生成标签、不读取 frozen holdout/test.h5、不写入训练产物。

这里使用的真实证据来自：

- `p5/_outputs/stage3_cv/leaderboards/T1.json` … `T7.json`
- `p5/_outputs/stage4_confirmation/p5_stage4_summary.json`
- `p7/_outputs/t3_chronos2_cv/summary.json`
- `p8/_outputs/t3_chronos2_calendar_cv/summary.json`

这些文件都在 `p10-results-sweetspot` 工作树中，且本章节只引用其摘要与哈希，不复述任何 frozen test 或 test.h5 内容。

## 真实结果快照

| 目标 | 当前状态 | 近期模型 / 家族 | 关键指标 |
|---|---|---|---|
| T1 | rankable / confirmed on known holdout | LightGBM, 64 trees | holdout MAE `0.2161617856064557`, Spearman `0.9931289666595325` |
| T2 | rankable / confirmed on known holdout | CatBoost, 64 iters | holdout AP `0.9990776475633495`, Brier `0.02286081524589408`, F1@0.5 `0.9777260638297872`, thickness MAE `26.19999999997617 m` |
| T3 | rankable / confirmed on known holdout | XGBoost, 64 trees; Chronos-2 foundation evidence exists | holdout MAE `47.701963387379216`, Spearman `0.6355692295523352`; p7/p8 development evidence supports Chronos blend but not default promotion |
| T4 | rankable / confirmed on known holdout | CatBoost, 64 iters | holdout AP `0.13146311725259094`, Brier `0.745222265148662`, F1@0.5 `0.1951219512195122` |
| T5 | not_feasible | no approved label | no label is defined; simulation proxy must not be field truth |
| T6 | blocked | no development-only feature source | test.h5 fallback forbidden |
| T7 | blocked | no development-only feature source | test.h5 fallback forbidden |

## 真实来源哈希

| source file | sha256 | note |
|---|---:|---|
| `.../p5/_outputs/stage3_cv/leaderboards/T1.json` | `7c992062b83f12da7a6de244ce543c898477a1bcd28765943fb47022f2c12d8b` | stage-3 winner evidence |
| `.../p5/_outputs/stage3_cv/leaderboards/T2.json` | `69ed4d406b7d71b2bd65716c4e9afeb76b0b46eeeaa266bf3d9c27b83377ca9b` | stage-3 winner evidence |
| `.../p5/_outputs/stage3_cv/leaderboards/T3.json` | `a68d7cee0bd9ede4907ede0f59dd55530821524999bac429bb7feb1824e772d7` | stage-3 winner evidence |
| `.../p5/_outputs/stage3_cv/leaderboards/T4.json` | `0749130e47b6c7645fc2a2b497531d14d2c2ce29827fd9354ab4401a1bb7d048` | stage-3 winner evidence |
| `.../p5/_outputs/stage3_cv/leaderboards/T5.json` | `869fb5494e0063e0251b7662b29f52672f95765d83fbfec44f629de5155f0d6e` | not_rankable |
| `.../p5/_outputs/stage3_cv/leaderboards/T6.json` | `8c39b306bbaa2a0e3ffc8c1d341a4a1c7d2b6e316fd8eb5325d05cb9360c720c` | not_rankable |
| `.../p5/_outputs/stage3_cv/leaderboards/T7.json` | `caa70a96ffb048d38cc8956cf0d211d4f6c42150f3a44eeb3c43d84d2f330b01` | not_rankable |
| `.../p5/_outputs/stage4_confirmation/p5_stage4_summary.json` | `1db1b5e15cf53a567dcc922f7d68b73bc6de2e987cc05f14eb9f6e36e47f992c` | known-holdout confirmation bundle |
| `.../p7/_outputs/t3_chronos2_cv/summary.json` | `bb27792bcc4d34fb622ebaa5eea908b7ae4c6a08a5fa1ff19fda4d3dc97b7d3b` | Chronos dev-fold evidence |
| `.../p8/_outputs/t3_chronos2_calendar_cv/summary.json` | `cfd1b2955638fa14ec9d00d9b169769fb8dbd2392eac47415c12fab0e4202783` | calendar Chronos evidence |

## DeepSeek prompt摘要

输入给 DeepSeek 的结构化 prompt 保持了以下约束：

1. 逐目标分析 T1–T7，不合并成综合甜点分。
2. 明确当前可用结果：
   - T1/T2/T3/T4 的 stage-3 CV winner；
   - T1/T2/T3/T4 的 stage-4 known-holdout confirmation；
   - T5 not feasible；
   - T6/T7 blocked。
3. 不许建议任何 holdout/test.h5 调参。
4. 所有不能低成本验证的建议必须标 `未验证`。

## DeepSeek 常识性建议摘要

### T1

- 当前已经接近平台期，建议优先做小规模树深/树数检查和特征稳定性检查。
- `更多树` 只应视作低成本鲁棒性验证，不应预设能显著提升。
- 目标变换相关建议被标为 `未验证`。

### T2

- 当前 AP 接近饱和，Brier 也较低，继续大幅调参的收益预期很低。
- 低成本可做阈值扫描或折内校准诊断，但不要把 holdout 上的阈值当成优化依据。

### T3

- 这是唯一被认为还有较强改进空间的目标。
- DeepSeek 认为 p7/p8 的 Chronos blend 是最有证据的候选路径。
- 小范围 XGBoost 深度/学习率网格可作为便宜的对照。
- 任何辅助特征方案如果要引入 T1 预测，都被标为 `未验证`。

### T4

- 当前 holdout 表现明显退化，样本量也很小，因此更像噪声/接口问题，而不是一个值得继续复杂化的模型问题。
- 可做折内校准与分折 Brier 分布诊断，但不要把阈值搜索包装成改进。

### T5

- 保持 `not_feasible`，不再投入计算。

### T6/T7

- 保持 `blocked`，直到出现真实的 development-only feature source。

## 诚实验证结论

- **已被既有开发证据支持**：T3 的 Chronos blend 路径。
  - p7 显示 selected macro-fold MAE `186.57151779454128` 优于 archived XGBoost macro-fold MAE `267.1179548547381`。
  - p8 的结论是 `EFFECT_SUPPORTED_NOT_PROMOTED`。
- **本工作树未新增训练验证**：T1/T2/T4 的“便宜建议”尚未在本工作树内重训，因此都保持 `未验证`。
- **未触碰禁区**：没有读取 frozen holdout/test.h5，也没有写入任何训练输出。

## 结论

本章的实际价值是把“还能试什么”和“哪些只是常识性建议”分开：T3 有一条真正可追的候选路径，T1/T2 主要是稳健性收尾，T4 更像接口和校准诊断，T5/T6/T7 仍应保持现状，不要造标签或偷用 test 数据。
