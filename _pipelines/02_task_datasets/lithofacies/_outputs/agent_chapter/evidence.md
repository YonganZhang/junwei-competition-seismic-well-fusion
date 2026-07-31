# Lithofacies agent analysis chapter evidence

## Outcome first

The best newly tested low-cost suggestion was `depth3_eta01_rounds60`. Mean fixed-nine Macro-F1 changed from `0.194938` to `0.213349` (`+0.018411`).

This exceeds the existing `0.005` development materiality line and the existing prior-calibrated P11 result `0.202187`. It is retained only as a development candidate; default enablement and holdout claims remain forbidden.

These are adaptive exploratory development results, not an unbiased holdout estimate. The estimator settings are seed-independent here (`subsample=1`, `colsample_bytree=1`), so 12 cells contain four distinct held-out well-family outcomes, each repeated across three nominal seeds.

## Real LOGO4 × 3-seed comparisons

| variant | mean fixed-9 Macro-F1 | std | delta vs baseline | wins/12 | verdict |
|---|---:|---:|---:|---:|---|
| baseline_archived | 0.194938 | 0.013858 | +0.000000 | 0/12 | reference |
| baseline_reproduced | 0.194938 | 0.013858 | +0.000000 | 0/12 | alignment check |
| weight_alpha_075 | 0.194817 | 0.030624 | -0.000121 | 3/12 | reject / diagnostic only |
| weight_alpha_100 | 0.005187 | 0.004002 | -0.189750 | 0/12 | reject / diagnostic only |
| well_and_mask_only_858 | 0.212508 | 0.055701 | +0.017571 | 6/12 | development candidate |
| depth3_eta01_rounds60 | 0.213349 | 0.024882 | +0.018411 | 12/12 | development candidate |
| depth3_eta01_rounds60_prior025 | 0.208138 | 0.021552 | +0.013201 | 12/12 | development candidate |

The reproduced baseline has identical validation argmax decisions to the archived baseline; its maximum absolute logit difference is `0`.

## Per-class diagnostic requested by DeepSeek S4

| class id | development support | baseline F1 | best candidate F1 |
|---:|---:|---:|---:|
| 0 | 11 | 0.000000 | 0.000000 |
| 1 | 104 | 0.307356 | 0.375406 |
| 2 | 7 | 0.000000 | 0.000000 |
| 3 | 40 | 0.163406 | 0.184295 |
| 4 | 27 | 0.000000 | 0.000000 |
| 5 | 124 | 0.510763 | 0.554543 |
| 6 | 127 | 0.772915 | 0.805896 |
| 7 | 6 | 0.000000 | 0.000000 |
| 8 | 1 | 0.000000 | 0.000000 |

## Suggestion disposition

- **Verified:** S1/E1 weight exponents `0.75` and `1.0` under the otherwise unchanged baseline.
- **Verified:** S2/E2 used the prompt's explicit `well + mask` interpretation (26 × 33 = 858 features) and removed seismic.
- **Verified:** S3/E3 used `max_depth=3`, `eta=0.1`, and 60 rounds.
- **Adaptive verified follow-up:** the already documented P11 fold-train prior correction was applied to the best S3/E3 logits after the uncalibrated result was observed.
- **Verified diagnostic:** S4 per-fold/per-class F1 is retained in the result rows and summarized above.
- **未验证:** S5 SMOTE/ADASYN and S6 three-point input smoothing were not tested in this bounded chapter.
- **未验证:** larger MOMENT, additional seismic attributes, focal loss, class-balanced loss, and any frozen-holdout effect remain untested.

## Leakage and attribution boundary

- Only the immutable development LOGO4 batch was opened. Every reported row records `known_holdout_accessed=false` and `frozen_test_accessed=false`.
- Feature transforms and class weights use fold-train arrays only. Validation labels are used only for evaluation.
- The DeepSeek consultation is common-sense advice, not empirical evidence. Only the explicitly listed experiments above were run.
- No result in this chapter measures a MOMENT contribution. 大模型贡献占比待下一轮消融确认。

## DeepSeek call metadata

- Requested model: `deepseek-chat`.
- Provider response model: `deepseek-v4-flash`.
- Response id: `0e28a5de-d51e-4f92-858e-30e498500088`.
- Prompt SHA-256: `370eb88002fc48207e66a69997fa1291fd976d56ddf9c3ec0cc7d6167213ae08`.
- The API credential was process-local and was not persisted.

## Structured user prompt sent to DeepSeek

```text
# 任务描述
GM09岩相预测：根据同一深度中心附近的测井序列和3x3地震空间patch，预测固定schema的9类成因岩相。
主指标为fixed_schema_macro_f1（9类始终进入macro平均，某fold缺类也不删类）。
评测是development-only严格LOGO4：按4个母井家族leave-one-group-out，4折×3个固定seed；禁止读取frozen holdout/test.h5。

# 数据特点
development唯一覆盖共447样本，9类支持度依次为[11, 104, 7, 40, 27, 124, 127, 6, 1]，最大/最小非零为127:1。
各fold训练样本315或320，验证样本87到132；部分验证fold缺类，fold 2训练集第9类支持度为0。
输入为13条真实测井曲线×33深度点、13条二值缺失mask×33点、3x3×33地震patch。
当前XGBoost把26×33测井（含mask）和3×3×33地震全部展平为1155维。

# 当前模型与真实指标
强baseline：XGBoost multi:softprob，40轮、max_depth=2、eta=0.2、hist、全特征，fold-train内按1/sqrt(class_count)做样本权重。
严格LOGO4×3seed mean fixed_schema_macro_f1=0.1949377021，std=0.0138577941。
fold-train class-prior校准（0.25×centered log count）=0.2021869757，相对baseline +0.0072492736。
完整系统（上述校准+以XGBoost叶索引为query、native-33 MOMENT token为key/value的gated residual cross-attention）=0.2015903888，相对baseline +0.0066526868，但相对calibration-only -0.0005965868；不能归因于MOMENT。
早期native-33 pretrained-vs-random差仅+0.0028758973，未达到0.005阈值。
深度logit smoothing曾测试多组，单独最好约+0.00382，但与prior calibration组合不稳定，已拒绝。

# 已知限制
小样本、极端类不均衡、按井/家族domain shift、部分fold训练缺类、当前特征是高维展平。
LOGO development已用于多轮探索，任何新数字只能作为探索性证据，不能当无偏holdout估计。
本轮需要低成本验证，不训练更大的MOMENT，不访问holdout，也不要求提升必须来自大模型。

# 请输出
请用中文按以下固定结构给出：
A. 诊断（3-6条，指出最可能瓶颈）
B. 建议清单（编号S1...；每条含类别=参数调整/特征设计/训练策略/评测诊断，具体做法，预期方向但不要编数字，成本=低/中/高，主要风险）
C. 低成本优先实验（最多3个；给出明确候选参数/特征和同口径LOGO4比较方式）
D. 暂不建议（说明原因）
E. 归因边界（明确不能从当前结果断言MOMENT贡献）
```

## DeepSeek original response

A. 诊断（3-6条，指出最可能瓶颈）

1. **类不均衡与缺类问题严重**：9类中支持度从1到127，且fold 2训练集第9类支持度为0，导致模型对少数类几乎无法学习，macro-F1被严重拉低。当前1/sqrt(count)权重虽缓解，但不足以应对极端不均衡。
2. **特征维度高但信息密度低**：1155维展平特征中，大量为缺失mask和地震patch冗余信息，XGBoost max_depth=2难以捕捉测井曲线与地震patch间的非线性交互，模型容量受限。
3. **domain shift主导误差**：按母井家族LOGO4下，不同家族测井响应和岩相分布差异大，模型可能过度拟合训练家族的地质特征，验证家族上泛化差。
4. **校准增益不稳定**：prior calibration单独提升+0.0072，但与深度logit smoothing组合后不稳定，说明校准对类先验敏感，且可能过拟合特定fold的类分布。
5. **样本量过小（447）**：9类、1155维特征下，模型自由度远高于有效样本量，过拟合风险高，且LOGO4下每折训练仅315-320样本，统计功效低。

B. 建议清单

- **S1**（类别=训练策略）
  做法：将样本权重从1/sqrt(count)改为更激进的1/count或1/count^0.75，并配合class-prior校准（保留现有0.25×centered log count）。
  预期方向：少数类（如第9类）权重提升，可能提高macro-F1，但需监控多数类（第6、7类）是否下降。
  成本=低；主要风险：权重过大导致多数类过拟合，整体F1下降。

- **S2**（类别=特征设计）
  做法：仅使用13条测井曲线（26×33=858维），去掉3×3×33地震patch（297维），对比全特征结果。
  预期方向：降低维度、减少噪声，可能提升泛化；若地震patch有信息，则下降。
  成本=低；主要风险：地震patch可能包含岩相空间相关性，去掉后信息损失。

- **S3**（类别=参数调整）
  做法：将max_depth从2增至3或4，同时降低eta至0.1，增加轮数至60-80，观察验证集macro-F1变化。
  预期方向：增加模型容量，可能捕捉更复杂交互，但过拟合风险上升。
  成本=低；主要风险：小样本下更易过拟合，需严格同口径比较。

- **S4**（类别=评测诊断）
  做法：按fold分别报告macro-F1和各类别F1，识别哪些fold/类别贡献最大误差，并检查fold 2（训练缺第9类）的预测分布。
  预期方向：定位瓶颈fold或类别，指导后续针对性调整。
  成本=低；主要风险：无直接提升，但为后续实验提供依据。

- **S5**（类别=训练策略）
  做法：对训练集做SMOTE或ADASYN过采样少数类（仅对训练fold内，不触碰验证），再训练XGBoost。
  预期方向：增加少数类样本多样性，可能提升少数类F1。
  成本=中；主要风险：过采样可能引入噪声，且需确保不泄漏验证信息。

- **S6**（类别=特征设计）
  做法：对测井曲线按深度做滑动窗口均值或差分（如3点平滑），减少噪声，保持维度不变。
  预期方向：平滑可能提升信噪比，但可能丢失薄层信息。
  成本=低；主要风险：过度平滑导致岩相边界模糊。

C. 低成本优先实验（最多3个）

1. **实验1：样本权重敏感性**
   候选参数：权重=1/count（替换当前1/sqrt(count)），其余设置不变（40轮、max_depth=2、eta=0.2、全特征）。
   比较方式：同口径LOGO4×3seed，记录mean fixed_schema_macro_f1及std，与baseline（0.1949）和calibration-only（0.2022）对比。

2. **实验2：特征消融**
   候选特征：仅测井+mask（858维），去掉地震patch；其余设置同baseline。
   比较方式：同口径LOGO4×3seed，对比baseline和实验1结果，判断地震patch贡献。

3. **实验3：深度与轮数调整**
   候选参数：max_depth=3, eta=0.1, 轮数=60，其余同baseline。
   比较方式：同口径LOGO4×3seed，对比baseline，观察是否提升。

D. 暂不建议

- **不建议训练更大MOMENT或增加模型复杂度**：当前样本量447，大模型极易过拟合，且已有证据显示MOMENT贡献不显著（pretrained-vs-random差仅+0.0029），投入产出比低。
- **不建议深度logit smoothing与prior calibration组合**：已测试不稳定，且单独增益有限（+0.0038），组合风险高。
- **不建议尝试复杂集成或深度学习架构**：小样本下统计功效不足，任何复杂模型都难以在LOGO4下获得可靠提升。

E. 归因边界

- 当前结果不能断言MOMENT对系统有正向贡献。完整系统（含MOMENT）相对calibration-only为-0.0006，且早期pretrained-vs-random差仅+0.0029，未达0.005阈值。因此，任何提升应归因于prior calibration或XGBoost本身，而非MOMENT组件。
- 所有新实验结果仅作为探索性证据，不能视为无偏holdout估计；最终结论需在frozen holdout上验证（但当前禁止访问）。
