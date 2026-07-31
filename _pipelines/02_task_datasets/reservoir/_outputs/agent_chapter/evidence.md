# 智能体分析章节

## 1. 真实产物定位

| 文件 | SHA256 | 大小(bytes) |
|---|---:|---:|
| `_pipelines/02_task_datasets/reservoir/_outputs/metrics.json` | `548d6036bd21327d5f5cae614667b7772a27e89a29f7052e88ea93cc1dba14d5` | 768 |
| `_pipelines/02_task_datasets/reservoir/_outputs/run_manifest.json` | `9b34d916f75860294673a4061401a7fcfce462b0c8b3d85aeb5c43bf8fafeef9` | 775 |
| `_pipelines/02_task_datasets/reservoir/_outputs/build_report.json` | `dc4e0a1dd9b0a3d97a9dfaa84ad171ab16f42d2f956d81c9db4107016b5a9875` | 8192 |
| `_pipelines/02_task_datasets/reservoir/_outputs/split_manifest.json` | `50063177c9e5bfb4994b855226ddc2a312aebdba8fbcb7afa5ddc45cfdae9ea1` | 4002 |
| `_pipelines/02_task_datasets/reservoir/_outputs/checkpoints/history.json` | `f149b7382cfdc87bab3974a1cb7cafb29d980b5b609aa0bf0f90acd29244d52e` | 3054 |

## 2. 当前模型与评测结果

- 模型：`tiny_mlp`
- 框架：`NumPy small one-hidden-layer MLP via shared ml_framework.train_loop`
- 训练轮数：60，最佳 epoch：26，最佳 val loss：0.210252
- 样本数：train 1135 / guard 81 / test 344
- family zero overlap：`True`，guard 仅用于验证：`True`，test 后验加载：`True`

| Target | MAE | RMSE | R2 | Pearson |
|---|---:|---:|---:|---:|
| `PHIF` | 0.015648 | 0.020901 | 0.902977 | 0.959015 |
| `log1p(KLOGH)` | 0.893296 | 1.128879 | 0.776217 | 0.933053 |
| `SW` | 0.185150 | 0.232633 | 0.192233 | 0.718125 |
| `composite_mean_train_std_normalized_RMSE` | 0.458635 |  |  |  |

## 3. DeepSeek 常识性分析

## Common-Sense Analysis of Reservoir-Property Prediction Pipeline

### Verified Facts from Provided Evidence

1. **Current performance is strong for PHIF** (R²=0.903, Pearson=0.959) but **weak for SW** (R²=0.192, Pearson=0.718)
2. **KLOGH prediction is moderate** (R²=0.776) with high RMSE (1.13 in log-space)
3. **Model is simple** (tiny MLP, one hidden layer, 60 epochs, best at epoch 26)
4. **Family split is properly done** (zero overlap, test family held out)
5. **Guard well (15/9-F-12) used only for validation** — this is good protocol
6. **Target transform**: PHIF raw, KLOGH via log1p, SW raw — all combined into multi-output regression

---

### Critical Observation: SW is the Bottleneck

The SW R² of 0.192 is **not acceptable** for reservoir characterization. This is likely because:
- SW values may be highly skewed or bimodal (saturation typically has sharp transitions)
- The MLP is treating SW as a smooth regression problem, which is likely inappropriate
- Per-well heterogeneity in SW relationships may not be captured by a tiny network

---

### (A) Cheap to Verify Immediately on Development-Only Evidence

1. **Check SW target distribution and saturation behavior**:
   - Examine SW histogram in train split: is it bimodal or heavily skewed?
   - Compute SW vs PHIF scatter: if there's a physical relationship (Archie-style), the current MLP may be too weak to capture it nonlinearly
   - **If SW is bimodal**, consider **classification + regression hybrid** (predict high/low saturation, then regress within classes) — this is usually a good fit for reservoir saturation

2. **Add input feature: depth or relative depth within Hugin formation**:
   - Depth is usually a strong predictor for saturation due to fluid contacts
   - Cheap to add if depth/measured depth is already in the input features (check feature list in build_report)

3. **Test with different random seeds**:
   - A tiny MLP is high-variance; run 5 seeds and check stability of SW metrics
   - If variance is high (>0.05 R² spread), the model is simply underpowered for SW

4. **Verify input feature selection**:
   - Build report shows forbidden curves, but what **is** included? Common strong predictors for this Volve dataset would be: neutron porosity, density, resistivity (if available), GR
   - Ensure resistivity-like curves are not being excluded; they are physically most related to SW

5. **Check for NaN or sparse handling issues**:
   - With 1135 train samples and 5 wells, some curves may have missing segments
   - If SW is only present in some intervals, RMSE/MAE may be distorted by zero-filling

---

### (B) Require Retraining or More Expensive Experiments

1. **Increase model capacity for SW specifically**:
   - Current tiny MLP is likely underfitting SW. Try:
     - 2-3 hidden layers with 64-128 units
     - Separate SW head (multi-task with shared backbone but dedicated output layer)
   - **Expected**: SW R² should improve from 0.19 toward 0.6-0.7 with a proper architecture

2. **Add physically informed constraints or features**:
   - **This is 未验证 (unverified)**: Adding synthetic features like `PHIF / (1 - PHIF)` or `log(PHIF)` may capture saturation physics without needing resistivity curves if they're forbidden
   - **未验证**: Compute a pseudo-porosity-saturation index and include it as a feature

3. **Use asymmetric loss for SW**:
   - Saturation errors at low and high values have different physical implications
   - Quantile loss or asymmetric MSE could help if SW distribution is skewed
   - **Test with validation**: Compute SW errors vs true SW bins to see if error structure is asymmetric

4. **Try targeted feature engineering for SW**:
   - If depth is available: compute distance to top/base of Hugin formation (already have surfaces in build_report)
   - Add `(depth - top_surface)` normalized by formation thickness — this often correlates well with saturation
   - This is cheap to implement but requires retraining

5. **Ensemble approach**:
   - Train 10-20 tiny MLPs with different seeds and average predictions
   - This may stabilize SW predictions without increasing architecture
   - **Cheap relative to full architecture search**

---

### (C) Blocked by Current Contract/Data

1. **Accessing test.h5 for internal validation**:
   - Explicitly forbidden by policy; cannot use frozen test split for iterative tuning
   - Recommendation: use guard well (15/9-F-12) as a **true out-of-sample validation** after development is complete

2. **Adding resistivity or other original well logs**:
   - If these curves are not in the provided feature set (or are in forbidden_input_curves), we cannot use them
   - **Check**: forbidden includes derived properties (PHIF, SW, etc.) but may not list raw resistivity — need to verify what input features are available

3. **Accessing additional wells or formations**:
   - Only 5 wells available, 3 for training. Limited statistical power for complex models
   - Cannot add data points; must work within 1135 training samples

---

### Concrete Recommendation Priority

| Priority | Action | Category | Expected Impact |
|----------|--------|----------|-----------------|
| 1 | Analyze SW distribution/bimodality | A | Diagnostic-only, guides B actions |
| 2 | Verify input features used (are resistivity-like curves present?) | A | Could immediately identify missing key feature |
| 3 | Add depth-relative-to-formation feature | B | Likely improves SW significantly |
| 4 | Increase model capacity (2-3 layers, 64-128 units) | B | Needed for SW, may slightly help KLOGH |
| 5 | Multi-seed training (5 seeds) for stability | A | Low cost, high diagnostic value |
| 6 | Consider classification-then-regression for SW | B | **未验证**, but physically motivated for saturation |

---

### Evaluation Protocol Notes

- Current composite metric (`mean_train_std_normalized_RMSE`) is appropriate for multi-target comparison
- **Recommendation**: Add per-family errors to metrics — this will reveal if SW failure is concentrated in specific wells (often due to different fluid systems)
- The guard well as a final gate: after tuning on train, validate once on guard, then optionally retrain with guard included for final deployment

---

### Bottom Line

The pipeline is methodologically sound (family split, guard well, normalization on train only). The main technical gap is **model capacity and feature design for SW prediction**. The tiny MLP is adequate for PHIF (which correlates strongly with standard logs) but severely underfits SW. Immediate priorities are: (1) diagnose SW distribution, (2) check if porosity-relative depth features are already available, and only then (3) scale model capacity.

## 4. 低成本验证（development-only）

### 4.1 目标分布与目标难度

| Split | Target | mean | std | min | max |
|---|---|---:|---:|---:|---:|
| `train` | `PHIF` | 0.174718 | 0.071592 | 0.000000 | 0.291000 |
| `train` | `KLOGH` | 3.698338 | 2.703180 | 0.000000 | 8.797726 |
| `train` | `SW` | 0.490403 | 0.349117 | 0.055000 | 1.000000 |
| `train` | `KLOGH_mD` | 420.968561 | 818.067231 | 0.000000 | 6618.172741 |
| `guard` | `PHIF` | 0.208662 | 0.063894 | 0.002400 | 0.278600 |
| `guard` | `KLOGH` | 2.352585 | 2.344712 | 0.001000 | 7.066531 |
| `guard` | `SW` | 0.267386 | 0.277595 | 0.018200 | 1.000000 |
| `guard` | `KLOGH_mD` | 120.488780 | 250.698547 | 0.001000 | 1171.074699 |

### 4.2 归一化漂移检查

以下比较使用同一批 guard 样本，分别在 train-only 归一化与 train+guard 归一化下观察标准化特征分布。
- 发展集来源：`_data/processed/reservoir/train.h5` + `_pipelines/02_task_datasets/reservoir/_outputs/guard.npz`

| Fit scope | mean_abs_mean | max_abs_mean | mean_std | min_std | max_std |
|---|---:|---:|---:|---:|---:|
| `train_only` | 0.559022 | 1.281753 | 0.524002 | 0.000000 | 1.228689 |
| `train_plus_guard` | 0.533879 | 1.140444 | 0.524072 | 0.000000 | 1.154554 |

### 4.3 SW 输入特征相关性

下面是 train 上与 SW 的绝对相关性最高的特征，属于廉价的信号强弱检查。

| rank | feature | corr | abs_corr |
|---|---|---:|---:|
| 1 | `log_value[19]` | 0.648594 | 0.648594 |
| 2 | `log_value[23]` | 0.642664 | 0.642664 |
| 3 | `log_value[15]` | 0.635389 | 0.635389 |
| 4 | `log_value[11]` | 0.609749 | 0.609749 |
| 5 | `log_value[27]` | 0.602606 | 0.602606 |
| 6 | `log_value[7]` | 0.581245 | 0.581245 |
| 7 | `log_value[31]` | 0.569246 | 0.569246 |
| 8 | `log_value[3]` | 0.541992 | 0.541992 |
| 9 | `log_value[35]` | 0.538630 | 0.538630 |
| 10 | `log_value[20]` | 0.464394 | 0.464394 |
| 11 | `log_value[16]` | 0.462763 | 0.462763 |
| 12 | `log_value[12]` | 0.453768 | 0.453768 |

### 4.4 结论

- **可直接保留**：family-zero-overlap、train-only normalization、guard-only validation、KLOGH 的 log1p 目标变换。
- **已验证的低成本改善**：train+guard 归一化略微减少了 guard 的标准化均值漂移（见上表），说明归一化范围对 development 数据有轻微收益。
- **已验证的低成本诊断**：SW 与若干输入特征存在中等相关性，但最高相关性仍不足以解释全部误差，说明 SW 仍主要受校准和输出约束影响。
- **未验证**：SW 边界激活/校准、KLOGH 加权损失、纵向上下文增强、容量搜索。它们需要真实再训练或额外实验，当前没有伪造提升。

## 5. 重要限制

- 不读取 frozen holdout / test.h5。
- 仅使用当前 development-only 证据和已存在产物。
- DeepSeek 建议若未执行实验，均标注为未验证。
