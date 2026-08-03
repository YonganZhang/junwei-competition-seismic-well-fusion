# P30 CIG-Bench vs audited_v2 lift and tolerance analysis

- Generated at: 2026-08-03T18:41:35.658947+00:00
- Source commit: `66e98707b612147202fab8887da4070594ae393d`
- Asset root: `_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_st10010`

## Scope and gate

- P30 gate manifest: `_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_st10010/manifest.json`
- Development only: `True`
- Group isolated: `True`
- Frozen holdout accessed: `False`

## Prior-normalized lift

- Guard positive prior: 0.008679
- CIG-Bench precision lift: 1.180x
- Baseline precision lift: 0.787x
- CIG-Bench AP lift: 1.424x
- Baseline AP lift: 0.761x
- CIG-Bench recall / prior ratio: 94.305x
- Baseline recall / prior ratio: 3.644x
- CIG-Bench predicted-positive fraction: 0.693781
- Baseline predicted-positive fraction: 0.040189
- CIG-Bench coverage ratio vs prior: 79.942x
- Baseline coverage ratio vs prior: 4.631x
- CIG-Bench fit threshold: 3.64363270458e-13

## Guard ordinary metrics

- CIG-Bench: precision=0.010238, recall=0.818430, AP=0.012357, F1=0.020223
- Baseline: precision=0.006829, recall=0.031625, AP=0.006605, F1=0.011233

## Tolerance radius 2 voxels

- CIG-Bench tolerance: precision=0.010238, recall=0.871320, F1=0.020238
- Baseline tolerance: precision=0.006829, recall=0.117230, F1=0.012906

## Radius sweep

- Guard CIG-Bench: {"radius_1": {"f1": 0.020229844310699872, "matched_prediction_voxels": 1501, "matched_truth_voxels": 1546, "precision": 0.010237767198221179, "predicted_positive_voxels": 146614, "radius": 1, "recall": 0.8429661941112323, "truth_positive_voxels": 1834}, "radius_2": {"f1": 0.020237746384917106, "matched_prediction_voxels": 1501, "matched_truth_voxels": 1598, "precision": 0.010237767198221179, "predicted_positive_voxels": 146614, "radius": 2, "recall": 0.871319520174482, "truth_positive_voxels": 1834}, "radius_3": {"f1": 0.02539342683212625, "matched_prediction_voxels": 1888, "matched_truth_voxels": 1660, "precision": 0.012877351412552689, "predicted_positive_voxels": 146614, "radius": 3, "recall": 0.9051254089422028, "truth_positive_voxels": 1834}}
- Guard baseline: {"radius_1": {"f1": 0.012466415905427188, "matched_prediction_voxels": 58, "matched_truth_voxels": 131, "precision": 0.006829153420463911, "predicted_positive_voxels": 8493, "radius": 1, "recall": 0.07142857142857142, "truth_positive_voxels": 1834}, "radius_2": {"f1": 0.012906451000249952, "matched_prediction_voxels": 58, "matched_truth_voxels": 215, "precision": 0.006829153420463911, "predicted_positive_voxels": 8493, "radius": 2, "recall": 0.11723009814612868, "truth_positive_voxels": 1834}, "radius_3": {"f1": 0.01923978207419909, "matched_prediction_voxels": 86, "matched_truth_voxels": 353, "precision": 0.01012598610620511, "predicted_positive_voxels": 8493, "radius": 3, "recall": 0.19247546346782987, "truth_positive_voxels": 1834}}
- Union CIG-Bench: {"radius_1": {"f1": 0.014649809215491975, "matched_prediction_voxels": 18115, "matched_truth_voxels": 18696, "precision": 0.007393627073042015, "predicted_positive_voxels": 2450083, "radius": 1, "recall": 0.7880627212948913, "truth_positive_voxels": 23724}, "radius_2": {"f1": 0.014656165200954628, "matched_prediction_voxels": 18115, "matched_truth_voxels": 19611, "precision": 0.007393627073042015, "predicted_positive_voxels": 2450083, "radius": 2, "recall": 0.8266312594840668, "truth_positive_voxels": 23724}, "radius_3": {"f1": 0.020314464353446977, "matched_prediction_voxels": 25176, "matched_truth_voxels": 20924, "precision": 0.010275570256191321, "predicted_positive_voxels": 2450083, "radius": 3, "recall": 0.8819760580003372, "truth_positive_voxels": 23724}}
- Union baseline: {"radius_1": {"f1": 0.013585000118282361, "matched_prediction_voxels": 837, "matched_truth_voxels": 1648, "precision": 0.007528671014166854, "predicted_positive_voxels": 111175, "radius": 1, "recall": 0.06946552014837296, "truth_positive_voxels": 23724}, "radius_2": {"f1": 0.014069178937786248, "matched_prediction_voxels": 837, "matched_truth_voxels": 2543, "precision": 0.007528671014166854, "predicted_positive_voxels": 111175, "radius": 2, "recall": 0.10719103018040803, "truth_positive_voxels": 23724}, "radius_3": {"f1": 0.020402490436073544, "matched_prediction_voxels": 1204, "matched_truth_voxels": 4170, "precision": 0.010829772880593658, "predicted_positive_voxels": 111175, "radius": 3, "recall": 0.17577137076378352, "truth_positive_voxels": 23724}}

## Verdict

- Default recommendation: `do_not_advance`
- Model classification: `diagnostic_high_recall_proposal_only`
- Summary: CIG-Bench should not be promoted as the default fault detector yet. It is a diagnostic/high-recall proposal: precision lift is about 1.18x, AP lift about 1.42x, fit threshold is near zero, predicted positive fraction is about 69.4%, and radius-2 tolerance F1 is about 0.0202.
- Minimum advancement conditions: ["Lift precision materially beyond the current ~1.18x level without collapsing recall.", "Reduce predicted positive fraction substantially below the current ~69.4% coverage.", "Raise tolerance F1 beyond the current radius-2 ~0.0202 while keeping guard and validation stable.", "Demonstrate calibration stability across the development folds with the same fit threshold."]

## Interpretation

CIG-Bench remains the higher-recall model, but the prior-normalized lift is still modest and the tolerance-based scores do not transform it into a clean, high-precision detector. The per-fold tolerance sweep is included to show whether the gain survives a near-boundary match criterion instead of exact voxel equality, and the union result is micro-aggregated from per-fold 3-D counts rather than flattened across fold boundaries.
