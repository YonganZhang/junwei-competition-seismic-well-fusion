# P37 real-well–seismic supervision closure

Decision: `BLOCKED_REAL_ALIGNED_SUPERVISION`. No pilot was run and P21 remains default.

## Evidence

- Three independent parent wells were audited: 15/9-19, 15/9-F-11, and 15/9-F-15.
- Only 15/9-19 contains native `PHIE`. F11T2 and F15A publish `PHIF`, so the required common PHIE target is absent.
- On 15/9-19 A, aligned PHIE versus PHIF has 1967 rows, MAE 0.052289496275, RMSE 0.077908254227, correlation 0.689834267040, and max absolute difference 0.263197110308; PHIF is not a legal silent alias for PHIE.
- Actual survey PDFs and direct checkshots close MD→TVDSS→TWT→UTM/ILXL, but the existing legal train.h5 active-KJI gate retains 879 19A rows, 0 F11T2 rows, and 2594 F15A rows.
- F11T2 has zero legal development KJI rows. Opening frozen test/holdout geometry to recover it is forbidden.
- P30 remains sparse Eclipse-grid proxy history only and was not substituted for missing real-well supervision.

## Consequence

The minimum pilot gate fails twice: fewer than three native-PHIE parent wells, and fewer than three parent wells with legal development KJI support. No normalization, encoder adaptation, cache generation, training, agent action, calibration, ablation, or promotion test was run.

Exact commands are in `rerun_commands.json`; machine checks are in `verification.json` and hashes in `artifact_manifest.json`.
