# P37 real-well–seismic cross-modal foundation

## Decision

`BLOCKED_REAL_ALIGNED_SUPERVISION`. Phase 0 audited the real Volve well-log,
actual-survey, checkshot, ST0202 seismic-index, and legal development KJI
assets. The preregistered minimum-pilot gate did not close, so no foundation
encoder adaptation, fusion training, agent action, calibration, ablation, or
promotion experiment was run. P21 remains the reconstruction default.

## Supervision closure

Three independent parent wells were audited: `15/9-19`, `15/9-F-11`, and
`15/9-F-15`. All three expose the same six legal input families (`GR`, `RHOB`,
`NPHI`, `DT`, `RT`, and `CALI`), actual survey stations, and direct checkshots.
Their actual-survey reports make MD→TVDSS and UTM coordinates auditable; the
checkshots and ST0202 index make TVDSS→TWT→IL/XL auditable.

The required common target is absent. Only `15/9-19` contains native
`LFP_PHIE`; F11T2 and F15A publish `PHIF`. This is not a naming-only mismatch:
on 1,967 overlapping 15/9-19 A rows, aligned PHIE versus PHIF has MAE
`0.05228949627462001`, RMSE `0.07790825422663222`, correlation
`0.6898342670399774`, and maximum absolute difference
`0.2631971103083672`. PHIF therefore cannot be silently renamed to PHIE.

The legal-development KJI gate also fails independently. Using the existing
anisotropic nearest-active-cell contract `(50 m, 50 m, 2 m)` with maximum
scaled distance `6.0`, the full alignment retains:

- 15/9-19 A: 879 rows, 76 unique legal development KJI;
- F11T2: 0 rows, 0 unique legal development KJI; minimum scaled distance
  `9.05590655452919`;
- F15A: 2,594 rows, 153 unique legal development KJI.

F11T2 lies outside the accepted active-KJI support of the legal `train.h5`
development surface. Frozen test/holdout geometry was not opened and cannot be
used to recover a third group.

## Boundary

P30 remains historical sparse Eclipse-grid proxy evidence only. It was not
substituted for missing real-well supervision. Its ordinary/regression RMSEs
remain `0.030569516403486055` and `0.030093884155904194`, with
`FEASIBLE_NO_PROMOTION`; P21 remains default at RMSE
`0.027734374378067677`.

The authoritative machine-readable evidence is under
`_pipelines/02_task_datasets/reconstruction/_outputs/p37_real_well_seismic_supervision_closure/`:
`summary.json`, `asset_inventory.json`, `split_manifest.json`,
`verification.json`, `artifact_manifest.json`, `rerun_commands.json`, and
`finding.md`.

## Unblock contract

A future pilot requires at least three independent parent wells with the same
audited native PHIE semantics, common legal input curves, and nonzero support
through MD→TVDSS→TWT→UTM/ILXL→legal-development-KJI. New data or a new split
authority must be reviewed explicitly; PHIF relabelling, opening frozen
holdouts, changing the KJI threshold, or falling back to an Eclipse proxy is
not an acceptable unblock.
