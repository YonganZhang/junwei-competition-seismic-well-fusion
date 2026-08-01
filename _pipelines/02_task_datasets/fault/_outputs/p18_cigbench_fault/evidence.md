# Fault CIG-Bench incremental comparison audit

- Generated at: 2026-08-01T06:49:55.588662+00:00
- Status: DATA_GATE_BLOCKED
- Reason code: NO_VALID_FAULT_3D_DEVELOPMENT_VOLUME

## Install and weight proof

- Package: `cig_bench` `0.2.0`
- Predictor: `cig_bench.predictor.fault.FaultPredictor`
- Weight path: `/mnt/data/yongan-admin-2/.cache/modelscope/models/douyimin--CIG-Bench/snapshots/master/CIG-Bench-Fault.pth`
- Weight bytes: `138135635`
- Weight sha256: `9d6e8668f0fd27cf0f0131d2b600d79e26bd5cd8f483c3ca2d1614d448351a36`

## Gate verdict

- `contiguous_3d_development_blocks_missing`: blocked
  - fault/train.h5 is a 2-D patch bundle with sample_shape=[1, 33, 65]; no contiguous 3-D development volume is present in the fault track.
- `coverage_audited_verified_background_missing`: blocked
  - fault_points.npz contains sparse fault sticks only (fields=['crossline', 'fault_name', 'inline', 'stick_no', 'twt_ms', 'utmx', 'utmy']); no verified background mask asset is registered.
- `explicit_unknown_mask_provenance_missing`: blocked
  - No explicit unknown-mask artifact or provenance record is registered for a 3-D development volume.
- `group_isolated_development_split_missing`: blocked
  - The audited fault split in build_summary.json is the 2-D train/test patch split with split_plan={'train': [9985, 10284], 'guard': [10285, 10292], 'test': [10293, 10369]}; no group-isolated 3-D development split exists.

## Current baseline reference

- Model: `fault_local_logistic`
- Run: `audited_v2`
- Validation F1 at selected threshold: 0.029329675915713147
- Threshold: 0.4505760073661804 (validation_max_f1)
- Test metrics: `{"dice": 0.015530801391690788, "f1": 0.015530801391690788, "fn": 144, "fp": 168215, "iou": 0.0078261740734411, "precision": 0.00783282117221, "recall": 0.9021739130434783, "tn": 25949, "tp": 1328}`

## Asset probe

- Train HDF5 summary: `{"label_shape": [33, 65], "n_samples": 256, "path": "_data/processed/fault/train.h5", "sample_kind": "non_fault", "sample_shape": [1, 33, 65], "split": "train"}`
- Fault points summary: `{"count": 3998, "fields_present": ["crossline", "fault_name", "inline", "stick_no", "twt_ms", "utmx", "utmy"], "keys": ["utmx", "utmy", "twt_ms", "inline", "crossline", "fault_name", "stick_no"], "mask_fields_present": [], "path": "_pipelines/01_common_preprocess/outputs/fault_points.npz"}`
- Frozen holdout accessed: `False`

## Minimum unblock contract

- A contiguous 3-D fault development volume with explicit tline/iline/xline coordinates.
- A verified-background mask provenance record for that development volume.
- An explicit unknown-mask provenance record for the same volume.
- A group-isolated development split manifest that keeps train/validation/dev disjoint by group.
- A fault-adapter path that consumes the dev volume without opening frozen holdout/test.h5.

## Conclusion

The CIG-Bench package installs and the default FaultPredictor checkpoint downloads successfully, but the current fault track does not expose a legal contiguous 3-D development volume with explicit audited background/unknown masks and a group-isolated dev split. The comparison is therefore blocked.
