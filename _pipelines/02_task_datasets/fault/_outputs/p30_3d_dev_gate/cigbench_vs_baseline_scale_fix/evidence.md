# P30 CIG-Bench vs audited_v2 fault baseline comparison

- Generated at: 2026-08-02T13:56:17.485071+00:00
- Source commit: `59aecb0236a9bce60ca5597354e3d729b962da9c`
- P30 manifest: `_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate/manifest.json`
- P30 split manifest: `_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate/split_manifest.json`

## Gate and scope

- P30 gate status: `READY`
- P30 gate reason: `LEGAL_CONTIGUOUS_3D_DEVELOPMENT_VOLUME_READY`
- Development only: `True`
- Group isolated: `True`
- Frozen holdout accessed: `False`

## Model provenance

- CIG-Bench package: `cig_bench 0.2.0`
- CIG-Bench weight path: `/mnt/data/yongan-admin-2/.cache/modelscope/models/douyimin--CIG-Bench/snapshots/master/CIG-Bench-Fault.pth`
- CIG-Bench weight sha256: `9d6e8668f0fd27cf0f0131d2b600d79e26bd5cd8f483c3ca2d1614d448351a36`
- audited_v2 baseline model: `_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_model.joblib`
- audited_v2 baseline model sha256: `6d5668eb73f01bfe0294ba3dc23dee67edeb519263be99fa034156dd40395c84`

## Fit-selected thresholds

- CIG-Bench threshold: 0.000000
- Baseline threshold: 0.535703

## Guard metrics

- CIG-Bench: precision=0.002146, recall=0.010360, f1=0.003555, iou=0.001781
- Baseline: precision=0.008904, recall=0.941112, f1=0.017641, iou=0.008899

## Guard deltas (CIG - baseline)

- F1 delta: -0.014086
- Precision delta: -0.006758
- Recall delta: -0.930752
- IoU delta: -0.007118

## Fold summary

- fit: inline=[10095, 10175], shape=[181, 81, 176], scoreable_voxels=2218775, positive_voxels=8934, unknown_voxels=361561
- guard: inline=[10176, 10183], shape=[181, 8, 176], scoreable_voxels=211325, positive_voxels=1834, unknown_voxels=43523
- validation: inline=[10184, 10235], shape=[181, 52, 176], scoreable_voxels=1267293, positive_voxels=12956, unknown_voxels=389219

## Conclusion

This run compares the CIG-Bench FaultPredictor and the audited_v2 logistic baseline on the same P30 continuous 3-D development asset. Unknown voxels are excluded; thresholds are selected on the fit fold and reused on guard. The saved comparison report records the exact measured metrics.
