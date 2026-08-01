# Fault contiguous 3-D development asset

- Generated at: 2026-08-01T13:48:28.791618+00:00
- Gate status: READY
- Reason code: LEGAL_CONTIGUOUS_3D_DEVELOPMENT_VOLUME_READY
- Frozen holdout accessed: `False`

## Subvolume

- Coordinate order: `tline, iline, xline`
- Inline range: [10095, 10235]
- Crossline range: [2175, 2350]
- Time index range: [605, 785]
- Time ms range: [2420.0, 3140.0]
- Saved subvolume: `_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate/dev_subvolume.npz`

## Mask logic

- Positive mask: exact rasterized fault-stick voxels from 3998 sparse points and 66 faults.
- Unknown mask: dilation radius {'tline': 4, 'iline': 2, 'xline': 2} plus 2-voxel boundary halo; positives excluded from unknown.
- Verified background mask: complement of positive and unknown within the selected subvolume.

## Split manifest

- Development-only: `True`
- Group-isolated: `True`
- Blocks: `[{"fault_point_count": 584, "inline": [10095, 10175], "n_inline": 81, "name": "fit", "positive_voxels": 8934}, {"fault_point_count": 107, "inline": [10176, 10183], "n_inline": 8, "name": "guard", "positive_voxels": 1834}, {"fault_point_count": 752, "inline": [10184, 10235], "n_inline": 52, "name": "validation", "positive_voxels": 12956}]`

## Gate verdict

- Status: `READY`
- Reason codes: `[]`
- Verified background voxels: 3673669
- Unknown voxels: 794303
- Positive voxels: 23724

## Provenance

- ST0202 SEG-Y: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/_sandbox/volve_data/_extracted_seismic/ST0202/Stacks/ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME.MIG_FIN.POST_STACK.3D.JS-017534.segy`
- Seismic index path: `_pipelines/01_common_preprocess/outputs/seismic_index.npz`
- Fault points path: `_pipelines/01_common_preprocess/outputs/fault_points.npz`
