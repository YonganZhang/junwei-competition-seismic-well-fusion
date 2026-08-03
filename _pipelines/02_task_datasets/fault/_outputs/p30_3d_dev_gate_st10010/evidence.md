# Fault contiguous 3-D development asset from ST10010

- Generated at: 2026-08-03T13:21:04.304800+00:00
- Gate status: READY
- Reason code: LEGAL_CONTIGUOUS_3D_DEVELOPMENT_VOLUME_READY
- Frozen holdout accessed: `False`

## Source stack

- Seismic archive: `_sandbox/volve_data/Volve_Seismic_ST10010.zip`
- Extracted stack: `_sandbox/volve_data/_extracted_seismic/ST10010/Stacks/ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy`
- Stack member: `ST10010/Stacks/ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy`
- Input sample interval ms: 4.0
- Input time range ms: [12.0, 3408.0]

## Subvolume

- Coordinate order: `tline, iline, xline`
- Inline range: [10095, 10235]
- Crossline range: [2175, 2350]
- Time index range: [605, 785]
- Time ms range: [2432.0, 3152.0]
- Saved subvolume: `_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_st10010/dev_subvolume.npz`

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
- Verified background voxels: 3673671
- Unknown voxels: 794301
- Positive voxels: 23724
