# Facies P10 model-results audit

## Interface audit matrix

| Item | Code evidence | Archived evidence | Status | Note |
|---|---|---|---|---|
| Input channels | `_models/facies/sam2_semantic.py:forward` | `_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json` | audited | SAM2 consumes [B,1,H,W] and repeats to 3 channels before image-encoder normalization. |
| Amplitude scaling | `_models/facies/sam2_semantic.py:forward` | `_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json` | audited | Clamps to [-5, 5], rescales to [0,1], and applies ImageNet mean/std. |
| Native SAM2 preprocessing | `_models/facies/sam2_semantic.py:build_model` | `_models/gaia_dagt/foundation_routes.v1.json` | audited | Official SAM2 build is loaded with apply_postprocessing=False. |
| Prompt leakage | `_models/facies/sam2_semantic.py; no prompt input exists` | `_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json` | audited | No validation truth prompt path exists; conditioning is explicitly spatial_prompt:none. |
| Label mapping | `_pipelines/02_task_datasets/facies/pipeline_contract.py` | `_pipelines/02_task_datasets/facies/_outputs/p5_stage3/p5_stage3_results.jsonl` | audited | F3 stays 10-class, Penobscot stays 8-class; independent TaskSpecs remain separate. |
| Decoder / head | `_models/facies/sam2_semantic.py; _pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py` | `_pipelines/02_task_datasets/facies/_outputs/p10_model_results/p10_sam2_repair_audit/repair_blocker.json` | data_blocked | The intended gated residual repair head is blocked until hydra is available in the audited SAM2 checkout. |
| PEFT / freeze policy | `_pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py` | `_pipelines/02_task_datasets/facies/_outputs/p10_model_results/p10_sam2_repair_audit/repair_blocker.json` | data_blocked | Base adapter freeze policy is implemented, but the repair probe is blocked before execution. |
| Loss | `_pipelines/02_task_datasets/facies/p9_sam2_effect.py` | `_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json` | audited | Weighted cross-entropy on raw logits; softmax is inference/evaluation only. |
| Postprocess | `_pipelines/02_task_datasets/facies/p4_metrics.py` | `_pipelines/02_task_datasets/facies/_outputs/p5_stage3/*.json` | audited | Argmax/softmax are used only at evaluation and visualization; no threshold tuning occurs. |
| Eval parity | `_pipelines/02_task_datasets/facies/facies_p5_stage3.py; _pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py` | `_pipelines/02_task_datasets/facies/_outputs/p5_stage3/*.json` | audited | Same locked development folds, same sample caps, same seed discipline, no frozen-test access. |

## Conclusion

The previous p10 artifact only framed SAM2 against the locked baseline. This revision keeps the archived reference comparison but records the actual repair attempt as data_blocked rather than pretending a repaired metric exists. No non_beneficial claim is made from an incomplete audit.

## Repair probe blocker

- Status: `data_blocked`
- Command: `CUDA_VISIBLE_DEVICES=3 /mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python _pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py --task-id facies_f3 --manifest /mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p4-training-integration/_tmp/p4-acceptance/facies_f3/split_manifest.json --processed-root /mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/track-facies/_data/processed --device cuda:0`
- Error: `ModuleNotFoundError: No module named 'hydra'`
- Checked environments: system python, torch-common env

## Archived reference comparison

| Dataset | Archived pretrained adapter mIoU | Archived locked baseline mIoU | Delta |
|---|---:|---:|---:|
| F3 | 0.079364 | 0.113056 | -0.033691 |
| Penobscot | 0.046141 | 0.155897 | -0.109756 |

## Evidence boundary

- Frozen test and known holdout were not reopened for tuning.
- The workbook and manifests reference archived evidence plus a blocker file for the attempted repair probe.
- Checkpoint paths are recorded as runtime references where the checkout does not contain a persisted weight file for the historical stage-3 baselines.

## Residual risk

- The blocked repair probe is intentionally not papered over with a local hydra stub or a package install.
- If a future promoted model is desired, the next step is to add the missing dependency into the audited SAM2 source environment or switch to a different audited foundation checkout, then rerun the same fixed-dev comparison.
