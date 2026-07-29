# Facies P10 model-results audit

## Conclusion

The archived evidence shows a non-beneficial SAM2 integration: same-split SAM2 remains below the locked strong baselines on both F3 and Penobscot. No reproducible code defect was proven from the archived artifacts, so no model repair was applied in this pass.

## Before / after primary metric

| Dataset | Before (SAM2 pretrained mIoU) | After (locked strong baseline mIoU) | Delta |
|---|---:|---:|---:|
| F3 | 0.082017 | 0.131316 | -0.049299 |
| Penobscot | 0.076754 | 0.132021 | -0.055267 |

## Root cause / fix

- Root cause: no gain on the locked same-split development evidence; the integration is honest but non-beneficial.
- Fix applied: none in this pass; the right conclusion is `non_beneficial`, not a fabricated repair.

## Evidence boundary

- Frozen test and known holdout were not reopened for tuning.
- The workbook and manifests reference archived evidence only.
- Checkpoint paths are recorded as runtime references where the checkout does not contain a persisted weight file.

## Residual risk

- Because no persisted checkpoint files exist in this checkout, the workbook uses logical runtime checkpoint references from the archived JSON evidence rather than a file on disk.
