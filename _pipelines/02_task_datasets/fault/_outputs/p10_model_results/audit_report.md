# Fault P10 model-results audit

- Source commit: `5b0f23c00599e64557e9fdf4c77c08c8803da22b`
- Primary metric: `average_precision`
- Workbook: `_outputs/p10_model_results/track_model_metrics.xlsx`
- Figures manifest: `_outputs/p10_model_results/figures_manifest.csv`
- Tables manifest: `_outputs/p10_model_results/tables_manifest.csv`
- Primary figure: `_outputs/p10_model_results/before_after_primary_metric.png`

## Conclusion

The fault lane remains `data_blocked` for any real 3D SAM-Med3D scoring attempt.
No production checkpoint exists in this checkout, and no fabricated 3D improvement is claimed.

## Evidence summary

- Baseline-only rows: 7
- Blocked foundation rows: 1
- Production checkpoint present: False

| role | path | note |
|---|---|---|
| canonical baseline metrics | `_outputs/runs/audited_v2/baseline_metrics.json` | preserved reference evidence |
| blocked SAM-Med3D gate | `_outputs/p9_sammed3d_gate/summary.json` | no legal 3D development fold |
| workbook | `_outputs/p10_model_results/track_model_metrics.xlsx` | single-sheet metrics ledger |
| figure | `_outputs/p10_model_results/before_after_primary_metric.png` | before/after comparison with blocked after-side |

## Preserved scientific conclusion

- `fault_local_logistic` remains the canonical audited reference.
- `blueyo0/SAM-Med3D:sam_med3d_turbo.pth` stays `data_blocked` because verified negatives are still 0 and there is no explicit unknown mask with a legal 3D development fold.
- The report intentionally reuses the blocked evidence rather than inventing a score.

## Validation scope

- workbook can be reopened by openpyxl;
- workbook has exactly one sheet named `模型指标`;
- row-level evidence paths are existing local files;
- figures/tables manifests index the rendered artifacts.
