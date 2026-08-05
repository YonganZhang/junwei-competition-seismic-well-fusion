---
phase_id: P39
status: accepted
severity: major
owner_col: COL4
source: experiment
closure_evidence: _pipelines/02_task_datasets/reconstruction/_outputs/p39_query_local_well_seismic_fusion/artifact_manifest.json
---

# P39.1 fixed-base query-local real-well PHIF well-seismic fusion

Decision: `FEASIBLE_NO_PROMOTION`; default: `locked_p38_well_only`.

P39.1 fixes the P39 attribution interface so every Iteration-3 weight control shares the exact pretrained/locked P38 well-only base in every outer fold. MOMENT and GFM modes now change only correction-head features. Target, split, action allowlist, selected steps, thresholds and the three-iteration limit are unchanged.

## Results

| model | 15/9-19 | 15/9-F-11 | 15/9-F-15 | macro RMSE |
|---|---:|---:|---:|---:|
| locked P38 well-only | 0.069284112717 | 0.064867699667 | 0.091791486719 | 0.075314433034 |
| moment_random_gfm_random | 0.070640321249 | 0.065445885806 | 0.091574477140 | 0.075886894732 |
| moment_pretrained_gfm_random | 0.070648525879 | 0.065460817830 | 0.091474327506 | 0.075861223738 |
| moment_random_gfm_pretrained | 0.070797109072 | 0.065420397813 | 0.091598762905 | 0.075938756597 |
| moment_pretrained_gfm_pretrained | 0.070768236209 | 0.065447913171 | 0.091509302980 | 0.075908484120 |

## Fixed-base attribution audit

| control | pre-fix macro RMSE | fixed-base macro RMSE | fixed-base well wins |
|---|---:|---:|---:|
| moment_random_gfm_random | 0.074681686544163 | 0.075886894731805 | 1/3 |
| moment_pretrained_gfm_random | 0.075861223738384 | 0.075861223738384 | 1/3 |
| moment_random_gfm_pretrained | 0.074734595356787 | 0.075938756596606 | 1/3 |
| moment_pretrained_gfm_pretrained | 0.075908484120144 | 0.075908484120144 | 1/3 |

Every per-control/per-fold base hash matches the shared pretrained P38 base; all full, outer-train cross-fitted, and held-row max-absolute differences are exactly zero. The row-aligned base arrays are retained in `predictions.npz`.

Final both-pretrained macro RMSE: `0.075908484120144`. Paired 20 m block-bootstrap CI95 candidate-minus-base: `[-0.000125432553584, 0.001197943276227]`.

Cyclic mismatch delta: `-0.000054045177568` with `0/3` parents degraded. Full +160 ms query mismatch delta: `-0.000038990671034` with `0/3` parents degraded.

Train-only calibration is finite and non-vacuous, and zero-gate fallback exactly equals the locked well-only prediction.

Failed promotion gates: `bootstrap_ci95_upper_below_zero`, `both_pretrained_below_both_random`, `both_pretrained_below_moment_pretrained_gfm_random`, `cyclic_mismatch_degrades_macro_and_two_parents`, `full_twt_mismatch_degrades_macro_and_two_parents`, `macro_below_locked_well_only`, `wins_at_least_two_parents`.

## Boundary

This is a three-parent Volve native-PHIF experiment. PHIF remains distinct from PHIE and Eclipse PORO; P21/P30 are history, not cross-target controls. The result does not establish field-wide generalization or disprove traditional geostatistics.

Exact rerun commands, row-aligned predictions, verification, and artifact hashes are stored beside this finding.
