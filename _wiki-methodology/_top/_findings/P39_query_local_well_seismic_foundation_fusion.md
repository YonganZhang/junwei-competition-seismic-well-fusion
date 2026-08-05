# P39 query-local real-well PHIF well-seismic fusion

Decision: `FEASIBLE_NO_PROMOTION`; default: `locked_p38_well_only`.

P39 kept the exact P38 three-parent native-PHIF LOGO3 rows and replaced no target, split, or gate. Iteration 2 changed only the prediction interface; Iteration 3 changed only the seismic representation. No fourth iteration was run.

## Results

| model | 15/9-19 | 15/9-F-11 | 15/9-F-15 | macro RMSE |
|---|---:|---:|---:|---:|
| locked P38 well-only | 0.069284112717 | 0.064867699667 | 0.091791486719 | 0.075314433034 |
| moment_random_gfm_random | 0.071566575997 | 0.062725143704 | 0.089753339932 | 0.074681686544 |
| moment_pretrained_gfm_random | 0.070648525879 | 0.065460817830 | 0.091474327506 | 0.075861223738 |
| moment_random_gfm_pretrained | 0.071661868687 | 0.062716001297 | 0.089825916087 | 0.074734595357 |
| moment_pretrained_gfm_pretrained | 0.070768236209 | 0.065447913171 | 0.091509302980 | 0.075908484120 |

Final both-pretrained macro RMSE: `0.075908484120144`. Paired 20 m block-bootstrap CI95 candidate-minus-base: `[-0.000125432553584, 0.001197943276227]`.

Cyclic mismatch delta: `-0.000054045177568` with `0/3` parents degraded. Full +160 ms query mismatch delta: `-0.000038990671034` with `0/3` parents degraded.

Train-only calibration is finite and non-vacuous, and zero-gate fallback exactly equals the locked well-only prediction.

Failed promotion gates: `bootstrap_ci95_upper_below_zero`, `both_pretrained_below_both_random`, `both_pretrained_below_moment_pretrained_gfm_random`, `both_pretrained_below_moment_random_gfm_pretrained`, `cyclic_mismatch_degrades_macro_and_two_parents`, `full_twt_mismatch_degrades_macro_and_two_parents`, `macro_below_locked_well_only`, `wins_at_least_two_parents`.

## Boundary

This is a three-parent Volve native-PHIF experiment. PHIF remains distinct from PHIE and Eclipse PORO; P21/P30 are history, not cross-target controls. The result does not establish field-wide generalization or disprove traditional geostatistics.

Exact rerun commands, row-aligned predictions, verification, and artifact hashes are stored beside this finding.
