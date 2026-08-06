---
phase_id: P38
status: accepted
severity: major
owner_col: COL2
source: runtime
created_at: 2026-08-05
closed_at: 2026-08-05
closure_evidence: _pipelines/02_task_datasets/reconstruction/_outputs/p38_real_well_phif_direct_seismic/verification.json
---

# P38 real-well PHIF and direct-seismic fusion

## Decision

`FEASIBLE_NO_PROMOTION`. P38 closes a real-well, real-seismic three-parent
Volve pilot, but the frozen pretrained MOMENT+GFM fusion does not beat the
strongest budget-matched control. No fourth scientific iteration was opened.

## Phase 0 closure

The target is each parent's native published CPI `PHIF` in V/V fraction. It
is neither `PHIE` nor Eclipse `PORO`: on 1,967 same-well 15/9-19 A rows,
PHIF versus PHIE has RMSE `0.07790825422663222`, MAE
`0.05228949627462001`, correlation `0.6898342670399774`, and maximum
absolute difference `0.2631971103083672`. The 42 finite PHIF zero rows in
15/9-19 A were retained and kept distinct from the explicit `-999.25` null.

Direct ST0202 extraction uses
MD → actual-survey TVDSS/ED50 UTM 31N → checkshot TWT → affine IL/XL →
nearest native 4 ms sample. It does not use `train.h5` KJI, frozen test data,
padding, waveform interpolation, or target-derived alignment. The native
seismic section is `[3, 400, 160]`: amplitude, five-sample local RMS, and
two-sample vertical gradient.

| parent | physical PHIF rows | retained joint rows | coverage |
|---|---:|---:|---:|
| 15/9-19 | 1,967 | 1,962 | 0.9974580579562786 |
| 15/9-F-11 | 2,017 | 1,877 | 0.9305899851264254 |
| 15/9-F-15 | 9,668 | 8,846 | 0.9149772445179976 |

The fixed LOGO3 split holds out one independent parent well at a time. All
normalization, imputation, feature preparation, model selection, calibration,
and agent actions are outer-train-only; the held-parent PHIF is prediction and
scoring evidence only.

## Fixed-budget result

All controls use the same small-head parameter budget and one frozen seed
(`2693`). MOMENT and GFM are frozen; the same-architecture random-init route
separates architecture from pretrained-weight contribution.

| model | 15/9-19 RMSE | 15/9-F-11 RMSE | 15/9-F-15 RMSE | equal-parent macro RMSE |
|---|---:|---:|---:|---:|
| well only | 0.06928411271673622 | 0.06486769966725374 | 0.09179148671870635 | 0.0753144330342321 |
| seismic only | 0.1574980502723219 | 0.07486964857860623 | 0.13367679616319436 | 0.12201483167137417 |
| raw-feature fusion | 0.16166115697372058 | 0.03729678822160508 | 0.09709288915493805 | 0.0986836114500879 |
| random-init MOMENT+GFM fusion | 0.11958165606704944 | 0.13117030877037925 | 0.1442549275632852 | 0.1316689641335713 |
| frozen pretrained MOMENT+GFM fusion | 0.0941056638970108 | 0.0625533913763209 | 0.08268463026874197 | 0.07978122851402457 |

The pretrained fusion beats random initialization and beats well-only on two
of three held parents, but loses the primary equal-parent macro gate to
well-only (`0.07978122851402457` versus `0.0753144330342321`). The paired
20 m MD-block bootstrap fusion-minus-well-only delta is
`0.004472364098755622`, with 95% interval
`[0.0005355760193219085, 0.008768653250911774]` and probability of fusion
improvement `0.0129`; this independently supports no promotion.

Train-only Gaussian calibration is finite and non-vacuous. For pretrained
fusion, equal-parent macro 50%/90% coverage is
`0.5659412514147172`/`0.9250444736949236`, macro Gaussian NLL is
`-1.1095621616950224`, and mean 90% interval width is
`0.2902522237778033` PHIF fraction.

## Agent and alignment ablations

The bounded agent selected only among the three Phase-0-frozen actions using
outer-train inner-LOGO evidence. Its held-parent actions were `default`,
`stronger_regularization`, and `stronger_regularization`; selected-action
macro RMSE is `0.07978122851402457` versus fixed-default
`0.08788942536657483`, a delta of `-0.008108196852550265` with two parent
wins. This demonstrates a bounded decision effect, not a promotion result.

The cyclic-well mismatch raises macro RMSE by `0.012225937719753194`, and the
fixed +160 ms TWT shift raises it by `0.00045834253943619063`. Each degrades
only one of three parents, so both preregistered two-parent robustness gates
fail. These failures are preserved rather than tuned away.

## Claim boundary and evidence

This is a three-parent Volve native-PHIF LOGO3 pilot. It does not establish
field-wide generalization, does not rank PHIF RMSE against P21/P30 Eclipse
PORO RMSE, and does not disprove traditional geostatistics. P21 remains the
Eclipse-PORO reconstruction default; P30 remains historical sparse-grid proxy
evidence with `FEASIBLE_NO_PROMOTION`.

**2026-08-06 independent re-audit note:** unlike P40/P41 (lithofacies/property,
see their finding files), P38's GFM seismic input is a native `[3, 400, 160]`
whole-trace-plus-window section, not a single collapsed CLS token, and the
independent re-audit found no trace-level label-collision defect here. But the
GFM branch still only encodes the whole trace plus a coarse 3-dim time
position, not a query-point-local waveform window — a weaker interface than
P39's later query-local design. This negative result is credible for that
specific interface; it does not establish that a depth-local seismic token
would fail the same way.

Authoritative machine-readable evidence is under
`_pipelines/02_task_datasets/reconstruction/_outputs/p38_real_well_phif_direct_seismic/`.
`predictions.npz` contains row-aligned predictions, `summary.json` records the
fixed results and gates, `verification.json` recomputes them, and
`artifact_manifest.json` locks durable hashes. Exact Phase 0, run,
verify-only, compile, and focused-test commands are in `rerun_commands.json`.
