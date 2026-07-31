# Sweetspot P5.1 R0/R1

This directory is a track-private, development-only protocol check for seven
independent targets. It never produces an aggregate sweetspot label or score.

## R0 contracts

`contracts/T1.v1.json` through `contracts/T7.v1.json` are separately versioned
and content-hashed. The validator fails closed if a formula, causal window,
censoring rule, split, head isolation, proxy warning, or test firewall changes.

- T1 is only the continuous `0.0314*sqrt(KLOGH/PHIF)` RQI proxy.
- T2 is only the near-binary `SAND_FLAG` net-reservoir/sand proxy, not pay.
- T3 is mean observed `BORE_OIL_VOL` over exactly `t0+1..t0+30` calendar days;
  explicit zero is retained, missing is not zero, and fewer than 24 observed
  days or an incomplete right boundary is censored.
- T4 is only the next-30-day, seven-consecutive-positive-calendar-day water
  onset proxy. Explicit zero interrupts a run; indeterminate missingness is
  censored. The formal failure/survival lane remains unapproved.
- T5 remains `not_feasible`; simulation proxies are forbidden.
- T6 is PHIF and T7 is KLOGH in mD with a fixed `log1p` model transform. They
  use separate heads and mother-well splits. R1 is explicitly a development-only
  raw-well-log mechanism lane, not a replacement for the blocked historical
  multimodal lane.

## R1 command and limits

Use the caller-provided shared tabular environment; do not install another environment:

```bash
TABULAR_PYTHON="${VOLVE_P5_TABULAR_PYTHON:?shared tabular interpreter}" \
PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" \
  -m _pipelines.02_task_datasets.sweetspot.p5.r01.runner
```

The runner opens only raw ZIP members/workbook rows authorized by P4
development groups. It has no physical-test or known-holdout argument. It uses
one fixed Huber linear model for regression and one fixed logistic model for
binary tasks, root seed 2693, no HPO, and fold-train-only preprocessing. The
random 80/20 lane is a leakage diagnostic and is always
`invalid_for_selection/not_rankable`; the legal lane is group/mother-well LOGO
with causal 30-day production windows. R1 publishes seven separate status
boards, never a final ranking. At least ten-model fair comparison is deferred
to R2.

Portable evidence is written only to this directory's `_outputs/`:

- `r0_contract_registry.json`
- `r0_data_audit.json`
- `r1_results.jsonl`
- `r1_summary.json`
- `artifact_manifest.json`

No labels, HDF5, predictions, checkpoint, model, cache, or frozen-test metrics
are written. Re-running refuses to overwrite evidence unless the operator
explicitly passes `--overwrite`; that flag can replace only the five files
above.

Tests:

```bash
TABULAR_PYTHON="${VOLVE_P5_TABULAR_PYTHON:?shared tabular interpreter}" \
PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" \
  -m unittest _pipelines/02_task_datasets/sweetspot/tests/test_sweetspot_p5_r01.py -v
```
