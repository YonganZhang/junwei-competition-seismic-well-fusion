# Facies P11 residual fusion experiment

## Summary

This fixed-development run keeps the strong small-model logits as the main route and tests a cached SAM2 residual branch with a bounded sigmoid gate.
The residual correction is exactly `sigmoid(gate) × 0.05 × tanh(residual_logits)`, so its absolute per-logit contribution cannot exceed 0.05.
The gate sees only inference-time summaries of the main logits and frozen cached SAM2 features; labels are used only by the training loss.

## Result table

| Task | Variant | mIoU | Δ vs baseline | Accuracy | Macro F1 | Gate mean | Correction mean abs | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| F3 | strong_small_baseline | 0.137977 | +0.000000 | 0.318844 | 0.223113 | 0.000000 | 0.00000000 | trained |
| F3 | direct_sam2 | 0.089780 | -0.048198 | 0.310902 | 0.143401 | 0.000000 | 0.00000000 | trained |
| F3 | pretrained_residual | 0.137977 | -0.000000 | 0.318844 | 0.223113 | 0.016855 | 0.00000250 | non_beneficial |
| F3 | random_sam2_residual | 0.137978 | +0.000001 | 0.318846 | 0.223114 | 0.016851 | 0.00001241 | trained |
| F3 | gate_zero | 0.137977 | +0.000000 | 0.318844 | 0.223113 | 0.000000 | 0.00000000 | exact main-route control |
| Penobscot | strong_small_baseline | 0.124216 | +0.000000 | 0.431519 | 0.170191 | 0.000000 | 0.00000000 | trained |
| Penobscot | direct_sam2 | 0.064715 | -0.059501 | 0.476048 | 0.086316 | 0.000000 | 0.00000000 | trained |
| Penobscot | pretrained_residual | 0.124217 | +0.000001 | 0.431524 | 0.170192 | 0.017725 | 0.00000568 | non_beneficial |
| Penobscot | random_sam2_residual | 0.124219 | +0.000003 | 0.431540 | 0.170194 | 0.017694 | 0.00003712 | trained |
| Penobscot | gate_zero | 0.124216 | +0.000000 | 0.431519 | 0.170191 | 0.000000 | 0.00000000 | exact main-route control |

## Conclusion

- F3 decision: **NON_BENEFICIAL**.
- Penobscot decision: **NON_BENEFICIAL**.
- `NON_BENEFICIAL` means the pretrained residual did not beat both the exact same-run strong baseline and the random-SAM2 residual control by the fixed 0.005 mIoU minimum; tie-level numerical changes are not promoted.

## Residual gate interpretation

- small logits remain the main route;
- frozen SAM2 features are computed once per fold/encoder condition and cached in CPU float16;
- the SAM2 correction is bounded by tanh, a fixed 0.05 scale, and a sigmoid gate;
- gate=0 reproduces the main-route logits exactly (checked at `1e-7` tolerance);
- random-init SAM2 is kept as the structure control.

## Evidence boundary

- No holdout or frozen-test data were opened for this run.
- Only the locked development folds 0 and 4 were evaluated.
- F3 and Penobscot used separate manifests, TaskSpecs, label spaces, models, and output rows.
- Results are bounded development ablations, not fresh-blind or contest-test scores.
