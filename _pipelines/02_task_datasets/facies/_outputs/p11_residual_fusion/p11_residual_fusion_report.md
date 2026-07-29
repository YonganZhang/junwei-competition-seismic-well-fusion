# Facies P11 residual fusion experiment

## Summary

This fixed-development run keeps the strong small-model logits as the main route and tests a cached SAM2 residual branch with a bounded sigmoid gate.
The residual correction is exactly `sigmoid(gate) × 0.05 × tanh(residual_logits)`, so its absolute per-logit contribution cannot exceed 0.05.
The gate sees only inference-time summaries of the main logits and frozen cached SAM2 features; labels are used only by the training loss.

## Result table

| Task | Variant | mIoU | Δ vs baseline | Accuracy | Macro F1 | Gate mean | Correction mean abs | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| F3 | strong_small_baseline | 0.138472 | +0.000000 | 0.320658 | 0.224004 | 0.000000 | 0.00000000 | trained |
| F3 | direct_sam2 | 0.089779 | -0.048693 | 0.310900 | 0.143400 | 0.000000 | 0.00000000 | trained |
| F3 | pretrained_residual | 0.138472 | +0.000000 | 0.320658 | 0.224004 | 0.016842 | 0.00000244 | non_beneficial |
| F3 | random_sam2_residual | 0.138474 | +0.000001 | 0.320660 | 0.224006 | 0.016836 | 0.00001204 | trained |
| F3 | gate_zero | 0.138472 | +0.000000 | 0.320658 | 0.224004 | 0.000000 | 0.00000000 | exact main-route control |
| Penobscot | strong_small_baseline | 0.166713 | +0.000000 | 0.464989 | 0.224872 | 0.000000 | 0.00000000 | trained |
| Penobscot | direct_sam2 | 0.064715 | -0.101998 | 0.476048 | 0.086316 | 0.000000 | 0.00000000 | trained |
| Penobscot | pretrained_residual | 0.166714 | +0.000002 | 0.464996 | 0.224873 | 0.017736 | 0.00000588 | non_beneficial |
| Penobscot | random_sam2_residual | 0.166714 | +0.000001 | 0.465012 | 0.224867 | 0.017710 | 0.00003861 | trained |
| Penobscot | gate_zero | 0.166713 | +0.000000 | 0.464989 | 0.224872 | 0.000000 | 0.00000000 | exact main-route control |

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
