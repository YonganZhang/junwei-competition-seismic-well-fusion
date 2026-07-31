# P21 reconstruction fixed ensemble acceptance evidence

- Protocol: five unchanged spatial development folds; 512 legal train labels and 2,048 validation rows per fold.
- Holdout/test: not opened; no corresponding path argument is accepted by the predictor.
- Candidate: fixed mean of the three `z4_f0.1_s{0,0.1,0.2}_k64_p1.5_b0.75` foundation kernels.
- P19 RMSE: `0.027751397627827728`.
- P21 RMSE: `0.027734374378067677`.
- Fold outcome: 1 win, 4 ties, 0 losses.
- Whole-fold bootstrap: improvement probability `0.70075`, 95% interval `[-0.0000650175, 0]`.
- Interpretation: deterministic simplicity win only; no broad statistical or causal foundation-model claim.
- Verification: `verification.json` reports `PASSED` and independently recomputes the artifact metrics.
