# P21 fixed foundation ensemble

- P19 pooled OOF RMSE: `0.027751397628`.
- Fixed ensemble pooled OOF RMSE: `0.027734374378`.
- RMSE delta: `-0.000017023250`.
- Fold outcomes vs P19: `{'win': 1, 'loss': 0, 'tie': 4}`.

The promoted route removes per-fold meta-selection and always averages
the z=4, foundation=0.1, seismic={0,0.1,0.2}, k=64, p=1.5,
blend=0.75 kernels. Four folds are prediction-equivalent to P19; the
remaining fold improves. The whole-fold bootstrap does not establish a
broad statistical effect, so the decision is a deterministic simplicity
win rather than a causal foundation-model claim.

Target-free contrastive LoRA residual routes were rejected because their
inner calibration did not transfer across outer spatial folds.
