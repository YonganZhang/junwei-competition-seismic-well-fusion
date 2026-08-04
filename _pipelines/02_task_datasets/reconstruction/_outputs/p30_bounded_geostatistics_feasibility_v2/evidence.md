# P30 bounded geostatistics feasibility

- P21 development OOF RMSE: `0.027734374378`.
- Anisotropic ordinary kriging RMSE: `0.030569516403`.
- Regression-kriging proxy RMSE: `0.030093884156`.
- Decision: `FEASIBLE_NO_PROMOTION`; P21 remains default: `True`.

The pilot used only the legal Volve development train container and the existing
five buffered spatial folds. Classical co-kriging remains blocked because the P21
fold cache has no aligned independent well-log secondary variable; the implemented
regression-kriging route is therefore a bounded seismic-secondary feasibility proxy.
P21's historical vertical weight is dimensionless and must not be described as a
physical directional variogram. No frozen test or holdout was opened.
The covariance-form variance uses `C(0) - w^T c - mu`; correcting the
previous multiplier sign does not change weights, mean predictions, RMSE, or decision.
