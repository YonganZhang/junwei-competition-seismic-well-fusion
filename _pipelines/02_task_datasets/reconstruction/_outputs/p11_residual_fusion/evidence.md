# P11 Reconstruction Residual-Fusion Evidence

## Outcome

- Decision: `VERIFIED_NO_PROMOTION`; default enabled: `False`.
- PyKrige development OOF RMSE: `0.0284497281702`.
- Gated pretrained-OpenMind residual RMSE: `0.0288029396248`.
- Gated no-foundation structural residual RMSE: `0.0285397611875`.
- Relative gain vs PyKrige: `-1.241528%`.

## Holdout firewall

- The runner accepts only `train.h5`; it has no test-loader argument.
- OpenMind extraction read only `seismic_patch[0:3]` and `seismic_patch[8]`.
- Targets and PyKrige predictions came from hash-verified P5 Stage-3 development OOF archives.
- Every residual training base prediction was OOF; every gate was calibrated by inner OOF.
- No frozen-test path, array, label, prediction, or historical metric was opened.

## Legal ablations

- `pykrige_oof`: unchanged cross-fitted strong baseline.
- `gate_zero_exact`: bitwise-identical copy of PyKrige OOF.
- `pretrained_openmind_residual_ungated`: genuine cached-weight encoder features without gate.
- `pretrained_openmind_residual_gated`: same residual plus bounded `[0,1]` confidence gate.
- `no_foundation_structural_residual_gated`: same residual/gate algorithm using seismic+coordinate controls.
- `original_direct_foundation_path`: `not_comparable` (the archived direct OpenMind path is strict-lane evidence and does not persist row-aligned conditional OOF predictions).

## Genuine feature identity

- Checkpoint SHA-256: `7a847af785635335c00e711d16ff4d225d86ecd5992b14c059df2b520e3ee933`.
- Source revision: `7044864404315536e92e670ef2f0ca24f11e6175`.
- Multiscale layer channels: `[32, 64, 128, 256, 320, 320]`.
- Feature cache SHA-256: `ce81f00edbf5ca77133089f79a923117f4a0e0cb812b423836dc3fc3023bd946`.

Fold/seed metrics and gate/residual distributions are in `summary.json`.
