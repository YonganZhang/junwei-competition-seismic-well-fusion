# Reconstruction P5.1 R0/R1 development-only evidence

## Scope

This is `development_protocol_mechanism_only`. It is not a model ranking, fresh-blind test, known-holdout confirmation, or field-generalization claim.

B0 is formally `no_pseudo_test_PORO_condition`. It retains the fixed project-level weak MD→TWT well tie used for seismic sampling, so it is not a claim of zero well-derived information.

Pseudo-well PORO values are deterministic `synthetic/reference-revealed` Eclipse target samples, not independently measured PHIE.

## R0

- Split hash: `f6a9ce60c4e47bc3d961ca3ce3e7b6de38de923462fce695a8d98292c27aaaa7`
- Sample hash: `e370a03760b51bf1235e8d5b247769641586ac2ab3b0eb1e73a249e863e05687`
- Feature hash: `46377f4d758d40c94fb966686fd9655f11807f672d000b11b98dce90d3f64fed`
- Config hash: `a62f39f93b59b463a1945fda864c2a3033dba4529397226e057c2f1daafb65a2`
- Common metric-mask hash: `f1a5fef8b48b1fd3dd7e9db496fbd9bca76d6361c2e593f5ef1be72d89355ef1`
- Physical `test.h5`, global `well_log_seq`, known metrics and known predictions read: no.

## R1

- Status: `passed`; formal rankable: `false`.
- B0: RMSE=0.0435098345, MAE=0.0389806847, bias=-0.0367727929, R²=-1.8643896015.
- B1: RMSE=0.0414922914, MAE=0.0371929511, bias=-0.0351512319, R²=-1.6049060541.
- shuffled: RMSE=0.0420969200, MAE=0.0375084178, bias=-0.0348448309, R²=-1.6813769424.
- ΔRMSE(B1-B0): -0.0020175431.
- ΔRMSE(B1-shuffled): -0.0006046286.
- Well-information gain supported on this development block: `true`.
- Shared in-memory checkpoint hash: `1318443c0413ffcf4e60afe7aeb35cace4918f78e9b7067b0e4bb5ebf434b128`.
- At least one R² is non-positive; this cannot support a spatial-generalization-success claim.

Formal/fresh-blind/field-generalization lanes remain blocked. The historical holdout was not opened, consumed, scored, or relabelled by R0/R1.
