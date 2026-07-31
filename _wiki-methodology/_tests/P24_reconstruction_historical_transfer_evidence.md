# P24 reconstruction historical-version transfer evidence

## Claim boundary

P24 is an `unseen_same_field_historical_version_transfer` test. It is not a
fresh blind holdout, an independent field, or a competition-hidden test. The
historical target member had not been used by the reconstruction pipeline, but
it belongs to the same Volve geomodel and the same final grid.

## Frozen provenance

- Preregistration commit: `1a3a0566a043fb593a502d1410abfdcb909519cf`.
- Execution-code commit: `4ee7a32e9e3b460a4236a4391a623391f227d334`.
- RMS archive SHA-256: `e09d62cf5a33408234503621cafbe7b8b63c7011bbc8ba6be31ab97712adbe9e`.
- Historical member SHA-256: `86ae0cdc0619e0eb703f99f8badb256678d3878fd12add1666bbdf0809099044`.
- Historical payload SHA-256: `1950a9fd751e6a8122d8adb69980676a8af3b77e04615f041aa1c38481a3aeaf`.
- Grid: `63×100×108` KJI; mapped RMS points: 508,622.

## Preflight gate

Before opening historical target values, P24 reproduced all five existing
PyKrige conditional OOF folds. Maximum absolute prediction delta was
`7.450331213076922e-09`, below the registered `1e-07` tolerance. The genuine
pretrained GFM cache was hash-verified with shape `3×12583×96`; feature
extraction did not use the target.

## Live evaluation journey

- Rows: 10,240 across five unchanged spatial folds.
- PyKrige RMSE/MAE: `0.02823541000303589 / 0.02129357859238752`.
- Frozen P21 RMSE/MAE: `0.027825182662981594 / 0.020826337774812152`.
- Relative RMSE improvement: `1.4528825329973549%`.
- Fold outcomes: 4 wins, 1 loss, 0 ties.
- Whole-fold bootstrap probability candidate better: `0.94595`; 95% delta
  interval `[-0.0009158514504419742, 0.00008813574002320204]`.
- Registered gate: at least 1% pooled RMSE improvement and at most one fold
  loss. Result: passed.

The bootstrap interval still crosses zero, so this is transfer support under
the registered operational gate, not a claim of a broad population-level
effect.

## Trace / SSDO audit evidence

This is an offline scientific pipeline rather than an HTTP/DOM user journey.
The anti-fake-completion signals are the preregistration commit, immutable
opening record, exact RMS/Eclipse mapping assertion, structured prediction
archive, SHA-256 artifact manifest, metric recomputation and claim-boundary
assertion. A second normal execution fails closed after the opening record
exists; only read-only verification remains available.

## Command gate

```bash
/mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python -m unittest -v \
  _pipelines/02_task_datasets/reconstruction/_tests/test_p24_historical_transfer.py \
  _pipelines/02_task_datasets/reconstruction/_tests/test_p21_fixed_foundation_ensemble.py

/mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python \
  _pipelines/02_task_datasets/reconstruction/p24_historical_transfer.py \
  --verify-only \
  --data-dir ../track-reconstruction/_data/processed/reconstruction \
  --stage3-root ../p5-stage3-reconstruction/_tmp/p5_stage3_reconstruction \
  --rms-zip ../../../_sandbox/volve_data/Volve_Reservoir_Model-RMS_model.zip \
  --eclipse-zip ../../../_sandbox/volve_data/Volve_Reservoir_Model-Eclipse_model.zip \
  --pykrige-site /mnt/data/yongan-admin-2/.cache/volve-p5/envs/tabular-cpu/lib/python3.11/site-packages
```

The second command performs hash/metric verification only; the one-time target
opening cannot be rerun through the normal execution path.
