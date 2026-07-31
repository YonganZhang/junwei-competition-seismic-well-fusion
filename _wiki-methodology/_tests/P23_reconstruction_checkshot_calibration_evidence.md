# P23 reconstruction checkshot calibration evidence

## Command gate

```bash
python3 -m unittest -v \
  _pipelines/02_task_datasets/reconstruction/_tests/test_p23_checkshot_calibration.py
```

Formal rerun when the raw Volve archive is mounted:

```bash
VOLVE_RAW_PROJECT_ROOT="${VOLVE_RAW_PROJECT_ROOT:?set raw Volve project root}" \
  python3 _pipelines/02_task_datasets/reconstruction/p23_checkshot_calibration.py
```

## Live evaluation journey

- Asset: `Volve_Seismic_VSP.zip`, SHA-256
  `e3c7f0ce7fb2590bc2dc0a24be6df5d90af174c6fb782d95424463e615acc8f4`.
- Calibration fit wells: 19A, 19BT2, 19SR.
- Independent calibration wells: F11T2 and F15A; neither well participates in fitting,
  gating, GFM extraction or P21 model evaluation.
- Target interval: Eclipse reconstruction TVDSS range
  `2800.718505859375–3543.77587890625 m`.
- Independent rows: 80 checkshot samples across the two validation wells.
- Current weak tie pooled MAE/RMSE: `633.1867277468943/633.5426027839591 ms`.
- Checkshot candidate pooled MAE/RMSE: `8.738925152523326/11.187825208493493 ms`.
- Independent-well outcome: 2 wins, 0 losses.
- P21 downstream reference RMSE: `0.027734374378067677`.
- Full checkshot-aligned downstream RMSE: `0.02776854691114144` (3 wins, 2 losses).
- Direct-support-gated downstream RMSE: `0.027790989240265015` (3 wins, 2 losses).
- Firewall: calibration fitting reads no porosity target; downstream experiments use only the
  established development folds and do not open `test.h5` or the frozen holdout.
- Interpretation: the time-depth correction is independently validated; downstream porosity
  improvement is rejected under the frozen P21 protocol. No porosity blind-test claim is made.

## Trace / audit evidence

- The formal result is a machine-readable JSON artifact whose protocol records the fit wells,
  independent wells, target depth interval, opened datasets and claim boundary.
- All five source artifacts are SHA-256 locked in the result, including the VSP archive, well
  picks, weak-tie NPZ, seismic index and reconstruction build summary.
- This is a deterministic offline scientific evaluation rather than a live service; HTTP/DOM
  journey evidence is therefore not applicable. The command exit code, structured JSON and
  regression assertions are the relevant anti-fake-completion signals.
