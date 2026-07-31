# Sweetspot data readiness audit

> Validate-only evidence report. It does not define or generate labels.

## Decision boundary

- Sweetspot truth found: **false**
- Decision owner: Junwei / designated domain expert
- Hard blocker: No approved sweetspot label contract; evidence fields must not be combined into labels.

## Real source availability

| Source | Present | Audited fields |
|---|---:|---:|
| `layer1.seismic_index` | True | 9 |
| `layer1.fault_points` | True | 7 |
| `layer1.horizon_bcu_points` | True | 5 |
| `layer1.well_tie_weak` | True | 5 |
| `layer1.well_logs_clean.clean` | True | 9 |
| `las` | True | 185 |
| `production.daily` | True | 24 |
| `production.monthly` | True | 10 |

## Coverage and missingness

| Field | Coverage |
|---|---:|
| `Layer1 clean/LFP_CALI` | 2/3 tracks |
| `Layer1 clean/LFP_DT` | 3/3 tracks |
| `Layer1 clean/LFP_DTS` | 3/3 tracks |
| `Layer1 clean/LFP_GR` | 3/3 tracks |
| `Layer1 clean/LFP_NPHI` | 3/3 tracks |
| `Layer1 clean/LFP_PHIE` | 3/3 tracks |
| `Layer1 clean/LFP_RHOB` | 3/3 tracks |
| `Layer1 clean/LFP_RT` | 3/3 tracks |
| `Layer1 clean/LFP_VSH` | 3/3 tracks |
| `LAS LFP_GR` | 3/3 tracks |
| `LAS LFP_PHIE` | 3/3 tracks |
| `LAS LFP_VSH` | 3/3 tracks |
| `LAS LFP_OIL` | 3/3 tracks |
| `LAS LFP_GAS` | 3/3 tracks |
| `Production daily/WELL_BORE_CODE` | 100.0% non-null |
| `Production daily/BORE_OIL_VOL` | 58.6% non-null |
| `Production daily/BORE_GAS_VOL` | 58.6% non-null |
| `Production daily/BORE_WAT_VOL` | 58.6% non-null |

## Coordinate alignment

- `fault_points`: inline=1.0, crossline=1.0, TWT=1.0 within seismic bounds.
- `horizon_bcu_points`: inline=1.0, crossline=1.0, TWT=1.0 within seismic bounds.
- `well_tie_weak/15_9-19_A`: inline=1.0, crossline=1.0, TWT=1.0 within seismic bounds.
- `well_tie_weak/15_9-19_BT2`: inline=1.0, crossline=1.0, TWT=1.0 within seismic bounds.
- `well_tie_weak/15_9-19_SR`: inline=1.0, crossline=1.0, TWT=1.0 within seismic bounds.
- Warning: well_tie_weak is an interpolation/extrapolation approximation, not exact well tie truth

## Explicitly not produced

- No sweetspot label or proxy label.
- No `_data/processed/sweetspot` directory or train/test HDF5.
- No model, checkpoint, training run, metric, or synthetic dataset.
