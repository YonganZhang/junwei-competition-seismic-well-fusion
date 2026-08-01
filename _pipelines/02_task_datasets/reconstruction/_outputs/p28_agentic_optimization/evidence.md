# P28 agentic reconstruction optimization

## Development promotion result

- Frozen P21 A0 RMSE: `0.027734374378068`.
- A2L promotion RMSE: `0.027759842173392`; relative gain vs A0 `-0.091828%`; fold outcomes `{'win': 1, 'loss': 3, 'tie': 1}`; search AUC `0.001046603`.
- A2D promotion RMSE: `0.027757927384379`; relative gain vs A0 `-0.084924%`; fold outcomes `{'win': 2, 'loss': 2, 'tie': 1}`; search AUC `0.001672371`.
- A3 promotion RMSE: `0.027756577492464`; relative gain vs A0 `-0.080056%`; fold outcomes `{'win': 2, 'loss': 3, 'tie': 0}`; search AUC `0.001701132`.
- Decision: `RETAIN_FROZEN_BASELINE`; retained strategy `A0`.
- A2L gate: `{'pooled_gain_pass': False, 'fold_non_degradation_pass': True, 'auc_superiority_pass': False, 'passed': False}`.

## Agent and route controls

- A2L used the real DeepSeek provider with strict JSON validation.
- The independent A2L route gate rejected P18 RGT-KED, whose pooled RMSE was 0.6421439169% worse with 2/5 spatial-fold wins.
- This RGT-KED gate is confirmatory and preconstrained to a registered route; it is not autonomous route discovery.
- A2D and A3 each used the same four-trial budget per held fold. A3 is PCG64 random kernel search; it is not a random-init foundation arm.
- A1 replays the same action entrypoint, fold inputs and frozen seed, then requires exact fresh-A0/fresh-A1 array and prediction-hash equality.
- Historical P21 A0 remains read-only provenance; fresh replay differs from it by 8.3266726846886741e-17 at maximum absolute error and this difference is not used as the A1 gate.

## Selection firewall

P19 coordinate purge was applied independently for every held fold. The strategies saw only improved/flat/worse classifications from the other four purged folds. Held-fold and final promotion metrics were calculated only after all four actions were fixed and were never fed back to a strategy.

The evidence contains five genuinely independent spatial units. The four policy trials are search evaluations, not additional independent geological samples. No frozen test or holdout file was opened.

## Attribution boundary

All P28 arms reuse the same pretrained GFM feature cache; any overall change is a kernel-search result, not a causal claim about foundation pretraining. Existing matched random-init evidence remains separate.
The existing P15 matched pretrained/random-init audit is cited only as separate attribution evidence and is not reused as A3.

## Per-fold promotion outcomes

| strategy | fold | A0 RMSE | candidate RMSE | relative change | outcome |
|---|---:|---:|---:|---:|---|
| A2L | 0 | 0.026804535898 | 0.026821177281 | +0.062084% | loss |
| A2L | 1 | 0.028579678784 | 0.028595565942 | +0.055589% | loss |
| A2L | 2 | 0.016645911572 | 0.016645911572 | +0.000000% | tie |
| A2L | 3 | 0.027712695506 | 0.027705079621 | -0.027482% | win |
| A2L | 4 | 0.035575505088 | 0.035655356347 | +0.224456% | loss |
| A2D | 0 | 0.026804535898 | 0.026821177281 | +0.062084% | loss |
| A2D | 1 | 0.028579678784 | 0.028579022129 | -0.002298% | win |
| A2D | 2 | 0.016645911572 | 0.016645911572 | +0.000000% | tie |
| A2D | 3 | 0.027712695506 | 0.027712556692 | -0.000501% | win |
| A2D | 4 | 0.035575505088 | 0.035655356347 | +0.224456% | loss |
| A3 | 0 | 0.026804535898 | 0.026821177281 | +0.062084% | loss |
| A3 | 1 | 0.028579678784 | 0.028595565942 | +0.055589% | loss |
| A3 | 2 | 0.016645911572 | 0.016585048635 | -0.365633% | win |
| A3 | 3 | 0.027712695506 | 0.027709998816 | -0.009731% | win |
| A3 | 4 | 0.035575505088 | 0.035667185978 | +0.257708% | loss |
