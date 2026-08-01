# Lithofacies default-baseline adoption evidence

## Outcome

Decision: **ACCEPT_AS_DEFAULT**.

| Configuration | LOGO4 x 3 cells | Fixed-schema Macro-F1 | Delta | Wins |
|---|---:|---:|---:|---:|
| archived depth=2, eta=0.2, rounds=40 | 12 | 0.194937702076 | - | - |
| default depth=3, eta=0.1, rounds=60 | 12 | 0.213348797049 | +0.018411094973 | 12/12 |

The live recomputation matches the P17 recorded means within `1e-12`.
The fixed metric is the arithmetic mean of nine schema-class F1 values; absent validation classes contribute zero. The split is the immutable four-development-family LOGO4 contract, repeated with seeds `1867973658`, `2137841944`, and `3902865753`.

## Attribution and firewall

This adoption is solely an XGBoost hyperparameter change. It does not use MOMENT, pretrained embeddings, a large model, or an LLM at training/inference time, so no improvement is attributed to MOMENT or any large model.

Only the prebuilt development LOGO4 batch was opened. `frozen_test_accessed=false`, `known_holdout_accessed=false`, and no `test.h5` path is accepted by the runner. P17 agent-chapter artifacts were verified against frozen SHA-256 values and were not rewritten.

## Fold means

| Fold | Archived | Default |
|---:|---:|---:|
| 0 | 0.213580451428 | 0.241355505165 |
| 1 | 0.183572146807 | 0.196748000027 |
| 2 | 0.181275775192 | 0.183717282560 |
| 3 | 0.201322434875 | 0.231574400442 |

The three nominal seeds are deterministic duplicates here because `subsample=1.0` and `colsample_bytree=1.0`; 12/12 therefore represents four distinct family outcomes, each repeated three times.
