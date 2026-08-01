# P29 facies root-cause chain

## Causal wiring audit

| Link | Connected | Evidence |
|---|---:|---|
| observation/prompt | yes | observation v2; prompt-ablation hash `86cec0f4d68d77a8d85e7175e98de0008a2ae08393217295d35ade7a31964c6d` |
| selected action/executor | yes | action registry hash `84fac72e044f5b9074df70e275006e50369e741085867b71399194b08a1fd5ef` |
| optimizer and fusion movement | yes | five action-effect records `c72d8a19bba820b04796b6e510a03259dd7be8167546506aa7095521c64eb56d` |
| prediction endpoint | yes | all actions persist prediction hashes; no-op check `True` |
| primary metric | yes | `evaluate_probabilities` support hash `7e94f0607cb4d1c34be3087f5673de4f1ef8f1516f80c491d3929f2ca4b1df82` |
| promotion/endpoint | yes | disjoint per-dataset package hash `bc3c4380dc006d4d07e0555187395ee7a5b93d90ee9d5c59ccae54ea56c19ae0` |

## Root cause

P28's five actions were executable, but a single global action forced F3 and Penobscot to share a setting, while four of five trials made policy endpoints converge. P29 therefore tests two-action sample efficiency and allows each dataset to retain its best fold-0 choice, including A0. This repairs the comparison; it does not assume an LLM contribution.

Honest verdict: **RETAIN_HYBRID**.

No frozen holdout or `test.h5` was read.
