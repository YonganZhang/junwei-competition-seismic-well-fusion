# P29 facies action-effect repair evidence

## Frozen scope

A0 remains the same-invocation pretrained SAM2 cross-attention gate=0.2 route. Selection uses fold 0; promotion uses disjoint fold 4; every fold has 32 training and 16 development-validation samples. The primary metric remains `p4_metrics.evaluate_probabilities` mIoU with all configured classes.

No frozen holdout or `test.h5` was read. DeepSeek received only observation-v2 train diagnostics and clipped normalized signed fold-0 effects, never raw scores, labels, residuals, sample IDs, paths, predictions, fold-4 results, or frozen-test information.

## Observation and action effects

- A1 true replay hashes equal A0: `True`; metrics equal: `True`.
- Formal metric support hash: `7e94f0607cb4d1c34be3087f5673de4f1ef8f1516f80c491d3929f2ca4b1df82`.
- Development cells with every configured class observed: `2/4`; formal mIoU still averages all configured classes.
- Five-action effect registry hash: `c72d8a19bba820b04796b6e510a03259dd7be8167546506aa7095521c64eb56d`.
- Every action changed at least one prediction endpoint versus A0: `True`.
- Prompt information ablation hash: `86cec0f4d68d77a8d85e7175e98de0008a2ae08393217295d35ade7a31964c6d`.

## Two-of-five sample efficiency and promotion

- A2L_llm_agent_execute: actions `['FAC_FUSION_LR_1E4', 'FAC_SAM2_FROZEN']`; path score `0.229056145`; package `{'F3': 'A0_GATE_020', 'Penobscot': 'A0_GATE_020'}`; promotion endpoint mean mIoU `0.270541772`; preserved promotion guards `False`; status `OK`.
- A2D_deterministic_agent: actions `['FAC_GATE_035', 'FAC_SAM2_FROZEN']`; path score `0.236434599`; package `{'F3': 'FAC_GATE_035', 'Penobscot': 'A0_GATE_020'}`; promotion endpoint mean mIoU `0.289272253`; preserved promotion guards `True`; status `OK`.
- A3_random_policy: actions `['FAC_FUSION_LR_1E4', 'FAC_GATE_050']`; path score `0.236767023`; package `{'F3': 'FAC_GATE_050', 'Penobscot': 'A0_GATE_020'}`; promotion endpoint mean mIoU `0.301022749`; preserved promotion guards `True`; status `OK`.
- A4_deterministic_search: actions `['FAC_GATE_050', 'FAC_DICE_050']`; path score `0.244526148`; package `{'F3': 'FAC_GATE_050', 'Penobscot': 'A0_GATE_020'}`; promotion endpoint mean mIoU `0.301022749`; preserved promotion guards `True`; status `OK`.

The oracle is selection-fold-only and was not promoted or shown to any policy.

## Preserved promotion guards

- a1_replay_prediction_hash_equal_a0: `True`
- a1_replay_metrics_equal_a0: `True`
- a2l_uses_at_most_two_legal_distinct_actions: `True`
- a2l_stop_semantics_executable: `True`
- a2l_sample_efficiency_above_a2d: `False`
- a2l_sample_efficiency_above_a3: `False`
- promotion_endpoint_mean_delta_at_least_0p005: `False`
- f3_delta_at_least_minus_0p005: `True`
- penobscot_delta_at_least_minus_0p005: `True`
- not_worse_than_continued_cnn: `True`
- all_five_action_effects_persisted: `True`
- all_five_action_effects_change_config_and_prediction: `True`
- formal_metric_support_persisted_and_hashed: `True`
- no_frozen_test_access: `True`

## Dataset-conditioned hybrid gate

- a4_dataset_conditioned_package_is_gate050_plus_a0: `True`
- a4_package_passes_preserved_promotion_guards: `True`
- action_effect_chain_is_non_noop: `True`
- formal_metric_support_is_persisted: `True`
- no_frozen_test_access: `True`

Verdict: **RETAIN_HYBRID**.

`RETAIN_HYBRID` means the dataset-conditioned deterministic package passed the frozen guards. It does not retain the direct LLM policy and is not a claim that SAM2 or the LLM caused the endpoint gain. Agent retention and direct endpoint superiority are reported separately.

- Retain direct agent: `False`
- Retain dataset-conditioned hybrid: `True`
- Direct A2L endpoint superiority: `False`
- Provider credential persisted: `False`
