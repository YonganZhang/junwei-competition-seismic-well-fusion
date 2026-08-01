# P28 facies execution-agent pilot evidence

## Scope

This is a fresh Stage-1 development execution pilot, not a replay of P13/P17 metrics. A0 is the same-invocation pretrained SAM2 cross-attention gate=0.2 route. Selection used fold 0 and promotion used fold 4 with 32 train and 16 validation samples per task/fold.

No frozen holdout or `test.h5` was read. DeepSeek received only categorical fold-train aggregates and improved/flat/worse selection feedback; raw metrics, validation curves, labels, residuals, sample IDs, predictions and paths were excluded.

## Frozen controls

- A0 selection mean mIoU: `0.229007898`
- A0 promotion mean mIoU: `0.270541772`
- Continued-CNN promotion mean mIoU: `0.218438502`
- A1 prediction hash equals A0: `True`

## Policy results

- A2L_llm_agent_execute: actions `['FAC_GATE_050', 'FAC_GATE_035']`, optimization AUC `0.244526148`, promotion action `FAC_GATE_050`, promotion mean mIoU `0.296767850`.
- A2D_deterministic_agent: actions `['FAC_GATE_035', 'FAC_SAM2_FROZEN']`, optimization AUC `0.234251910`, promotion action `FAC_GATE_035`, promotion mean mIoU `0.281262810`.
- A3_random_policy: actions `['FAC_FUSION_LR_1E4', 'FAC_GATE_050']`, optimization AUC `0.236767023`, promotion action `FAC_GATE_050`, promotion mean mIoU `0.296767850`.

## Promotion gate

- a1_prediction_hash_equal_a0: `True`
- a2l_two_legal_distinct_trials: `True`
- a2l_auc_above_a2d: `True`
- a2l_auc_above_a3: `True`
- promotion_mean_delta_at_least_0p005: `True`
- f3_delta_at_least_minus_0p005: `True`
- penobscot_delta_at_least_minus_0p005: `False`
- not_worse_than_continued_cnn: `False`
- no_frozen_test_access: `True`

Verdict: **REJECT_A2L**.

This verdict concerns the policy under the frozen Stage-1 budget. It does not attribute any model gain to SAM2 and does not authorize frozen-test access or default enablement.

## Protocol identity

- Protocol hash: `812d5c7c7341934430d64d5b9c6ce8d15d97967ebddfdfe1db3f40ef00a3a670`
- Runner hash at evidence creation: `9902598893aec9d3e6e21d7975ac5eadf620e0a8b2e690b18815df5ecfebd852`
- Provider credential persisted: `False`
