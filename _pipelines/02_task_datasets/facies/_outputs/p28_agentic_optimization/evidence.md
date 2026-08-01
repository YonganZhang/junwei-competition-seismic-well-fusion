# P28 facies execution-agent pilot evidence

## Scope

This is a fresh Stage-1 development execution pilot, not a replay of P13/P17 metrics. A0 is the same-invocation pretrained SAM2 cross-attention gate=0.2 route. Selection used fold 0 and promotion used fold 4 with 32 train and 16 validation samples per task/fold.

No frozen holdout or `test.h5` was read. DeepSeek received only categorical fold-train aggregates and improved/flat/worse selection feedback; raw metrics, validation curves, labels, residuals, sample IDs, predictions and paths were excluded.

## Frozen controls

- A0 selection mean mIoU: `0.229007898`
- A0 promotion mean mIoU: `0.270541772`
- Continued-CNN promotion mean mIoU: `0.218438502`
- A1 fresh replay prediction hashes equal A0: `True`
- A1 fresh replay metrics equal A0: `True`

## Policy results

- A2L_llm_agent_execute: actions `['FAC_GATE_050', 'FAC_GATE_035', 'FAC_FUSION_LR_1E4', 'FAC_DICE_050']`, optimization path score (mean of the four-step running maximum selection mean mIoU; not area under a curve) `0.244526148`; endpoint promotion action `FAC_GATE_050`, endpoint promotion mean mIoU `0.296767850`.
- A2D_deterministic_agent: actions `['FAC_GATE_035', 'FAC_SAM2_FROZEN', 'FAC_GATE_050', 'FAC_FUSION_LR_1E4']`, optimization path score (mean of the four-step running maximum selection mean mIoU; not area under a curve) `0.239389029`; endpoint promotion action `FAC_GATE_050`, endpoint promotion mean mIoU `0.296767850`.
- A3_random_policy: actions `['FAC_SAM2_FROZEN', 'FAC_GATE_050', 'FAC_GATE_035', 'FAC_DICE_050']`, optimization path score (mean of the four-step running maximum selection mean mIoU; not area under a curve) `0.240646586`; endpoint promotion action `FAC_GATE_050`, endpoint promotion mean mIoU `0.296767850`.

## Promotion gate

- a1_replay_prediction_hash_equal_a0: `True`
- a1_replay_metrics_equal_a0: `True`
- a2l_four_legal_distinct_trials: `True`
- a2l_path_score_above_a2d: `True`
- a2l_path_score_above_a3: `True`
- promotion_endpoint_mean_delta_at_least_0p005: `True`
- f3_delta_at_least_minus_0p005: `True`
- penobscot_delta_at_least_minus_0p005: `False`
- not_worse_than_continued_cnn: `False`
- no_frozen_test_access: `True`

Previous two-trial verdict: **REJECT_A2L**. Corrected four-trial verdict: **REJECT_A2L**.

Verdict: **REJECT_A2L**.

This verdict concerns the policy under the frozen Stage-1 budget. It does not attribute any model gain to SAM2 and does not authorize frozen-test access or default enablement.

## Protocol identity

- Protocol hash: `8d6ba75504973cb7794ff9ea5dbef8cbc84ca7a1acdc484248d88913c99ad146`
- Runner hash at evidence creation: `cd8d2d435cff6fc3d090b79d23ac7283ad60495b52d2b30390ad30dd91a69535`
- Provider credential persisted: `False`
