# P13 cross-attention evidence

## Experiment

The accepted CNN deepest feature map supplies query tokens. The native-128 SAM2 feature map supplies key/value tokens. Four-head cross-attention writes a gated feature residual back before the original CNN decoder.

SAM2 loading is controlled by one `sam2_weight_mode` parameter with accepted values `pretrained` and `random`. This evidence was generated with `pretrained`; the opposite mode is not included in this package.

The candidate also uses 160 continuation updates, horizontal flips, mild intensity/noise augmentation, AdamW, cosine decay, and CE+Dice loss. A continued-CNN control receives the same optimization without the cross-attention/SAM2 path.

## Fixed-development results

| Task | Variant | mIoU | Δ vs baseline | Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| F3 | strong_small_baseline | 0.136263 | +0.000000 | 0.313400 | 0.221520 |
| F3 | continued_cnn_control | 0.206862 | +0.070598 | 0.432123 | 0.314526 |
| F3 | cross_attention_fusion | 0.292153 | +0.155890 | 0.532482 | 0.416360 |
| Penobscot | strong_small_baseline | 0.129101 | +0.000000 | 0.450842 | 0.176563 |
| Penobscot | continued_cnn_control | 0.189564 | +0.060463 | 0.534685 | 0.262397 |
| Penobscot | cross_attention_fusion | 0.205270 | +0.076169 | 0.539150 | 0.282640 |

## Objective conclusion

- F3 overall mIoU changed from 0.136263 to 0.292153 (+0.155890).
- Penobscot overall mIoU changed from 0.129101 to 0.205270 (+0.076169).
- Equal-task mean mIoU changed from 0.132682 to 0.248712 (+0.116029).

**大模型贡献占比待下一轮消融确认。** These numbers objectively describe the complete P13 package; they do not attribute the change to SAM2 because the random-weight ablation has not yet been run.

## Data and evaluation boundary

- mIoU uses the unchanged P11/P12 probability evaluator.
- Validation samples and locked folds 0/4 are unchanged.
- Augmentation and fitting use training samples/labels only.
- No frozen holdout or `test.h5` path was accepted or read.
- No dense prediction, checkpoint copy, or feature cache is persisted.
