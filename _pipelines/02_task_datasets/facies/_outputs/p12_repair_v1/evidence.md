# P12 repair-v1 evidence

## Diagnostic question

P11 was non-beneficial while expanding every 128×128 facies slice to 1024×1024 and freezing the complete SAM2 image encoder. P12 tests whether native-resolution input plus conservative top-block fine-tuning changes that result.

## Frozen repair contract

- SAM2 receives normalized `[B,3,128,128]` tensors with no spatial interpolation.
- A real Hiera-B+ forward produced finite feature maps at `32×32`, `16×16`, and `8×8`; the local Hiera positional encoding and window partitioning therefore accept this shape.
- Only Hiera blocks 22 and 23 are trainable; the encoder learning rate is `1e-05` and the head/core rate is `0.0001`.
- Folds, seeds, samples, 40-update budget, metrics, strong main route, residual formula, regularizers, and five variants are unchanged from P11.

## Results

| Task | Variant | P11 mIoU | P12 mIoU | P12 Δ vs baseline | Change in Δ vs P11 | Gate mean | Encoder update L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| F3 | strong_small_baseline | 0.137977 | 0.135625 | +0.000000 | +0.000000 | 0.000000 | 0.000000 |
| F3 | direct_sam2 | 0.089780 | 0.099006 | -0.036619 | +0.011579 | 0.000000 | 0.451014 |
| F3 | pretrained_residual | 0.137977 | 0.135626 | +0.000001 | +0.000001 | 0.017172 | 0.001282 |
| F3 | random_sam2_residual | 0.137978 | 0.135627 | +0.000002 | +0.000001 | 0.017123 | 0.115294 |
| F3 | gate_zero | 0.137977 | 0.135625 | +0.000000 | +0.000000 | 0.000000 | 0.000000 |
| Penobscot | strong_small_baseline | 0.124216 | 0.157666 | +0.000000 | +0.000000 | 0.000000 | 0.000000 |
| Penobscot | direct_sam2 | 0.064715 | 0.064711 | -0.092955 | -0.033454 | 0.000000 | 0.428445 |
| Penobscot | pretrained_residual | 0.124217 | 0.157664 | -0.000002 | -0.000003 | 0.017736 | 0.006707 |
| Penobscot | random_sam2_residual | 0.124219 | 0.157688 | +0.000022 | +0.000019 | 0.017612 | 0.417499 |
| Penobscot | gate_zero | 0.124216 | 0.157666 | +0.000000 | +0.000000 | 0.000000 | 0.000000 |

## Cross-run comparison caveat

The same seeded small-model baseline was not bitwise stable across the separately executed P11 and P12 GPU runs. P12 minus P11 baseline mIoU was -0.002352 for F3 and +0.033450 for Penobscot. Therefore same-run deltas versus the P12 baseline are the primary causal comparison; P12-minus-P11 same-variant values are descriptive only.

## Honest conclusion

- F3: decision **NON_BENEFICIAL**. Repaired direct SAM2 is -0.036619 mIoU versus the same-run baseline; repaired pretrained residual is +0.000001. Their descriptive same-variant P12-minus-P11 changes are +0.009227 and -0.002351; their baseline-relative deltas changed by +0.011579 and +0.000001, respectively.
- Penobscot: decision **NON_BENEFICIAL**. Repaired direct SAM2 is -0.092955 mIoU versus the same-run baseline; repaired pretrained residual is -0.000002. Their descriptive same-variant P12-minus-P11 changes are -0.000004 and +0.033447; their baseline-relative deltas changed by -0.033454 and -0.000003, respectively.

Promotion still requires pretrained residual to beat both the same-run baseline and the repaired random-SAM2 control by at least 0.005 mIoU.

## Encoder-update evidence

Every trainable SAM2 variant records its final gradient norm and the L2 displacement of blocks 22–23 from their initial values. The verifier rejects zero-gradient or zero-update evidence.

## Data boundary

- Only the locked F3 and Penobscot development manifests and each task's `train.h5` were used.
- Folds are exactly 0 and 4 with seed `2693 + fold_id`.
- No frozen holdout, `test.h5`, dense prediction, feature cache, or checkpoint copy was read or persisted.
- The committed P11 artifact manifest was hash-checked before and after this independent output was written.

## Reproduction command

```text
/mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python _pipelines/02_task_datasets/facies/p12_repair_v1.py run --f3-manifest /mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p4-training-integration/_tmp/p4-acceptance/facies_f3/split_manifest.json --penobscot-manifest /mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p4-training-integration/_tmp/p4-acceptance/facies_penobscot/split_manifest.json --processed-root /mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/track-facies/_data/processed --device cuda:0
```
