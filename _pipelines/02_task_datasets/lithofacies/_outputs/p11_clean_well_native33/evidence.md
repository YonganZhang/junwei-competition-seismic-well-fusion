# P11 clean well-log native-context diagnostic evidence

## Answer

Representation-control state: `NO_MATERIAL_PRETRAINED_RANDOM_SEPARATION`. Pretrained minus random-init fixed-nine Macro-F1 is `+0.002876` against the existing absolute materiality threshold `0.005`.

No larger MOMENT model was run or selected. The separate mask-residual and seismic-CNN late-fusion phase remains outside this minimal diagnostic, exactly so this comparison isolates the representation.

## Native representation contract

- MOMENT input: `13` real normalized log curves × `33` measured-depth samples.
- Resampling/interpolation: `none`.
- Pinned patch length/stride: `8/8`; `4` real tokens cover `32` samples; `1` trailing sample is reported as unpatched.
- The 13 binary observation-mask planes and 9 flattened seismic traces never enter MOMENT or the residual head in this clean-input run.

## Strict LOGO4 ablations

| variant | fixed-9 macro-F1 mean | std | accuracy | gate mean | mean abs residual |
|---|---:|---:|---:|---:|---:|
| baseline | 0.194938 | 0.013858 | 0.492678 | — | — |
| direct | 0.073445 | 0.021440 | 0.225047 | — | — |
| pretrained | 0.197572 | 0.014402 | 0.498697 | 0.017800 | 0.023095 |
| random | 0.194696 | 0.015305 | 0.491659 | 0.017848 | 0.034374 |
| gate0 | 0.194938 | 0.013858 | 0.492678 | 0.000000 | 0.000000 |

## Paired checks

- pretrained − baseline: `+0.002634`.
- pretrained − random: `+0.002876`.
- pretrained − direct: `+0.124127`.
- pretrained wins over baseline in `7/12` fold/seed pairs.
- gate0 − baseline: `+0.000000`; maximum logit error `0.0e+00`.

## Every fold and seed

| fold | seed | baseline | direct | pretrained | random | gate0 | pretrained gate | random gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1867973658 | 0.213580 | 0.070750 | 0.218938 | 0.213580 | 0.213580 | 0.017838 | 0.017961 |
| 0 | 2137841944 | 0.213580 | 0.071238 | 0.217465 | 0.213580 | 0.213580 | 0.017755 | 0.017819 |
| 0 | 3902865753 | 0.213580 | 0.055556 | 0.217621 | 0.217621 | 0.213580 | 0.017837 | 0.017881 |
| 1 | 1867973658 | 0.183572 | 0.093617 | 0.188101 | 0.183572 | 0.183572 | 0.017769 | 0.017862 |
| 1 | 2137841944 | 0.183572 | 0.089867 | 0.190691 | 0.188101 | 0.183572 | 0.017814 | 0.017792 |
| 1 | 3902865753 | 0.183572 | 0.091574 | 0.190691 | 0.172193 | 0.183572 | 0.017818 | 0.017882 |
| 2 | 1867973658 | 0.181276 | 0.040137 | 0.181276 | 0.181276 | 0.181276 | 0.017784 | 0.017818 |
| 2 | 2137841944 | 0.181276 | 0.056258 | 0.181276 | 0.181450 | 0.181276 | 0.017731 | 0.017752 |
| 2 | 3902865753 | 0.181276 | 0.063930 | 0.181276 | 0.181450 | 0.181276 | 0.017754 | 0.017747 |
| 3 | 1867973658 | 0.201322 | 0.072121 | 0.199196 | 0.199196 | 0.201322 | 0.017838 | 0.017882 |
| 3 | 2137841944 | 0.201322 | 0.058230 | 0.201322 | 0.201322 | 0.201322 | 0.017813 | 0.017903 |
| 3 | 3902865753 | 0.201322 | 0.118057 | 0.203006 | 0.203006 | 0.201322 | 0.017848 | 0.017872 |

## Leakage and preservation audit

- Inputs are the immutable four development mother-family folds. Every holdout-like input path is rejected before opening.
- Normalization, class weights, Stage-3 logits, LOGO4 membership, seeds, metric, batch size, and 40-update budget are unchanged.
- The committed original P11 60-cell artifacts were checked byte-for-byte before this run:
  - `artifact_manifest.json`: `d81c3dc0647020186004c3630825d732e8ad8d18c975131c20ec14be0e97be14`
  - `evidence.md`: `eac5c4e31c86dcd080993fc82fee5958b41402edc7dfee666f97b6c80bc2d1f6`
  - `primary_metric.png`: `c3a82df5905027027078c9545dc6afe0c74c9dd6174775a59cbc72bc32c739db`
  - `results.jsonl`: `9c73c5ead55a4ee3e472368dbfdadc811ce92da5a2ffa321601cc23b550e7e3c`
  - `summary.json`: `86d1cda65a9e67e92dff1278fe6bed423fd5d4a8de918547d20eac1b60a75466`
- This evidence is development-only and cannot authorize a frozen holdout read or competition claim.
