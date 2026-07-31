# P18 anisotropic foundation geostatistics

## Main result

- PyKrige development OOF RMSE: `0.028449728170`.
- Honest nested P18 RMSE: `0.027752680679`.
- Relative RMSE change: `-2.4501%`.
- Spatial-fold outcomes: 5 wins / 0 losses.
- Whole-fold bootstrap delta CI95: `[-0.001140994782, -0.000353924655]`.

## Correction to P17

P17 reported RMSE `0.028319907650` after selecting on the same pooled OOF archive. Re-running that exact candidate family with nested fold selection gives `0.028534404074`, so the old improvement is superseded as selection-biased.

## Method

P18 scales the vertical coordinate separately from the two horizontal coordinates, then combines physical position, seismic attributes and frozen GFM latent coordinates in a local inverse-distance kernel. For each reported fold, candidate ranking uses only the other four folds; the top three predictions are averaged.

## Boundary

- Exactly 512 training labels and 2,048 validation rows are used per fold.
- PCA and scaling are fitted independently inside each outer train fold.
- No frozen test or holdout path exists in the CLI.
- No ablation is run now; improvement is not causally attributed to pretraining.
- The bootstrap has only five coarse spatial units and is not a large-sample significance claim.
- Grid-edge sensitivity remains a preregistered external-validation item.
- The development winner remains disabled until the later promotion gate.
