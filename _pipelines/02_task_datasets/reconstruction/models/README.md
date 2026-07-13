# Reconstruction model plugin contract

Model selection is handled by `ml_framework.model_registry`.  To add a model:

1. Add `models/<registered-name>.py` with a same-name registered
   `build_model(**kwargs)`.
2. Run `baseline.py --model <registered-name>`; do not edit this package's
   `__init__.py` or the training script.

The returned track adapter supplies `train_batch`, `validation_loss`,
`predict`, `save_checkpoint`, and `load_checkpoint`.  Epoch iteration,
train/validation history, best-checkpoint selection and plotting remain in the
shared framework.

Available same-name plugins:

- `ridge_linear`: the unchanged reference baseline used by the canonical
  conditional and strict result files;
- `reconstruction_linear_sgd`: a NumPy linear mini-batch optimizer with
  optional light L2 regularisation;
- `reconstruction_tiny_mlp`: a deterministic NumPy MLP with one eight-unit
  `tanh` hidden layer and optional light L2 regularisation.

The alternatives implement the same adapter contract but have no canonical
track metrics until they are intentionally trained and evaluated.  Adding a
future model still requires only its same-name file and decorator line.
