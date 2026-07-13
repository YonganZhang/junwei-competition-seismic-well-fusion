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
