from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from data_pipeline import (  # noqa: E402
    FORBIDDEN_INPUT_CURVES,
    INPUT_CHANNELS,
    canonical_well_id,
    deterministic_family_split,
    inverse_targets,
    parent_well_family,
    transform_targets,
)
from ml_framework.model_registry import get_model  # noqa: E402
from train_baseline import (  # noqa: E402
    apply_statistics,
    fit_train_statistics,
    inverse_normalized_targets,
    make_batches_factory,
)


def test_parent_family_groups_all_sidetracks() -> None:
    assert parent_well_family("15_9-F-1") == "15/9-F-1"
    assert parent_well_family("15_9-F-1 A") == "15/9-F-1"
    assert parent_well_family("NO 15/9-F-1 C") == "15/9-F-1"
    assert parent_well_family("15_9-19 A") == "15/9-19"
    assert parent_well_family("15_9-19 BT2") == "15/9-19"
    assert canonical_well_id("NO 15/9-F-11 T2") == "15/9-F-11 T2"


def test_deterministic_split_has_zero_family_overlap() -> None:
    families = ["15/9-19", "15/9-F-1", "15/9-F-11", "15/9-F-12", "15/9-F-15"]
    first = deterministic_family_split(families)
    second = deterministic_family_split(reversed(families))
    assert first == second
    groups = {name: {family for family, split in first.items() if split == name} for name in ("train", "guard", "test")}
    assert len(groups["train"]) >= 2
    assert groups["guard"] and groups["test"]
    assert not (groups["train"] & groups["guard"] or groups["train"] & groups["test"] or groups["guard"] & groups["test"])


def test_target_transform_roundtrip() -> None:
    physical = np.asarray([[0.2, 0.0, 0.8], [0.3, 1234.5, 0.4]])
    transformed = transform_targets(physical[:, 0], physical[:, 1], physical[:, 2])
    assert np.allclose(inverse_targets(transformed), physical)


def test_input_channel_allowlist_excludes_target_leakage() -> None:
    assert not (set(INPUT_CHANNELS) & FORBIDDEN_INPUT_CURVES)
    assert {"PHIF", "KLOGH", "SW", "KLOGH_NEW", "LFP_PHIE"} <= FORBIDDEN_INPUT_CURVES


def test_mask_and_training_only_stats_roundtrip() -> None:
    seismic = np.arange(2 * 3 * 3 * 9, dtype=float).reshape(2, 3, 3, 9)
    values = np.ones((2, 9, 4), dtype=float)
    masks = np.ones_like(values)
    masks[1, :, 2] = 0.0
    values[1, :, 2] = 9999.0
    logs = np.concatenate([values, masks], axis=2)
    targets = np.asarray([[0.2, np.log1p(10.0), 0.8], [0.3, np.log1p(100.0), 0.4]])
    stats = fit_train_statistics(seismic, logs, targets)
    features, target_norm = apply_statistics(seismic, logs, targets, stats)
    assert np.isfinite(features).all()
    assert np.allclose(inverse_normalized_targets(target_norm, stats), targets)
    # The missing sentinel is never used to fit the channel and becomes zero after masking.
    channel_stats = stats["logs"][2]
    assert channel_stats.mean == 1.0


def test_nonempty_batch_factory_and_dynamic_model_contracts(tmp_path: Path) -> None:
    x = np.ones((5, 7), dtype=float)
    y = np.ones((5, 3), dtype=float)
    batches = make_batches_factory(x, y, batch_size=2, shuffle=True, seed=1)()
    assert batches and sum(len(batch[0]) for batch in batches) == 5
    for model_name in ("tiny_mlp", "reservoir_linear", "reservoir_ridge"):
        model_kwargs = {
            "n_features": 7,
            "n_outputs": 3,
            "hidden_dim": 4,
            "learning_rate": 0.002,
        }
        if model_name != "tiny_mlp":
            model_kwargs["unused_factory_option"] = True
        model = get_model(model_name, models_package="models", **model_kwargs)
        assert model.__class__.__module__ == f"models.{model_name}"
        initial_prediction = model.predict(x)
        assert initial_prediction.shape == (5, 3)
        assert np.isfinite(initial_prediction).all()
        assert np.isfinite(model.train_batch((x, y)))
        assert np.isfinite(model.validation_loss((x, y)))
        trained_prediction = model.predict(x)
        assert trained_prediction.shape == (5, 3)
        assert np.isfinite(trained_prediction).all()

        checkpoint = tmp_path / f"{model_name}.ckpt"
        model.save_checkpoint(checkpoint)
        restored = get_model(model_name, models_package="models", **model_kwargs)
        restored.load_checkpoint(checkpoint)
        assert np.array_equal(restored.predict(x), trained_prediction)
