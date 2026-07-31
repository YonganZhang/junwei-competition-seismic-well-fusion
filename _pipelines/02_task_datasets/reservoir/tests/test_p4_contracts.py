from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from ml_framework.model_registry import get_model  # noqa: E402
from p4_pipeline import (  # noqa: E402
    FORBIDDEN_INPUTS,
    PERMEABILITY_KLOGH,
    POROSITY_PHIF,
    FeatureStats,
    IndexedSample,
    LoadedSample,
    build_independent_manifest,
    evaluate_predictions,
    fit_fold_statistics,
    select_frozen_test_family,
    transform_inputs,
)
POROSITY_SPECS = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.targets.porosity.task_spec"
)
PERMEABILITY_SPECS = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.targets.permeability.task_spec"
)
build_phif_task_spec = POROSITY_SPECS.build_phif_task_spec
build_phie_task_spec = POROSITY_SPECS.build_phie_task_spec
permeability_spec = PERMEABILITY_SPECS.build_task_spec


def _indexed(family: str, index: int, task: str = "t") -> IndexedSample:
    return IndexedSample(
        sample_id=f"{task}-{family}-{index}",
        family_id=family,
        well_id=f"{family} A",
        depth_m=float(index),
        position={"time_ms": float(index)},
        source_kind="unit",
        source_path=Path("unit"),
        source_key=index,
        label_valid=True,
    )


def _loaded(family: str, index: int, log_value: float = 1.0, observed: float = 1.0) -> LoadedSample:
    record = _indexed(family, index)
    logs = np.ones((9, 8), dtype=float)
    logs[:, :4] *= log_value
    logs[:, 4:] *= observed
    return LoadedSample(record, np.full((3, 3, 9), float(index)), logs, float(index))


def test_target_specs_freeze_labels_whitelists_masks_and_hpo_direction() -> None:
    phif = build_phif_task_spec()
    phie = build_phie_task_spec()
    permeability = permeability_spec()
    assert phif.targets == ("PHIF",)
    assert phif.metadata["primary_label_version"] is True
    assert phie.targets == ("PHIE",)
    assert phie.metadata["mixed_with_PHIF"] is False
    assert "LFP_PHIE" in phie.metadata["never_alias"]
    assert permeability.targets == ("KLOGH",)
    assert permeability.units["KLOGH"] == "mD"
    assert permeability.target_transform["KLOGH"] == "log1p(KLOGH_mD)"
    for spec in (phif, phie, permeability):
        assert spec.hpo["direction"] == "minimize"
        assert set(spec.targets) == set(spec.target_masks)
        assert not (set(spec.input_whitelist) & set(spec.forbidden_inputs))
        assert set(FORBIDDEN_INPUTS) <= set(spec.forbidden_inputs)


def test_property_transforms_are_reversible_and_physical_predictions_bounded() -> None:
    permeability = np.asarray([0.0, 1.0, 1234.5])
    domain = PERMEABILITY_KLOGH.from_physical(permeability)
    assert np.allclose(PERMEABILITY_KLOGH.to_physical(domain, prediction=False), permeability)
    assert np.all(PERMEABILITY_KLOGH.to_physical(np.asarray([-3.0, 1.0]), prediction=True) >= 0.0)
    assert np.allclose(POROSITY_PHIF.to_physical(np.asarray([-0.2, 1.2]), prediction=True), [0.0, 1.0])


def test_independent_logo_manifest_prefers_f15_and_reports_effective_four() -> None:
    families = ("15/9-19", "15/9-F-1", "15/9-F-11", "15/9-F-12", "15/9-F-15")
    records = [_indexed(family, index, "phif") for family in families for index in range(3)]
    test_family, selection = select_frozen_test_family(records)
    manifest = build_independent_manifest(
        records,
        test_family=test_family,
        target_name="PHIF",
        label_version="v1",
        seed=2693,
        selection_record=selection,
    )
    assert test_family == "15/9-F-15"
    assert manifest.requested_n_splits == 5
    assert manifest.effective_n_splits == 4
    assert "only 4 independent development groups" in manifest.downgrade_reason
    assert not (set(manifest.test_groups) & set(manifest.development_groups))
    assert all(not (set(fold.train_groups) & set(fold.validation_groups)) for fold in manifest.folds)


def test_fold_train_preprocessing_uses_masks_and_exact_fit_ids() -> None:
    samples = [_loaded("A", 1), _loaded("B", 2)]
    # A missing channel carries a huge sentinel, but its explicit mask removes it.
    bad_logs = samples[1].well_log_seq.copy()
    bad_logs[:, 2] = 9999.0
    bad_logs[:, 6] = 0.0
    samples[1] = LoadedSample(samples[1].index, samples[1].seismic_patch, bad_logs, samples[1].target_domain)
    stats = fit_fold_statistics(samples)
    features = transform_inputs(samples, stats)
    assert stats.fit_sample_ids == tuple(sample.index.sample_id for sample in samples)
    assert np.allclose(stats.log_mean.reshape(9, 4)[:, 2], 1.0)
    assert np.allclose(features[1, 81 + 2 :: 4][:9], 0.0)
    assert np.isfinite(features).all()


def test_single_output_registered_models_train_validate_and_roundtrip(tmp_path: Path) -> None:
    x = np.arange(35, dtype=float).reshape(5, 7) / 35.0
    y = np.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    for name in ("reservoir_linear", "reservoir_ridge", "tiny_mlp"):
        kwargs = dict(n_features=7, n_outputs=1, hidden_dim=4, learning_rate=0.002, seed=17)
        model = get_model(name, models_package="models", **kwargs)
        assert model.predict(x).shape == (5, 1)
        assert np.isfinite(model.train_batch((x, y)))
        assert np.isfinite(model.validation_loss((x, y)))
        expected = model.predict(x)
        checkpoint = tmp_path / f"{name}.ckpt"
        model.save_checkpoint(checkpoint)
        restored = get_model(name, models_package="models", **kwargs)
        restored.load_checkpoint(checkpoint)
        assert np.array_equal(restored.predict(x), expected)


def test_metrics_are_physical_with_explicit_mathematical_undefined_case() -> None:
    metrics = evaluate_predictions(
        np.log1p(np.asarray([10.0, 100.0, 1000.0])),
        np.log1p(np.asarray([12.0, 80.0, 900.0])),
        PERMEABILITY_KLOGH,
    )
    assert metrics["physical_unit"] == "mD"
    assert metrics["physical"]["MAE"] > 0
    assert metrics["log1p_domain"]["RMSE"] > 0
    constant = evaluate_predictions(np.asarray([0.2, 0.2]), np.asarray([0.2, 0.3]), POROSITY_PHIF)
    assert constant["physical"]["R2"] is None
    assert "constant" in constant["physical"]["R2_reason"]
