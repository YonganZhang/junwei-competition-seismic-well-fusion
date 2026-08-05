from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


RESERVOIR = Path(__file__).resolve().parents[1]
if str(RESERVOIR) not in sys.path:
    sys.path.insert(0, str(RESERVOIR))

import p41_property_crossmodal_qualification as p41  # noqa: E402


def _metric(value: float, target_values: tuple[float, float, float] | None = None) -> dict:
    values = target_values or (value, value, value)
    return {
        "composite_equal_weight_train_std_normalized_RMSE": value,
        "per_target": {
            target: {"train_std_normalized_RMSE": values[index]}
            for index, target in enumerate(p41.TARGETS)
        },
    }


def test_p5_strong_baseline_lock_is_exact_and_target_specific() -> None:
    lock = p41.verify_p5_lock()
    assert lock["source_commit"] == p41.P5_COMMIT
    assert lock["winners"] == {
        "PHIF": "extra_trees_regressor",
        "KLOGH": "extra_trees_regressor",
        "SW": "xgboost_regressor",
    }
    assert lock["input"].startswith("153 =")


def test_linear_interp_extrap_reports_real_linear_extension() -> None:
    xp = np.array([10.0, 20.0, 30.0])
    fp = np.array([100.0, 200.0, 500.0])
    assert p41.linear_interp_extrap(15.0, xp, fp) == 150.0
    assert p41.linear_interp_extrap(0.0, xp, fp) == 0.0
    assert p41.linear_interp_extrap(40.0, xp, fp) == 800.0


def test_outer_and_inner_logo_contexts_are_isolated() -> None:
    families = np.array(["A", "B", "C", "D"] * 3)
    rows = p41.contexts(families)
    assert len(rows) == 16
    assert sum(row["level"] == "outer" for row in rows) == 4
    assert sum(row["level"] == "inner" for row in rows) == 12
    for row in rows:
        assert row["validation_family"] not in row["train_families"]
        assert row["outer"] not in row["train_families"]


def test_depth_spread_is_family_balanced_and_bounded() -> None:
    count = 400
    indices = np.arange(count)
    families = np.array(["A"] * 200 + ["B"] * 100 + ["C"] * 100)
    wells = np.array([f"{family}-well" for family in families])
    depths = np.arange(count, dtype=float)
    sample_ids = np.array([f"s{index:04d}" for index in indices])
    selected = p41.depth_spread_indices(indices, families, wells, depths, sample_ids)
    assert len(selected) == p41.TRAIN_LIMIT
    assert {family: int(np.sum(families[selected] == family)) for family in "ABC"} == {
        "A": 64, "B": 64, "C": 64
    }


def test_baseline_preprocessing_fits_training_rows_only() -> None:
    rng = np.random.default_rng(3)
    train_seismic = rng.normal(size=(8, 3, 3, 9))
    train_logs = np.concatenate(
        [rng.normal(size=(8, 9, 4)), np.ones((8, 9, 4))], axis=2
    )
    validation_seismic = rng.normal(size=(2, 3, 3, 9))
    validation_logs = np.concatenate(
        [rng.normal(size=(2, 9, 4)), np.ones((2, 9, 4))], axis=2
    )
    first = p41._baseline_preprocess(
        train_seismic, train_logs, validation_seismic, validation_logs
    )
    changed = p41._baseline_preprocess(
        train_seismic, train_logs, validation_seismic + 1e6,
        np.concatenate([validation_logs[:, :, :4] + 1e6, validation_logs[:, :, 4:]], axis=2),
    )
    train_view = p41._baseline_preprocess(train_seismic, train_logs, train_seismic, train_logs)
    replay = p41._baseline_preprocess(train_seismic, train_logs, train_seismic, train_logs)
    assert first.shape == changed.shape == (2, 153)
    assert np.array_equal(train_view, replay)


def test_shuffle_stays_inside_train_family() -> None:
    indices = np.arange(12)
    families = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
    shuffled = p41.within_family_shuffle(indices, families, seed=2693)
    assert np.all(families[shuffled] == families[indices])
    assert np.all(shuffled != indices)


def test_a8_native_time_shift_has_no_wraparound() -> None:
    sections = np.arange(2 * 400 * 160, dtype=np.float32).reshape(2, 400, 160)
    shifted = p41.shift_sections_one_sample(sections)
    assert np.array_equal(shifted[:, :-1], sections[:, 1:])
    assert np.array_equal(shifted[:, -1], sections[:, -1])


def test_gate_zero_is_exact_fallback() -> None:
    import torch

    model = p41.GatedResidual(4, 5, 2693).module
    well = torch.randn(7, 4)
    seismic = torch.randn(7, 5)
    base = torch.randn(7, 3)
    predicted, _ = model(well, seismic, base)
    forced, _ = model(well, seismic, base, force_off=True)
    assert torch.equal(predicted, base)
    assert torch.equal(forced, base)


def test_metric_direction_and_promotion_gate() -> None:
    actual = np.zeros((4, 3))
    std = np.ones(3)
    good = p41.regression_metrics(actual, np.full((4, 3), 0.5), std)
    bad = p41.regression_metrics(actual, np.ones((4, 3)), std)
    assert good["composite_equal_weight_train_std_normalized_RMSE"] < bad[
        "composite_equal_weight_train_std_normalized_RMSE"
    ]
    folds = {
        "B0": [_metric(1.0)] * 4,
        "W1": [_metric(0.95)] * 4,
        "S1": [_metric(0.96)] * 4,
        "F1": [_metric(0.90)] * 4,
        "A5": [_metric(0.98)] * 4,
        "A6": [_metric(0.98)] * 4,
        "A7": [_metric(1.0)] * 4,
        "A8": [_metric(0.98)] * 4,
    }
    gate = p41.promotion_gate(folds, [0.01, 0.02], True, 0.0, True)
    assert gate["passed"] is True
    folds["F1"] = [_metric(1.01)] * 4
    stopped = p41.promotion_gate(folds, [-0.1, 0.1], True, 0.0, True)
    assert stopped["verdict"] == "R0_STOP_NO_ATTRIBUTABLE_SIGNAL"
