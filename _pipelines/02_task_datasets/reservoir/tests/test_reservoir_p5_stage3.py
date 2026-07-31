"""Reservoir-prefixed fail-closed tests for P5 Stage-3."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

import reservoir_p5_stage3 as STAGE3  # noqa: E402


def _synthetic_archive(path: Path, *, frozen_family: bool = False) -> None:
    count = 10
    families = ["A"] * 3 + ["B"] * 3 + ["C"] * 4
    if frozen_family:
        families[-1] = STAGE3.FROZEN_TEST_FAMILY
    fold = {
        "fold_id": 0,
        "train_groups": ["A", "B"],
        "validation_groups": ["C"],
        "train_sample_ids": [f"s{index}" for index in range(6)],
        "validation_sample_ids": [f"s{index}" for index in range(6, 10)],
        "family_counts": {
            "train": {"A": 3, "B": 3},
            "validation": {"C": 4},
        },
        "independent_target_valid_counts": {
            target: {"train": 6, "validation": 3}
            for target in STAGE3.PROPERTY_TARGETS
        },
    }
    manifest = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 3,
        "kind": "mother_well_family_LOGO4",
        "source": {"paths_persisted": False},
        "selection_registered_before_modeling": True,
        "selection_policy": "unit",
        "development_groups": ["A", "B", "C"],
        "folds": [fold],
        "temporary_fractional_split_used": False,
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "test_metrics": False,
            "frozen_test_family": STAGE3.FROZEN_TEST_FAMILY,
            "frozen_test_ids_persisted": False,
        },
    }
    manifest["split_hash"] = STAGE3._hash_payload(manifest)
    rng = np.random.default_rng(2693)
    seismic = rng.normal(size=(count, 3, 3, 9))
    seismic[6:] += 100.0
    logs = np.concatenate(
        [rng.normal(size=(count, 9, 4)), np.ones((count, 9, 4))], axis=2
    )
    labels = np.column_stack(
        [
            np.linspace(0.1, 0.3, count),
            np.log1p(np.linspace(1.0, 100.0, count)),
            np.linspace(0.2, 0.9, count),
        ]
    )
    masks = np.ones((count, 3), dtype=np.uint8)
    masks[6, 0] = 0
    masks[7, 1] = 0
    masks[8, 2] = 0
    np.savez_compressed(
        path,
        fold_0_seismic_patch=seismic,
        fold_0_well_log_sequence=logs,
        fold_0_labels_model_domain=labels,
        fold_0_target_masks=masks,
        fold_0_split=np.asarray(["train"] * 6 + ["validation"] * 4),
        fold_0_sample_ids=np.asarray([f"s{index}" for index in range(count)]),
        fold_0_family_ids=np.asarray(families),
        fold_0_well_ids=np.asarray([f"w{index}" for index in range(count)]),
        fold_0_depths_m=np.arange(count, dtype=float),
        split_manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )


def _completed_rows(target: str) -> list[dict]:
    budget = STAGE3.load_budget()
    rows = []
    for index, cell in enumerate(
        value for value in STAGE3.expected_cells(budget) if value["target"] == target
    ):
        score = 0.1 + (index % 12) * 0.001
        rows.append(
            {
                **cell,
                "status": "completed",
                "validation": {
                    "metric": {
                        "physical": {"RMSE": score, "MAE": score / 2, "R2": 0.5},
                        "model_domain": {
                            "RMSE": score,
                            "MAE": score / 2,
                            "R2": 0.5,
                        },
                    }
                },
                "resources": {"wall_seconds": 1.0},
            }
        )
    return rows


def test_budget_freezes_exact_108_cells_seeds_top3_and_stage2_configs() -> None:
    budget = STAGE3.load_budget()
    cells = STAGE3.expected_cells(budget)
    assert len(cells) == 108
    assert len({cell["cell_id"] for cell in cells}) == 108
    assert {cell["repeat_seed"] for cell in cells} == set(STAGE3.REPEAT_SEEDS)
    assert {cell["lane"] for cell in cells} == {STAGE3.TABULAR_LANE}
    assert budget["targets"]["PHIF"] == [
        "extra_trees_regressor",
        "lightgbm_regressor",
        "hist_gradient_boosting_regressor",
    ]
    stage2_budget = json.loads(STAGE3.STAGE2_BUDGET.read_text())
    for model_id, frozen in budget["model_budgets"].items():
        inherited = stage2_budget["model_budgets"][model_id]
        assert frozen["config"] == inherited["config"]
        assert frozen["update_steps"] == inherited["update_steps"] == 32
        assert frozen["max_wall_seconds"] == inherited["max_wall_seconds"]


def test_p4_logo4_is_frozen_and_never_uses_fractional_split() -> None:
    budget = STAGE3.load_budget()
    manifest = json.loads(STAGE3.DEFAULT_P4_PHIF_SPLIT.read_text())
    folds = STAGE3._validate_p4_logo4(manifest, budget)
    assert [fold["validation_groups"] for fold in folds] == [
        ["15/9-F-12"],
        ["15/9-19"],
        ["15/9-F-11"],
        ["15/9-F-1"],
    ]
    assert all(
        STAGE3.FROZEN_TEST_FAMILY
        not in set(fold["train_groups"]) | set(fold["validation_groups"])
        for fold in folds
    )
    assert budget["fold_policy"]["temporary_fractional_split_forbidden"] is True


def test_fold_preprocessing_fits_train_only_and_masks_stay_independent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "logo4.npz"
    _synthetic_archive(path)
    train, validation, evidence = STAGE3.load_fold(path, 0)
    assert len(train.sample_ids) == 6 and len(validation.sample_ids) == 4
    assert abs(float(np.mean(train.inputs["seismic_patch"]))) < 1e-10
    assert float(np.mean(validation.inputs["seismic_patch"])) > 10.0
    assert evidence["preprocessing"]["fit"] == "fold_train_only"
    assert evidence["preprocessing"]["target_statistics_fitted"] is False
    assert evidence["preprocessing"]["calibration"] == "none"
    masks = [
        np.asarray(validation.target_masks[target])
        for target in STAGE3.PROPERTY_TARGETS
    ]
    assert [int(mask.sum()) for mask in masks] == [3, 3, 3]
    assert not np.array_equal(masks[0], masks[1])
    assert not np.array_equal(masks[1], masks[2])


def test_test_firewall_rejects_frozen_family_and_cli_has_no_test_option(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.npz"
    _synthetic_archive(path, frozen_family=True)
    with pytest.raises(RuntimeError, match="frozen-test firewall"):
        STAGE3.load_fold(path, 0)
    parser = STAGE3.build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    for command in ("prepare", "run", "render"):
        options = {
            option
            for action in subparsers.choices[command]._actions
            for option in action.option_strings
        }
        assert "--test-h5" not in options
        assert "--test-data" not in options
        assert "--test-metrics" not in options


def test_duplicate_result_cells_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    row = {"cell_id": "duplicate", "status": "failed"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="duplicate"):
        STAGE3._load_results(path)


def test_completion_below_80_percent_is_not_rankable() -> None:
    budget = STAGE3.load_budget()
    rows = _completed_rows("PHIF")[:-8]
    board = STAGE3._tabular_leaderboard("PHIF", rows, budget, "split")
    assert board["legal_completion_rate"] < 0.8
    assert board["status"] == "not_rankable"
    assert all("rank" not in entry for entry in board["entries"])


def test_cross_lane_pollution_fails_and_monai_remains_not_rankable() -> None:
    budget = STAGE3.load_budget()
    rows = _completed_rows("SW")
    rows[0]["lane"] = STAGE3.MONAI_LANE
    with pytest.raises(RuntimeError, match="cross-lane"):
        STAGE3._tabular_leaderboard("SW", rows, budget, "split")
    monai = STAGE3._monai_leaderboard("SW", budget, "split")
    assert monai["status"] == "not_rankable"
    assert monai["candidate_count"] == 1
    assert monai["stage3_expected_cells"] == 0
    assert monai["entries"] == []


def test_target_leaderboards_rank_separately_without_run_order_tiebreak() -> None:
    budget = STAGE3.load_budget()
    for target in STAGE3.PROPERTY_TARGETS:
        board = STAGE3._tabular_leaderboard(
            target, list(reversed(_completed_rows(target))), budget, "split"
        )
        assert board["status"] == "rankable"
        assert board["completed_cells"] == 36
        assert [entry["rank"] for entry in board["entries"]] == [1, 2, 3]
        assert all(math.isfinite(entry["mean_physical_RMSE"]) for entry in board["entries"])
        assert board["ranking_tiebreakers"][-1] == "model_id"


def test_portable_outputs_when_present_cover_all_cells_and_no_absolute_paths() -> None:
    output = STAGE3.DEFAULT_OUTPUT_DIR
    result_path = output / "p5_stage3_results.jsonl"
    if not result_path.is_file():
        pytest.skip("portable Stage-3 evidence has not been generated")
    rows = [json.loads(line) for line in result_path.read_text().splitlines() if line]
    assert len(rows) == 108
    assert len({row["cell_id"] for row in rows}) == 108
    assert all(row["test_firewall"]["test_access"] is False for row in rows)
    summary = json.loads((output / "p5_stage3_summary.json").read_text())
    assert summary["expected_cells"] == 108
    assert summary["cross_lane_ranking"] is False
    assert summary["lanes"][STAGE3.MONAI_LANE]["status"] == "not_rankable"
    for path in output.glob("*.json*"):
        assert "/mnt/data" not in path.read_text()
        assert ".claude/worktrees" not in path.read_text()
