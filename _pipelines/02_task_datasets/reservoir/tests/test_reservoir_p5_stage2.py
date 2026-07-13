"""Reservoir-prefixed contract tests for the P5 Stage-2 property pilot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

import reservoir_p5_stage2 as STAGE2  # noqa: E402
from _code.ml_framework.contracts import ModelBatch  # noqa: E402
from _models.property._p5_common import (  # noqa: E402
    Stage1GateError,
    property_output,
    require_approved_weight,
)


def _manifest(families: list[str]) -> dict:
    payload = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 2,
        "source": {"paths_persisted": False},
        "source_fold_id": 0,
        "selection_registered_before_modeling": True,
        "selection_policy": "unit",
        "train_groups": ["A", "B"],
        "validation_groups": sorted(set(families[6:])),
        "train_sample_ids": [f"s{i}" for i in range(6)],
        "validation_sample_ids": [f"s{i}" for i in range(6, 10)],
        "family_counts": {"train": {"A": 3, "B": 3}, "validation": {"C": 4}},
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "frozen_test_family": STAGE2.FROZEN_TEST_FAMILY,
            "frozen_test_ids_persisted": False,
        },
    }
    payload["split_hash"] = STAGE2._hash_payload(payload)
    return payload


def _write_fold(path: Path, *, frozen_family: bool = False) -> None:
    count = 10
    families = ["A"] * 3 + ["B"] * 3 + ["C"] * 4
    if frozen_family:
        families[-1] = STAGE2.FROZEN_TEST_FAMILY
    manifest = _manifest(families)
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
        seismic_patch=seismic,
        well_log_sequence=logs,
        labels_model_domain=labels,
        target_masks=masks,
        split=np.asarray(["train"] * 6 + ["validation"] * 4),
        sample_ids=np.asarray([f"s{i}" for i in range(count)]),
        family_ids=np.asarray(families),
        well_ids=np.asarray([f"w{i}" for i in range(count)]),
        depths_m=np.arange(count, dtype=float),
        split_manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )


def _metric_batch() -> ModelBatch:
    count = 6
    return ModelBatch(
        inputs={"tabular": np.zeros((count, 153))},
        targets={
            "PHIF": np.linspace(0.1, 0.3, count),
            "KLOGH": np.log1p(np.linspace(1.0, 50.0, count)),
            "SW": np.linspace(0.2, 0.8, count),
        },
        input_masks={},
        target_masks={
            "PHIF": np.asarray([1, 1, 0, 1, 1, 1], dtype=bool),
            "KLOGH": np.asarray([1, 0, 1, 1, 1, 1], dtype=bool),
            "SW": np.asarray([0, 1, 1, 1, 1, 1], dtype=bool),
        },
        sample_ids=[f"m{i}" for i in range(count)],
        groups={"mother_well_family": ["A"] * 3 + ["B"] * 3},
        coordinates={"depth_m": np.arange(count)},
    )


def test_budget_freezes_all_ten_candidates_and_protocol_caps() -> None:
    budget = STAGE2.load_budget()
    assert budget["root_seed"] == 2693
    assert len(budget["model_order"]) == 10
    assert budget["sample_budget"]["max_train_samples"] == 192
    assert budget["sample_budget"]["max_validation_samples"] == 81
    assert budget["metric_policy"]["targets"] == ["PHIF", "KLOGH", "SW"]
    assert budget["metric_policy"]["cross_lane_ranking_forbidden"] is True
    for model_id, cell in budget["model_budgets"].items():
        if cell["kind"] == "tree":
            assert cell["max_wall_seconds"] <= 300
        if cell["kind"] == "neural_tabular":
            assert cell["max_wall_seconds"] <= 600
            assert cell["update_steps"] <= 200
        if cell["kind"] == "neural_3d":
            assert cell["max_wall_seconds"] <= 900
            assert cell["update_steps"] <= 80
    assert budget["model_budgets"]["tabiclv2_regressor"]["update_steps"] == 0


def test_p4_first_fold_is_group_disjoint_and_hash_locked() -> None:
    path = STAGE2.DEFAULT_P4_SPLIT
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = STAGE2.load_budget()["joint_fold_policy"]["p4_split_manifest_sha256"]
    fold = STAGE2.validate_p4_fold(payload, expected, STAGE2._hash_file(path))
    assert fold["fold_id"] == 0
    assert set(fold["train_groups"]) == {"15/9-19", "15/9-F-1", "15/9-F-11"}
    assert fold["validation_groups"] == ["15/9-F-12"]
    assert not (set(fold["train_sample_ids"]) & set(fold["validation_sample_ids"]))


def test_seed_derivation_is_stable_and_role_specific() -> None:
    assert STAGE2.stable_seed(2693, "model", "tabm_regressor") == STAGE2.stable_seed(
        2693, "model", "tabm_regressor"
    )
    assert STAGE2.stable_seed(2693, "model", "tabm_regressor") != STAGE2.stable_seed(
        2693, "loader", "tabm_regressor"
    )


def test_fold_preprocessing_fits_train_only_and_preserves_independent_masks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fold.npz"
    _write_fold(path)
    train, validation, evidence = STAGE2.load_fixed_fold(path)
    assert len(train.sample_ids) == 6
    assert len(validation.sample_ids) == 4
    assert abs(float(np.mean(train.inputs["seismic_patch"]))) < 1e-10
    assert float(np.mean(validation.inputs["seismic_patch"])) > 10.0
    assert evidence["preprocessing"]["fit_validation_overlap"] is False
    masks = [np.asarray(validation.target_masks[target]) for target in ("PHIF", "KLOGH", "SW")]
    assert [int(mask.sum()) for mask in masks] == [3, 3, 3]
    assert not np.array_equal(masks[0], masks[1])
    assert not np.array_equal(masks[1], masks[2])


def test_metrics_are_target_specific_with_worst_family_and_klogh_inverse() -> None:
    batch = _metric_batch()
    prediction = np.column_stack(
        [
            np.asarray(batch.targets["PHIF"]) + 0.01,
            np.asarray(batch.targets["KLOGH"]) + 0.1,
            np.asarray(batch.targets["SW"]) - 0.02,
        ]
    )
    metrics = STAGE2.evaluate_targets(
        batch, property_output(prediction, STAGE2.build_task_spec())
    )
    assert tuple(metrics) == ("PHIF", "KLOGH", "SW")
    assert {target: metrics[target]["valid_count"] for target in metrics} == {
        "PHIF": 5,
        "KLOGH": 5,
        "SW": 5,
    }
    assert metrics["KLOGH"]["model_domain_name"] == "log1p(KLOGH_mD)"
    assert metrics["KLOGH"]["physical"]["RMSE"] > metrics["KLOGH"]["model_domain"]["RMSE"]
    assert all(metrics[target]["worst_mother_family"]["family_id"] in {"A", "B"} for target in metrics)
    assert len({metrics[target]["mask_sha256"] for target in metrics}) == 3


def test_firewall_rejects_frozen_family_and_cli_has_no_test_data_option(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.npz"
    _write_fold(path, frozen_family=True)
    with pytest.raises(RuntimeError, match="frozen-test firewall"):
        STAGE2.load_fixed_fold(path)
    parser = STAGE2.build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    for command in ("prepare", "run"):
        options = {
            option
            for action in subparsers.choices[command]._actions
            for option in action.option_strings
        }
        assert "--test-h5" not in options
        assert "--test-data" not in options
        assert "--test-metrics" not in options


def test_tabiclv2_remains_a_structured_license_skip() -> None:
    with pytest.raises(Stage1GateError) as caught:
        require_approved_weight("tabiclv2_regressor", {})
    assert caught.value.code == "weight_license_unconfirmed"
    assert caught.value.details["auto_download"] is False


def test_target_leaderboard_isolates_modality_lanes_and_excludes_nonpiloted(
    tmp_path: Path,
) -> None:
    def row(model_id: str, status: str, score: float, lane: str) -> dict:
        value = {
            "model_id": model_id,
            "lane": lane,
            "status": status,
            "resources": {"wall_seconds": 1.0},
            "input_budget": {"input_modalities": ["tabular"]},
            "validation": {"targets": {}},
        }
        if status == "development_piloted":
            value["validation"]["targets"]["PHIF"] = {
                "physical": {"RMSE": score, "MAE": score / 2, "R2": 0.5},
                "model_domain": {"RMSE": score, "R2": 0.5},
                "worst_mother_family": {
                    "family_id": "C",
                    "physical": {"RMSE": score},
                },
                "valid_count": 4,
            }
        return value

    board = STAGE2._leaderboard(
        "PHIF",
        [
            row("tab_b", "development_piloted", 0.2, "tabular_cpu"),
            row("skip", "skipped", 0.0, "tabular_cpu"),
            row("tab_a", "development_piloted", 0.1, "tabular_cpu"),
            row("monai", "development_piloted", 0.001, "seismic_3d_gpu"),
        ],
        "split",
        ["tabular_cpu", "seismic_3d_gpu"],
    )
    tabular = board["lanes"]["tabular_cpu"]
    seismic = board["lanes"]["seismic_3d_gpu"]
    assert tabular["status"] == "rankable"
    assert [entry["model_id"] for entry in tabular["entries"]] == ["tab_a", "tab_b"]
    assert seismic["status"] == "not_rankable"
    assert [entry["model_id"] for entry in seismic["entries"]] == ["monai"]
    assert "rank" not in seismic["entries"][0]
    assert board["cross_lane_ranking"] is False
    assert all(entry["model_id"] != "monai" for entry in tabular["entries"])
    assert board["test_access"] is False

    budget = STAGE2.load_budget()
    actual_rows = {
        "catboost_regressor": row(
            "catboost_regressor", "development_piloted", 0.2, "tabular_cpu"
        ),
        "lightgbm_regressor": row(
            "lightgbm_regressor", "development_piloted", 0.1, "tabular_cpu"
        ),
        "monai_densenet3d_regressor": row(
            "monai_densenet3d_regressor",
            "development_piloted",
            0.001,
            "seismic_3d_gpu",
        ),
    }
    summary = STAGE2.write_outputs(tmp_path, actual_rows, budget, _manifest(["C"] * 10))
    assert set(summary["modality_lanes"]) == {"tabular_cpu", "seismic_3d_gpu"}
    assert summary["modality_lanes"]["tabular_cpu"]["leaderboard_status_by_target"]["PHIF"] == "rankable"
    assert summary["modality_lanes"]["seismic_3d_gpu"]["leaderboard_status_by_target"]["PHIF"] == "not_rankable"
    written = json.loads((tmp_path / "leaderboard_phif.json").read_text())
    assert written["lanes"]["tabular_cpu"]["entries"][0]["model_id"] == "lightgbm_regressor"
    assert written["lanes"]["seismic_3d_gpu"]["entries"][0]["model_id"] == "monai_densenet3d_regressor"


def test_gpu_lane_requires_the_frozen_flock_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gpu0.lock"):
        with STAGE2.gpu_flock("cuda", tmp_path / "wrong.lock"):
            pass
    with STAGE2.gpu_flock("cpu", None) as waited:
        assert waited == 0.0
