"""Track-prefixed portable contracts for property P5.1 R0/R1."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np


R01 = importlib.import_module(
    "_pipelines.02_task_datasets.reservoir.reservoir_p5_r01"
)


def _synthetic_development() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(R01.ROOT_SEED)
    families = np.repeat(np.asarray(R01.DEVELOPMENT_FAMILIES), 4)
    count = len(families)
    seismic = rng.normal(size=(count, 3, 3, 9))
    log_values = rng.normal(size=(count, 9, 4))
    log_masks = np.ones((count, 9, 4), dtype=np.float64)
    logs = np.concatenate([log_values, log_masks], axis=2)
    labels = np.linspace(0.1, 0.4, count)
    sample_ids = [f"development-{index}" for index in range(count)]
    return seismic, logs, labels, families, sample_ids


def test_r0_freezes_targets_families_lanes_and_tabicl_official_identity() -> None:
    contract = R01.build_r0_contract()
    assert contract["status"] == "contract_frozen"
    assert contract["zero_training"] is True
    assert tuple(contract["targets"]) == ("PHIF", "KLOGH", "SW")
    assert contract["targets"]["PHIF"]["unit"] == "fraction"
    assert contract["targets"]["KLOGH"]["unit"] == "mD"
    assert contract["targets"]["KLOGH"]["target_transform"] == "log1p(KLOGH_mD)"
    assert contract["targets"]["SW"]["unit"] == "fraction"
    assert all(
        payload["independent_from_other_target_masks"]
        for payload in contract["targets"].values()
    )
    assert contract["source_joint_complete_case_filter"] is True
    assert tuple(contract["development"]["families"]) == R01.DEVELOPMENT_FAMILIES
    assert contract["frozen_test"] == {
        "family": "15/9-F-15",
        "access": False,
        "loader_implemented": False,
        "metrics_allowed": False,
        "fresh_blind_claim_allowed": False,
    }
    assert set(contract["lanes"]) == {
        "scratch_flat_fusion",
        "tabiclv2_tabular_pretrained",
        "monai_seismic_3d_gpu",
    }
    assert contract["tabiclv2"]["official_checkpoint_available"] is True
    assert contract["tabiclv2"]["local_status"] == "artifact_unavailable"
    assert contract["tabiclv2"]["source_lock_entry"]["license"] == "BSD-3-Clause"


def test_cli_has_no_test_surface_and_family_firewall_fails_closed() -> None:
    parser = R01.build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    for command in ("audit", "run"):
        options = {
            option
            for action in subparsers.choices[command]._actions
            for option in action.option_strings
        }
        assert not any("test" in option.lower() or "holdout" in option.lower() for option in options)
    allowed = [
        {
            "sample_id": family,
            "family_id": family,
            "seismic": np.zeros((3, 3, 9)),
            "logs": np.zeros((9, 8)),
            "label": np.zeros(3),
        }
        for family in R01.DEVELOPMENT_FAMILIES
    ]
    R01._validate_development_rows(allowed)
    bad = [dict(row) for row in allowed]
    bad[0]["family_id"] = R01.FROZEN_TEST_FAMILY
    try:
        R01._validate_development_rows(bad)
    except RuntimeError as error:
        assert "frozen-test family" in str(error)
    else:
        raise AssertionError("frozen-test family must fail closed")


def test_random_kfold_is_overlapping_diagnostic_and_logo4_is_group_disjoint() -> None:
    _, _, _, families, sample_ids = _synthetic_development()
    random_folds = R01._random_kfold(len(families))
    logo_folds = R01._logo4(families)
    random_manifest = R01._protocol_manifest(random_folds, families, sample_ids)
    logo_manifest = R01._protocol_manifest(logo_folds, families, sample_ids)
    assert len(random_manifest) == len(logo_manifest) == 4
    assert all(fold["mother_family_overlap"] for fold in random_manifest)
    assert all(not fold["mother_family_overlap"] for fold in logo_manifest)
    assert [fold["validation_groups"] for fold in logo_manifest] == [
        [family] for family in R01.LOGO_VALIDATION_ORDER
    ]
    assert sum(fold["validation_count"] for fold in random_manifest) == len(families)
    assert sum(fold["validation_count"] for fold in logo_manifest) == len(families)


def test_fold_preprocessing_is_train_only_and_ridge_prediction_is_finite() -> None:
    seismic, logs, labels, families, sample_ids = _synthetic_development()
    fold = R01._logo4(families)[0]
    prediction, evidence = R01._run_fold(
        target="PHIF",
        seismic=seismic,
        logs=logs,
        labels=labels,
        families=families,
        sample_ids=sample_ids,
        train_indices=fold["train"],
        validation_indices=fold["validation"],
        fold_id=0,
        protocol="mother_family_logo4",
    )
    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()
    assert evidence["preprocessing"]["fit"] == "fold_train_only"
    assert evidence["preprocessing"]["fit_validation_overlap"] is False
    assert evidence["preprocessing"]["target_statistics_fitted"] is False
    assert evidence["preprocessing"]["target_transform_fitted"] is False
    assert evidence["mother_family_overlap"] == []
    assert evidence["test_firewall"]["test_access"] is False


def test_target_metrics_are_independent_physical_and_klogh_has_log_diagnostic() -> None:
    families = np.asarray(["A", "A", "B", "B"])
    phif = R01._metrics_by_target(
        "PHIF",
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([-0.1, 0.22, 0.28, 1.2]),
        families,
    )
    assert phif["physical_unit"] == "fraction"
    assert phif["raw_prediction_boundary"]["invalid_count"] == 2
    assert phif["worst_mother_family"]["family_id"] in {"A", "B"}
    truth_k = np.log1p(np.asarray([1.0, 10.0, 100.0, 1000.0]))
    klogh = R01._metrics_by_target("KLOGH", truth_k, truth_k + 0.1, families)
    assert klogh["physical_unit"] == "mD"
    assert klogh["log1p_diagnostic"] == klogh["model_domain"]
    assert klogh["physical_unclipped"]["RMSE"] > klogh["model_domain"]["RMSE"]


def test_reservoir_ridge_single_target_checkpoint_roundtrip(tmp_path: Path) -> None:
    spec = R01._make_target_task_spec("SW")
    model = R01.build_ridge(
        spec,
        n_features=3,
        learning_rate=0.002,
        l2_strength=0.001,
        seed=R01.ROOT_SEED,
    )
    features = np.arange(18, dtype=float).reshape(6, 3) / 18.0
    targets = np.linspace(0.1, 0.6, 6)[:, None]
    model.train_batch((features, targets))
    expected = model.predict_array(features)
    checkpoint = tmp_path / "ridge.ckpt"
    model.save_checkpoint(checkpoint)
    restored = R01.build_ridge(
        spec,
        n_features=3,
        learning_rate=0.002,
        l2_strength=0.001,
        seed=R01.ROOT_SEED,
    )
    restored.load_checkpoint(checkpoint)
    assert np.array_equal(restored.predict_array(features), expected)


def test_portable_r01_outputs_when_present_are_complete_and_path_clean() -> None:
    output_dir = R01.DEFAULT_OUTPUT_DIR
    summary_path = output_dir / "r1_summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split = json.loads((output_dir / "r1_split_manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert summary["r1_status"] == "protocol_diagnostic_complete"
    assert tuple(summary["targets"]) == ("KLOGH", "PHIF", "SW")
    assert all(value["status"] == "protocol_diagnostic_complete" for value in summary["targets"].values())
    assert split["test_firewall"]["test_access"] is False
    assert split["frozen_test_family"] == R01.FROZEN_TEST_FAMILY
    assert artifacts["absolute_paths_persisted"] is False
    assert artifacts["large_runtime_artifacts_persisted"] is False
    portable_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv"}
    )
    assert ".claude/worktrees" not in portable_text
    assert "/mnt/data/" not in portable_text
