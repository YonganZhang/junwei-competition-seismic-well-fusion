"""Fail-closed tests for property P5 Stage-4 known-holdout confirmation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


RESERVOIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = RESERVOIR / "reservoir_p5_stage4.py"
SPEC = importlib.util.spec_from_file_location("reservoir_p5_stage4", RUNNER_PATH)
assert SPEC and SPEC.loader
STAGE4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE4)


def test_contract_freezes_unique_winners_seed_and_exact_32_update_budget() -> None:
    contract = STAGE4.load_contract()
    assert contract["root_seed"] == 2693
    assert contract["confirmation_kind"] == "previously_seen_reusable_holdout"
    assert contract["prior_test_consumed"] is True
    assert contract["fresh_blind"] is False
    assert {
        target: contract["winners"][target]["model_id"]
        for target in STAGE4.PROPERTY_TARGETS
    } == {
        "PHIF": "extra_trees_regressor",
        "KLOGH": "extra_trees_regressor",
        "SW": "xgboost_regressor",
    }
    for winner in contract["winners"].values():
        assert winner["rank"] == 1
        assert winner["lane"] == "tabular_cpu"
        assert winner["budget"]["n_estimators"] == 32
        assert winner["budget"]["update_steps"] == 32


def test_stage3_hashes_split_winners_whitelist_and_prior_exposure_verify() -> None:
    report = STAGE4.validate_stage3_and_contract()
    assert report["stage3_split_hash"] == (
        "2334f3cc301fc66d6b98c6edf3a4f9c920776469531003d62f5370e119426a18"
    )
    assert report["feature_whitelist_verified"] is True
    assert report["forbidden_feature_overlap"] == []
    assert report["prior_exposure"]["status"] == "verified_before_fitting"
    assert report["prior_exposure"]["prior_test_consumed"] is True
    assert report["prior_exposure"]["fresh_blind"] is False
    assert {target: row["model_id"] for target, row in report["winners"].items()} == STAGE4.FROZEN_WINNERS


def test_contract_tampering_fails_closed(tmp_path: Path) -> None:
    payload = STAGE4.load_contract()
    payload["winners"]["SW"]["budget"]["n_estimators"] = 33
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="budget changed"):
        STAGE4.load_contract(path)


def test_record_gate_enforces_family_shape_finiteness_and_separation() -> None:
    records = []
    for index, family in enumerate(STAGE4.DEVELOPMENT_FAMILIES):
        records.append(
            {
                "sample_id": f"sample-{index}",
                "family_id": family,
                "well_id": family,
                "depth_m": float(index),
                "position": {},
                "seismic": np.zeros((3, 3, 9)),
                "logs": np.zeros((9, 8)),
                "labels": np.asarray([0.2, np.log1p(100.0), 0.4]),
            }
        )
    STAGE4._validate_records(records, set(STAGE4.DEVELOPMENT_FAMILIES), "development")
    records[0]["family_id"] = STAGE4.KNOWN_HOLDOUT_FAMILY
    with pytest.raises(RuntimeError, match="families changed"):
        STAGE4._validate_records(records, set(STAGE4.DEVELOPMENT_FAMILIES), "development")


def test_target_metric_transforms_and_interval_contract_are_physical() -> None:
    truth_log = np.log1p(np.asarray([0.0, 10.0, 100.0]))
    prediction_log = np.log1p(np.asarray([1.0, 12.0, 80.0]))
    metrics, arrays = STAGE4._target_metrics("KLOGH", truth_log, prediction_log, 0.2)
    assert metrics["unit"] == "mD"
    assert metrics["model_domain_name"] == "log1p(KLOGH_mD)"
    assert np.allclose(arrays["truth_physical"], [0.0, 10.0, 100.0])
    assert np.isfinite(arrays["prediction_physical"]).all()
    assert metrics["interval"]["nominal_coverage"] == 0.9
    assert metrics["prior_test_consumed"] is True
    assert metrics["fresh_blind"] is False
    for target in ("PHIF", "SW"):
        _, bounded = STAGE4._target_metrics(
            target,
            np.asarray([0.1, 0.9]),
            np.asarray([-0.2, 1.2]),
            0.1,
        )
        assert np.all((bounded["prediction_physical"] >= 0) & (bounded["prediction_physical"] <= 1))


def test_single_use_state_refuses_any_second_confirmation(tmp_path: Path) -> None:
    state = {
        "state": "CONFIRMED",
        "prior_test_consumed": True,
        "fresh_blind": False,
    }
    (tmp_path / "confirmation_state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="single-use"):
        STAGE4.confirm_stage4(output_dir=tmp_path)


def test_confirm_cli_has_no_paths_or_model_selection_surface() -> None:
    parser = STAGE4.build_parser()
    confirm = next(action for action in parser._actions if action.dest == "command").choices["confirm"]
    options = {option for action in confirm._actions for option in action.option_strings}
    assert options == {"-h", "--help"}
    assert "finalize" not in parser._subparsers._group_actions[0].choices


def test_plot_source_has_no_forbidden_titles_or_host_paths() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "set_title" not in source
    assert "suptitle" not in source
    assert "plt.title" not in source
    for forbidden in ("/" + "mnt" + "/data", ".claude/" + "worktrees"):
        assert forbidden not in source


def test_real_portable_confirmation_when_present() -> None:
    root = RESERVOIR / "_outputs" / "p5_stage4_confirmation"
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        pytest.skip("real Stage-4 confirmation has not been executed")
    audit = STAGE4.audit_existing(root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    state = json.loads((root / "confirmation_state.json").read_text(encoding="utf-8"))
    assert audit["status"] == "verified"
    assert state["state"] == "CONFIRMED"
    assert summary["development_rows"] == 1216
    assert summary["known_holdout_rows"] == 344
    assert summary["confirmation_kind"] == "previously_seen_reusable_holdout"
    assert summary["prior_test_consumed"] is True
    assert summary["fresh_blind"] is False
    for target, model_id in STAGE4.FROZEN_WINNERS.items():
        target_summary = summary["target_summaries"][target]
        assert target_summary["model_id"] == model_id
        assert target_summary["known_holdout_rows"] == 344
        assert np.isfinite(target_summary["physical"]["MAE"])
        assert np.isfinite(target_summary["physical"]["RMSE"])
        assert (root / target.lower() / "predictions.csv").is_file()
        assert len(list((root / target.lower() / "figures").glob("*.png"))) == 3
