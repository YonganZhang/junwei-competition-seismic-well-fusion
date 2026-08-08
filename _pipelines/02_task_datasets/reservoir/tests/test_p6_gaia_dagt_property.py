"""Fail-closed tests for the private P6 property Gaia/DAGT evidence pack."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


RESERVOIR = Path(__file__).resolve().parents[1]
MODULE_PATH = RESERVOIR / "p6_gaia_dagt_property.py"
SPEC = importlib.util.spec_from_file_location("reservoir_p6_gaia_dagt_property", MODULE_PATH)
assert SPEC and SPEC.loader
P6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P6)


def test_evidence_bundle_contains_qc_only_agent_and_private_target_states(tmp_path: Path) -> None:
    bundle = P6.build_evidence_bundle()

    assert bundle["conclusion"]["status"] == "no_verified_gain"
    assert bundle["conclusion"]["agent_mode"] == "supervisory_qc_agent"
    assert bundle["conclusion"]["predictive_text_agent"] == "blocked_by_missing_sample_level_text"

    assert bundle["qc_track_spec"]["track_id"] == "property_p6_gaia_dagt_qc"
    assert bundle["qc_track_spec"]["task_type"] == "multitask"
    assert bundle["qc_track_spec"]["target_fields"] == ["PHIF", "KLOGH", "SW"]
    assert bundle["qc_track_spec"]["source_manifest_digest"]

    roundtrip = json.loads(json.dumps(bundle["model_batch"], sort_keys=True))
    assert roundtrip["task_targets"]["PHIF"] == [0.12, 0.18]
    assert roundtrip["task_masks"]["KLOGH"] == [True, True]
    assert roundtrip["feasibility"] == {"PHIF": True, "KLOGH": True, "SW": True}

    deny_list = bundle["agent_evidence"]["deny_list"]
    for token in ("checkpoint", "holdout", "test_metric", "inferred_ac_offset"):
        assert token in deny_list

    assert bundle["qc_dry_run"]["metric"]["loss"] >= 0.0
    assert bundle["qc_dry_run"]["svg_sha256"]
    assert bundle["qc_dry_run"]["svg_path"].endswith("qc_dry_run.svg")

    assert len(bundle["state_table"]) == 3
    assert all(row["status"] == "no_verified_gain" for row in bundle["state_table"])
    assert all(row["f2"] == "blocked_by_missing_sample_level_text" for row in bundle["state_table"])
    assert all(row["c1"] == "disabled" and row["c2"] == "disabled" for row in bundle["state_table"])

    candidates = bundle["candidate_state_index"]
    assert candidates["B0"]["model_id"] == "extra_trees_regressor"
    assert candidates["B1"]["model_id"] == "xgboost_regressor"
    assert candidates["tabm_regressor"]["state"] == "development_piloted"
    assert candidates["realmlp_regressor"]["state"] == "development_piloted"
    assert candidates["ft_transformer_regressor"]["state"] == "development_piloted"
    assert candidates["tabiclv2_regressor"]["state"] == "skipped"
    assert candidates["monai_densenet3d_regressor"]["state"] == "development_piloted"

    written = P6._write_portable_outputs(bundle, tmp_path)
    for path in written.values():
        assert path.is_file()
        if path.suffix in {".json", ".md", ".svg"}:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".svg":
                assert text.startswith("<svg")
            else:
                assert "/mnt/data/" not in text
                assert "file://" not in text


def test_science_manifest_indexes_existing_target_figures() -> None:
    manifest = P6._science_manifest()
    assert manifest["track_id"] == "property"
    assert manifest["figure_count"] == 9
    targets = {figure["target"] for figure in manifest["figures"]}
    assert targets == {"PHIF", "KLOGH", "SW"}
    assert all(figure["path"].endswith(".png") for figure in manifest["figures"])


def test_candidate_state_index_records_tabiclv2_and_monai_blockers() -> None:
    tabicl = P6._candidate_state("tabiclv2_regressor")
    monai = P6._candidate_state("monai_densenet3d_regressor")
    assert tabicl["state"] == "skipped"
    assert monai["state"] == "development_piloted"
    assert any(row["reason"] for row in tabicl["records"])
    assert not any(row["reason"] for row in monai["records"])
