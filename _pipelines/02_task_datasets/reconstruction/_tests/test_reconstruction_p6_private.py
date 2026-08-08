from __future__ import annotations

import importlib
import json
from dataclasses import asdict
import sys
from pathlib import Path

import numpy as np


RECON_DIR = Path(__file__).resolve().parents[1]
if str(RECON_DIR) not in sys.path:
    sys.path.insert(0, str(RECON_DIR))

mod = importlib.import_module("reconstruction_p6_private")


def test_build_mode_makes_qc_and_track_specs() -> None:
    strict = mod.build_mode("strict")
    conditional = mod.build_mode("conditional")
    assert strict.qc_evidence.agent_mode == "supervisory_qc_agent"
    assert conditional.qc_evidence.agent_mode == "supervisory_qc_agent"
    assert strict.conclusion["state"] == "no_verified_gain"
    assert conditional.conclusion["state"] == "no_verified_gain"
    assert strict.track_spec.task_type == "volume_3d"
    assert "global_well_constraints" in conditional.track_spec.input_fields
    assert "well_constraints" not in strict.framework_task_spec.input_whitelist
    assert "pseudo_test_well_constraints" in conditional.framework_task_spec.input_whitelist


def test_model_batch_is_real_3d_and_lane_separated() -> None:
    strict = mod.build_mode("strict")
    conditional = mod.build_mode("conditional")
    strict_batch = asdict(strict.model_batch)
    conditional_batch = asdict(conditional.model_batch)
    assert strict_batch["inputs"]["mode"] == ["strict"]
    assert conditional_batch["inputs"]["mode"] == ["conditional"]
    assert strict_batch["metadata"]["strict_no_test_well_constraints"] is True
    assert conditional_batch["metadata"]["conditional_reuses_pseudo_test_constraints"] is True
    assert strict_batch["metadata"]["prior_test_consumed"] is True
    assert conditional_batch["metadata"]["prior_test_consumed"] is True
    assert strict_batch["coordinates"]["kji"]
    assert conditional_batch["coordinates"]["kji"]
    assert strict_batch["target_masks"]["poro"]
    assert conditional_batch["target_masks"]["poro"]


def test_write_package_creates_manifest_and_references_real_figures(tmp_path: Path) -> None:
    package = mod.write_package(tmp_path)
    assert package["modes"]["strict"]["conclusion"] == "strict/conclusion.json"
    assert package["modes"]["conditional"]["lane_table"] == "conditional/lane_table.json"
    conclusion = json.loads((tmp_path / "strict" / "conclusion.json").read_text(encoding="utf-8"))
    assert conclusion["state"] == "no_verified_gain"
    sci = json.loads((tmp_path / "conditional" / "sci_manifest.json").read_text(encoding="utf-8"))
    assert sci["figure_count"] >= 1
    assert all(figure["exists"] for figure in sci["figures"])
    assert any("prediction_comparison" in figure["path"] for figure in sci["figures"])
    artifact = json.loads((tmp_path / "p6_private_artifact_manifest.json").read_text(encoding="utf-8"))
    assert artifact["files"]
    assert any(entry["path"].endswith("qc_evidence.json") for entry in artifact["files"])
    assert package["package_hash"] == json.loads((tmp_path / "p6_private_package.json").read_text(encoding="utf-8"))["package_hash"]
