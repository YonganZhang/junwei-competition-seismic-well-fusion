from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
MODULE_PATH = TRACK_DIR / "build_dataset.py"
SCHEMA_PATH = TRACK_DIR / "label_spec.schema.v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("sweetspot_build_dataset", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline():
    return _load_module()


@pytest.fixture(scope="module")
def real_inventory(pipeline):
    source_root = pipeline.resolve_source_root(PROJECT_ROOT)
    return pipeline.audit_sources(PROJECT_ROOT, source_root)


def _approved_spec() -> dict:
    """只用于验证合同闸门，不是军伟批准的真实标签定义。"""
    return {
        "schema_version": "sweetspot-label-spec/v1",
        "spec_version": "1.0.0-test-fixture",
        "status": "approved",
        "target_semantics": "geological",
        "output": {
            "type": "binary",
            "classes": ["negative", "positive"],
            "units": "class",
            "probability_interpretation": "future model output only",
        },
        "allowed_source_fields": [
            {"source": "las", "field": "LFP_PHIE", "role": "candidate_evidence"},
            {"source": "las", "field": "LFP_GR", "role": "inference_input"},
        ],
        "label_construction": {
            "formula": "domain-expert-reviewed fixture rule",
            "formula_field_refs": [{"source": "las", "field": "LFP_PHIE"}],
            "thresholds": [],
            "thresholds_not_applicable_reason": "fixture uses an externally fixed expert rule",
            "weights": [],
            "weights_not_applicable_reason": "fixture has no weighted combination",
            "fit_domain": {
                "statistics_scope": "external_fixed",
                "population": "test fixture only",
                "uses_test_statistics": False,
            },
        },
        "time_window": {
            "definition": "static interpretation snapshot",
            "start": None,
            "end": None,
            "timezone": None,
            "leakage_cutoff": "all label evidence fixed before split evaluation",
        },
        "spatial_scale": {
            "support": "well_interval",
            "coordinate_system": "MD plus weak-tie inline/crossline/TWT",
            "vertical_domain": "measured_depth_m",
            "resolution": "0.5 m",
            "alignment_tolerance": "one seismic grid cell and one time sample",
        },
        "class_rules": {
            "positive": "expert fixture positive",
            "negative": "expert fixture negative",
            "unlabeled": "exclude from supervised samples",
        },
        "split_strategy": {
            "strategy": "well_holdout",
            "group_key": "well_name",
            "train_rule": "explicit approved training-well list",
            "validation_rule": "explicit approved validation-well list",
            "test_rule": "explicit approved held-out test-well list",
            "fit_statistics_scope": "train_only",
            "leakage_guards": ["no well occurs in more than one split"],
        },
        "inference_allowed_inputs": [
            {"source": "las", "field": "LFP_GR"},
        ],
        "metrics": [
            {"name": "PR-AUC", "aggregation": "held-out wells", "decision_threshold": None},
        ],
        "approval": {
            "approved": True,
            "approved_by": "TEST FIXTURE - NOT A REAL APPROVER",
            "approved_role": "unit test",
            "approved_at": "2026-01-01T00:00:00Z",
            "decision_record": "tests/test_validate_only.py",
        },
        "notes": "Contract-gate fixture only; never use to generate labels.",
    }


def test_real_audit_finds_candidates_but_no_sweetspot_label(real_inventory):
    catalog = real_inventory["field_catalog"]
    assert "LFP_PHIE" in catalog["las"]
    assert "BORE_OIL_VOL" in catalog["production.daily"]
    assert "inline" in catalog["layer1.fault_points"]
    assert "twt_est_ms" in catalog["layer1.well_tie_weak"]
    assert "LFP_PHIE" in catalog["layer1.well_logs_clean.clean"]
    assert real_inventory["artifacts"]["layer1.well_logs_clean"]["n_tracks"] == 3
    for track in real_inventory["artifacts"]["layer1.well_logs_clean"]["tracks"].values():
        assert track["attributes"]["las_path"].startswith("_sandbox/")
    assert not any(
        "sweetspot" in field.lower()
        for fields in catalog.values()
        for field in fields
    )
    assert real_inventory["label_readiness"]["sweetspot_truth_found"] is False


def test_complete_approved_contract_validates_but_never_creates_dataset(
    pipeline, real_inventory, tmp_path
):
    fake_project_root = tmp_path / "project"
    forbidden_dir = fake_project_root / "_data" / "processed" / "sweetspot"
    report_path = tmp_path / "contract_validation.json"

    result = pipeline.run_validate_only(
        _approved_spec(),
        real_inventory,
        project_root=fake_project_root,
        schema_path=SCHEMA_PATH,
        validation_report_path=report_path,
    )

    assert result["valid"] is True
    assert result["dataset_write_attempted"] is False
    assert not forbidden_dir.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["valid"] is True
    assert not list(tmp_path.rglob("*.h5"))


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (lambda s: s["approval"].update(approved=False), "approved=false"),
        (
            lambda s: s["allowed_source_fields"].append(
                {"source": "las", "field": "NOT_A_REAL_CURVE", "role": "candidate_evidence"}
            ),
            "未知或不存在的真实字段",
        ),
        (
            lambda s: s["label_construction"]["fit_domain"].update(
                statistics_scope="test", uses_test_statistics=True
            ),
            "test 统计",
        ),
        (lambda s: s["class_rules"].update(negative=""), "负样本规则"),
        (lambda s: s["spatial_scale"].update(resolution="<REQUIRED>"), "空间尺度"),
        (lambda s: s["split_strategy"].update(test_rule=""), "split 规则"),
        (lambda s: s.update(unexpected_key="not allowed"), "Additional properties"),
    ],
)
def test_fail_closed_contract_rejections(
    pipeline, real_inventory, tmp_path, mutate, expected_error
):
    spec = copy.deepcopy(_approved_spec())
    mutate(spec)
    forbidden_dir = tmp_path / "project" / "_data" / "processed" / "sweetspot"

    result = pipeline.run_validate_only(
        spec,
        real_inventory,
        project_root=tmp_path / "project",
        schema_path=SCHEMA_PATH,
        validation_report_path=tmp_path / "validation.json",
    )

    assert result["valid"] is False
    assert any(expected_error in error for error in result["errors"])
    assert result["dataset_write_attempted"] is False
    assert not forbidden_dir.exists()


def test_missing_spec_fails_without_creating_processed_directory(
    pipeline, real_inventory, tmp_path
):
    result = pipeline.run_validate_only(
        None,
        real_inventory,
        project_root=tmp_path / "project",
        schema_path=SCHEMA_PATH,
        validation_report_path=tmp_path / "validation.json",
    )
    assert result["valid"] is False
    assert any("缺少 label spec" in error for error in result["errors"])
    assert not (tmp_path / "project" / "_data" / "processed" / "sweetspot").exists()


def test_validate_only_module_cannot_call_dataset_writer():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "import dataset_io" not in source
    assert "from dataset_io" not in source
    assert "save_split(" not in source
