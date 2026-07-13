"""Portable unit gates for the P5 property model-adapter lane."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _models.property._p5_common import (  # noqa: E402
    PROPERTY_TARGETS,
    Stage1GateError,
    property_output,
    require_approved_weight,
    seed_torch_runtime,
    source_lock_sha256,
    target_arrays,
)


CONTRACT = importlib.import_module("_pipelines.02_task_datasets.reservoir.p5_contract")
STAGE1 = importlib.import_module("_pipelines.02_task_datasets.reservoir.p5_stage1")

MODEL_IDS = (
    "catboost_regressor",
    "lightgbm_regressor",
    "tabm_regressor",
    "xgboost_regressor",
    "extra_trees_regressor",
    "hist_gradient_boosting_regressor",
    "realmlp_regressor",
    "ft_transformer_regressor",
    "tabiclv2_regressor",
    "monai_densenet3d_regressor",
)


def test_source_lock_and_dynamic_discovery_cover_exact_first_ten() -> None:
    lock_path = PROJECT_ROOT / "_models/property/source_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert tuple(lock["model_order"]) == MODEL_IDS
    assert len(lock["models"]) == 10
    assert len(source_lock_sha256()) == 64
    for expected in MODEL_IDS:
        discovered = discover_model("property", expected)
        assert discovered.model_id == expected
        assert discovered.module.__name__ == f"_models.property.{expected}"
        assert set(discovered.capabilities) >= {
            "task_types",
            "input_modalities",
            "supports_missing_mask",
            "supports_uncertainty",
        }


def test_property_task_spec_masks_and_klogh_transform_are_independent() -> None:
    spec = CONTRACT.build_task_spec()
    assert spec.targets == PROPERTY_TARGETS
    assert tuple(spec.target_masks) == PROPERTY_TARGETS
    assert spec.target_transform == {
        "PHIF": "identity",
        "KLOGH": "log1p(KLOGH_mD)",
        "SW": "identity",
    }
    physical = np.asarray([0.0, 1.0, 10.0, 1234.5])
    domain = CONTRACT.physical_to_model_domain("KLOGH", physical)
    assert np.allclose(CONTRACT.model_to_physical("KLOGH", domain, prediction=False), physical)

    batch = STAGE1.synthetic_contract_batch(sample_count=12)
    _, masks = target_arrays(batch, spec)
    assert masks.shape == (12, 3)
    assert np.array_equal(masks.sum(axis=0), [11, 11, 11])
    assert not np.array_equal(masks[:, 0], masks[:, 1])
    assert not np.array_equal(masks[:, 1], masks[:, 2])
    assert not np.array_equal(masks[:, 0], masks[:, 2])


def test_raw_output_is_preserved_and_physical_view_is_bounded() -> None:
    raw_matrix = np.asarray([[-0.2, -2.0, 1.2], [1.4, np.log1p(9.0), -0.1]])
    output = property_output(raw_matrix, CONTRACT.build_task_spec())
    assert np.array_equal(output.raw["PHIF"], raw_matrix[:, 0])
    assert np.array_equal(output.raw["KLOGH"], raw_matrix[:, 1])
    assert np.array_equal(output.raw["SW"], raw_matrix[:, 2])
    assert np.allclose(output.transformed["PHIF"], [0.0, 1.0])
    assert np.allclose(output.transformed["KLOGH"], [0.0, 9.0])
    assert np.allclose(output.transformed["SW"], [1.0, 0.0])


def test_tabiclv2_weight_license_gate_is_structured_and_precedes_import() -> None:
    with pytest.raises(Stage1GateError) as caught:
        require_approved_weight("tabiclv2_regressor", {})
    assert caught.value.to_dict() == {
        "code": "weight_license_unconfirmed",
        "message": (
            "tabiclv2_regressor checkpoint license is not approved in source_lock.json"
        ),
        "details": {
            "model_id": "tabiclv2_regressor",
            "checkpoint": "tabicl-regressor-v2-20260212.ckpt",
            "license_status": "unconfirmed",
            "auto_download": False,
        },
    }


def test_stage1_cli_has_no_frozen_test_argument_and_rejects_test_family(tmp_path: Path) -> None:
    parser = STAGE1.build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    for command in ("prepare", "run"):
        option_strings = {
            option
            for action in subparsers.choices[command]._actions
            for option in action.option_strings
        }
        assert not any("test" in option.lower() for option in option_strings)

    batch_path = tmp_path / "development.npz"
    count = 2
    np.savez_compressed(
        batch_path,
        seismic_patch=np.zeros((count, 3, 3, 9)),
        well_log_sequence=np.ones((count, 9, 8)),
        tabular=np.zeros((count, 153)),
        labels_model_domain=np.zeros((count, 3)),
        target_masks=np.ones((count, 3), dtype=np.uint8),
        sample_ids=np.asarray(["a", "b"]),
        family_ids=np.asarray(["15/9-F-15", "15/9-F-15"]),
        well_ids=np.asarray(["15/9-F-15", "15/9-F-15 A"]),
        depths_m=np.asarray([1.0, 2.0]),
        source_manifest_json=np.asarray(
            json.dumps({"test_access": False, "test_loader_implemented": False})
        ),
    )
    with pytest.raises(RuntimeError, match="frozen-test firewall"):
        STAGE1.load_development_batch(batch_path)


def test_synthetic_batch_shapes_are_finite_for_every_adapter_input() -> None:
    batch = STAGE1.synthetic_contract_batch()
    assert batch.inputs["tabular"].shape == (12, 153)
    assert batch.inputs["seismic_patch"].shape == (12, 3, 3, 9)
    assert batch.inputs["well_log_sequence"].shape == (12, 9, 8)
    assert all(np.isfinite(values).all() for values in batch.inputs.values())
    assert tuple(batch.targets) == PROPERTY_TARGETS
    assert tuple(batch.target_masks) == PROPERTY_TARGETS


def test_torch_seed_gate_enables_fail_closed_determinism() -> None:
    torch = pytest.importorskip("torch")
    seed_torch_runtime(torch, 2693)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
