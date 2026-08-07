"""Fail-closed tests for P5.2 / protocol R2 property learning curves."""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


RESERVOIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = RESERVOIR / "reservoir_p5_r2.py"
SPEC = importlib.util.spec_from_file_location("reservoir_p5_r2", RUNNER_PATH)
assert SPEC and SPEC.loader
R2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R2)


def _write_audit_envelope(
    output_dir: Path,
    *,
    expected_cells: int,
    figures: list[dict] | None = None,
) -> None:
    rows = [
        json.loads(line)
        for line in (output_dir / "p5_r2_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    payloads = {
        "p5_r2_oof_manifest.json": {
            "schema_version": 1,
            "entries": [
                {
                    key: row[key]
                    for key in (
                        "cell_id",
                        "model_id",
                        "phase",
                        "budget",
                        "fold_id",
                        "repeat_id",
                        "status",
                        "oof",
                        "checkpoint",
                        "split",
                    )
                }
                for row in rows
            ],
        },
        "p5_r2_visualization_manifest.json": {
            "schema_version": 1,
            "figures": [] if figures is None else figures,
        },
    }
    manifests = {}
    for filename, payload in payloads.items():
        payload["sha256"] = R2._hash_payload(payload)
        (output_dir / filename).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
        manifests[filename] = payload["sha256"]
    summary = {
        "expected_cells": expected_cells,
        "oof_manifest_sha256": manifests["p5_r2_oof_manifest.json"],
        "visualization_manifest_sha256": manifests["p5_r2_visualization_manifest.json"],
        "artifact_hashes": {
            "results": R2._hash_file(output_dir / "p5_r2_results.jsonl"),
            "oof_manifest": manifests["p5_r2_oof_manifest.json"],
            "visualization_manifest": manifests["p5_r2_visualization_manifest.json"],
        },
    }
    summary["summary_sha256"] = R2._hash_payload(summary)
    (output_dir / "p5_r2_summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )


def test_r2_contract_freezes_seed_budget_and_records_historical_source_lock_mismatch() -> None:
    contract = R2.load_r2_contract()
    assert contract["root_seed"] == 2693
    assert contract["baseline_head"] == "af8c066de0c3fc24fce024abb350b4f2c9e82d9b"
    assert contract["development_families"] == [
        "15/9-19",
        "15/9-F-1",
        "15/9-F-11",
        "15/9-F-12",
    ]
    assert contract["lanes"] == {
        "scratch_flat_fusion": "tabm_regressor",
        "tabular_pretrained": "tabiclv2_regressor",
        "seismic_3d_gpu": "monai_densenet3d_regressor",
    }
    assert contract["curve_budgets"]["tabm_regressor"] == [40, 160, 640]
    assert contract["curve_budgets"]["tabiclv2_regressor"] == [1, 2, 4]
    assert contract["curve_budgets"]["monai_densenet3d_regressor"] == [12, 48, 192]
    assert len(contract["repeat_seeds"]) == 3
    assert sum(len(values) for values in contract["curve_budgets"].values()) * 4 * 3 == 108
    assert contract["preflight"]["historical_source_lock_mismatch"] is True
    assert contract["preflight"]["current_source_lock_sha256"] == (
        "fa109524a662fdc23cb793580206b9ca9dbcf58cc18a2947db0d6d3bb68a8269"
    )
    assert contract["preflight"]["historical_source_lock_sha256"] == (
        "a25789166f0a9eae7bbfe6716009fa65aa45c2a1f93118e1e10ffb9711bb0db8"
    )
    assert contract["preflight"]["stage4_validation_status"] == "historical_source_lock_mismatch"
    assert contract["historical_stage4_contract"]["known_holdout_family"] == "15/9-F-15"


def test_r2_plateau_uses_pooled_physical_metrics_and_detects_selection() -> None:
    curve_metrics = {
        target: {
            "40": {
                "mean_physical_RMSE": 2.00,
                "mean_worst_family_RMSE": 2.50,
                "completed_cells": 12,
            },
            "160": {
                "mean_physical_RMSE": 1.99,
                "mean_worst_family_RMSE": 2.49,
                "completed_cells": 12,
            },
            "640": {
                "mean_physical_RMSE": 1.985,
                "mean_worst_family_RMSE": 2.485,
                "completed_cells": 12,
            },
        }
        for target in ("PHIF", "KLOGH", "SW")
    }
    pooled, plateau = R2._selected_budget_for_targets(
        model_id="tabm_regressor", curve_metrics=curve_metrics
    )
    assert pooled["40"]["mean_physical_RMSE"] == pytest.approx(2.0)
    assert plateau["status"] == "plateaued"
    assert plateau["budget"] == 40
    assert plateau["reason"] == "all higher budgets <1% improvement"


def test_curve_metrics_are_grouped_by_model_before_budget() -> None:
    def row(model_id: str, rmse: float) -> dict:
        return {
            "model_id": model_id,
            "phase": "curve",
            "budget": 40,
            "status": "completed",
            "validation": {
                "targets": {
                    target: {
                        "physical": {"RMSE": rmse},
                        "worst_mother_family": {"physical": {"RMSE": rmse + 1.0}},
                    }
                    for target in ("PHIF", "KLOGH", "SW")
                }
            },
        }

    grouped = R2._group_curve_metrics(
        [row("tabm_regressor", 1.0), row("monai_densenet3d_regressor", 9.0)]
    )
    assert grouped["tabm_regressor"]["PHIF"]["40"]["mean_physical_RMSE"] == 1.0
    assert grouped["monai_densenet3d_regressor"]["PHIF"]["40"]["mean_physical_RMSE"] == 9.0
    assert grouped["tabiclv2_regressor"]["PHIF"] == {}


def test_torch_adapter_supports_mae_bounded_roundtrip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from _code.ml_framework.contracts import ModelBatch
    from _models.property._p5_common import TorchMultiTargetAdapter

    task_spec = R2.build_task_spec()
    module = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        module.weight.copy_(
            torch.tensor(
                [[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4]],
                dtype=torch.float32,
            )
        )

    batch = ModelBatch(
        inputs={"tabular": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
        targets={
            "PHIF": np.asarray([0.2, 0.8], dtype=np.float64),
            "KLOGH": np.asarray([np.log1p(5.0), np.log1p(9.0)], dtype=np.float64),
            "SW": np.asarray([0.1, 0.9], dtype=np.float64),
        },
        input_masks={},
        target_masks={
            "PHIF": np.asarray([1, 1], dtype=bool),
            "KLOGH": np.asarray([1, 1], dtype=bool),
            "SW": np.asarray([1, 1], dtype=bool),
        },
        sample_ids=["s0", "s1"],
        groups={"mother_well_family": ["A", "B"]},
        coordinates={"depth_m": np.asarray([1000.0, 1001.0])},
    )

    adapter = TorchMultiTargetAdapter(
        model_id="tabm_regressor",
        task_spec=task_spec,
        torch_module=module,
        input_builder=lambda current: (current.inputs["tabular"],),
        torch=torch,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        config={"seed": 2693, "loss_name": "mae", "output_activation": "bounded"},
    )
    fit_report = adapter.fit(batch)
    assert fit_report["loss_name"] == "mae"
    assert fit_report["output_activation"] == "bounded"
    prediction = adapter.predict_array(batch)
    assert prediction.shape == (2, 3)
    assert np.isfinite(prediction).all()

    checkpoint = tmp_path / "checkpoint.bin"
    adapter.save_checkpoint(checkpoint)
    restored = TorchMultiTargetAdapter(
        model_id="tabm_regressor",
        task_spec=task_spec,
        torch_module=torch.nn.Linear(2, 3, bias=False),
        input_builder=lambda current: (current.inputs["tabular"],),
        torch=torch,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        config={"seed": 2693, "loss_name": "mae", "output_activation": "bounded"},
    )
    restored.load_checkpoint(checkpoint)
    assert np.allclose(prediction, restored.predict_array(batch))


def test_torch_adapter_explicitly_expands_targets_for_ensemble_huber_loss() -> None:
    torch = pytest.importorskip("torch")
    from _code.ml_framework.contracts import ModelBatch
    from _models.property._p5_common import TorchMultiTargetAdapter

    class EnsembleHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 3, bias=False)

        def forward(self, features):
            return self.linear(features).unsqueeze(1).expand(-1, 4, -1)

    batch = ModelBatch(
        inputs={"tabular": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
        targets={target: np.asarray([0.2, 0.8]) for target in ("PHIF", "KLOGH", "SW")},
        input_masks={},
        target_masks={target: np.asarray([1, 1], dtype=bool) for target in ("PHIF", "KLOGH", "SW")},
        sample_ids=["s0", "s1"],
        groups={"mother_well_family": ["A", "B"]},
        coordinates={"depth_m": np.asarray([1000.0, 1001.0])},
    )
    adapter = TorchMultiTargetAdapter(
        model_id="tabm_regressor",
        task_spec=R2.build_task_spec(),
        torch_module=EnsembleHead(),
        input_builder=lambda current: (current.inputs["tabular"],),
        torch=torch,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        config={"seed": 2693, "loss_name": "huber", "output_activation": "identity"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        report = adapter.fit(batch)
    assert np.isfinite(report["loss"])


def test_tabiclv2_missing_checkpoint_is_structured_skip_without_test_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_discovered = SimpleNamespace(
        model_id="tabiclv2_regressor",
        module=SimpleNamespace(__name__="_models.property.tabiclv2_regressor"),
        capabilities={
            "task_types": ["regression"],
            "input_modalities": ["tabular"],
            "supports_missing_mask": True,
            "supports_uncertainty": False,
        },
        build=lambda *_args, **_kwargs: pytest.fail("build should not be reached"),
    )
    monkeypatch.setattr(R2, "discover_model", lambda *_args, **_kwargs: fake_discovered)
    dummy_batch = SimpleNamespace()
    evidence = {
        "split_hash": "deadbeef",
        "train_groups": ["A"],
        "validation_groups": ["B"],
        "train_sample_ids_sha256": "1" * 64,
        "validation_sample_ids_sha256": "2" * 64,
        "preprocessing": {"fit_validation_overlap": False},
    }
    row = R2._run_model_cell(
        model_id="tabiclv2_regressor",
        phase="curve",
        budget=1,
        loss_name="mse",
        output_activation="identity",
        fold_id=0,
        repeat_id=0,
        repeat_seed=2693,
        train_batch=dummy_batch,
        validation_batch=dummy_batch,
        fold_evidence=evidence,
        output_dir=tmp_path,
        tabicl_checkpoint=None,
    )
    assert row["status"] == "skipped"
    assert row["reason"]["code"] == "weight_checkpoint_missing"
    assert row["test_firewall"]["test_access"] is False


def test_ablation_runtime_paths_are_variant_specific() -> None:
    common = {
        "output_dir": Path("portable"),
        "model_id": "tabm_regressor",
        "phase": "ablation",
        "budget": 160,
        "fold_id": 0,
        "repeat_id": 0,
    }
    mse_checkpoint, mse_oof = R2._runtime_cell_paths(
        **common, loss_name="mse", output_activation="identity"
    )
    huber_checkpoint, huber_oof = R2._runtime_cell_paths(
        **common, loss_name="huber", output_activation="bounded"
    )
    assert mse_checkpoint != huber_checkpoint
    assert mse_oof != huber_oof
    assert "loss-mse__output-identity" in mse_oof.as_posix()
    assert "loss-huber__output-bounded" in huber_oof.as_posix()


def test_audit_existing_accepts_portable_relative_oof_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "portable"
    output_dir.mkdir(parents=True)
    oof_path = output_dir / "runtime/oof/tabm_regressor/curve/budget40/fold0__repeat0.npz"
    oof_path.parent.mkdir(parents=True)
    np.savez_compressed(oof_path, prediction=np.asarray([1.0]))
    row = {
        "cell_id": "tabm_regressor__curve__budget40__fold0__repeat0__loss-mse__output-identity",
        "model_id": "tabm_regressor",
        "phase": "curve",
        "budget": 40,
        "fold_id": 0,
        "repeat_id": 0,
        "status": "completed",
        "oof": {
            "relative_path": "runtime/oof/tabm_regressor/curve/budget40/fold0__repeat0.npz",
            "sha256": R2._hash_file(oof_path),
        },
        "checkpoint": {"roundtrip": True},
        "split": {"split_hash": "abc"},
    }
    (output_dir / "p5_r2_results.jsonl").write_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_audit_envelope(output_dir, expected_cells=1)
    audit = R2.audit_existing(output_dir)
    assert audit["status"] == "verified"
    assert audit["rows"] == 1


def test_audit_existing_accepts_and_hashes_manifest_figures(tmp_path: Path) -> None:
    output_dir = tmp_path / "portable"
    output_dir.mkdir(parents=True)
    (output_dir / "p5_r2_results.jsonl").write_text("", encoding="utf-8")
    figure_path = output_dir / "figures" / "curve.png"
    figure_path.parent.mkdir(parents=True)
    figure_path.write_bytes(b"not-a-real-png-but-hashable-audit-evidence")
    _write_audit_envelope(
        output_dir,
        expected_cells=0,
        figures=[
            {
                "path": "figures/curve.png",
                "sha256": R2._hash_file(figure_path),
            }
        ],
    )
    audit = R2.audit_existing(output_dir)
    assert audit["status"] == "verified"


def test_audit_existing_rejects_completed_cells_that_share_oof_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "portable"
    oof_path = output_dir / "runtime/oof/tabm_regressor/ablation/shared.npz"
    oof_path.parent.mkdir(parents=True)
    np.savez_compressed(oof_path, prediction=np.asarray([1.0]))
    shared_oof = {
        "relative_path": "runtime/oof/tabm_regressor/ablation/shared.npz",
        "sha256": R2._hash_file(oof_path),
    }
    rows = [
        {
            "cell_id": f"cell-{index}",
            "model_id": "tabm_regressor",
            "phase": "ablation",
            "budget": 160,
            "fold_id": 0,
            "repeat_id": index,
            "status": "completed",
            "oof": shared_oof,
            "checkpoint": {"roundtrip": True},
            "split": {"split_hash": "abc"},
        }
        for index in range(2)
    ]
    (output_dir / "p5_r2_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_audit_envelope(output_dir, expected_cells=2)
    with pytest.raises(RuntimeError, match="share OOF path"):
        R2.audit_existing(output_dir)
