from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RECON_DIR = Path(__file__).resolve().parents[1]
if str(RECON_DIR) not in sys.path:
    sys.path.insert(0, str(RECON_DIR))

mod = importlib.import_module("reconstruction_p5_r2")


def test_train_path_accepts_only_train_h5(tmp_path: Path) -> None:
    train_path = tmp_path / "train.h5"
    train_path.write_bytes(b"train")
    (tmp_path / "test.h5").write_bytes(b"test")

    assert mod._train_path(tmp_path) == train_path


def test_build_point_selection_uses_validation_local_indices(monkeypatch) -> None:
    def fake_select_spatial_points(coordinates: np.ndarray, indices_kji: np.ndarray, count: int) -> np.ndarray:
        assert coordinates.shape[0] == indices_kji.shape[0]
        return np.arange(count, dtype=np.int64)

    monkeypatch.setattr(mod.r01, "select_spatial_points", fake_select_spatial_points)

    n = 80
    cell_k_blocks = np.concatenate(
        [
            np.full(10, 0, dtype=np.int16),
            np.full(10, 1, dtype=np.int16),
            np.full(10, 2, dtype=np.int16),
            np.full(10, 3, dtype=np.int16),
            np.full(40, 4, dtype=np.int16),
        ]
    )
    geometry = SimpleNamespace(
        cell_k_blocks=cell_k_blocks,
        coordinates=np.column_stack(
            (
                np.linspace(0.0, 1.0, n),
                np.linspace(1.0, 2.0, n),
                np.linspace(2.0, 3.0, n),
            )
        ).astype(np.float64),
        indices_kji=np.column_stack(
            (
                np.arange(n, dtype=np.int64),
                np.arange(n, dtype=np.int64) % 7,
                np.arange(n, dtype=np.int64) % 5,
            )
        ),
    )

    selection = mod._build_point_selection("strict", geometry)
    assert selection["train_local_indices"].shape[0] == 30
    assert selection["validation_local_indices"].shape[0] == 40
    assert selection["pseudo_well_local_indices"].shape[0] == 32
    assert np.all(np.isin(selection["pseudo_well_global_indices"], selection["validation_global_indices"]))


@dataclass
class FakeHistory:
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    best_val_loss: float

    def to_dict(self) -> dict[str, object]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }


class FakeModel:
    def train_batch(self, batch) -> float:
        features, target = batch
        return float(np.mean(features) + np.mean(target))

    def validation_loss(self, batch) -> float:
        features, target = batch
        return float(np.mean(features) - np.mean(target)) ** 2

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], 0.5, dtype=np.float64)

    def save_checkpoint(self, path: Path) -> None:
        path.write_text("checkpoint", encoding="utf-8")


def _fake_bundle(mode: str) -> mod.ModeBundle:
    n = 80
    validation_mask = np.zeros(n, dtype=bool)
    validation_mask[40:] = True
    train_mask = ~validation_mask
    indices = np.column_stack(
        (
            np.arange(n, dtype=np.int64),
            np.arange(n, dtype=np.int64) % 7,
            np.arange(n, dtype=np.int64) % 5,
        )
    )
    coordinates = np.column_stack(
        (
            np.linspace(0.0, 1.0, n),
            np.linspace(1.0, 2.0, n),
            np.linspace(2.0, 3.0, n),
        )
    ).astype(np.float64)
    geometry = SimpleNamespace(
        indices_kji=indices,
        coordinates=coordinates,
        seismic=np.column_stack(
            (
                np.linspace(0.0, 0.5, n),
                np.linspace(0.1, 0.6, n),
                np.linspace(0.2, 0.7, n),
            )
        ).astype(np.float64),
        access_audit={"physical_test_h5_opened": False},
        volume_shape_kji=(n, 1, 1),
    )
    train_global = np.flatnonzero(train_mask)
    validation_global = np.flatnonzero(validation_mask)
    pseudo_local = np.arange(32, dtype=np.int64)
    pseudo_global = validation_global[pseudo_local]
    target = np.linspace(0.0, 1.0, n, dtype=np.float64)
    split_manifest = {"split_hash": "split-hash"}
    manifest = {
        "selection_hashes": {
            "train_selection_hash": "train-hash",
            "validation_selection_hash": "validation-hash",
            "pseudo_well_selection_hash": "pseudo-hash",
        }
    }
    common_metric_mask = validation_mask.copy()
    common_metric_mask[pseudo_global] = False
    pseudo_distances = np.linspace(1.0, 2.0, validation_global.size, dtype=np.float64)
    return mod.ModeBundle(
        mode=mode,
        geometry=geometry,
        split_manifest=split_manifest,
        train_mask=train_mask,
        validation_mask=validation_mask,
        train_local_indices=np.arange(train_global.size, dtype=np.int64),
        validation_local_indices=np.arange(validation_global.size, dtype=np.int64),
        train_global_indices=train_global,
        validation_global_indices=validation_global,
        pseudo_well_local_indices=pseudo_local,
        pseudo_well_global_indices=pseudo_global,
        train_target=target[train_global],
        validation_target=target[validation_global],
        pseudo_well_values=target[pseudo_global],
        common_metric_mask=common_metric_mask,
        pseudo_test_distances=pseudo_distances,
        distance_edges=(0.0, 1.0, 2.0, np.inf),
        train_constraints=np.column_stack([coordinates[train_global], target[train_global]]).astype(np.float64),
        pseudo_test_constraints=np.column_stack([coordinates[pseudo_global], target[pseudo_global]]).astype(np.float64),
        access_audit=geometry.access_audit,
        manifest=manifest,
    )


def test_run_one_cell_smoke_for_strict_and_conditional(monkeypatch, tmp_path: Path) -> None:
    bundle = _fake_bundle("strict")
    fake_model = FakeModel()

    def fake_discover_model(track_id: str, model_id: str):
        assert track_id == "reconstruction"
        return SimpleNamespace(build=lambda task_spec, **config: fake_model)

    def fake_train_loop(**kwargs):
        checkpoint_dir = kwargs["checkpoint_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        kwargs["save_checkpoint_fn"](kwargs["model"], checkpoint_dir / "best.ckpt")
        kwargs["save_checkpoint_fn"](kwargs["model"], checkpoint_dir / "last.ckpt")
        return FakeHistory(
            train_loss=[1.0, 0.8, 0.7],
            val_loss=[1.2, 0.9, 0.95],
            best_epoch=1,
            best_val_loss=0.9,
        )

    monkeypatch.setattr(mod, "discover_model", fake_discover_model)
    monkeypatch.setattr(mod, "train_loop", fake_train_loop)
    monkeypatch.setattr(mod, "_load_best_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(mod, "_distance_band_metrics", lambda *args, **kwargs: [{"band_id": 0, "voxel_count": 1}])
    monkeypatch.setattr(mod, "_render_visualization", lambda **kwargs: kwargs["output_path"].write_text("viz", encoding="utf-8"))

    strict_result = mod._run_one_cell(
        mode="strict",
        model_id="reconstruction_linear_sgd",
        loss_name="mse",
        budget=25,
        feature_variant="full",
        bundle=bundle,
        output_root=tmp_path,
    )
    assert strict_result["status"] == "passed"
    assert strict_result["condition_status"]["B1"] == "not_applicable"
    assert (tmp_path / "cells" / "strict" / "reconstruction_linear_sgd" / "mse" / "full" / "updates_025" / "status.json").is_file()

    conditional_result = mod._run_one_cell(
        mode="conditional",
        model_id="reconstruction_linear_sgd",
        loss_name="huber",
        budget=25,
        feature_variant="drop_well",
        bundle=_fake_bundle("conditional"),
        output_root=tmp_path,
    )
    assert conditional_result["status"] == "passed"
    assert conditional_result["condition_status"]["B1"] == "passed"
    assert "B1" in conditional_result["prediction_hashes"]
    assert "shuffled" in conditional_result["prediction_hashes"]
