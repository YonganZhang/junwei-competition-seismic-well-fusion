from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.artifacts import ArtifactManifest, atomic_write_json, hash_file
from _code.ml_framework.checkpoint import load_checkpoint, save_checkpoint
from _code.ml_framework.preprocess import NormStats, denormalize, normalize
from _models.facies._p5_common import source_lock

import facies_p5_r01 as r01
from p4_tasks import get_task_spec


def _load_track_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


stage = _load_track_module("facies_p5_r2", "facies_p5_r2.py")


def _section(task_id: str, inline: int, value_offset: float = 0.0) -> r01.FullSection:
    size = 8
    seismic = np.linspace(-1.0, 1.0, size * size, dtype=np.float32).reshape(size, size)
    seismic = seismic + value_offset
    num_classes = int(get_task_spec(task_id).metadata["num_classes"])
    labels = np.tile(np.arange(num_classes, dtype=np.uint8), (size, (size + num_classes - 1) // num_classes))
    labels = labels[:size, :size]
    return r01.FullSection(task_id=task_id, inline=inline, seismic=seismic, label=labels)


class _DummyModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        score = x.sum(dim=1, keepdim=True) + self.bias
        logits = [score, -score]
        while len(logits) < self.num_classes:
            logits.append(torch.zeros_like(score))
        return torch.cat(logits[: self.num_classes], dim=1)


class FaciesP5R2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_track_prefixed_module_and_signature(self) -> None:
        self.assertEqual(Path(__file__).name, "test_facies_p5_r2.py")
        self.assertEqual(stage.__name__, "facies_p5_r2")
        self.assertEqual(Path(stage.__file__).name, "facies_p5_r2.py")
        self.assertNotIn("test", inspect.signature(stage.run_r2).parameters)
        self.assertNotIn("frozen_test", inspect.signature(stage.run_r2).parameters)
        self.assertEqual(stage.CONTROL_MODELS["facies_f3"], "smp_fpn_r18")
        self.assertEqual(stage.CONTROL_MODELS["facies_penobscot"], "smp_deeplabv3plus_r18")

    def test_run_specs_are_frozen_and_split_by_task(self) -> None:
        specs = stage.build_run_specs()
        self.assertEqual(len(specs), 24)
        self.assertEqual({spec.task_id for spec in specs}, {"facies_f3", "facies_penobscot"})
        self.assertEqual({spec.recipe_id for spec in specs}, {
            "ce_2d",
            "ce_plus_dice_2d",
            "ce_to_lovasz_2d",
            "ce_2p5d",
        })
        self.assertEqual(stage.ENDPOINTS, (40, 400, 1000))
        self.assertEqual(stage.MODEL_SEEDS, (1867973658, 2137841944, 3902865753))

    def test_zscore_roundtrip_and_two_point_five_d_batch_factory(self) -> None:
        sections = {inline: _section("facies_f3", inline, 0.1 * index) for index, inline in enumerate((100, 101, 102))}
        refs = (
            r01.WindowRef("facies_f3", 101, 0, 0, 8),
        )
        stats = NormStats(method="zscore", mean=0.0, std=1.0)
        images, labels = stage._batch_from_refs(
            sections,
            refs,
            stats,
            context_mode="2p5d",
            split_bounds=(100, 102),
        )
        self.assertEqual(images.shape, (1, 3, 8, 8))
        self.assertEqual(labels.shape, (1, 8, 8))
        self.assertTrue(np.isfinite(images).all())
        restored = denormalize(normalize(images[0, 1], stats), stats)
        np.testing.assert_allclose(restored, images[0, 1], atol=1e-6)
        factory = stage._EpochBatchFactory(
            sections=sections,
            refs=refs,
            stats=stats,
            context_mode="2d",
            schedule=np.array([[0], [0]], dtype=np.int64),
            split_bounds=(100, 102),
        )
        first = list(factory())
        second = list(factory())
        self.assertEqual(first[0]["epoch"], 1)
        self.assertEqual(second[0]["epoch"], 2)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            list(factory())

    def test_real_model_in_channels_three_forward_and_checkpoint_roundtrip(self) -> None:
        task_spec = get_task_spec("facies_f3")
        model = stage.discover_model("facies", "smp_fpn_r18").build(
            task_spec,
            num_classes=int(task_spec.metadata["num_classes"]),
            lane="scratch",
            in_channels=3,
        )
        self.assertIsInstance(model, nn.Module)
        sample = torch.randn(2, 3, 32, 32)
        output = model(sample)
        self.assertEqual(tuple(output.shape), (2, int(task_spec.metadata["num_classes"]), 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.ckpt"
            save_checkpoint(
                path,
                epoch=3,
                model_state=model.state_dict(),
                optimizer_state={},
                scheduler_state=None,
                scaler_state=None,
                config_hash="c" * 64,
                split_hash="s" * 64,
                trainer_state={
                    "next_epoch": 4,
                    "global_step": 3,
                    "best_epoch": 2,
                    "best_val_loss": 0.25,
                    "epochs_without_improvement": 0,
                    "stopped_early": False,
                    "history": [{"epoch": 1, "train_loss": 1.0, "val_loss": 1.0}],
                },
                seed_report={"root_seed": 2693, "seed_tree": {"model": 1}},
                environment={"python": sys.version.split()[0], "torch": torch.__version__},
                extra={"schema_version": stage.RESULT_SCHEMA},
            )
            loaded = load_checkpoint(path)
            restored = stage.discover_model("facies", "smp_fpn_r18").build(
                task_spec,
                num_classes=int(task_spec.metadata["num_classes"]),
                lane="scratch",
                in_channels=3,
            )
            restored.load_state_dict(loaded["model_state"])
            self.assertEqual(tuple(restored(sample).shape), tuple(output.shape))

    def test_section_forward_accepts_window_logits_smaller_than_full_section(self) -> None:
        task_id = "facies_penobscot"
        num_classes = int(get_task_spec(task_id).metadata["num_classes"])
        section = _section(task_id, 200, 0.25)
        refs = tuple(
            r01.WindowRef(task_id, 200, row, col, 4)
            for row in (0, 4)
            for col in (0, 4)
        )
        prediction, probabilities, coverage, arrays = stage._section_forward(
            _DummyModel(num_classes=num_classes),
            {200: section},
            section,
            refs,
            NormStats(method="zscore", mean=0.0, std=1.0),
            context_mode="2d",
            split_bounds=(200, 200),
            batch_size=2,
            num_classes=num_classes,
            device=torch.device("cpu"),
            arrays=True,
        )
        self.assertEqual(prediction.shape, section.seismic.shape)
        self.assertEqual(probabilities.shape, (num_classes, *section.seismic.shape))
        self.assertEqual(coverage["covered_unique_voxels"], section.seismic.size)
        self.assertEqual(coverage["min_predictions_per_voxel"], 1)
        self.assertIsNotNone(arrays)
        metrics, section_stats, evaluated = stage._evaluate_full_validation(
            _DummyModel(num_classes=num_classes),
            SimpleNamespace(
                validation_sections={200: section},
                validation_window_refs=refs,
                normalization=NormStats(method="zscore", mean=0.0, std=1.0),
                validation_range=(200, 200),
            ),
            context_mode="2d",
            batch_size=2,
            num_classes=num_classes,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(section_stats), 1)
        self.assertEqual(set(evaluated), {200})
        self.assertEqual(metrics["coverage_by_inline"][0]["inline"], 200)

    def test_jsonl_writer_and_plateau_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = stage._atomic_write_jsonl(root / "records.jsonl", [{"a": 1}, {"b": 2}])
            self.assertEqual(jsonl.read_text(encoding="utf-8").count("\n"), 2)
            manifest_root = root / "manifest"
            manifest_root.mkdir()
            payload = atomic_write_json(manifest_root / "artifact.txt", {"ok": True})
            self.assertTrue(payload.is_file())
            manifest = ArtifactManifest(run_id=stage.RESULT_SCHEMA, root=manifest_root)
            manifest.register("artifact.txt", role="evidence")
            manifest.verify()
        rows = [
            {
                "task_id": "facies_f3",
                "recipe_id": "ce_2d",
                "repeat_id": repeat,
                "status": "completed",
                "endpoint_epoch": 400,
                "validation_metrics": {"miou": 0.3, "macro_f1": 0.2, "accuracy": 0.4},
            }
            for repeat in range(3)
        ] + [
            {
                "task_id": "facies_f3",
                "recipe_id": "ce_2d",
                "repeat_id": repeat,
                "status": "completed",
                "endpoint_epoch": 1000,
                "validation_metrics": {"miou": 0.301, "macro_f1": 0.2, "accuracy": 0.4},
            }
            for repeat in range(3)
        ]
        plateau = stage._plateau_decision(rows)
        self.assertTrue(plateau["ready_for_r3"])
        self.assertFalse(plateau["not_ready_for_r3"])

    def test_learning_curve_accepts_persisted_train_history_dict(self) -> None:
        records = []
        for repeat_id in range(3):
            records.append(
                {
                    "task_id": "facies_f3",
                    "recipe_id": "ce_2d",
                    "status": "completed",
                    "repeat_id": repeat_id,
                    "run_key": f"facies_f3/ce_2d/seed-{repeat_id}",
                    "endpoint_epoch": 1000,
                    "endpoint_epochs": [40, 400, 1000],
                    "history": {
                        "train_loss": [1.0, 0.5, 0.25],
                        "val_loss": [1.1, 0.6, 0.3],
                        "best_epoch": 2,
                        "best_val_loss": 0.3,
                    },
                    "endpoint_results": [
                        {"epoch": epoch, "metrics": {"miou": value}}
                        for epoch, value in ((40, 0.1), (400, 0.2), (1000, 0.3))
                    ],
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "learning_curve.png"
            rendered = stage._render_learning_curve(
                task_id="facies_f3",
                records=records,
                output_path=output_path,
            )
            self.assertEqual(rendered, output_path)
            self.assertTrue(output_path.is_file())

    def test_resume_loader_requires_complete_frozen_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(
                root / "p5_r2_config.json",
                {"budget": asdict(stage.R2Budget())},
            )
            (root / "p5_r2_results.jsonl").write_text(
                json.dumps({"status": "completed"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(stage.ProtocolBlocked, "exactly 36 completed"):
                stage._load_resume_records(root / "p5_r2_results.jsonl", stage.R2Budget())

    def test_diagnostic_renderer_with_stubbed_model(self) -> None:
        task_id = "facies_f3"
        num_classes = int(get_task_spec(task_id).metadata["num_classes"])
        section = _section(task_id, 200, 0.25)
        validation_sections = {200: section}
        refs = (r01.WindowRef(task_id, 200, 0, 0, 8),)
        material = stage.TaskMaterial(
            task_id=task_id,
            label_version=get_task_spec(task_id).label_version,
            num_classes=num_classes,
            development_range=(100, 200),
            train_range=(100, 150),
            guard_range=(151, 175),
            validation_range=(176, 200),
            source_fingerprints=({"name": "synthetic", "sha256": "0" * 64},),
            source_lock_sha256="1" * 64,
            r01_source_sha256="2" * 64,
            adapter_sha256="3" * 64,
            train_sections={200: section},
            validation_sections=validation_sections,
            train_window_refs=refs,
            validation_window_refs=refs,
            validation_proxy_refs=refs,
            normalization=NormStats(method="zscore", mean=0.0, std=1.0),
            class_weights=(1.0,) * num_classes,
            class_histogram=(1,) * num_classes,
            roundtrip_max_abs_error=0.0,
            split_hash="4" * 64,
            firewall={
                "test_archive_opened": False,
                "test_labels_read": False,
                "known_holdout_predictions_or_metrics_read": False,
                "fresh_blind": False,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.ckpt"
            dummy = _DummyModel(num_classes=num_classes)
            save_checkpoint(
                checkpoint,
                epoch=0,
                model_state=dummy.state_dict(),
                optimizer_state={},
                scheduler_state=None,
                scaler_state=None,
                config_hash="5" * 64,
                split_hash=material.split_hash,
                trainer_state={
                    "next_epoch": 1,
                    "global_step": 0,
                    "best_epoch": 0,
                    "best_val_loss": 1.0,
                    "epochs_without_improvement": 0,
                    "stopped_early": False,
                    "history": [{"epoch": 1, "train_loss": 1.0, "val_loss": 1.0}],
                },
                seed_report={"root_seed": 2693, "seed_tree": {"model": 1}},
                environment={"python": sys.version.split()[0], "torch": torch.__version__},
                extra={"schema_version": stage.RESULT_SCHEMA},
            )

            class _StubDiscovery:
                def build(self, task_spec, **config):
                    return _DummyModel(num_classes=num_classes)

            diagnostic = stage.SectionPrediction(
                inline=200,
                sample_id="facies_f3:validation:inline=200",
                gt_class_count=2,
                correct_pixels=32,
                error_pixels=32,
                boundary_pixels=20,
                boundary_error_pixels=10,
                total_pixels=64,
                error_fraction=0.5,
                coverage={"duplicate_prediction_assignments_before_blend": 0},
                has_informative_signal=True,
            )
            with patch.object(stage, "discover_model", return_value=_StubDiscovery()):
                manifest = stage._render_diagnostics(
                    task_id=task_id,
                    checkpoint_path=checkpoint,
                    material=material,
                    diagnostic=diagnostic,
                    context_mode="2d",
                    runtime_root=root,
                    device=torch.device("cpu"),
                    output_path=root / "diag.png",
                )
            self.assertEqual(manifest["checkpoint_runtime_relative_path"], "best.ckpt")
            self.assertTrue((root / "diag.png").is_file())
            self.assertEqual(manifest["selection_rule"], "prefer_validation_section_with_>=2_gt_classes_and_correct_plus_error_pixels")
            self.assertEqual(manifest["selection_sample_id"], diagnostic.sample_id)
            self.assertEqual(manifest["firewall"]["fresh_blind"], False)


if __name__ == "__main__":
    unittest.main()
