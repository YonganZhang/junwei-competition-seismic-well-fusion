from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.seeding import derive_seed
from _code.ml_framework.splits import Fold, SplitManifest, validate_manifest
from _code.ml_framework.artifacts import atomic_write_json, hash_file

import facies_p5_stage2 as stage2
from p4_tasks import LABEL_VERSIONS, get_task_spec


def _load_track_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


stage3 = _load_track_module("facies_p5_stage3", "facies_p5_stage3.py")


def _write_five_fold_archive(root: Path, task_id: str) -> Path:
    classes = int(get_task_spec(task_id).metadata["num_classes"])
    start = 100 if task_id == "facies_f3" else 1000
    buffer_groups = 25 if task_id == "facies_f3" else 23
    groups = [start + index * 100 for index in range(5)]
    sample_ids: list[str] = []
    task_root = root / "processed" / task_id
    task_root.mkdir(parents=True)
    with h5py.File(task_root / "train.h5", "w") as archive:
        archive.attrs["task"] = task_id
        archive.attrs["split"] = "train"
        for index, inline in enumerate(groups):
            key = f"sample_{index:07d}"
            sample_id = f"{task_id}:train:{key}"
            sample_ids.append(sample_id)
            group = archive.create_group(key)
            label = np.tile(
                np.arange(classes, dtype=np.uint8),
                (32, (32 + classes - 1) // classes),
            )[:, :32]
            raw = (
                np.linspace(-1.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
                + index * 0.05
            )
            group.create_dataset("seismic_patch", data=raw)
            group.create_dataset("label", data=label)
            group.attrs["position"] = json.dumps(
                {"inline": inline, "crossline": 300, "time_ms": 0.0}
            )
            group.attrs["meta"] = json.dumps(
                {"task": task_id, "split": "train", "source": "synthetic-development"}
            )
    folds: list[Fold] = []
    support = {
        "train_per_class_pixels": [100] * classes,
        "validation_per_class_pixels": [100] * classes,
    }
    for fold_id in range(5):
        validation_ids = (sample_ids[fold_id],)
        train_ids = tuple(value for index, value in enumerate(sample_ids) if index != fold_id)
        validation_groups = (str(groups[fold_id]),)
        train_groups = tuple(str(value) for index, value in enumerate(groups) if index != fold_id)
        nearest = min(abs(groups[fold_id] - value) for value in groups if value != groups[fold_id])
        folds.append(
            Fold(
                fold_id=fold_id,
                train_groups=train_groups,
                validation_groups=validation_groups,
                train_sample_ids=train_ids,
                validation_sample_ids=validation_ids,
                purge={
                    "strategy": "permanent_contiguous_inline_buffer",
                    "buffer_groups": buffer_groups,
                    "nearest_train_validation_inline_distance": nearest,
                },
                support=support,
            )
        )
    manifest = SplitManifest(
        manifest_version="facies-p4-v1",
        group_key="inline",
        requested_n_splits=5,
        effective_n_splits=5,
        downgrade_reason=None,
        test_groups=("9999",),
        test_sample_ids=(f"{task_id}:test:not-opened",),
        development_groups=tuple(map(str, groups)),
        development_sample_ids=tuple(sample_ids),
        folds=tuple(folds),
        metadata={
            "track_id": "facies",
            "task_id": task_id,
            "label_version": LABEL_VERSIONS[task_id],
            "num_classes": classes,
        },
    )
    validate_manifest(manifest)
    path = root / f"{task_id}_split_manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path


def _manifest_hash(path: Path) -> str:
    return stage2._manifest_from_dict(json.loads(path.read_text())).stable_hash()


def _dummy_metrics(classes: int, score: float) -> dict[str, object]:
    confusion = np.eye(classes, dtype=np.int64) * 10
    return {
        "accuracy": score,
        "miou": score,
        "macro_f1": score,
        "per_class_support": [10] * classes,
        "per_class_iou": [score] * classes,
        "per_class_f1": [score] * classes,
        "confusion_matrix": confusion.tolist(),
        "evaluated_pixels": classes * 10,
        "all_classes_supported": True,
        "finite_logits": True,
    }


def _dummy_records(*, completed: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, cell in enumerate(stage3.expected_cells()):
        classes = int(get_task_spec(cell.task_id).metadata["num_classes"])
        score = 0.2 + 0.01 * (2 - stage3.TOP_MODELS[cell.task_id].index(cell.model_id))
        row: dict[str, object] = {
            **asdict(cell),
            "schema_version": stage3.RESULT_SCHEMA,
            "track_id": "facies",
            "cell_key": cell.key,
            "status": "completed" if completed else "blocked",
            "test_archive_opened": False,
            "test_labels_read": False,
            "test_metrics_computed": False,
        }
        if completed:
            row.update(
                {
                    "validation_metrics": _dummy_metrics(classes, score),
                    "resources": {
                        "wall_seconds": 1.0 + index / 1000,
                        "cuda_peak_allocated_bytes": 1024,
                    },
                }
            )
        rows.append(row)
    return rows


class FaciesP5Stage3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_track_prefixed_module_and_test_names(self) -> None:
        self.assertEqual(Path(__file__).name, "test_facies_p5_stage3.py")
        self.assertEqual(stage3.__name__, "facies_p5_stage3")
        self.assertEqual(Path(stage3.__file__).name, "facies_p5_stage3.py")

    def test_exact_ninety_cell_matrix_and_frozen_seeds(self) -> None:
        cells = stage3.expected_cells()
        self.assertEqual(len(cells), 90)
        self.assertEqual(len({cell.key for cell in cells}), 90)
        observed = tuple(
            derive_seed(2693, "model", "p5-stage3", repeat_id)
            for repeat_id in range(3)
        )
        self.assertEqual(observed, stage3.REPEAT_SEEDS)
        self.assertEqual(observed, (1867973658, 2137841944, 3902865753))
        for task_id in stage3.TASK_IDS:
            task_cells = [cell for cell in cells if cell.task_id == task_id]
            self.assertEqual(len(task_cells), 45)
            self.assertEqual({cell.lane for cell in task_cells}, {"scratch"})
            self.assertEqual({cell.fold_id for cell in task_cells}, set(range(5)))
            self.assertEqual(
                {cell.model_id for cell in task_cells}, set(stage3.TOP_MODELS[task_id])
            )

    def test_budget_is_exact_stage2_copy_and_fail_closed(self) -> None:
        stage3_budget = asdict(stage3.Stage3Budget())
        stage2_budget = asdict(stage2.PilotBudget())
        stage2_budget.pop("fold_id")
        self.assertEqual(stage3_budget, stage2_budget)
        with self.assertRaisesRegex(ValueError, "reuse Stage-2"):
            stage3.Stage3Budget(max_updates=41)
        with self.assertRaisesRegex(ValueError, "reuse Stage-2"):
            stage3.Stage3Budget(loss_id="focal")

    def test_gpu_contract_rejects_cpu_missing_lock_and_wrong_lock(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CPU is forbidden"):
            stage3.validate_gpu_contract(torch.device("cpu"), str(stage3.EXPECTED_GPU_LOCK))
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
        ):
            with self.assertRaisesRegex(RuntimeError, "must be set"):
                stage3.validate_gpu_contract(torch.device("cuda:0"), None)
            with self.assertRaisesRegex(RuntimeError, "must equal"):
                stage3.validate_gpu_contract(torch.device("cuda:0"), "/tmp/wrong.lock")
            self.assertEqual(
                stage3.validate_gpu_contract(
                    torch.device("cuda:0"), str(stage3.EXPECTED_GPU_LOCK)
                ),
                stage3.EXPECTED_GPU_LOCK,
            )

    def test_runner_and_visualizer_have_no_frozen_test_surface(self) -> None:
        for function in (stage3.run_stage3, stage3.render_visualizations):
            parameters = inspect.signature(function).parameters
            for forbidden in ("test", "test_loader", "test_root", "frozen_test"):
                self.assertNotIn(forbidden, parameters)
        with tempfile.TemporaryDirectory() as directory:
            archive = stage2.Stage2DevelopmentArchive("facies_f3", Path(directory))
            with self.assertRaisesRegex(RuntimeError, "frozen-test"):
                archive.split_path("test")

    def test_each_fold_preprocessor_uses_only_locked_fold_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_five_fold_archive(root, "facies_f3")
            locked = {
                **stage2.LOCKED_MANIFEST_STABLE_HASHES,
                "facies_f3": _manifest_hash(manifest),
            }
            with patch.object(stage2, "LOCKED_MANIFEST_STABLE_HASHES", locked):
                fold = stage3.prepare_fold(
                    task_id="facies_f3",
                    fold_id=2,
                    manifest_path=manifest,
                    processed_root=root / "processed",
                    budget=stage3.Stage3Budget(),
                )
            self.assertEqual(fold.fold_id, 2)
            self.assertFalse(set(fold.train_sample_ids) & set(fold.validation_sample_ids))
            self.assertFalse(set(fold.train_groups) & set(fold.validation_groups))
            self.assertGreater(fold.nearest_inline_distance, fold.buffer_groups)
            self.assertEqual(
                fold.preprocessor.fit_sample_ids_hash,
                stage3.hash_payload(list(fold.train_sample_ids)),
            )
            self.assertLessEqual(fold.preprocessor.roundtrip_max_abs_error, 1e-2)
            self.assertEqual(fold.update_schedule.shape, (40, 2))

    def test_duplicate_missing_and_cross_lane_cells_fail_closed(self) -> None:
        cells = list(stage3.expected_cells())
        with self.assertRaisesRegex(ValueError, "exactly 90"):
            stage3.validate_cell_specs(cells[:-1])
        duplicate = cells[:-1] + [cells[0]]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            stage3.validate_cell_specs(duplicate)
        contaminated = list(_dummy_records(completed=False))
        contaminated[0]["lane"] = "pretrained"
        with self.assertRaisesRegex(ValueError, "frozen lane"):
            stage3.validate_records(contaminated)

    def test_under_eighty_percent_is_not_rankable_without_imputation(self) -> None:
        records = _dummy_records(completed=False)
        task_indices = [
            index for index, row in enumerate(records) if row["task_id"] == "facies_f3"
        ]
        for index in task_indices[:35]:
            row = records[index]
            classes = int(row["head_num_classes"]) if "head_num_classes" in row else 10
            row["status"] = "completed"
            row["validation_metrics"] = _dummy_metrics(classes, 0.2)
            row["resources"] = {
                "wall_seconds": 1.0,
                "cuda_peak_allocated_bytes": 1024,
            }
        board = stage3.build_leaderboard("facies_f3", records)
        self.assertEqual(board["completed_cells"], 35)
        self.assertLess(board["completion_rate"], 0.8)
        self.assertEqual(board["status"], "not_rankable")
        self.assertTrue(all(entry["rank"] is None for entry in board["entries"]))

    def test_diagnostic_selection_rejects_uninformative_seeded_sample(self) -> None:
        records = _dummy_records(completed=True)
        task_id = "facies_penobscot"
        winner_id = stage3.TOP_MODELS[task_id][0]
        winner_rows = [
            row
            for row in records
            if row["task_id"] == task_id and row["model_id"] == winner_id
        ][:2]
        classes = int(get_task_spec(task_id).metadata["num_classes"])
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            entries = []
            for index, row in enumerate(winner_rows):
                path = runtime / f"candidate_{index}.npz"
                if index == 0:
                    labels = np.zeros((1, 8, 8), dtype=np.uint8)
                    prediction = labels.copy()
                    sample_id = "seeded-single-class"
                else:
                    labels = np.resize(
                        np.arange(classes, dtype=np.uint8), (1, 8, 8)
                    )
                    prediction = labels.copy()
                    prediction[0, 0, 0] = (int(labels[0, 0, 0]) + 1) % classes
                    sample_id = "informative-development-oof"
                error = np.asarray(prediction != labels, dtype=np.uint8)
                stage3._atomic_save_npz(
                    path,
                    sample_ids=np.asarray([sample_id], dtype=str),
                    inline=np.asarray([100 + index], dtype=np.int64),
                    seismic=np.zeros((1, 8, 8), dtype=np.float16),
                    labels=labels,
                    prediction=prediction,
                    confidence=np.ones((1, 8, 8), dtype=np.float16),
                    entropy=np.zeros((1, 8, 8), dtype=np.float16),
                    error=error,
                )
                entries.append(
                    {
                        "cell_key": row["cell_key"],
                        "runtime_relative_path": path.relative_to(runtime).as_posix(),
                        "sha256": hash_file(path),
                    }
                )
            selected = stage3._select_diagnostic_sample(
                task_id=task_id,
                winner_id=winner_id,
                winner_rows=winner_rows,
                oof_manifest={"entries": entries},
                runtime_root=runtime,
            )
            self.assertEqual(
                selected["selection_outcome"], "global_informative_candidate"
            )
            self.assertEqual(
                selected["statistics"]["sample_id"],
                "informative-development-oof",
            )
            self.assertTrue(selected["statistics"]["eligible"])
            self.assertGreaterEqual(selected["statistics"]["gt_class_count"], 2)
            self.assertGreater(selected["statistics"]["correct_pixels"], 0)
            self.assertGreater(selected["statistics"]["error_pixels"], 0)

    def test_visualization_rebuild_reads_only_archived_oof(self) -> None:
        records = _dummy_records(completed=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "portable"
            runtime = root / "runtime"
            output.mkdir()
            stage2._atomic_write_jsonl(output / "p5_stage3_results.jsonl", records)
            manifest_entries = []
            for task_id in stage3.TASK_IDS:
                board = stage3.build_leaderboard(task_id, records)
                atomic_write_json(output / f"{task_id}_scratch_leaderboard.json", board)
                winner = next(entry for entry in board["entries"] if entry["rank"] == 1)
                representative = next(
                    row
                    for row in records
                    if row["task_id"] == task_id
                    and row["model_id"] == winner["model_id"]
                    and row["fold_id"] == 0
                    and row["repeat_id"] == 0
                )
                classes = int(get_task_spec(task_id).metadata["num_classes"])
                prediction_path = runtime / f"{task_id}.npz"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                labels = np.tile(np.arange(classes, dtype=np.uint8), (16, 1))
                labels = np.resize(labels, (1, 16, 16))
                prediction = labels.copy()
                prediction[0, 0, 0] = (int(labels[0, 0, 0]) + 1) % classes
                error = np.asarray(prediction != labels, dtype=np.uint8)
                stage3._atomic_save_npz(
                    prediction_path,
                    sample_ids=np.asarray([f"{task_id}:train:sample"], dtype=str),
                    inline=np.asarray([100], dtype=np.int64),
                    seismic=np.zeros((1, 16, 16), dtype=np.float16),
                    labels=labels,
                    probabilities=np.full(
                        (1, classes, 16, 16), 1 / classes, dtype=np.float16
                    ),
                    prediction=prediction,
                    confidence=np.ones((1, 16, 16), dtype=np.float16),
                    entropy=np.zeros((1, 16, 16), dtype=np.float16),
                    error=error,
                )
                manifest_entries.append(
                    {
                        "cell_key": representative["cell_key"],
                        "runtime_relative_path": prediction_path.relative_to(runtime).as_posix(),
                        "sha256": hash_file(prediction_path),
                    }
                )
            atomic_write_json(
                output / "p5_stage3_oof_manifest.json",
                {
                    "schema_version": stage3.RESULT_SCHEMA,
                    "entries": manifest_entries,
                    "frozen_test_consumed": False,
                },
            )
            manifest_path = stage3.render_visualizations(
                output_root=output, runtime_root=runtime
            )
            payload = json.loads(manifest_path.read_text())
            self.assertTrue(all(item["status"] == "generated" for item in payload["figures"]))
            self.assertTrue(all(item["no_model_or_dataset_loaded"] for item in payload["figures"]))
            self.assertTrue(
                all(item["selection_statistics"]["eligible"] for item in payload["figures"])
            )
            self.assertFalse(payload["frozen_test_consumed"])
            for task_id in stage3.TASK_IDS:
                self.assertTrue((output / f"{task_id}_stage3_oof_diagnostics.png").is_file())


if __name__ == "__main__":
    unittest.main()
