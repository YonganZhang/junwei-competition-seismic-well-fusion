from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)



def _load_track_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_p5_stage1 = _load_track_module("facies_p5_stage1_firewall", "p5_stage1.py")
DevelopmentOnlyArchive = _p5_stage1.DevelopmentOnlyArchive
prepare_development_batch = _p5_stage1.prepare_development_batch
run_stage1 = _p5_stage1.run_stage1


def _write_development_archive(root: Path, task_id: str, classes: int) -> None:
    task_root = root / task_id
    task_root.mkdir(parents=True)
    with h5py.File(task_root / "train.h5", "w") as archive:
        archive.attrs["task"] = task_id
        archive.attrs["split"] = "train"
        inline_start = 100 if task_id == "facies_f3" else 1000
        for index in range(12):
            group = archive.create_group(f"sample_{index:07d}")
            label = np.tile(np.arange(classes, dtype=np.uint8), (16, 1))
            raw = label.astype(np.float32) + index * 0.05
            group.create_dataset("seismic_patch", data=raw)
            group.create_dataset("label", data=label)
            group.attrs["position"] = json.dumps(
                {
                    "inline": inline_start + index,
                    "crossline": 300,
                    "time_ms": 0.0,
                    "well_name": None,
                }
            )
            group.attrs["meta"] = json.dumps(
                {"task": task_id, "split": "train", "source": "synthetic-development"}
            )


class _TinyModel(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(1, classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)


class _FakeDiscovered:
    capabilities = {
        "task_types": ["multiclass"],
        "input_modalities": ["seismic_amplitude_2d"],
        "supports_missing_mask": False,
        "supports_uncertainty": False,
    }

    def build(self, task_spec, **config):
        self.last_config = config
        return _TinyModel(int(task_spec.metadata["num_classes"]))


class P5Stage1FirewallTests(unittest.TestCase):
    def test_runner_surface_has_no_test_or_frozen_test_argument(self) -> None:
        parameters = inspect.signature(run_stage1).parameters
        self.assertNotIn("test", parameters)
        self.assertNotIn("test_loader", parameters)
        self.assertNotIn("frozen_test", parameters)

    def test_archive_rejects_test_path_before_filesystem_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = DevelopmentOnlyArchive("facies_f3", Path(directory))
            with self.assertRaisesRegex(RuntimeError, "frozen-test"):
                archive.split_path("test")

    def test_real_preparation_and_results_remain_task_separate(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            output = root / "output"
            _write_development_archive(processed, "facies_f3", 10)
            _write_development_archive(processed, "facies_penobscot", 8)
            f3 = prepare_development_batch("facies_f3", processed)
            pen = prepare_development_batch("facies_penobscot", processed)
            self.assertEqual(f3.labels.max(), 9)
            self.assertEqual(pen.labels.max(), 7)
            self.assertNotEqual(f3.development_batch_hash, pen.development_batch_hash)
            self.assertLessEqual(max(map(int, f3.inline_groups)), 463)
            self.assertLessEqual(max(map(int, pen.inline_groups)), 1335)
            self.assertEqual(f3.guard_inline_range, (464, 488))
            self.assertEqual(pen.guard_inline_range, (1336, 1358))

            with patch("p5_stage1.discover_model", return_value=_FakeDiscovered()):
                summary = run_stage1(
                    tasks=("facies_f3", "facies_penobscot"),
                    models=("smp_unet_r18",),
                    processed_root=processed,
                    output_root=output,
                    device=torch.device("cpu"),
                )
            self.assertTrue(summary["tasks_are_independent"])
            self.assertFalse(summary["test_archive_opened"])
            self.assertEqual(summary["tasks"]["facies_f3"]["head_num_classes"], 10)
            self.assertEqual(summary["tasks"]["facies_penobscot"]["head_num_classes"], 8)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIn("summary.json", manifest["artifacts"])
            for task_id, classes in (("facies_f3", 10), ("facies_penobscot", 8)):
                result_path = output / task_id / "smp_unet_r18" / "scratch" / "stage1.json"
                result = json.loads(result_path.read_text())
                self.assertEqual(result["status"], "contract_smoked")
                self.assertEqual(result["head_num_classes"], classes)
                self.assertFalse(result["test_archive_opened"])
                self.assertFalse(result["test_labels_read"])
                self.assertTrue((output / result["checkpoint"]["path"]).is_file())
                self.assertIn(
                    result["checkpoint"]["path"],
                    manifest["artifacts"],
                )


if __name__ == "__main__":
    unittest.main()
