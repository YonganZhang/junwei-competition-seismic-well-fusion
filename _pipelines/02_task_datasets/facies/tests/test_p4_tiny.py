from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.checkpoint import load_checkpoint
from _code.ml_framework.model_registry import get_model

from p4_data import FaciesArchive
from p4_losses import build_loss
from p4_tasks import fixed_baseline_config
from p4_training import train_development_model


def _write_train_archive(root: Path) -> None:
    task_root = root / "facies_f3"
    task_root.mkdir(parents=True)
    with h5py.File(task_root / "train.h5", "w") as archive:
        archive.attrs["task"] = "facies_f3"
        archive.attrs["split"] = "train"
        for index in range(4):
            group = archive.create_group(f"sample_{index:07d}")
            label = np.tile(np.arange(10, dtype=np.uint8), (8, 1))
            raw = label.astype(np.float32) + index * 0.01
            group.create_dataset("seismic_patch", data=raw)
            group.create_dataset("label", data=label)
            group.attrs["position"] = json.dumps(
                {"inline": 100 + index, "crossline": 300, "time_ms": 0.0, "well_name": None}
            )
            group.attrs["meta"] = json.dumps(
                {"task": "facies_f3", "split": "train", "source": "synthetic-tiny"}
            )


class TinyOverfitAndCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_linear_pixel_tiny_overfit_decreases_loss(self) -> None:
        labels = torch.arange(10, dtype=torch.long).repeat(1, 8, 1)
        inputs = labels.float().unsqueeze(1)
        inputs = (inputs - inputs.mean()) / inputs.std()
        model = get_model(
            "facies_linear_pixel", models_package="models", num_classes=10
        )
        criterion = build_loss(
            "cross_entropy", num_classes=10, class_weights=torch.ones(10)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
        with torch.no_grad():
            initial = float(criterion(model(inputs), labels))
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        final = float(criterion(model(inputs), labels).detach())
        self.assertLess(final, initial * 0.4)

    def test_shared_trainer_writes_complete_resumable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            _write_train_archive(processed)
            archive = FaciesArchive("facies_f3", processed)
            records = archive.development_index()
            preprocessing = archive.fit_preprocessor(records[:2])
            config = fixed_baseline_config("facies_f3")
            config["batch_size"] = 2
            _, state, best_path = train_development_model(
                archive=archive,
                train_records=records[:2],
                validation_records=records[2:],
                preprocessor=preprocessing,
                run_config=config,
                split_hash="synthetic-split",
                output_dir=root / "run",
                epochs=1,
                fold_id=0,
                device=torch.device("cpu"),
            )
            payload = load_checkpoint(best_path)
            self.assertEqual(state.next_epoch, 1)
            self.assertIn("model_state", payload)
            self.assertIn("optimizer_state", payload)
            self.assertIn("rng_state", payload)
            self.assertEqual(payload["split_hash"], "synthetic-split")
            self.assertEqual(payload["extra"]["label_version"], "f3-zenodo-1471548-ids-0-9-v1")


if __name__ == "__main__":
    unittest.main()
