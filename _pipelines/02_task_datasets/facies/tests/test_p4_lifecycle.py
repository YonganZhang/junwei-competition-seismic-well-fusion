from __future__ import annotations

import argparse
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

from p4_experiment import (
    cv_run,
    freeze_run,
    frozen_test_run,
    prepare_run,
    refit_run,
    smoke_run,
    visualize_run,
)


def _write_split(path: Path, split: str, lines: range) -> None:
    with h5py.File(path, "w") as archive:
        archive.attrs["task"] = "facies_f3"
        archive.attrs["split"] = split
        for index, line in enumerate(lines):
            group = archive.create_group(f"sample_{index:07d}")
            label = np.tile(np.arange(10, dtype=np.uint8), (4, 1))
            seismic = label.astype(np.float32) + (line % 7) * 0.01
            group.create_dataset("seismic_patch", data=seismic)
            group.create_dataset("label", data=label)
            group.attrs["position"] = json.dumps(
                {"inline": line, "crossline": 300, "time_ms": 0.0, "well_name": None}
            )
            group.attrs["meta"] = json.dumps(
                {"task": "facies_f3", "split": split, "source": "synthetic-lifecycle"}
            )


class LifecycleEndToEndTests(unittest.TestCase):
    def test_full_synthetic_lifecycle_and_single_test_consumption(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed" / "facies_f3"
            processed.mkdir(parents=True)
            _write_split(processed / "train.h5", "train", range(100, 587))
            _write_split(processed / "test.h5", "test", range(620, 751))
            run_root = root / "run"
            prepared = prepare_run(
                argparse.Namespace(
                    task="facies_f3",
                    run_id="synthetic-f3",
                    run_root=run_root,
                    processed_root=root / "processed",
                    requested_n_splits=5,
                    buffer_groups=1,
                )
            )
            self.assertEqual(prepared["state"], "SPLIT_LOCKED")
            smoked = smoke_run(
                argparse.Namespace(
                    run_root=run_root,
                    processed_root=root / "processed",
                    device="cpu",
                    epochs=1,
                    max_train_records=4,
                    max_validation_records=2,
                )
            )
            self.assertEqual(smoked["state"], "SMOKE_PASSED")
            cv = cv_run(
                argparse.Namespace(
                    run_root=run_root,
                    processed_root=root / "processed",
                    device="cpu",
                    epochs=1,
                )
            )
            self.assertEqual(cv["state"], "CV_COMPLETE")
            frozen = freeze_run(argparse.Namespace(run_root=run_root))
            self.assertEqual(frozen["state"], "CONFIG_FROZEN")
            refit = refit_run(
                argparse.Namespace(
                    run_root=run_root,
                    processed_root=root / "processed",
                    device="cpu",
                )
            )
            self.assertEqual(refit["state"], "REFIT_COMPLETE")
            tested = frozen_test_run(
                argparse.Namespace(
                    run_root=run_root,
                    processed_root=root / "processed",
                    device="cpu",
                )
            )
            self.assertEqual(tested["state"], "TEST_CONSUMED")
            self.assertTrue(np.isfinite(tested["miou"]))
            with self.assertRaisesRegex(RuntimeError, "REFIT_COMPLETE"):
                frozen_test_run(
                    argparse.Namespace(
                        run_root=run_root,
                        processed_root=root / "processed",
                        device="cpu",
                    )
                )
            visualized = visualize_run(
                argparse.Namespace(run_root=run_root, diagnostic_seed=2693)
            )
            self.assertTrue(visualized["read_archived_predictions_only"])
            self.assertTrue((run_root / "visualizations" / "facies_diagnostics.png").is_file())


if __name__ == "__main__":
    unittest.main()
