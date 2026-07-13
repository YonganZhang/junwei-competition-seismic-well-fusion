from __future__ import annotations

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from p4_data import FaciesArchive
from p4_tasks import TASK_IDS
from p4_training import run_fast_real_data_smoke


PROCESSED_ROOT = os.environ.get("FACIES_P4_PROCESSED_ROOT")


@unittest.skipUnless(
    PROCESSED_ROOT,
    "set FACIES_P4_PROCESSED_ROOT for the explicit real-data integration smoke",
)
class RealDataSmokeTests(unittest.TestCase):
    def test_both_tasks_run_development_only_smoke(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            for task_id in TASK_IDS:
                with self.subTest(task_id=task_id):
                    archive = FaciesArchive(task_id, Path(PROCESSED_ROOT))
                    evidence = run_fast_real_data_smoke(
                        task_id=task_id,
                        archive=archive,
                        output_dir=Path(directory) / task_id,
                        device=torch.device("cpu"),
                        epochs=1,
                        max_candidates=64,
                        max_train_records=32,
                        max_validation_records=16,
                    )
                    self.assertFalse(evidence["test_archive_opened"])
                    self.assertFalse(evidence["test_labels_read"])
                    self.assertFalse(evidence["test_inference_run"])
                    self.assertTrue(np.isfinite(evidence["metrics"]["miou"]))
                    self.assertTrue((Path(directory) / task_id / "checkpoint_best.pkl").is_file())
                    print(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "train_records": evidence["train_records"],
                                "validation_records": evidence["validation_records"],
                                "test_archive_opened": evidence["test_archive_opened"],
                                "accuracy": evidence["metrics"]["accuracy"],
                                "miou_observed_support_smoke": evidence["metrics"]["miou"],
                                "macro_f1_observed_support_smoke": evidence["metrics"]["macro_f1"],
                                "observed_class_ids": evidence["metrics"]["observed_class_ids"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main()
