"""Explicit data-dependent P4 reconstruction smoke; no training or test inference."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

import p4_reconstruction as p4  # noqa: E402


class P4RealDataSmokeTest(unittest.TestCase):
    def test_real_unified_hdf5_split_and_first_fold_preprocessing(self):
        data_dir = p4.resolve_data_dir(
            Path(os.environ["RECONSTRUCTION_DATA_DIR"])
            if "RECONSTRUCTION_DATA_DIR" in os.environ
            else None
        )
        missing = [data_dir / "train.h5", data_dir / "test.h5"]
        missing = [path for path in missing if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "P4 real reconstruction smoke skipped; provision RECONSTRUCTION_DATA_DIR with train.h5/test.h5"
            )
        report = p4.real_data_smoke(data_dir)
        self.assertEqual(report["catalog_patches"], 210)
        conditional = report["modes"]["conditional"]
        strict = report["modes"]["strict"]
        self.assertEqual(conditional["requested_n_splits"], 5)
        self.assertEqual(conditional["development_patches"], 140)
        self.assertEqual(conditional["test_patches"], 70)
        self.assertEqual(strict["development_patches"], 70)
        self.assertEqual(strict["test_patches"], 105)
        self.assertTrue(conditional["finite_features"])
        self.assertTrue(strict["finite_features"])
        self.assertGreater(strict["first_fold_constraints"], 0)
        self.assertEqual(strict["first_fold_constraints_supplied_to_model"], 0)


if __name__ == "__main__":
    unittest.main()
