from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
for root in (str(PROJECT_ROOT), str(HERE)):
    if root not in sys.path:
        sys.path.insert(0, root)

p40 = importlib.import_module("lithofacies_p40_crossmodal_foundation_pilot")
features = importlib.import_module("lithofacies_p40_foundation_features")


class P40ContractTests(unittest.TestCase):
    def test_native_widths_and_no_p38_random_projection(self) -> None:
        self.assertEqual(features.MOMENT_WEIGHTS_SHA256, "1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825")
        self.assertEqual(features.GFM_WEIGHTS_SHA256, "c905945267bbbc58f0e1848106d182f40b5dc61273959b666a49b384cfcb7446")
        source = Path(features.__file__).read_text(encoding="utf-8")
        self.assertNotIn("PROJECTION_DIM = 16", source)
        self.assertIn("[N,2,1200]", source)
        self.assertIn("[13,4,768]", source)

    def test_firewall_rejects_test_and_holdout_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbids"):
            p40._safe_paths((Path("known_holdout/runtime.npz"),))
        with self.assertRaisesRegex(ValueError, "forbids"):
            p40._safe_paths((Path("data/test.h5"),))

    def test_native_window_contains_requested_support(self) -> None:
        start = p40._window_start(2200, 2210, 160, 1900, 2500)
        self.assertLessEqual(start, 2200)
        self.assertGreaterEqual(start + 159, 2210)
        with self.assertRaises(RuntimeError):
            p40._window_start(2000, 2200, 160, 1900, 2500)

    def test_instance_normalization_is_sample_local_and_masked(self) -> None:
        logs = np.zeros((2, 13, 33), dtype=np.float32)
        masks = np.zeros_like(logs, dtype=np.uint8)
        logs[0, 0, :3] = [1, 2, 3]
        logs[1, 0, :3] = [100, 200, 300]
        masks[:, 0, :3] = 1
        result = p40._instance_normalize_logs(logs, masks)
        np.testing.assert_allclose(result[0, 0, :3], result[1, 0, :3], atol=1e-6)
        self.assertEqual(float(result[:, 1:].max()), 0.0)

    def test_pca_fits_only_explicit_train_rows(self) -> None:
        rng = np.random.default_rng(2693)
        tokens = rng.normal(size=(12, 2, 20)).astype(np.float32)
        train = np.arange(8, dtype=np.int64)
        transformed, audit = p40._pca_features(tokens, train, stream=1)
        self.assertEqual(transformed.shape, (12, 7))
        self.assertEqual(audit["fit_rows"], 8)
        self.assertFalse(audit["held_rows_used"])
        self.assertIn("not random projection", audit["method"])

    def test_shuffle_is_within_family_and_non_identity(self) -> None:
        indices = np.arange(12, dtype=np.int64)
        families = np.asarray(["a"] * 6 + ["b"] * 6)
        shuffled = p40._shuffle_within_family(indices, families, stream=3)
        self.assertFalse(np.array_equal(shuffled, indices))
        np.testing.assert_array_equal(np.sort(shuffled[:6]), indices[:6])
        np.testing.assert_array_equal(np.sort(shuffled[6:]), indices[6:])


if __name__ == "__main__":
    unittest.main()
