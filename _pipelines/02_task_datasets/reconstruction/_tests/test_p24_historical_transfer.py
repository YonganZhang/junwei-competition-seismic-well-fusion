from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
MODULE_PATH = HERE / "p24_historical_transfer.py"
SPEC = importlib.util.spec_from_file_location("reconstruction_p24", MODULE_PATH)
assert SPEC and SPEC.loader
p24 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p24
SPEC.loader.exec_module(p24)


class P24HistoricalTransferTests(unittest.TestCase):
    def test_preregistration_preserves_claim_boundary_and_p21_lock(self) -> None:
        payload = json.loads(p24.PREREGISTRATION.read_text(encoding="utf-8"))
        p24._validate_preregistration(payload)
        self.assertFalse(payload["evidence_class"]["fresh_blind"])
        self.assertFalse(payload["evidence_class"]["cross_field"])
        self.assertFalse(
            payload["evaluation_lock"]["hyperparameter_search_after_target_open"]
        )

    def test_rms_mapping_uses_frozen_ijk_positive_order(self) -> None:
        original_shape = p24.GRID_SHAPE_KJI
        try:
            p24.GRID_SHAPE_KJI = (2, 2, 3)
            current_kji = np.asarray(
                [
                    [[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]],
                    [[5.0, 6.0, 0.0], [7.0, 8.0, 9.0]],
                ],
                dtype=np.float32,
            )
            current_ijk = np.transpose(current_kji, (2, 1, 0)).ravel()
            current_rms = current_ijk[current_ijk > 0.0]
            historical_rms = current_rms + 10.0
            historical, audit = p24._historical_kji_volume(
                current_kji, current_rms, historical_rms
            )
            expected = np.where(current_kji > 0.0, current_kji + 10.0, 0.0)
            np.testing.assert_array_equal(historical, expected)
            self.assertTrue(audit["current_reference_exact_elementwise_match"])
        finally:
            p24.GRID_SHAPE_KJI = original_shape

    def test_success_gate_is_frozen_to_effect_size_and_fold_robustness(self) -> None:
        target = np.zeros(10, dtype=np.float64)
        baseline = np.ones(10, dtype=np.float64)
        candidate = np.full(10, 0.98, dtype=np.float64)
        fold_ids = np.repeat(np.arange(5), 2)
        comparison = p24._comparison(
            target,
            baseline,
            candidate,
            fold_ids,
            {
                "minimum_relative_rmse_improvement_vs_pykrige": 0.01,
                "maximum_fold_losses_vs_pykrige": 1,
            },
        )
        self.assertTrue(comparison["gate_passed"])
        self.assertEqual(comparison["outcomes_vs_pykrige"]["win"], 5)

    def test_missing_historical_cell_fails_closed(self) -> None:
        volume = np.ones((2, 2, 2), dtype=np.float32)
        volume[1, 1, 1] = 0.0
        with self.assertRaises(RuntimeError):
            p24._sample_kji(volume, np.asarray([[1, 1, 1]], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
