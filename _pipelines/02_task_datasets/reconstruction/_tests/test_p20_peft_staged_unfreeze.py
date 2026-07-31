from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np


TRACK = Path(__file__).resolve().parents[1]
OUTPUT = TRACK / "_outputs" / "p20_peft_staged_unfreeze"
PROJECT_ROOT = TRACK.parents[2]
sys.path[:0] = [str(TRACK), str(PROJECT_ROOT)]

import p20_peft_staged_unfreeze as p20  # noqa: E402


class P20PEFTContractTest(unittest.TestCase):
    def test_coordinate_only_calibration_is_deterministic(self) -> None:
        raw = np.zeros((80, 6), dtype=np.float64)
        raw[:, 3] = np.linspace(0.0, 1.0, 80)
        raw[:, 4] = np.sin(np.linspace(0.0, 4.0, 80))
        raw[:, 5] = np.cos(np.linspace(0.0, 4.0, 80))
        first = p20._farthest_calibration(raw, 16)  # noqa: SLF001
        second = p20._farthest_calibration(raw, 16)  # noqa: SLF001
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(len(np.unique(first)), 16)

    def test_lora_rank_and_staged_trainability_are_exact(self) -> None:
        import torch

        tail = p20.gfm_ft.build_tail_module(trainable_block_count=1)
        state = {name: tensor.detach().clone() for name, tensor in tail.state_dict().items()}
        model = p20._make_model(  # noqa: SLF001
            tail_state=state,
            query_width=7,
            route="staged_lora_r4",
            seed=2693,
        )
        groups = p20._parameter_groups(model)  # noqa: SLF001
        self.assertEqual(sum(parameter.numel() for parameter in groups["peft"]), 76_800)
        self.assertEqual(
            sum(parameter.numel() for parameter in groups["base_tail"]),
            17_295_600,
        )
        self.assertEqual(p20._set_phase(groups, "staged_lora_r4", 1), "head_warmup")  # noqa: SLF001
        self.assertEqual(p20._set_phase(groups, "staged_lora_r4", 9), "head+peft")  # noqa: SLF001
        self.assertEqual(
            p20._set_phase(groups, "staged_lora_r4", 17),  # noqa: SLF001
            "head+peft+terminal_norm",
        )
        self.assertEqual(
            p20._set_phase(groups, "staged_lora_r4", 25),  # noqa: SLF001
            "head+lora+terminal_norm+full_final_block",
        )
        self.assertTrue(any(parameter.requires_grad for parameter in groups["base_tail"]))
        self.assertGreater(
            float(torch.linalg.vector_norm(model.head_out.weight).detach()), 0.0
        )

    def test_source_exposes_no_holdout_cli_argument(self) -> None:
        source = (TRACK / "p20_peft_staged_unfreeze.py").read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--test', source)
        self.assertNotIn('add_argument("--holdout', source)


class P20PEFTArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        cls.verification = json.loads(
            (OUTPUT / "verification.json").read_text(encoding="utf-8")
        )

    def test_four_routes_and_strict_label_budget_are_recorded(self) -> None:
        self.assertEqual(set(self.summary["routes"]), set(p20.ROUTES))
        protocol = self.summary["protocol"]
        self.assertEqual(protocol["train_labels_per_fold"], 512)
        self.assertEqual(protocol["validation_rows_per_fold"], 2048)
        self.assertEqual(protocol["outer_spatial_folds"], list(range(5)))
        self.assertFalse(protocol["holdout_opened"])
        self.assertFalse(protocol["test_h5_opened"])

    def test_artifact_manifest_hashes_every_portable_output(self) -> None:
        manifest = json.loads(
            (OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertIn("summary.json", manifest)
        self.assertIn("verification.json", manifest)
        self.assertIn("predictions.npz", manifest)
        for name, expected in manifest.items():
            digest = hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
            self.assertEqual(digest, expected)

    def test_peft_and_all_staged_gradients_are_real(self) -> None:
        checks = self.verification["optimization_checks"]
        self.assertTrue(checks["lora_gradient_nonzero"])
        self.assertTrue(checks["adapter_gradient_nonzero"])
        self.assertTrue(checks["terminal_norm_gradient_nonzero"])
        self.assertTrue(checks["full_tail_gradient_nonzero"])
        staged = self.verification["routes_recomputed"]["staged_lora_r4"]
        self.assertEqual(staged["tensor_shapes"]["prefix_batch"], [8, 3, 161, 1200])
        self.assertEqual(staged["tensor_shapes"]["query_batch"][1], 7)
        self.assertTrue(
            all(row["peft"] > 0.0 for row in staged["parameter_update_l2_by_fold"])
        )
        self.assertTrue(
            all(
                row["base_tail"] > 0.0
                for row in staged["parameter_update_l2_by_fold"]
            )
        )

    def test_p20_beats_pykrige_but_not_p19_and_stays_disabled(self) -> None:
        verification = self.verification
        p19_rmse = verification["p19_reference"]["rmse"]
        staged_rmse = verification["routes_recomputed"]["staged_lora_r4"][
            "metrics"
        ]["rmse"]
        baseline_rmse = verification["baseline"]["rmse"]
        self.assertLess(staged_rmse, baseline_rmse)
        self.assertGreater(staged_rmse, p19_rmse)
        self.assertEqual(
            verification["routes_recomputed"]["staged_lora_r4"][
                "fold_wins_vs_pykrige"
            ],
            5,
        )
        self.assertFalse(verification["decision"]["default_enabled"])
        self.assertEqual(
            verification["decision"]["state"], "VERIFIED_NO_PROMOTION"
        )

    def test_longer_budget_and_fixed_blends_do_not_repair_generalization(self) -> None:
        extended = self.verification["extended_80_update_check"]
        self.assertEqual(extended["maximum_updates"], 80)
        self.assertEqual(extended["selected_updates"], [80, 80, 80, 72, 80])
        self.assertGreater(
            extended["metrics"]["rmse"],
            self.verification["routes_recomputed"]["staged_lora_r4"][
                "metrics"
            ]["rmse"],
        )
        complementarity = self.verification["p19_complementarity"]
        self.assertGreater(complementarity["error_correlation"], 0.999)
        self.assertEqual(
            complementarity["best_fixed_blend"]["staged_lora_weight"], 0.0
        )
        self.assertFalse(complementarity["fixed_blend_improves_p19"])


if __name__ == "__main__":
    unittest.main()
