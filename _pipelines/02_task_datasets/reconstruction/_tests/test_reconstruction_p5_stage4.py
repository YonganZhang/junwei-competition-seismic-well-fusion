from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRACK_DIR = Path(__file__).resolve().parents[1]
SOURCE = TRACK_DIR / "reconstruction_p5_stage4.py"
MODULE_NAME = "reconstruction_p5_stage4_contract"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SOURCE)
stage4 = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = stage4
assert SPEC.loader is not None
SPEC.loader.exec_module(stage4)


class ReconstructionStage4FrozenContractTest(unittest.TestCase):
    def test_both_frozen_winners_are_rankable_pykrige_with_exact_budget(self) -> None:
        for mode in stage4.MODES:
            frozen = stage4.frozen_stage3_winner(mode)
            self.assertEqual(frozen["winner_entry"]["model_id"], "pykrige_ok3d")
            self.assertEqual(
                frozen["budget"],
                {"model_class": "traditional_cpu", "max_updates": 1, "max_wall_seconds": 300},
            )
            self.assertEqual(frozen["config_without_seed"]["n_training_samples"], 512)
            self.assertEqual(frozen["config_without_seed"]["device"], "cpu")
            self.assertEqual(frozen["config_without_seed"]["variogram_model"], "linear")
            self.assertEqual(frozen["config_without_seed"]["nlags"], 4)

    def test_mode_splits_match_historical_exposure_and_stage3(self) -> None:
        expected = {"strict": [0, 1, 2], "conditional": [4, 5]}
        summary = json.loads((TRACK_DIR / "p5_stage3_summary.json").read_text())
        for mode, blocks in expected.items():
            result = json.loads((TRACK_DIR / f"results_{mode}.json").read_text())
            frozen = stage4.frozen_stage3_winner(mode)
            self.assertEqual(result["test"]["patch_i_blocks"], blocks)
            self.assertEqual(list(stage4.p4.protocol(mode).test_i_blocks), blocks)
            self.assertEqual(frozen["split_hash"], summary["split_hashes"][mode])
            self.assertTrue(math.isfinite(result["models"][result["primary_baseline"]]["rmse"]))

    def test_cli_has_only_explicit_known_holdout_run_and_collation(self) -> None:
        parser = stage4.build_parser()
        commands = next(
            action.choices
            for action in parser._actions
            if hasattr(action, "choices") and isinstance(action.choices, dict)
        )
        self.assertEqual(set(commands), {"run-mode", "collate"})
        self.assertNotIn("test", commands)
        self.assertNotIn("refit", commands)

    def test_source_fail_closed_labels_and_timeout_path_are_explicit(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('EVIDENCE_CLASS = "previously_seen_reusable_holdout"', source)
        self.assertIn('"prior_test_consumed": True', source)
        self.assertIn('"fresh_blind": False', source)
        self.assertIn("except stage2.PilotTimeout", source)
        self.assertIn('"code": "budget_timeout"', source)
        self.assertNotIn("run_frozen_test_once(", source)


class ReconstructionStage4PortableEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output = TRACK_DIR / "p5_stage4_confirmation"
        cls.summary_path = cls.output / "summary.json"
        if not cls.summary_path.is_file():
            raise unittest.SkipTest("canonical Stage-4 confirmation has not been executed")

    def test_canonical_modes_validate_and_never_claim_fresh_blind(self) -> None:
        summary = json.loads(self.summary_path.read_text())
        self.assertTrue(summary["prior_test_consumed"])
        self.assertFalse(summary["fresh_blind"])
        self.assertEqual(summary["evidence_class"], stage4.EVIDENCE_CLASS)
        self.assertEqual(set(summary["results"]), set(stage4.MODES))
        for mode in stage4.MODES:
            status = stage4.validate_mode_output(mode, self.output)
            self.assertEqual(status["status"], "passed")
            self.assertTrue(status["prior_test_consumed"])
            self.assertFalse(status["fresh_blind"])

    def test_predictions_reproduce_primary_and_spectral_metrics(self) -> None:
        for mode in stage4.MODES:
            status = json.loads((self.output / mode / "status.json").read_text())
            with np.load(self.output / mode / "predictions.npz", allow_pickle=False) as archive:
                truth = np.asarray(archive["truth"], dtype=np.float64)
                prediction = np.asarray(archive["prediction"], dtype=np.float64)
                indices = np.asarray(archive["indices_kji"], dtype=np.int64)
                shape = tuple(int(value) for value in archive["volume_shape_kji"])
            names = stage4.p4.metric_names(mode)
            rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
            mae = float(np.mean(np.abs(prediction - truth)))
            spectral = stage4.p4.spectral_log_rmse(truth, prediction, indices, shape)
            self.assertAlmostEqual(rmse, status["metrics"][names["rmse"]], places=12)
            self.assertAlmostEqual(mae, status["metrics"][names["mae"]], places=12)
            self.assertAlmostEqual(
                spectral, status["metrics"][names["spectral_log_rmse"]], places=12
            )

    def test_cdf_is_diagnostic_only_and_conditional_caveat_is_preserved(self) -> None:
        strict = json.loads((self.output / "strict" / "metrics.json").read_text())
        conditional = json.loads((self.output / "conditional" / "metrics.json").read_text())
        for payload in (strict, conditional):
            self.assertEqual(payload["cdf"]["status"], "diagnostic_only")
            self.assertFalse(payload["cdf"]["used_for_selection"])
            self.assertFalse(payload["cdf"]["numeric_score_reported"])
        self.assertEqual(strict["constraint_audit"]["test_constraints_used"], 0)
        self.assertGreater(conditional["constraint_audit"]["test_constraints_used"], 0)
        self.assertTrue(
            conditional["constraint_audit"]["conditional_reconstruction_not_strict_holdout"]
        )

    def test_validation_rejects_a_fresh_blind_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(self.output / "strict", root / "strict")
            status_path = root / "strict" / "status.json"
            status = json.loads(status_path.read_text())
            status["fresh_blind"] = True
            status["result_hash"] = stage4.hash_payload(
                {key: value for key, value in status.items() if key != "result_hash"}
            )
            status_path.write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fresh blind"):
                stage4.validate_mode_output("strict", root)

    def test_all_canonical_paths_are_project_relative(self) -> None:
        summary = json.loads(self.summary_path.read_text())
        for mode, result in summary["results"].items():
            self.assertIn(mode, stage4.MODES)
            for key in ("manifest", "predictions", "figure"):
                self.assertFalse(Path(result[key]).is_absolute())
        serialized = self.summary_path.read_text()
        self.assertNotIn("/mnt/", serialized)
        self.assertNotIn(".claude/worktrees", serialized)


if __name__ == "__main__":
    unittest.main()
