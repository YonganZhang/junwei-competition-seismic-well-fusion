from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path


BASE = "_pipelines.02_task_datasets.sweetspot.targets"


def _temporary_directory():
    root = Path.cwd() / "_tmp"
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


class SweetspotBaselineLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.baseline = importlib.import_module(f"{BASE}.baseline")

    def test_frozen_test_gate_is_single_use(self):
        gate = self.baseline.FrozenTestGate(["dev", "test"], ["test"])
        self.assertEqual(gate.consume().tolist(), [1])
        self.assertTrue(gate.consumed)
        with self.assertRaisesRegex(RuntimeError, "already been consumed"):
            gate.consume()

    def test_eight_sanity_and_twenty_pilot_configs_are_preregistered(self):
        for runtime in self.baseline.RUNTIMES.values():
            self.assertEqual(len(self.baseline._candidate_configs(runtime)), 8)
            self.assertEqual(len(self.baseline._pilot_configs(runtime)), 20)

    def test_real_water_case_runs_without_pilot_or_test_tuning(self):
        with _temporary_directory() as directory:
            root = Path(directory) / "water"
            status = self.baseline.run_target(
                "water_breakthrough", root, sanity_train_limit=128, run_pilot=False,
            )
            self.assertTrue(status["test_consumed_once"])
            self.assertEqual(status["hpo_sanity_trials"], 8)
            self.assertEqual(status["hpo_pilot_trials"], 0)
            self.assertEqual(status["hpo_pilot_status"], "skipped_by_explicit_runner_option")
            lifecycle = json.loads((root / "lifecycle.json").read_text(encoding="utf-8"))
            self.assertEqual(lifecycle["state"], "VERIFIED")
            plan = json.loads((root / "hpo" / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["test_access"], "forbidden")
            visualization = json.loads(
                (root / "visualizations" / "visualization_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(visualization["prediction_path"], "predictions.csv")
            self.assertNotIn(str(root.resolve()), json.dumps(visualization))
            self.assertTrue((root / "manifest.json").is_file())


class SevenTargetRegistryTests(unittest.TestCase):
    def test_registry_declares_seven_independent_cases(self):
        registry = importlib.import_module(f"{BASE}.registry")
        with _temporary_directory() as directory:
            payload = registry.build_registry(Path(directory) / "p4-seven-target-registry-test.json")
        self.assertEqual(payload["target_count"], 7)
        self.assertTrue(payload["all_targets_independent"])
        self.assertEqual([item["target_number"] for item in payload["targets"]], list(range(1, 8)))
        fifth = payload["targets"][4]
        self.assertEqual(fifth["status"], "not_feasible")
        self.assertIn("not_feasible", fifth["artifacts"])
        self.assertEqual(payload["targets"][5]["target_id"], "porosity")
        self.assertEqual(payload["targets"][6]["target_id"], "permeability")


if __name__ == "__main__":
    unittest.main()
