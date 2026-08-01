from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import importlib

p28 = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p28.sweetspot_p28_agentic_optimization"
)


class SweetspotP28AgenticOptimizationTests(unittest.TestCase):
    def test_observation_redacts_raw_metrics_labels_and_paths(self) -> None:
        observation = p28._observation()
        prompt = p28._build_prompt(observation)
        prompt_text = json.dumps(prompt, ensure_ascii=False)

        self.assertIn("improved", prompt_text)
        self.assertIn("flat", prompt_text)
        self.assertIn("worse", prompt_text)  # schema contract mentions it
        self.assertNotIn("0.2161617856064557", prompt_text)
        self.assertNotIn("186.57151779454128", prompt_text)
        self.assertNotIn("/mnt/data", prompt_text)
        self.assertNotIn("predictions.csv", prompt_text)
        self.assertNotIn("test.h5", prompt_text)

    def test_generate_report_blocks_when_no_ported_executor_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = p28.generate_report(Path(tmp), call_deepseek=False)

        self.assertEqual(summary["verdict"], "BLOCKED_EXECUTOR")
        self.assertFalse(summary["executor_available"])
        self.assertEqual(summary["gates"]["T5"], "not_feasible")
        self.assertEqual(summary["gates"]["T6"], "blocked")
        self.assertEqual(summary["gates"]["T7"], "blocked")
        self.assertEqual(summary["selection_fold_ids"], [0, 1, 2])
        self.assertEqual(summary["promotion_fold_ids"], [3])
        self.assertEqual(summary["a0_reference"]["split_hash"], "c44277ffc1f6fb6b5dd740952e921c732d45193c7cb3b5c3dfc061e79025c62a")

    def test_protocol_and_summary_files_are_written_portably(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summary = p28.generate_report(output_dir, call_deepseek=False)
            protocol = output_dir / "protocol.json"
            protocol_jsonl = output_dir / "protocol.jsonl"
            summary_path = output_dir / "summary.json"
            evidence = output_dir / "evidence.md"
            manifest = output_dir / "manifest.json"

            for path in (protocol, protocol_jsonl, summary_path, evidence, manifest):
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0)

            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest_data["schema_version"], p28.SCHEMA_VERSION)
            self.assertEqual(manifest_data["track_id"], "sweetspot")
            self.assertFalse(manifest_data["manual_review"]["reviewed"])
            self.assertIsNone(manifest_data["manual_review"]["reviewer"])
            self.assertEqual(manifest_data["artifact_count"], 4)
            self.assertEqual(len(manifest_data["inputs"]), 6)
            protocol_row = next(row for row in manifest_data["outputs"] if row["role"] == "protocol")
            self.assertEqual(summary["protocol_sha256"], protocol_row["sha256"])

    def test_source_does_not_reference_frozen_holdout_or_test_h5(self) -> None:
        source = Path(p28.__file__).read_text(encoding="utf-8")
        self.assertNotIn("test.h5", source)
        self.assertNotIn("frozen holdout", source)
        self.assertIn("BLOCKED_EXECUTOR", source)


if __name__ == "__main__":
    unittest.main()
