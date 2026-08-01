from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any


p29 = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p29_agent_action_effect")


class SweetspotP29AgentActionEffectTests(unittest.TestCase):
    def _path_like_strings(self, payload: Any) -> list[str]:
        values: list[str] = []
        if isinstance(payload, dict):
            for value in payload.values():
                values.extend(self._path_like_strings(value))
        elif isinstance(payload, list):
            for value in payload:
                values.extend(self._path_like_strings(value))
        elif isinstance(payload, str):
            if payload.startswith(("_pipelines/", "_wiki-methodology/", "_sandbox/", ".claude/worktrees/", "./", "../", "/mnt/data/")):
                values.append(payload)
        return values

    def test_generate_report_writes_local_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = p29.generate_report(Path(tmp), call_deepseek=False)
            output_dir = Path(tmp)

            for filename in ["protocol.json", "protocol.jsonl", "action_effects.json", "root_cause.md", "summary.json", "evidence.md", "manifest.json"]:
                path = output_dir / filename
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["track_id"], "sweetspot")
            self.assertEqual(manifest["schema_version"], p29.SCHEMA_VERSION)
            self.assertEqual(manifest["artifact_count"], 6)
            self.assertFalse(manifest["manual_review"]["reviewed"])
            self.assertEqual(len(manifest["inputs"]), 8)
            self.assertEqual([item["role"] for item in manifest["outputs"]], ["protocol", "protocol_jsonl", "action_effects", "root_cause", "summary", "evidence"])
            self.assertTrue(all("path" in item and "sha256" in item for item in manifest["outputs"]))
            self.assertEqual(summary["baseline_kind"], "same_fold_same_executor_a0")
            self.assertIn(summary["verdict"], {"RETAIN_AGENT", "RETAIN_HYBRID", "REJECT_AGENT", "DATA_GATE_BLOCKED"})
            self.assertEqual(summary["selection_folds"], [0, 1, 2])
            self.assertEqual(summary["promotion_folds"], [3])

    def test_prompt_uses_same_fold_baseline_and_signed_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            p29.generate_report(output_dir, call_deepseek=False)
            protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
            action_effects = json.loads((output_dir / "action_effects.json").read_text(encoding="utf-8"))

            self.assertEqual(protocol["feedback_baseline"]["kind"], "same_fold_same_executor_a0")
            self.assertTrue(protocol["prompt_contract"]["same_fold_same_executor_a0"])
            self.assertIn("remaining_budget_trials", protocol["prompt_contract"])
            self.assertNotIn("promotion_mae", json.dumps(protocol["candidate_feedback"], ensure_ascii=False))
            self.assertNotIn("historical_stage3_aggregate", json.dumps(protocol["candidate_feedback"], ensure_ascii=False))

            baseline_row = action_effects["baseline"]["fold_rows"][0]
            candidate = next(item for item in protocol["candidate_feedback"] if item["action_id"] != p29.A0_ACTION_ID)
            candidate_row = next(row for row in next(item for item in action_effects["actions"] if item["action_id"] == candidate["action_id"])["fold_rows"] if row["fold_id"] == baseline_row["fold_id"])
            expected_delta = (baseline_row["selection_mae"] - candidate_row["candidate_mae"]) / abs(baseline_row["selection_mae"])
            actual_delta = next(row["signed_normalized_delta"] for row in candidate["fold_feedback"] if row["fold_id"] == baseline_row["fold_id"])
            self.assertAlmostEqual(actual_delta, expected_delta, places=12)

    def test_controls_are_independent_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summary = p29.generate_report(output_dir, call_deepseek=False)
            protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["a0_prediction_hash"], summary["a1_prediction_hash"])
            self.assertTrue(summary["a1_same_hash"])
            self.assertEqual(summary["a1_prediction_hash_contract"], "EXECUTED_IDENTITY_CHECK")
            self.assertEqual(summary["a2d"]["trial_budget"], 4)
            self.assertEqual(summary["a3"]["control_seed"], p29.ROOT_SEED + 17)
            self.assertIn(summary["a3"]["trial_budget"], {2, 3, 4})
            self.assertEqual(summary["selection_folds"], [0, 1, 2])
            self.assertEqual(summary["promotion_folds"], [3])
            self.assertEqual([row["fold_id"] for row in protocol["candidate_feedback"][0]["fold_feedback"]], [0, 1, 2])

    def test_canonical_artifacts_have_only_relative_path_like_fields(self) -> None:
        output_dir = p29.OUTPUT_DIR
        protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
        action_effects = json.loads((output_dir / "action_effects.json").read_text(encoding="utf-8"))
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        protocol_jsonl = (output_dir / "protocol.jsonl").read_text(encoding="utf-8").splitlines()

        for blob in (protocol, action_effects, summary, manifest, [json.loads(row) for row in protocol_jsonl]):
            for value in self._path_like_strings(blob):
                self.assertFalse(Path(value).is_absolute(), value)
                self.assertNotIn(".claude/worktrees", value)
                self.assertNotIn("p10-results-sweetspot", value)

        for record in manifest["outputs"]:
            path = Path(record["path"])
            self.assertFalse(path.is_absolute(), record["path"])
            self.assertTrue(str(path).startswith("_pipelines/02_task_datasets/sweetspot/"), record["path"])
            resolved = p29.WORKTREE_ROOT / path
            self.assertTrue(resolved.is_file(), record["path"])
            self.assertEqual(record["sha256"], p29._sha256_file(resolved))

        for record in manifest["inputs"]:
            self.assertTrue(record["scientific_source_id"].startswith(("_pipelines/", "_wiki-methodology/")))
            self.assertNotIn(".claude/worktrees", record["scientific_source_id"])
            self.assertNotIn("p10-results-sweetspot", record["scientific_source_id"])


if __name__ == "__main__":
    unittest.main()
