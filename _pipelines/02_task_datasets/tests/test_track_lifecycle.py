from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "_pipelines/02_task_datasets/track_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("track_lifecycle", MODULE_PATH)
assert SPEC and SPEC.loader
LIFECYCLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIFECYCLE
SPEC.loader.exec_module(LIFECYCLE)


class TrackLifecycleTests(unittest.TestCase):
    def test_all_tracks_pass_all_stages(self) -> None:
        for track in LIFECYCLE.TRACKS:
            with self.subTest(track=track):
                result = LIFECYCLE.evaluate(track, "verify")
                self.assertEqual(result["status"], "PASS")
                self.assertEqual({row["stage"] for row in result["checks"]}, set(LIFECYCLE.STAGES[:-1]))
                self.assertTrue(all(row["passed"] for row in result["checks"]))

    def test_cli_writes_hashed_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fault.json"
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--track", "fault", "--stage", "verify", "--output", str(output)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "PASS")
            self.assertRegex(payload["evidence"]["summary"]["sha256"], r"^[0-9a-f]{64}$")

    def test_missing_field_is_fail_loud(self) -> None:
        with self.assertRaisesRegex(AssertionError, "missing evidence field"):
            LIFECYCLE._resolve({}, "not.present")


if __name__ == "__main__":
    unittest.main()
