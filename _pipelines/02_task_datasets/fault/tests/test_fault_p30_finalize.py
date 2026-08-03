from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK))

import fault_p30_finalize as finalize  # noqa: E402


class FaultP30FinalizeTest(unittest.TestCase):
    def test_asset_and_comparison_are_one_fail_closed_command(self) -> None:
        built = {
            "gate_result": {"status": "READY"},
            "manifest_path": "asset/manifest.json",
        }
        evaluated = {
            "outputs": {"comparison_path": "comparison.json"},
            "report": {"decision": {"default_recommendation": "do_not_advance"}},
        }
        with patch.object(finalize.asset, "build_dev_asset", return_value=built), patch.object(
            finalize.comparison, "run", return_value=evaluated
        ):
            result = finalize.run(device="cpu")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["decision"], "do_not_advance")

    def test_failed_asset_gate_stops_before_model_comparison(self) -> None:
        built = {
            "gate_result": {"status": "BLOCKED"},
            "manifest_path": "asset/manifest.json",
        }
        with patch.object(finalize.asset, "build_dev_asset", return_value=built), patch.object(
            finalize.comparison, "run"
        ) as compare:
            with self.assertRaisesRegex(RuntimeError, "data gate"):
                finalize.run()
        compare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
