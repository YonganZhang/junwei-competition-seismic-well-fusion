from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import lithofacies_agent_chapter as agent  # noqa: E402


class LithofaciesAgentChapterTests(unittest.TestCase):
    def test_holdout_like_paths_are_rejected_before_open(self) -> None:
        for path in (
            Path("dataset/test.h5"),
            Path("runtime/frozen-family/data.npz"),
            Path("known_holdout/results.json"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "forbidden holdout path"):
                    agent.ensure_development_only_paths((path,))
        agent.ensure_development_only_paths((Path("runtime/development_logo4.npz"),))

    def test_well_only_ablation_keeps_logs_and_masks_but_drops_seismic(self) -> None:
        well = np.arange(2 * 26 * 33, dtype=np.float32).reshape(2, 26, 33)
        seismic_a = np.zeros((2, 3, 3, 33), dtype=np.float32)
        seismic_b = np.ones((2, 3, 3, 33), dtype=np.float32)
        features_a = agent.well_and_mask_only_features(well, seismic_a)
        features_b = agent.well_and_mask_only_features(well, seismic_b)
        self.assertEqual(features_a.shape, (2, 858))
        np.testing.assert_array_equal(features_a, well.reshape(2, -1))
        np.testing.assert_array_equal(features_a, features_b)

    def test_structured_prompt_contains_measured_context(self) -> None:
        summary = {
            "variants": {
                name: {
                    "metrics": {
                        "fixed_schema_macro_f1": {"mean": mean, "std": 0.01}
                    }
                }
                for name, mean in (
                    ("baseline", agent.BASELINE_MEAN),
                    ("prior_calibrated", agent.CURRENT_BEST_MEAN),
                    ("cross_attention", 0.20159038882981442),
                )
            },
            "comparison": {
                "prior_calibrated_minus_baseline": 0.007249273594053696,
                "cross_attention_minus_baseline": 0.006652686754176795,
                "cross_attention_minus_prior_calibrated": -0.0005965868398769003,
            },
        }
        system, user = agent.build_structured_prompt(
            p11_summary=summary,
            class_counts=[11, 104, 7, 40, 27, 124, 127, 6, 1],
            sample_count=447,
        )
        self.assertIn("不得虚构提升数字", system)
        self.assertIn("447", user)
        self.assertIn("127:1", user)
        self.assertIn("fixed_schema_macro_f1", user)

    def test_deepseek_client_does_not_return_the_api_key(self) -> None:
        response = {
            "id": "response-id",
            "model": "provider-model",
            "usage": {"total_tokens": 9},
            "choices": [
                {
                    "message": {
                        "content": "A.诊断 B.建议 C.实验 D.暂缓 E.归因"
                    }
                }
            ],
        }
        stream = io.BytesIO(json.dumps(response).encode("utf-8"))
        with mock.patch.object(agent.urllib.request, "urlopen", return_value=stream):
            payload = agent.call_deepseek(
                system_prompt="system",
                user_prompt="user",
                api_key="secret-value-that-must-not-return",
            )
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("secret-value-that-must-not-return", serialized)
        self.assertFalse(payload["credential_persisted"])
        self.assertEqual(payload["request_model"], "deepseek-chat")

    def test_summary_keeps_all_six_variants_and_twelve_cells(self) -> None:
        rows = []
        values = {
            "baseline_archived": agent.BASELINE_MEAN,
            "baseline_reproduced": agent.BASELINE_MEAN,
            "weight_alpha_075": agent.BASELINE_MEAN - 0.001,
            "weight_alpha_100": agent.BASELINE_MEAN - 0.1,
            "well_and_mask_only_858": agent.BASELINE_MEAN - 0.01,
            "depth3_eta01_rounds60": agent.BASELINE_MEAN + 0.001,
            "depth3_eta01_rounds60_prior025": agent.BASELINE_MEAN + 0.006,
        }
        for variant, value in values.items():
            for fold_id in agent.FOLD_IDS:
                for repeat_id in range(len(agent.REPEAT_SEEDS)):
                    rows.append(
                        {
                            "variant": variant,
                            "fold_id": fold_id,
                            "repeat_id": repeat_id,
                            "metrics": {
                                "fixed_schema_macro_f1": value,
                                "per_class_f1": [0.0] * 9,
                            },
                        }
                    )
        summary = agent.summarize_rows(rows)
        self.assertEqual(set(summary["variants"]), set(agent.VARIANTS))
        self.assertEqual(
            summary["decision"]["best_low_cost_candidate"],
            "depth3_eta01_rounds60_prior025",
        )
        self.assertTrue(summary["decision"]["promotion_passed"])
        self.assertFalse(summary["decision"]["default_enabled"])
        self.assertEqual(
            summary["decision"]["state"],
            "DEVELOPMENT_CANDIDATE_KEEP_FOR_CONFIRMATION",
        )


if __name__ == "__main__":
    unittest.main()
