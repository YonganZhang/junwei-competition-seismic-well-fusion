from __future__ import annotations

import json
from pathlib import Path
import unittest

from _models.gaia_dagt.foundation import load_foundation_routes


ROOT = Path(__file__).resolve().parents[3]


def _read(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise unittest.SkipTest(f"missing archived P9 evidence: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


class FoundationEffectProtocolTests(unittest.TestCase):
    def test_protocol_is_fail_closed_and_covers_every_track(self) -> None:
        protocol = _read("_models/gaia_dagt/foundation_effect_protocol.v1.json")
        rules = protocol["global_rules"]
        self.assertTrue(rules["same_sample_universe_required"])
        self.assertTrue(rules["same_fold_required"])
        self.assertTrue(rules["same_metric_required"])
        self.assertTrue(rules["random_init_same_architecture_required_for_default_promotion"])
        self.assertFalse(rules["frozen_test_access"])
        self.assertFalse(rules["known_holdout_access"])
        self.assertEqual(
            set(protocol["tracks"]),
            {"fault", "facies", "property", "lithofacies", "sweetspot", "reconstruction"},
        )

    def test_route_states_match_archived_effect_decisions(self) -> None:
        routes = load_foundation_routes()
        self.assertEqual(routes["fault"]["state"], "CONNECTED_UNVERIFIED")
        self.assertEqual(routes["property"]["state"], "CONNECTED_UNVERIFIED")
        self.assertEqual(routes["sweetspot"]["state"], "CONNECTED_UNVERIFIED")
        for track in ("facies", "lithofacies", "reconstruction"):
            self.assertEqual(routes[track]["state"], "VERIFIED_NO_GAIN")
        self.assertTrue(all(not route["default_enabled"] for route in routes.values()))


class FoundationEffectEvidenceTests(unittest.TestCase):
    def test_property_has_same_split_gain_but_remains_fail_closed(self) -> None:
        summary = _read(
            "_pipelines/02_task_datasets/reservoir/_outputs/"
            "p9_tabicl_effect/summary.json"
        )
        for comparison in summary["comparisons"].values():
            self.assertTrue(comparison["pretrained_better_than_strong_baseline"])
            self.assertTrue(comparison["pretrained_better_than_target_shuffle"])
            self.assertLess(comparison["relative_rmse_change_vs_strong_baseline"], 0)
        self.assertEqual(
            summary["decision"]["state"], "EFFECT_SUPPORTED_NOT_PROMOTED"
        )
        self.assertFalse(summary["decision"]["default_enabled"])
        self.assertFalse(summary["evaluation"]["frozen_test_accessed"])

    def test_sweetspot_clears_registered_causal_controls(self) -> None:
        summary = _read(
            "_pipelines/02_task_datasets/sweetspot/p8/_outputs/"
            "t3_chronos2_calendar_cv/summary.json"
        )
        methods = summary["methods"]
        foundation = methods["F0_chronos2_calendar"]["macro_fold_mean"]["mae"]
        for control in (
            "B1_calendar_history_mean",
            "B2_calendar_extra_trees",
            "C0_calendar_extra_trees_target_shuffle",
            "C1_chronos2_history_order_shuffle",
        ):
            self.assertLess(
                foundation, methods[control]["macro_fold_mean"]["mae"]
            )
        self.assertEqual(
            summary["decision"]["state"], "EFFECT_SUPPORTED_NOT_PROMOTED"
        )
        self.assertFalse(summary["decision"]["default_enabled"])

    def test_facies_models_do_not_replace_locked_small_models(self) -> None:
        for task in ("facies_f3", "facies_penobscot"):
            summary = _read(
                "_pipelines/02_task_datasets/facies/_outputs/"
                f"p9_sam2_effect/{task}/summary.json"
            )
            comparison = summary["comparison"]
            self.assertLess(
                comparison["pretrained_macro_fold_miou"],
                comparison["strong_baseline_macro_fold_miou"],
            )
            self.assertEqual(summary["decision"]["state"], "CONNECTED_NO_PROMOTION")
            self.assertFalse(summary["evaluation"]["frozen_test_accessed"])

    def test_lithofacies_and_reconstruction_show_pretraining_effect_not_baseline_win(self) -> None:
        lithofacies = _read(
            "_pipelines/02_task_datasets/lithofacies/_outputs/"
            "p9_moment_effect/summary.json"
        )
        litho = lithofacies["comparison"]
        self.assertGreater(
            litho["pretrained_macro_fold_f1"],
            litho["same_architecture_random_init_macro_fold_f1"],
        )
        self.assertLess(
            litho["pretrained_macro_fold_f1"], litho["strong_baseline_macro_f1"]
        )

        reconstruction = _read(
            "_pipelines/02_task_datasets/reconstruction/_outputs/"
            "p9_openmind_effect/summary.json"
        )
        recon = reconstruction["comparison"]
        self.assertLess(
            recon["pretrained_macro_fold_rmse"],
            recon["same_architecture_random_init_macro_fold_rmse"],
        )
        self.assertGreater(
            recon["pretrained_macro_fold_rmse"],
            recon["strong_baseline_macro_fold_rmse"],
        )
        self.assertTrue(
            reconstruction["evaluation"][
                "same_validation_sample_universe_as_strong_baseline"
            ]
        )
        self.assertFalse(reconstruction["evaluation"]["frozen_test_accessed"])

    def test_fault_checkpoint_is_connected_but_scoring_is_data_gate_blocked(self) -> None:
        summary = _read(
            "_pipelines/02_task_datasets/fault/_outputs/"
            "p9_sammed3d_gate/summary.json"
        )
        self.assertTrue(summary["foundation_route"]["synthetic_forward_passed"])
        audit = summary["development_data_audit"]
        self.assertEqual(audit["contiguous_3d_volume_count"], 0)
        self.assertEqual(audit["verified_negative_voxels"], 0)
        self.assertGreater(audit["unannotated_patch_assumed_negative_samples"], 0)
        self.assertEqual(
            summary["decision"]["state"], "CONNECTED_DATA_GATE_BLOCKED"
        )
        self.assertFalse(summary["decision"]["default_enabled"])


if __name__ == "__main__":
    unittest.main()
