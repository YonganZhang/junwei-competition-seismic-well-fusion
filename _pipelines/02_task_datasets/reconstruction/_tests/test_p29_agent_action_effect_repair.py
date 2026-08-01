import json
import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np

TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK))
import p29_agent_action_effect_repair as p29


class P29Test(unittest.TestCase):
    def test_prompt_ablation_contains_only_allowed_quantitative_field(self):
        histories = {f: [{"round": 1, "action_id": "x", "feedback":
                          {"classification": "improved", "relative_rmse_change": -0.12,
                           "fold_outcomes": {"win": 1, "loss": 0},
                           "uncertainty": {"stderr": 0.01}, "rmse": 2.0}}] for f in range(5)}
        categorical = json.dumps(p29.build_prompt_observation(mode="categorical", round_id=2, histories=histories))
        quantitative = json.dumps(p29.build_prompt_observation(mode="safe_quantitative", round_id=2, histories=histories))
        self.assertNotIn("relative_rmse_change", categorical)
        self.assertIn("relative_rmse_change", quantitative)
        self.assertIn("fold_outcomes", quantitative); self.assertIn("uncertainty", quantitative)
        self.assertIn("remaining_budget", quantitative); self.assertIn("promotion_threshold", quantitative)
        self.assertNotIn('"rmse"', quantitative)

    def test_action_effect_and_serializable_replay(self):
        p29.validate_action_registry()
        c = np.array([[0., 0., 0.], [1., 0., 1.], [0., 1., 2.]])
        v = np.array([1., 2., 4.]); q = np.array([[.2, .2, .3]])
        seismic = np.array([[.1,.2,.3],[.4,.2,.1],[.7,.1,.2]])
        latent = np.array([[.1],[.9],[.3]])
        a0 = p29.replay_predictor(p29.predictor_config({"distance_power": 1.5, "vertical_weight": 4., "seismic_weights":[0.,.1,.2], "foundation_weight":.1}), coordinates=c, values=v, query=q, seismic=seismic, latent=latent)
        changed = p29.replay_predictor(p29.predictor_config({"distance_power": 1.5, "vertical_weight": 8., "seismic_weights":[0.,.1,.2], "foundation_weight":.1}), coordinates=c, values=v, query=q, seismic=seismic, latent=latent)
        self.assertFalse(np.array_equal(a0, changed))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"; path.write_text(json.dumps(p29.predictor_config({"distance_power": 1.5, "vertical_weight": 8., "seismic_weights":[0.,.1,.2], "foundation_weight":.1})))
            loaded = json.loads(path.read_text())
            replay = p29.replay_predictor(loaded, coordinates=c, values=v, query=q, seismic=seismic, latent=latent)
            mutated = json.loads(path.read_text()); mutated["parameters"]["vertical_weight"] = 2.
            self.assertFalse(np.array_equal(replay, p29.replay_predictor(mutated, coordinates=c, values=v, query=q, seismic=seismic, latent=latent)))

    def test_probe_writes_owned_output(self):
        with tempfile.TemporaryDirectory() as td:
            result = p29.run_probe(Path(td))
            self.assertGreater(result["different_action_count"], 1)
            self.assertTrue((Path(td) / "summary.json").is_file())

    def test_real_evidence_contract_rejects_empty_histories_and_fake_purge(self):
        histories = {f: [] for f in range(5)}
        with self.assertRaises(ValueError):
            p29.build_prompt_observation(mode="safe_quantitative", round_id=2, histories=histories)
        self.assertIn("p19", p29.p19._without_coordinates.__module__)

    def test_real_summary_purge_calls_and_no_held_fold_prompt_leak(self):
        path = TRACK / "_outputs" / "p29_agent_action_effect_repair" / "summary.json"
        if not path.is_file(): self.skipTest("real probe not run")
        summary = json.loads(path.read_text())
        self.assertEqual(len(summary["purge_audits"]), 5)
        self.assertTrue(all(a["p19_rows_called"] and a["p19_without_coordinates_called"] for a in summary["purge_audits"]))
        for held, rows in summary["outer_fold_observations"].items():
            self.assertTrue(rows)
            self.assertTrue(all(int(held) not in row["selection_fold_ids"] for row in rows))


if __name__ == "__main__":
    unittest.main()
