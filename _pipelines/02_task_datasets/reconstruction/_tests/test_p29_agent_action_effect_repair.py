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
                           "outcomes": {"win": 1}, "rmse": 2.0}}] for f in range(5)}
        categorical = json.dumps(p29.build_prompt_observation(mode="categorical", round_id=2, histories=histories))
        quantitative = json.dumps(p29.build_prompt_observation(mode="safe_quantitative", round_id=2, histories=histories))
        self.assertNotIn("relative_rmse_change", categorical)
        self.assertIn("relative_rmse_change", quantitative)
        self.assertNotIn("outcomes", quantitative); self.assertNotIn('"rmse"', quantitative)

    def test_action_effect_and_serializable_replay(self):
        p29.validate_action_registry()
        c = np.array([[0., 0., 0.], [1., 0., 1.], [0., 1., 2.]])
        v = np.array([1., 2., 4.]); q = np.array([[.2, .2, .3]])
        a0 = p29.replay_predictor(p29.predictor_config({"distance_power": 1.5, "vertical_weight": 4.}), coordinates=c, values=v, query=q)
        changed = p29.replay_predictor(p29.predictor_config({"distance_power": 1.5, "vertical_weight": 8.}), coordinates=c, values=v, query=q)
        self.assertFalse(np.array_equal(a0, changed))

    def test_probe_writes_owned_output(self):
        with tempfile.TemporaryDirectory() as td:
            result = p29.run_probe(Path(td))
            self.assertGreater(result["different_action_count"], 1)
            self.assertTrue((Path(td) / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
