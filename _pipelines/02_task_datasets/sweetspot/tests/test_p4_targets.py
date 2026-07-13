from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from _code.ml_framework.model_discovery import discover_model


BASE = "_pipelines.02_task_datasets.sweetspot.targets"


class SweetspotRealDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts = {
            name: importlib.import_module(f"{BASE}.{name}.contract")
            for name in ("reservoir_quality", "hydrocarbon_pay", "productivity", "water_breakthrough")
        }
        cls.built = {
            name: module.build_dataset_and_manifest()
            for name, module in cls.contracts.items()
        }

    def test_target_specs_forbid_label_and_future_leakage(self):
        expected = {
            "reservoir_quality": {"PHIF", "KLOGH", "RQI"},
            "hydrocarbon_pay": {"SAND_FLAG", "SAND_FLAG_PROXY"},
            "productivity": {"future_BORE_OIL_VOL", "FUTURE_30D_MEAN_OIL"},
            "water_breakthrough": {"future_BORE_WAT_VOL", "event_date", "days_to_event"},
        }
        for name, contract in self.contracts.items():
            spec = contract.task_spec()
            self.assertTrue(expected[name].issubset(spec.forbidden_inputs))
            self.assertFalse(set(spec.input_whitelist) & set(spec.forbidden_inputs))

    def test_frozen_test_is_disjoint_and_cv_downgrades_honestly(self):
        expected_splits = {
            "reservoir_quality": 3,
            "hydrocarbon_pay": 3,
            "productivity": 4,
            "water_breakthrough": 3,
        }
        for name, (_, manifest, evidence) in self.built.items():
            self.assertEqual(manifest.requested_n_splits, 5)
            self.assertEqual(manifest.effective_n_splits, expected_splits[name])
            self.assertTrue(manifest.downgrade_reason)
            self.assertTrue(set(manifest.test_sample_ids).isdisjoint(manifest.development_sample_ids))
            self.assertEqual(evidence["effective_n_splits"], expected_splits[name])

    def test_real_targets_have_support(self):
        rqi, _, _ = self.built["reservoir_quality"]
        sand, _, _ = self.built["hydrocarbon_pay"]
        productivity, _, _ = self.built["productivity"]
        water, _, evidence = self.built["water_breakthrough"]
        self.assertGreater(len(rqi["sample_ids"]), 1000)
        self.assertEqual(set(np.unique(sand["target"])), {0.0, 1.0})
        self.assertGreater(len(productivity["sample_ids"]), 100)
        self.assertEqual(set(water["event_within_30d"]), {0, 1})
        self.assertIn("NO 15/9-F-12 H", evidence["excluded_groups"])

    def test_canonical_baselines_fit_without_test_access(self):
        rqi, manifest, _ = self.built["reservoir_quality"]
        train = set(manifest.folds[0].train_sample_ids)
        index = [i for i, sid in enumerate(rqi["sample_ids"]) if sid in train][:1000]
        model = discover_model("sweetspot", "robust_linear").build(self.contracts["reservoir_quality"].task_spec())
        x = [rqi["features"][i] for i in index]
        y = [rqi["target"][i] for i in index]
        model.fit(x, {"RQI": y}, {"RQI": [True] * len(y)})
        self.assertEqual(len(model.predict(x[:7]).raw["RQI"]), 7)

        sand, manifest, _ = self.built["hydrocarbon_pay"]
        train = set(manifest.folds[0].train_sample_ids)
        index = [i for i, sid in enumerate(sand["sample_ids"]) if sid in train]
        # Preserve both classes while keeping the test quick.
        zero = [i for i in index if sand["target"][i] == 0.0][:500]
        one = [i for i in index if sand["target"][i] == 1.0][:500]
        index = zero + one
        model = discover_model("sweetspot", "logistic_classifier").build(self.contracts["hydrocarbon_pay"].task_spec())
        x = [sand["features"][i] for i in index]; y = [sand["target"][i] for i in index]
        model.fit(x, {"SAND_FLAG_PROXY": y}, {"SAND_FLAG_PROXY": [True] * len(y)})
        output = model.predict(x[:7])
        self.assertEqual(len(output.raw["SAND_FLAG_PROXY"]), 7)
        self.assertTrue(np.all((output.transformed["SAND_FLAG_PROXY"] >= 0) & (output.transformed["SAND_FLAG_PROXY"] <= 1)))

    def test_remaining_oil_is_fail_closed(self):
        contract = importlib.import_module(f"{BASE}.remaining_oil_infill.contract")
        evidence = contract.not_feasible_evidence()
        self.assertEqual(evidence["status"], "not_feasible")
        self.assertTrue(evidence["dynamic_members"])
        self.assertIsNone(evidence["baseline"])
        self.assertTrue(evidence["no_synthetic_fallback"])


class SweetspotVisualizationTests(unittest.TestCase):
    def test_target_specific_visualizers_are_read_only(self):
        visualizer = importlib.import_module(f"{BASE}.visualize")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(2693)
            cases = {
                "reservoir_quality": pd.DataFrame({
                    "well": np.repeat(["w1", "w2"], 20), "depth_m": np.tile(np.arange(20), 2),
                    "observed": rng.uniform(0, 1, 40), "prediction": rng.uniform(0, 1, 40),
                }),
                "hydrocarbon_pay": pd.DataFrame({
                    "well": np.repeat(["w1", "w2"], 20), "depth_m": np.tile(np.arange(20), 2),
                    "observed": np.tile([0, 1], 20), "probability": rng.uniform(0, 1, 40),
                }),
                "productivity": pd.DataFrame({
                    "well": np.repeat(["w1", "w2"], 20),
                    "cutoff_date": pd.date_range("2020-01-01", periods=20).tolist() * 2,
                    "observed": rng.uniform(1, 100, 40), "prediction": rng.uniform(1, 100, 40),
                }),
                "water_breakthrough": pd.DataFrame({
                    "well": np.repeat(["w1", "w2"], 20),
                    "cutoff_date": pd.date_range("2020-01-01", periods=20).tolist() * 2,
                    "observed": np.tile([0, 1], 20), "probability": rng.uniform(0, 1, 40),
                }),
            }
            for name, frame in cases.items():
                csv = root / f"{name}.csv"; frame.to_csv(csv, index=False)
                before = csv.read_bytes()
                manifest = visualizer.render(name, csv, root / name, frozen_threshold=0.5)
                self.assertEqual(csv.read_bytes(), before)
                self.assertEqual(len(manifest["figures"]), 4)
                for figure in manifest["figures"]:
                    self.assertGreater((root / name / figure["name"]).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
