from __future__ import annotations

import importlib.util
import importlib
import unittest

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from _code.ml_framework.model_discovery import discover_model
from _models.sweetspot.p5_common import AdapterSkip
MODEL_ORDER = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.matrix"
).MODEL_ORDER


def task_spec(task_id: str = "T6", task_type: str = "regression") -> TaskSpec:
    head = f"{task_id}_HEAD"
    return TaskSpec(
        track_id="sweetspot", task_id=f"p5-test-{task_id.lower()}", task_type=task_type,
        input_modalities=("unit_test",), targets=(head,), units={head: "fixture"},
        label_version=f"{task_id.lower()}-test-only", target_masks={head: "finite fixture labels"},
        group_keys=("fixture_group",), target_transform={head: "identity"},
        inverse_transform={head: "identity"}, train_loss={head: "fixture loss"},
        inference_transform={head: "identity"}, threshold_policy={}, calibration_policy={},
        primary_metrics=("mae",), metric_directions={"mae": "minimize"},
        visualizer_id=f"p5_test_{task_id.lower()}", required_figures=(f"{task_id.lower()}_fixture",),
        input_whitelist=("fixture.input",), forbidden_inputs=("fixture.label",),
        metadata={"single_target_head": True, "target_id": task_id, "class_count": 2, "no_proxy_fallback": True},
    )


class AdapterDiscoveryTests(unittest.TestCase):
    def test_all_ten_modules_are_lazy_discoverable(self):
        for model_id in MODEL_ORDER:
            discovered = discover_model("sweetspot", model_id)
            self.assertEqual(discovered.model_id, model_id)
            self.assertIn("stage1_input_key", discovered.capabilities)

    def test_each_build_creates_an_independent_estimator_and_head(self):
        if importlib.util.find_spec("xgboost") is None:
            self.skipTest("xgboost is not in this shared interpreter")
        t6 = discover_model("sweetspot", "xgboost").build(task_spec("T6"))
        t7 = discover_model("sweetspot", "xgboost").build(task_spec("T7"))
        self.assertIsNot(t6, t7)
        self.assertIsNot(t6.estimator, t7.estimator)
        self.assertNotEqual(t6.target, t7.target)

    def test_locked_source_candidates_fail_closed_without_checkout(self):
        for model_id in ("patchtst", "seg_spatial_tcn"):
            with self.assertRaises(AdapterSkip) as raised:
                discover_model("sweetspot", model_id).build(task_spec("T6"))
            self.assertEqual(raised.exception.reason_code, "locked_source_checkout_unavailable")


class InstalledAdapterSmokeTests(unittest.TestCase):
    def test_installed_tree_families_fit_and_roundtrip(self):
        rng = np.random.default_rng(2693)
        x = rng.normal(size=(32, 6)).astype(np.float32)
        y = (0.3 * x[:, 0] - 0.2 * x[:, 1]).astype(np.float32)
        for model_id, module in (("xgboost", "xgboost"), ("catboost", "catboost"), ("lightgbm", "lightgbm")):
            if importlib.util.find_spec(module) is None:
                continue
            adapter = discover_model("sweetspot", model_id).build(task_spec("T6"))
            evidence = adapter.stage1_smoke({"tabular": x}, y, np.ones(32, dtype=bool), seed=2693)
            self.assertTrue(evidence["finite_output"])
            self.assertLessEqual(evidence["checkpoint_roundtrip_max_abs_delta"], 1e-10)
            self.assertFalse(evidence["test_accessed"])

    def test_installed_inceptiontime_runs_backward_and_roundtrip(self):
        if importlib.util.find_spec("tsai") is None:
            self.skipTest("tsai is not in this shared interpreter")
        rng = np.random.default_rng(2693)
        x = rng.normal(size=(8, 2, 32)).astype(np.float32)
        y = rng.normal(size=8).astype(np.float32)
        adapter = discover_model("sweetspot", "inceptiontime").build(task_spec("T6"), c_in=2, seq_len=32, nf=4)
        evidence = adapter.stage1_smoke({"sequence": x}, y, np.ones(8, dtype=bool), seed=2693)
        self.assertTrue(evidence["backward_completed"])
        self.assertTrue(evidence["finite_output"])

    def test_installed_tft_uses_real_upstream_model(self):
        if importlib.util.find_spec("pytorch_forecasting") is None:
            self.skipTest("pytorch_forecasting is not in this shared interpreter")
        import pandas as pd

        rows = []
        for group in ("g0", "g1"):
            for index in range(14):
                rows.append({"sample_id": f"{group}-{index}", "time_idx": index, "group_id": group, "known": float(index % 7), "target": float(index + (group == "g1"))})
        frame = pd.DataFrame(rows)
        adapter = discover_model("sweetspot", "temporal_fusion_transformer").build(task_spec("T3"), encoder_length=6, prediction_length=2)
        evidence = adapter.stage1_smoke({"time_series_frame": frame}, np.zeros(len(frame)), np.ones(len(frame), dtype=bool), seed=2693)
        self.assertTrue(evidence["backward_completed"])
        self.assertTrue(evidence["finite_output"])

    def test_installed_monai_runs_scratch_backward_and_roundtrip(self):
        if importlib.util.find_spec("monai") is None:
            self.skipTest("MONAI is not in this shared interpreter")
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        rng = np.random.default_rng(2693)
        x = rng.normal(size=(2, 1, 8, 8, 8)).astype(np.float32)
        y = rng.normal(size=(2, 8, 8, 8)).astype(np.float32)
        adapter = discover_model("sweetspot", "monai_unet3d").build(
            task_spec("T6"), in_channels=1, channels=(2, 4), strides=(2,), device=device,
        )
        evidence = adapter.stage1_smoke({"volume": x}, y, np.ones_like(y, dtype=bool), seed=2693)
        self.assertTrue(evidence["backward_completed"])
        self.assertTrue(evidence["finite_output"])
        self.assertEqual(evidence["device"], device)


if __name__ == "__main__":
    unittest.main()
