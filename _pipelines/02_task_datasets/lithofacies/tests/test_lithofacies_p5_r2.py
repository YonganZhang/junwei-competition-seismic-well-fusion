"""Lithofacies-prefixed P5.2 R2 contract and smoke tests."""
from __future__ import annotations

import inspect
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import lithofacies_p5_r2 as r2  # noqa: E402


def _synthetic_tensors(batch: int = 4, length: int = 33) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(r2.ROOT_SEED)
    well = rng.normal(size=(batch, 26, length)).astype(np.float32)
    seismic = rng.normal(size=(batch, 3, 3, length)).astype(np.float32)
    labels = np.arange(batch, dtype=np.int64) % r2.NUM_CLASSES
    return well, seismic, labels


class LithofaciesR2ContractTests(unittest.TestCase):
    def test_track_prefixed_module_and_source_lock(self) -> None:
        self.assertEqual(Path(r2.__file__).name, "lithofacies_p5_r2.py")
        self.assertEqual(Path(__file__).name, "test_lithofacies_p5_r2.py")
        self.assertEqual(r2.MODEL_ROSTER, (
            "multimodal_mlp",
            "xgboost_multisoftprob_window",
            "inceptiontime_window",
        ))
        self.assertEqual(r2.BUDGET_POINTS, (40, 200, 1000))
        self.assertEqual(r2.REPEAT_SEEDS, (1867973658, 2137841944, 3902865753))
        source = (TRACK_DIR / "lithofacies_p5_r2.py").read_text(encoding="utf-8")
        self.assertNotIn('"test.h5"', source)
        self.assertNotIn("load_frozen_test", source)
        self.assertNotIn("P5_r01_reproduction_20260715", source)
        self.assertEqual(tuple(inspect.signature(r2.main).parameters), ())

    def test_s_lane_is_structurally_not_rankable_without_finite_md(self) -> None:
        samples = [
            {"position": {"center_md_m": None}},
            {"position": {"center_md_m": float("nan")}},
        ]
        lane = r2._evaluate_s_lane(samples)
        self.assertEqual(lane["status"], "not_rankable")
        self.assertEqual(lane["finite_center_md_count"], 0)
        self.assertIn("forbidden", lane["reason"])


class LithofaciesR2SmokeTests(unittest.TestCase):
    def test_torch_models_support_forward_step_and_state_roundtrip(self) -> None:
        import torch
        from torch.nn import functional as F

        well, seismic, labels = _synthetic_tensors()
        for model_id in ("multimodal_mlp", "inceptiontime_window"):
            model = r2.get_model(
                model_id,
                models_package="models",
                **r2._model_config(model_id, tuple(well.shape[1:]), tuple(seismic.shape[1:])),
            )
            self.assertIsInstance(model, torch.nn.Module)
            logits = model(torch.as_tensor(well), torch.as_tensor(seismic))
            self.assertEqual(tuple(logits.shape), (len(labels), r2.NUM_CLASSES))
            self.assertTrue(torch.isfinite(logits).all())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = F.cross_entropy(logits, torch.as_tensor(labels))
            loss.backward()
            optimizer.step()
            with tempfile.TemporaryDirectory(dir=TRACK_DIR / "_outputs") as directory:
                checkpoint = Path(directory) / f"{model_id}.pt"
                torch.save(model.state_dict(), checkpoint)
                roundtrip = r2.get_model(
                    model_id,
                    models_package="models",
                    **r2._model_config(model_id, tuple(well.shape[1:]), tuple(seismic.shape[1:])),
                )
                roundtrip.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                rt_logits = roundtrip(torch.as_tensor(well), torch.as_tensor(seismic))
                self.assertEqual(tuple(rt_logits.shape), (len(labels), r2.NUM_CLASSES))
                self.assertTrue(torch.isfinite(rt_logits).all())

    def test_xgboost_adapter_pickle_roundtrip_and_logits(self) -> None:
        well, seismic, labels = _synthetic_tensors()
        adapter = r2.get_model(
            "xgboost_multisoftprob_window",
            models_package="models",
            num_classes=r2.NUM_CLASSES,
            well_log_shape=tuple(well.shape[1:]),
            seismic_shape=tuple(seismic.shape[1:]),
        )
        loss = adapter.fit_stage1(well, seismic, labels, class_counts=np.bincount(labels, minlength=r2.NUM_CLASSES))
        self.assertTrue(np.isfinite(loss))
        logits = adapter.predict_logits(well, seismic)
        self.assertEqual(logits.shape, (len(labels), r2.NUM_CLASSES))
        self.assertTrue(np.isfinite(logits).all())
        payload = pickle.dumps(adapter)
        restored = pickle.loads(payload)
        rt_logits = restored.predict_logits(well, seismic)
        self.assertEqual(rt_logits.shape, (len(labels), r2.NUM_CLASSES))
        self.assertTrue(np.isfinite(rt_logits).all())


if __name__ == "__main__":
    unittest.main()
