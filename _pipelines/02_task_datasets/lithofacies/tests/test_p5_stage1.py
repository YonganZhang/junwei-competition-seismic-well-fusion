"""P5 source-lock, adapter, split, firewall, and Stage-1 contract tests."""
from __future__ import annotations

import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path

import numpy as np


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _models.lithofacies.p5_adapter_common import OptionalDependencyUnavailable  # noqa: E402
from p4_contract import CLASS_NAMES, DEVELOPMENT_FAMILIES, TEST_FAMILY, lithofacies_task_spec  # noqa: E402
from p5_stage1 import (  # noqa: E402
    FIRST_TEN,
    SOURCE_LOCK_PATH,
    build_development_logo4,
    load_source_lock,
)


TORCH_MODEL_IDS = (
    "inceptiontime_window",
    "tcn_center_head",
    "balanced_softmax_tcn",
    "moderntcn_window",
    "ms_tcn2_dense",
    "embracenet_missing_modal",
    "multibench_lowrank_tensor_fusion",
)
ESTIMATOR_MODEL_IDS = (
    "xgboost_multisoftprob_window",
    "catboost_multiclass_window",
    "minirocket_ridge_window",
)


def _synthetic_split_samples() -> list[dict]:
    samples = []
    for family_index, family in enumerate(DEVELOPMENT_FAMILIES):
        for index in range(3):
            samples.append(
                {
                    "label": (family_index + index) % len(CLASS_NAMES),
                    "position": {
                        "well_name": f"{family}-TRACK",
                        "center_md_m": 1000.0 + index,
                        "time_ms": 2000.0 + index,
                    },
                    "meta": {
                        "family_id": family,
                        "label_trace": {"member": f"{family}.xlsx", "excel_row": index + 2},
                    },
                }
            )
    return samples


class P5SourceLockTests(unittest.TestCase):
    def test_source_lock_exactly_matches_frozen_first_ten(self) -> None:
        lock = load_source_lock()
        self.assertEqual(lock["class_count"], 9)
        self.assertEqual(tuple(CLASS_NAMES), tuple(lithofacies_task_spec().metadata["class_names"]))
        self.assertEqual(tuple(model["model_id"] for model in lock["models"]), FIRST_TEN)
        self.assertEqual(len(set(model["revision"] for model in lock["models"])), 10)
        for model in lock["models"]:
            self.assertTrue(model["source_url"].startswith("https://"))
            self.assertTrue(model["paper_url"].startswith("https://"))
            self.assertTrue(model["license"])
            self.assertFalse(model["pretrained_weights"]["used"])
            adapter = PROJECT_ROOT / "_models" / "lithofacies" / f"{model['model_id']}.py"
            self.assertTrue(adapter.is_file(), adapter)

    def test_all_adapters_are_dynamically_discoverable(self) -> None:
        lock_by_id = {model["model_id"]: model for model in load_source_lock()["models"]}
        for model_id in FIRST_TEN:
            with self.subTest(model_id=model_id):
                if any(
                    importlib.util.find_spec(name) is None
                    for name in lock_by_id[model_id]["required_imports"]
                ):
                    continue
                discovered = discover_model("lithofacies", model_id)
                self.assertEqual(discovered.model_id, model_id)
                self.assertEqual(discovered.capabilities["fixed_class_count"], 9)
                self.assertEqual(
                    discovered.capabilities["input_modalities"],
                    ["well_log_sequence", "st0202_seismic_patch"],
                )
                self.assertTrue(discovered.capabilities["supports_missing_mask"])
                self.assertIn(discovered.capabilities["leaderboard_lane"], {"P", "S"})

    def test_p_and_s_leaderboard_rosters_are_separate(self) -> None:
        lock = load_source_lock()
        p_models = [model["model_id"] for model in lock["models"] if model["leaderboard_lane"] == "P"]
        s_models = [model["model_id"] for model in lock["models"] if model["leaderboard_lane"] == "S"]
        self.assertEqual(len(p_models), 9)
        self.assertEqual(s_models, ["ms_tcn2_dense"])
        self.assertFalse(set(p_models) & set(s_models))


class P5FirewallTests(unittest.TestCase):
    def test_logo4_is_a_disjoint_mother_family_cover(self) -> None:
        samples = _synthetic_split_samples()
        folds = build_development_logo4(samples)
        self.assertEqual(len(folds), 4)
        validation_groups = []
        for fold in folds:
            train = set(fold["train_groups"])
            validation = set(fold["validation_groups"])
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, set(DEVELOPMENT_FAMILIES))
            self.assertNotIn(TEST_FAMILY, train | validation)
            validation_groups.extend(validation)
        self.assertEqual(tuple(validation_groups), DEVELOPMENT_FAMILIES)

    def test_logo4_rejects_frozen_test_family(self) -> None:
        samples = _synthetic_split_samples()
        samples[0]["meta"]["family_id"] = TEST_FAMILY
        with self.assertRaisesRegex(ValueError, "non-development family"):
            build_development_logo4(samples)

    def test_stage1_source_has_no_frozen_test_loader_or_filename(self) -> None:
        source = (TRACK_DIR / "p5_stage1.py").read_text(encoding="utf-8")
        self.assertNotIn("load_frozen_test", source)
        self.assertNotIn('"test.h5"', source)
        self.assertNotIn("'test.h5'", source)
        self.assertIn('dataset_root.resolve() / "train.h5"', source)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is unavailable")
class P5TorchAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch
        torch.manual_seed(2693)

    def _inputs(self, lane: str):
        torch = self.torch
        length = 17 if lane == "S" else 33
        batch = 2 if lane == "S" else 4
        well = torch.randn(batch, 26, length)
        well[:, 13:, :] = torch.randint(0, 2, (batch, 13, length), dtype=torch.int64).float()
        seismic = torch.randn(batch, 3, 3, length)
        labels = (
            torch.randint(0, 9, (batch, length), dtype=torch.long)
            if lane == "S"
            else torch.tensor([0, 1, 2, 3], dtype=torch.long)
        )
        return well, seismic, labels

    def test_real_torch_forward_backward_softmax_and_checkpoint_roundtrip(self) -> None:
        torch = self.torch
        functional = torch.nn.functional
        lock_by_id = {model["model_id"]: model for model in load_source_lock()["models"]}
        for model_id in TORCH_MODEL_IDS:
            with self.subTest(model_id=model_id):
                lock = lock_by_id[model_id]
                if any(importlib.util.find_spec(name) is None for name in lock["required_imports"]):
                    continue
                lane = lock["leaderboard_lane"]
                well, seismic, labels = self._inputs(lane)
                config = {
                    **lock["smoke_config"],
                    "num_classes": 9,
                    "well_log_shape": tuple(well.shape[1:]),
                    "seismic_shape": tuple(seismic.shape[1:]),
                }
                discovered = discover_model("lithofacies", model_id)
                model = discovered.build(lithofacies_task_spec(), **config)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(well, seismic)
                expected = (len(well), 9, well.shape[-1]) if lane == "S" else (len(well), 9)
                self.assertEqual(tuple(logits.shape), expected)
                self.assertTrue(bool(torch.isfinite(logits).all()))
                custom_loss = getattr(discovered.module, "stage1_loss", None)
                if custom_loss is None:
                    loss = functional.cross_entropy(logits, labels)
                else:
                    loss = custom_loss(
                        logits,
                        labels,
                        class_counts=torch.tensor([4, 4, 3, 3, 2, 2, 1, 1, 0]),
                    )
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    before = model(well, seismic)
                    changed_seismic = model(well, seismic + 0.25)
                    changed_masks = well.clone()
                    changed_masks[:, 13:, :] = 1.0 - changed_masks[:, 13:, :]
                    changed_mask_output = model(changed_masks, seismic)
                self.assertGreater(float(torch.max(torch.abs(before - changed_seismic))), 0.0)
                self.assertGreater(float(torch.max(torch.abs(before - changed_mask_output))), 0.0)
                probability = torch.softmax(before, dim=1)
                self.assertTrue(bool(torch.allclose(probability.sum(dim=1), torch.ones_like(probability.sum(dim=1)))))
                buffer = io.BytesIO()
                torch.save({"state_dict": model.state_dict(), "config": config}, buffer)
                buffer.seek(0)
                payload = torch.load(buffer, weights_only=True)
                reloaded = discovered.build(lithofacies_task_spec(), **payload["config"])
                reloaded.load_state_dict(payload["state_dict"])
                reloaded.eval()
                with torch.no_grad():
                    after = reloaded(well, seismic)
                self.assertTrue(bool(torch.equal(before, after)))

    def test_embracenet_handles_each_available_modality(self) -> None:
        torch = self.torch
        lock = next(
            model for model in load_source_lock()["models"]
            if model["model_id"] == "embracenet_missing_modal"
        )
        well, seismic, _ = self._inputs("P")
        discovered = discover_model("lithofacies", lock["model_id"])
        model = discovered.build(
            lithofacies_task_spec(),
            **lock["smoke_config"],
            num_classes=9,
            well_log_shape=tuple(well.shape[1:]),
            seismic_shape=tuple(seismic.shape[1:]),
        ).eval()
        with torch.no_grad():
            full = model(well, seismic)
            logs_only = model(well, seismic, torch.tensor([[1.0, 0.0]] * len(well)))
            seismic_only = model(well, seismic, torch.tensor([[0.0, 1.0]] * len(well)))
        self.assertEqual(tuple(full.shape), (len(well), 9))
        self.assertTrue(bool(torch.isfinite(logs_only).all()))
        self.assertTrue(bool(torch.isfinite(seismic_only).all()))
        self.assertFalse(bool(torch.equal(logs_only, seismic_only)))


class P5EstimatorAdapterTests(unittest.TestCase):
    def test_available_estimators_fit_real_code_path_and_pickle_roundtrip(self) -> None:
        rng = np.random.default_rng(2693)
        well = rng.normal(size=(18, 26, 33)).astype(np.float32)
        well[:, 13:, :] = rng.integers(0, 2, size=(18, 13, 33)).astype(np.float32)
        seismic = rng.normal(size=(18, 3, 3, 33)).astype(np.float32)
        labels = np.tile(np.arange(9, dtype=np.int64), 2)
        counts = np.bincount(labels, minlength=9)
        lock_by_id = {model["model_id"]: model for model in load_source_lock()["models"]}
        for model_id in ESTIMATOR_MODEL_IDS:
            with self.subTest(model_id=model_id):
                lock = lock_by_id[model_id]
                if any(importlib.util.find_spec(name) is None for name in lock["required_imports"]):
                    continue
                discovered = discover_model("lithofacies", model_id)
                try:
                    model = discovered.build(
                        lithofacies_task_spec(),
                        **lock["smoke_config"],
                        num_classes=9,
                        well_log_shape=(26, 33),
                        seismic_shape=(3, 3, 33),
                    )
                except OptionalDependencyUnavailable:
                    continue
                loss = model.fit_stage1(well, seismic, labels, class_counts=counts)
                self.assertTrue(np.isfinite(loss))
                before = np.asarray(model.predict_logits(well[:4], seismic[:4]))
                self.assertEqual(before.shape, (4, 9))
                self.assertTrue(np.isfinite(before).all())
                buffer = io.BytesIO()
                import pickle

                pickle.dump(model, buffer, protocol=pickle.HIGHEST_PROTOCOL)
                buffer.seek(0)
                after = pickle.load(buffer).predict_logits(well[:4], seismic[:4])
                np.testing.assert_array_equal(before, after)


class P5RealBatchIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LITHOFACIES_P5_REAL_BATCH") == "1", "real batch gate disabled")
    def test_existing_development_batch_preparation_never_reads_frozen_test(self) -> None:
        from p5_stage1 import load_batch, prepare_batch

        dataset_root = Path(os.environ["LITHOFACIES_P5_DATASET_ROOT"])
        output = TRACK_DIR / "_outputs" / "p5_stage1_test" / "real_batch.npz"
        manifest = prepare_batch(
            dataset_root, output, max_train=32, max_validation=8, sequence_length=8
        )
        _, stored = load_batch(output)
        self.assertEqual(manifest["loaded_files"], ["train.h5"])
        self.assertFalse(stored["frozen_test_accessed"])
        self.assertEqual(stored["effective_n_splits"], 4)


if __name__ == "__main__":
    unittest.main()
