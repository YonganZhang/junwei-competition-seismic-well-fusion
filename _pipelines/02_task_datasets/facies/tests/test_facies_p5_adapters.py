from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.model_discovery import discover_model
from _models.facies._p5_common import P5AdapterSkip, source_lock, source_locks

from p4_tasks import get_task_spec


def _load_track_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, TRACK_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MODEL_IDS = _load_track_module("facies_p5_stage1_adapters", "p5_stage1.py").MODEL_IDS


AVAILABLE_2D_IDS = (
    "smp_unet_r18",
    "smp_deeplabv3plus_r18",
    "smp_unetpp_r18",
    "smp_fpn_r18",
    "torchvision_lraspp_mbv3",
    "hf_segformer_b0",
)
LEGACY_IDS = (
    "deepseismic_patch_skip",
    "deepseismic_seresnet_unet",
    "sfm_base_facies",
)


class P5SourceAndDiscoveryTests(unittest.TestCase):
    def test_exact_ten_source_locks_and_dynamic_plugins(self) -> None:
        self.assertEqual(tuple(source_locks()), MODEL_IDS)
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                discovered = discover_model("facies", model_id)
                self.assertEqual(discovered.model_id, model_id)
                lock = source_lock(model_id)
                self.assertTrue(lock["source_url"].startswith("https://"))
                self.assertTrue(lock["source_tag"] or lock["source_commit"])
                self.assertTrue(lock["code_license"])
                self.assertEqual(lock["allowed_lanes"], ["scratch"])
                self.assertNotEqual(lock["weights"]["status"], "approved")

    def test_pretrained_lane_is_fail_closed_for_every_candidate(self) -> None:
        spec = get_task_spec("facies_f3")
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                with self.assertRaises(P5AdapterSkip) as raised:
                    discover_model("facies", model_id).build(spec, lane="pretrained")
                self.assertEqual(raised.exception.code, "weight_lane_not_approved")

    def test_f3_and_penobscot_heads_cannot_be_crossed(self) -> None:
        model = discover_model("facies", "smp_unet_r18")
        with self.assertRaisesRegex(ValueError, "independent 10-class head"):
            model.build(get_task_spec("facies_f3"), num_classes=8)
        with self.assertRaisesRegex(ValueError, "independent 8-class head"):
            model.build(get_task_spec("facies_penobscot"), num_classes=10)


class P5ExecutableAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def _exercise_2d(self, model_id: str, classes: int) -> None:
        spec = get_task_spec("facies_f3" if classes == 10 else "facies_penobscot")
        try:
            model = discover_model("facies", model_id).build(spec, lane="scratch")
        except P5AdapterSkip as skip:
            self.assertIn(
                skip.code,
                {"dependency_unavailable", "dependency_import_failed", "runtime_version_mismatch"},
            )
            return
        self.assertIsInstance(model, nn.Module)
        model.train()
        inputs = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, classes, (2, 64, 64))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2, classes, 64, 64))
        self.assertTrue(torch.isfinite(logits).all())
        loss = nn.CrossEntropyLoss()(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(inputs)

        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = discover_model("facies", model_id).build(spec, lane="scratch")
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        restored.eval()
        with torch.no_grad():
            actual = restored(inputs)
        torch.testing.assert_close(actual, expected)

    def test_available_2d_adapters_forward_backward_and_checkpoint(self) -> None:
        for model_id in AVAILABLE_2D_IDS:
            for classes in (10, 8):
                with self.subTest(model_id=model_id, classes=classes):
                    self._exercise_2d(model_id, classes)

    def test_legacy_adapters_skip_instead_of_substituting_architectures(self) -> None:
        spec = get_task_spec("facies_f3")
        for model_id in LEGACY_IDS:
            with self.subTest(model_id=model_id):
                with self.assertRaises(P5AdapterSkip) as raised:
                    discover_model("facies", model_id).build(spec, lane="scratch")
                self.assertEqual(raised.exception.code, "legacy_source_port_not_available")

    def test_monai_3d_adapter_is_real_but_fixed_2d_io_is_gated(self) -> None:
        spec = get_task_spec("facies_penobscot")
        discovered = discover_model("facies", "monai_unet3d")
        with self.assertRaises(P5AdapterSkip) as raised:
            discovered.build(spec, lane="scratch")
        self.assertEqual(
            raised.exception.code,
            "contiguous_3d_development_blocks_unavailable",
        )
        try:
            model = discovered.build(
                spec,
                lane="scratch",
                allow_3d_contract=True,
                channels=(4, 8, 16),
                strides=(2, 2),
            )
        except P5AdapterSkip as skip:
            self.assertIn(
                skip.code,
                {"dependency_unavailable", "dependency_import_failed", "runtime_version_mismatch"},
            )
            return
        inputs = torch.randn(1, 1, 8, 32, 32)
        target = torch.randint(0, 8, (1, 8, 32, 32))
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1, 8, 8, 32, 32))
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
