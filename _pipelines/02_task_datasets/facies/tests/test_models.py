from __future__ import annotations

import importlib
import io
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.ml_framework.model_registry import MODEL_REGISTRY, get_model


MODEL_NAMES = ("facies_linear_pixel", "facies_tiny_fcn")


class SwappableModelContractTests(unittest.TestCase):
    def test_dynamic_discovery_and_real_forward(self) -> None:
        inputs = torch.randn(2, 1, 9, 11)

        for name in MODEL_NAMES:
            with self.subTest(model=name):
                module_name = f"models.{name}"
                MODEL_REGISTRY.pop(name, None)
                sys.modules.pop(module_name, None)
                importlib.invalidate_caches()

                model = get_model(
                    name,
                    models_package="models",
                    num_classes=4,
                    base_channels=99,
                )

                self.assertIsInstance(model, nn.Module)
                self.assertEqual(MODEL_REGISTRY[name].__module__, module_name)
                logits = model(inputs)
                self.assertEqual(tuple(logits.shape), (2, 4, 9, 11))
                self.assertTrue(torch.isfinite(logits).all().item())
                with self.assertRaises(ValueError):
                    model(torch.randn(2, 2, 9, 11))

    def test_optimizer_loss_step_and_state_dict_roundtrip(self) -> None:
        torch.manual_seed(2693)
        inputs = torch.randn(2, 1, 8, 8)
        targets = torch.randint(0, 4, (2, 8, 8))
        criterion = nn.CrossEntropyLoss()

        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = get_model(
                    name,
                    models_package="models",
                    num_classes=4,
                )
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                before = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                optimizer.step()
                self.assertTrue(
                    any(
                        not torch.equal(before[key], value)
                        for key, value in model.state_dict().items()
                    )
                )

                checkpoint = io.BytesIO()
                torch.save({"model_state_dict": model.state_dict()}, checkpoint)
                checkpoint.seek(0)
                payload = torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
                restored = get_model(
                    name,
                    models_package="models",
                    num_classes=4,
                )
                restored.load_state_dict(payload["model_state_dict"])

                model.eval()
                restored.eval()
                with torch.no_grad():
                    expected = model(inputs)
                    actual = restored(inputs)
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
