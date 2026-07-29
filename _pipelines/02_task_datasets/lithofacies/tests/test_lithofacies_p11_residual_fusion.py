from __future__ import annotations

import copy
import importlib
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

p11 = importlib.import_module(
    "_pipelines.02_task_datasets.lithofacies."
    "lithofacies_p11_residual_fusion"
)


class _DummyMoment(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.pipeline = torch.nn.Module()
        self.pipeline.head = torch.nn.Linear(3, 9)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        pooled = values.mean(dim=-1)[:, :3]
        return self.pipeline.head(self.backbone(pooled))


class P11ResidualFusionContractTest(unittest.TestCase):
    def test_holdout_paths_fail_before_open(self) -> None:
        for path in (
            Path("/does/not/exist/test.h5"),
            Path("/does/not/exist/known-holdout"),
            Path("/does/not/exist/frozen_bundle.npz"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "forbidden holdout path"):
                    p11.ensure_development_only_paths([path])
        p11.ensure_development_only_paths(
            [Path("/safe/development_logo4.npz")]
        )

    def test_gate0_is_exact_baseline_degeneration(self) -> None:
        model = p11.GatedEmbeddingResidual(7)
        embeddings = torch.randn(4, 7)
        baseline = torch.randn(4, 9)
        fused, gate, residual = model(
            embeddings, baseline, force_gate_zero=True
        )
        self.assertTrue(torch.equal(fused, baseline))
        self.assertTrue(torch.equal(gate, torch.zeros(9)))
        self.assertEqual(tuple(residual.shape), (4, 9))
        self.assertLessEqual(
            float(residual.detach().abs().max()), p11.MAX_RESIDUAL_LOGIT
        )

    def test_randomization_preserves_head_and_is_deterministic(self) -> None:
        model_a = _DummyMoment()
        model_b = copy.deepcopy(model_a)
        head_before = {
            name: value.detach().clone()
            for name, value in model_a.named_parameters()
            if name.startswith("pipeline.head.")
        }
        p11._randomize_frozen_backbone(model_a, seed=2693)
        p11._randomize_frozen_backbone(model_b, seed=2693)
        for name, value in model_a.named_parameters():
            self.assertTrue(torch.equal(value, dict(model_b.named_parameters())[name]))
            if name in head_before:
                self.assertTrue(torch.equal(value, head_before[name]))
        self.assertFalse(
            torch.equal(model_a.backbone.weight, _DummyMoment().backbone.weight)
        )

    def test_strict_matrix_summary_and_missing_cell_rejection(self) -> None:
        rows = []
        values = {
            "baseline": 0.20,
            "direct": 0.05,
            "pretrained": 0.21,
            "random": 0.19,
            "gate0": 0.20,
        }
        for fold_id in p11.FOLD_IDS:
            for repeat_id in range(len(p11.REPEAT_SEEDS)):
                for variant in p11.ABLATIONS:
                    metrics = {
                        name: values[variant]
                        for name in p11.PRIMARY_METRICS
                    }
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "repeat_id": repeat_id,
                            "seed": p11.REPEAT_SEEDS[repeat_id],
                            "variant": variant,
                            "metrics": metrics,
                            "training": {
                                "gate_mean": (
                                    0.02
                                    if variant in {"pretrained", "random"}
                                    else (0.0 if variant == "gate0" else None)
                                ),
                                "gate0_max_abs_error": 0.0,
                                "residual_contribution_mean_abs": (
                                    0.01
                                    if variant in {"pretrained", "random"}
                                    else (
                                        0.0
                                        if variant == "gate0"
                                        else None
                                    )
                                ),
                            },
                        }
                    )
        summary = p11.summarize_results(rows)
        self.assertEqual(summary["evaluation"]["completed_cells"], 60)
        self.assertEqual(summary["decision"]["state"], "PROMOTE_RESIDUAL")
        self.assertEqual(
            summary["comparison"]["gate0_max_abs_logit_error"], 0.0
        )
        with self.assertRaisesRegex(ValueError, "complete strict"):
            p11.summarize_results(rows[:-1])

    def test_selection_rejects_partial_out_of_contract_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict LOGO4"):
            p11._selection((4,), (0,))
        with self.assertRaisesRegex(ValueError, "three frozen seeds"):
            p11._selection((0,), (3,))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            p11._selection((0, 0), (0,))

    def test_manifest_verifier_detects_hash_tampering(self) -> None:
        rows = []
        for fold_id in (0,):
            for repeat_id in (0,):
                for variant in p11.ABLATIONS:
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "repeat_id": repeat_id,
                            "seed": p11.REPEAT_SEEDS[repeat_id],
                            "variant": variant,
                            "metrics": {
                                name: 0.1 for name in p11.PRIMARY_METRICS
                            },
                            "training": {
                                "gate_mean": (
                                    0.01
                                    if variant in {"pretrained", "random"}
                                    else (
                                        0.0
                                        if variant == "gate0"
                                        else None
                                    )
                                ),
                                "gate0_max_abs_error": 0.0,
                                "residual_contribution_mean_abs": (
                                    0.01
                                    if variant in {"pretrained", "random"}
                                    else (
                                        0.0
                                        if variant == "gate0"
                                        else None
                                    )
                                ),
                            },
                        }
                    )
        summary = p11.summarize_results(
            rows, fold_ids=(0,), repeat_ids=(0,)
        )
        summary["decision"] = {
            "state": "KEEP_BASELINE_GATE_CLOSED",
            "default_enabled": False,
        }
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "_tmp") as tmp:
            output = (Path(tmp) / "p11").relative_to(PROJECT_ROOT)
            p11._write_artifacts(
                rows=rows, summary=summary, output_dir=output
            )
            verified = p11.verify_artifacts(output)
            self.assertEqual(verified["rows"], 5)
            (output / "evidence.md").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                p11.verify_artifacts(output)


if __name__ == "__main__":
    unittest.main()
