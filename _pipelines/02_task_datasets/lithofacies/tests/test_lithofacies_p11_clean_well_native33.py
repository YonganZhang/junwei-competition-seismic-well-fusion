from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


TRACK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

diagnostic = importlib.import_module(
    "_pipelines.02_task_datasets.lithofacies."
    "lithofacies_p11_clean_well_native33"
)
p11 = diagnostic.p11


class _RecordingPipeline(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recorded: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def embed(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor,
        reduction: str,
    ) -> SimpleNamespace:
        self.recorded.append((tuple(x_enc.shape), tuple(input_mask.shape)))
        assert reduction == "mean"
        return SimpleNamespace(embeddings=x_enc.mean(dim=(1, 2))[:, None])

    def forward(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> SimpleNamespace:
        self.recorded.append((tuple(x_enc.shape), tuple(input_mask.shape)))
        logits = torch.zeros((len(x_enc), 9), device=x_enc.device)
        return SimpleNamespace(logits=logits)


class P11CleanWellNativeContractTest(unittest.TestCase):
    def test_clean_selector_excludes_masks_and_has_native_shape(self) -> None:
        well = np.arange(2 * 26 * 33, dtype=np.float32).reshape(2, 26, 33)
        selected = diagnostic._clean_log_inputs(well)
        self.assertEqual(selected.shape, (2, 13, 33))
        self.assertTrue(np.array_equal(selected, well[:, :13, :]))
        changed_masks = well.copy()
        changed_masks[:, 13:, :] = -12345.0
        self.assertTrue(
            np.array_equal(
                selected,
                diagnostic._clean_log_inputs(changed_masks),
            )
        )

    def test_native_embedding_path_never_resamples(self) -> None:
        pipeline = _RecordingPipeline()
        model = diagnostic.NativeLogMomentClassifier(pipeline)
        values = np.ones((3, 13, 33), dtype=np.float32)
        embeddings = diagnostic._extract_native_embeddings(
            model,
            values,
            device="cpu",
        )
        self.assertEqual(embeddings.shape, (3, 1))
        self.assertEqual(pipeline.recorded, [((3, 13, 33), (3, 33))])
        source = inspect.getsource(diagnostic._extract_native_embeddings)
        self.assertNotIn("interpolate", source)
        self.assertNotIn("512", source)
        self.assertEqual(diagnostic.PATCH_COUNT, 4)
        self.assertEqual(diagnostic.EFFECTIVE_CONTEXT_STEPS, 32)
        self.assertEqual(diagnostic.TRAILING_UNPATCHED_STEPS, 1)

    def test_native_classifier_forwards_exact_tensor_and_mask(self) -> None:
        pipeline = _RecordingPipeline()
        model = diagnostic.NativeLogMomentClassifier(pipeline)
        logits = model(torch.ones(2, 13, 33))
        self.assertEqual(tuple(logits.shape), (2, 9))
        self.assertEqual(pipeline.recorded, [((2, 13, 33), (2, 33))])
        with self.assertRaisesRegex(ValueError, "native MOMENT input"):
            model(torch.ones(2, 35, 33))

    def test_strict_matrix_reports_representation_control(self) -> None:
        rows = []
        values = {
            "baseline": 0.20,
            "direct": 0.05,
            "pretrained": 0.206,
            "random": 0.199,
            "gate0": 0.20,
        }
        for fold_id in p11.FOLD_IDS:
            for repeat_id in range(len(p11.REPEAT_SEEDS)):
                for variant in p11.ABLATIONS:
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "repeat_id": repeat_id,
                            "seed": p11.REPEAT_SEEDS[repeat_id],
                            "variant": variant,
                            "metrics": {
                                metric: values[variant]
                                for metric in p11.PRIMARY_METRICS
                            },
                            "training": {
                                "gate_mean": (
                                    0.02
                                    if variant == "pretrained"
                                    else (
                                        0.015
                                        if variant == "random"
                                        else (
                                            0.0
                                            if variant == "gate0"
                                            else None
                                        )
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
        summary = diagnostic._summarize(
            rows,
            fold_ids=p11.FOLD_IDS,
            repeat_ids=tuple(range(len(p11.REPEAT_SEEDS))),
        )
        self.assertEqual(summary["evaluation"]["completed_cells"], 60)
        self.assertTrue(
            summary["representation_diagnostic"][
                "material_separation_detected"
            ]
        )
        self.assertEqual(
            summary["representation_diagnostic"]["state"],
            "MATERIAL_PRETRAINED_RANDOM_SEPARATION",
        )

    def test_legacy_artifacts_remain_byte_identical(self) -> None:
        self.assertEqual(
            diagnostic._verify_legacy_outputs(),
            diagnostic.LEGACY_P11_HASHES,
        )

    def test_artifact_verifier_detects_tampering(self) -> None:
        rows = []
        for variant in p11.ABLATIONS:
            rows.append(
                {
                    "fold_id": 0,
                    "repeat_id": 0,
                    "seed": p11.REPEAT_SEEDS[0],
                    "variant": variant,
                    "metrics": {
                        metric: 0.1 for metric in p11.PRIMARY_METRICS
                    },
                    "training": {
                        "gate_mean": (
                            0.01
                            if variant in {"pretrained", "random"}
                            else (0.0 if variant == "gate0" else None)
                        ),
                        "gate0_max_abs_error": 0.0,
                        "residual_contribution_mean_abs": (
                            0.01
                            if variant in {"pretrained", "random"}
                            else (0.0 if variant == "gate0" else None)
                        ),
                    },
                }
            )
        summary = diagnostic._summarize(
            rows,
            fold_ids=(0,),
            repeat_ids=(0,),
        )
        summary["preserved_p11_artifacts"] = diagnostic.LEGACY_P11_HASHES
        summary["representation"] = {
            "resampling": "none",
            "real_log_channels": 13,
            "observation_mask_channels": 0,
            "seismic_channels": 0,
        }
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "_tmp") as tmp:
            output = Path(tmp) / "native33"
            diagnostic._write_artifacts(
                rows=rows,
                summary=summary,
                output_dir=output,
            )
            verified = diagnostic.verify_artifacts(output)
            self.assertEqual(verified["rows"], 5)
            (output / "evidence.md").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                diagnostic.verify_artifacts(output)


if __name__ == "__main__":
    unittest.main()
