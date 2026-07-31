"""P4 fault baseline, HPO, lifecycle, checkpoint, and artifact adapters."""
from __future__ import annotations

import inspect
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACK_DIR = Path(__file__).resolve().parent
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "_code", TRACK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from _code.ml_framework.artifacts import (  # noqa: E402
    ArtifactManifest,
    atomic_write_json,
    hash_file,
    hash_payload,
)
from _code.ml_framework.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from _code.ml_framework.contracts import ModelBatch, ModelOutput, TaskSpec  # noqa: E402
from _code.ml_framework.hpo import HPOPlan, TrialResult, run_fixed_trials  # noqa: E402
from _code.ml_framework.lifecycle import ExperimentLifecycle, ExperimentState  # noqa: E402
from _code.ml_framework.run_layout import create_run_layout  # noqa: E402
from _code.ml_framework.seeding import SeedTree  # noqa: E402
from _code.ml_framework.trainer import TrainerState  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from p4_contract import TARGET_NAME, adapt_fault_arrays, fault_task_spec, validate_fault_batch  # noqa: E402
from p4_split import BufferedCVPlan  # noqa: E402


LEGACY_MODELS = (
    "fault_local_logistic",
    "fault_raw_logistic",
    "fault_local_huber",
)
OFFICIAL_MODE = "official"
PROXY_REGRESSION_MODE = "proxy_regression"


def fixed_baseline_configs() -> tuple[dict[str, Any], ...]:
    """Return deterministic simple baselines without importing Optuna."""

    return tuple(
        {
            "model_id": model_id,
            "supervision_mode": OFFICIAL_MODE,
            "metric_direction": "maximize",
        }
        for model_id in LEGACY_MODELS
    )


def fault_hpo_plan() -> HPOPlan:
    return HPOPlan(
        sanity_trials=8,
        pilot_trials=20,
        top_configs=3,
        confirm_seeds=3,
        sampler="random_then_tpe",
        pruner="nop",
        direction="maximize",
    )


def run_fault_fixed_trials(
    configs: Sequence[Mapping[str, Any]],
    objective: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
    *,
    root_seed: int,
    output_dir: Path,
) -> list[TrialResult]:
    """Run development-only fixed trials; this API intentionally has no test input."""

    return run_fixed_trials(
        configs,
        objective,
        root_seed=root_seed,
        output_dir=output_dir,
        metric_direction="maximize",
    )


def masked_fault_metrics(
    batch: ModelBatch,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Compute formal metrics on valid labels and proxy diagnostics separately."""

    validate_fault_batch(batch)
    if not 0.0 < threshold < 1.0:
        raise ValueError("fault threshold must lie strictly inside (0,1)")
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(batch.targets[TARGET_NAME], dtype=bool)
    valid = np.asarray(batch.target_masks[TARGET_NAME], dtype=bool)
    proxy = np.asarray(batch.input_masks["proxy_mask"], dtype=bool)
    if values.shape != targets.shape or not np.isfinite(values).all():
        raise ValueError("fault probabilities must be finite and match [B,D,H,W] targets")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("fault probabilities must lie in [0,1]")
    truth = targets[valid]
    scores = values[valid]
    if not len(truth) or len(np.unique(truth)) < 2:
        raise RuntimeError("formal binary metrics require audited positive and negative labels")
    prediction = scores >= threshold
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    proxy_scores = values[proxy]
    return {
        "formal": {
            "role": "formal_valid_label_only",
            "valid_label_count": int(valid.sum()),
            "average_precision": float(average_precision_score(truth.astype(np.uint8), scores)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "iou": iou,
            "threshold": threshold,
        },
        "proxy": {
            "role": "proxy_regression_only",
            "proxy_count": int(proxy.sum()),
            "mean_probability": float(proxy_scores.mean()) if len(proxy_scores) else None,
            "predicted_positive_fraction": (
                float(np.mean(proxy_scores >= threshold)) if len(proxy_scores) else None
            ),
        },
        "unknown_label_count": int((~valid).sum()),
    }


class LegacyFaultBaselineAdapter:
    """Wrap the existing probability baseline in the strict P4 logits envelope.

    Unknown targets always receive zero sample weight.  Proxy voxels are usable
    only through the explicitly named regression mode, never by the official
    mode or formal metrics.
    """

    def __init__(self, model_id: str = "fault_local_logistic", *, seed: int = 2693) -> None:
        if model_id not in LEGACY_MODELS:
            raise ValueError(f"unsupported fault legacy model {model_id!r}")
        self.model_id = model_id
        self.seed = seed
        self.model = discover_model("fault", model_id).build(fault_task_spec(), seed=seed)

    @staticmethod
    def _legacy_arrays(
        batch: ModelBatch,
        *,
        supervision_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | str]]:
        validate_fault_batch(batch)
        amplitude = np.asarray(batch.inputs["seismic_amplitude"], dtype=np.float32)
        if amplitude.shape[1] != 1:
            raise ValueError("legacy fault baselines support only depth=1 patch batches")
        targets = np.asarray(batch.targets[TARGET_NAME], dtype=np.uint8)
        valid = np.asarray(batch.target_masks[TARGET_NAME], dtype=bool)
        proxy = np.asarray(batch.input_masks["proxy_mask"], dtype=bool)
        if supervision_mode == OFFICIAL_MODE:
            supervision = valid
            metric_role = "formal"
        elif supervision_mode == PROXY_REGRESSION_MODE:
            supervision = valid | proxy
            metric_role = "proxy_regression_only"
        else:
            raise ValueError(f"unsupported supervision_mode={supervision_mode!r}")

        supervised_targets = targets[supervision]
        positives = int(np.sum(supervised_targets == 1))
        negatives = int(np.sum(supervised_targets == 0))
        if positives == 0 or negatives == 0:
            raise RuntimeError(
                f"{supervision_mode} baseline requires positive and negative supervised labels; "
                f"observed positive={positives}, negative={negatives}"
            )
        weights = supervision.astype(np.float32)
        return (
            amplitude,
            targets[:, 0],
            weights[:, 0],
            {
                "supervision_mode": supervision_mode,
                "metric_role": metric_role,
                "positive_labels": positives,
                "negative_labels": negatives,
                "unknown_zero_weight": int(np.sum(~supervision)),
            },
        )

    def train_batch(self, batch: ModelBatch, *, supervision_mode: str = OFFICIAL_MODE) -> dict[str, Any]:
        amplitudes, labels, weights, audit = self._legacy_arrays(
            batch,
            supervision_mode=supervision_mode,
        )
        loss = float(self.model.train_batch(amplitudes, labels, weights))
        if not math.isfinite(loss):
            raise RuntimeError("legacy fault baseline returned non-finite training loss")
        return {**audit, "loss": loss}

    def loss_batch(self, batch: ModelBatch, *, supervision_mode: str = OFFICIAL_MODE) -> dict[str, Any]:
        amplitudes, labels, weights, audit = self._legacy_arrays(
            batch,
            supervision_mode=supervision_mode,
        )
        loss = float(self.model.loss_batch(amplitudes, labels, weights))
        if not math.isfinite(loss):
            raise RuntimeError("legacy fault baseline returned non-finite validation loss")
        return {**audit, "loss": loss}

    def predict(self, batch: ModelBatch) -> ModelOutput:
        validate_fault_batch(batch)
        amplitudes = np.asarray(batch.inputs["seismic_amplitude"], dtype=np.float32)
        if amplitudes.shape[1] != 1:
            raise ValueError("legacy fault baselines support only depth=1 patch batches")
        probabilities_2d = np.asarray(self.model.predict_batch(amplitudes), dtype=np.float64)
        probabilities = probabilities_2d[:, None, :, :]
        if probabilities.shape != amplitudes.shape or not np.isfinite(probabilities).all():
            raise RuntimeError("legacy baseline produced invalid probability output")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise RuntimeError("legacy baseline probabilities must lie in [0,1]")
        clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        logits = np.log(clipped / (1.0 - clipped)).astype(np.float32)
        return ModelOutput(
            raw={TARGET_NAME: logits},
            transformed={TARGET_NAME: probabilities.astype(np.float32)},
            aux={
                "model_id": self.model_id,
                "adapter": "legacy_probability_to_logits",
                "scientific_role": "simple_baseline",
            },
        )


def tiny_verified_batch() -> ModelBatch:
    """Deterministic tiny batch containing explicit positive/negative supervision."""

    amplitudes = np.zeros((10, 1, 3, 3), dtype=np.float32)
    labels = np.zeros((10, 3, 3), dtype=np.uint8)
    verified_negative = np.zeros_like(labels, dtype=bool)
    positions: list[dict[str, int]] = []
    for index in range(10):
        amplitudes[index, 0, 1, 1] = 2.0 + 0.01 * index
        amplitudes[index, 0, 0, 0] = -2.0 - 0.01 * index
        labels[index, 1, 1] = 1
        verified_negative[index, 0, 0] = True
        positions.append({"inline": 100 + index * 20, "crossline": 200, "time_index": 300})
    return adapt_fault_arrays(
        amplitudes,
        labels,
        positions,
        ["fault"] * len(labels),
        verified_negative_mask=verified_negative,
    )


def run_tiny_baseline_smoke(*, seed: int = 2693, epochs: int = 12) -> dict[str, Any]:
    if epochs <= 0 or epochs > 50:
        raise ValueError("tiny smoke epochs must be in [1,50]")
    batch = tiny_verified_batch()
    adapter = LegacyFaultBaselineAdapter("fault_raw_logistic", seed=seed)
    losses = [adapter.train_batch(batch)["loss"] for _ in range(epochs)]
    output = adapter.predict(batch)
    probabilities = np.asarray(output.transformed[TARGET_NAME])
    positives = np.asarray(batch.targets[TARGET_NAME], dtype=bool)
    negatives = np.asarray(batch.input_masks["verified_negative_mask"], dtype=bool)
    positive_mean = float(probabilities[positives].mean())
    negative_mean = float(probabilities[negatives].mean())
    if not all(math.isfinite(value) for value in (*losses, positive_mean, negative_mean)):
        raise RuntimeError("tiny baseline smoke produced non-finite evidence")
    if positive_mean <= negative_mean:
        raise RuntimeError("tiny baseline failed to rank verified positives above verified negatives")
    return {
        "status": "passed",
        "model_id": adapter.model_id,
        "seed": seed,
        "epochs": epochs,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "positive_probability_mean": positive_mean,
        "verified_negative_probability_mean": negative_mean,
        "output_shape": list(probabilities.shape),
        "unknown_voxels_used_for_training": 0,
    }


@dataclass
class FaultRunContext:
    """Track wrapper around the shared one-way lifecycle and artifact envelope."""

    run_id: str
    root: Path
    task_spec: TaskSpec
    split_plan: BufferedCVPlan
    blind_audit: Mapping[str, Any]
    lifecycle: ExperimentLifecycle
    manifest: ArtifactManifest
    seed_report: Mapping[str, Any]
    environment: Mapping[str, Any]

    @classmethod
    def initialize(
        cls,
        run_root: Path,
        *,
        run_id: str,
        split_plan: BufferedCVPlan,
        blind_audit: Mapping[str, Any],
        root_seed: int = 2693,
    ) -> "FaultRunContext":
        if not run_id.strip():
            raise ValueError("run_id must not be blank")
        root = create_run_layout(run_root)
        spec = fault_task_spec()
        tree = SeedTree(root_seed)
        seed_report = {"root_seed": root_seed, "seed_tree": tree.to_dict()["derived"]}
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "shared_contract_commit": "954e06c8d6d5454891c77aa370d244ae0b7453fc",
            "fault_source_sha256": {
                name: hash_file(TRACK_DIR / name)
                for name in (
                    "p4_contract.py",
                    "p4_split.py",
                    "p4_workflow.py",
                    "p4_visualization.py",
                )
            },
        }
        atomic_write_json(root / "task_spec.json", spec.to_dict())
        atomic_write_json(root / "seed_report.json", seed_report)
        atomic_write_json(root / "environment.json", environment)
        atomic_write_json(root / "split_manifest.json", split_plan.to_dict())
        atomic_write_json(root / "blind_test_audit.json", blind_audit)

        lifecycle = ExperimentLifecycle(run_id)
        lifecycle.advance(
            ExperimentState.SPLIT_LOCKED,
            {
                "split_hash": split_plan.stable_hash(),
                "blind_test_status": blind_audit.get("status", "unknown"),
            },
        )
        atomic_write_json(root / "lifecycle.json", lifecycle.to_dict())
        manifest = ArtifactManifest(run_id, root)
        for relative, role in (
            ("task_spec.json", "task_spec"),
            ("seed_report.json", "seed_report"),
            ("environment.json", "environment"),
            ("split_manifest.json", "split_manifest"),
            ("blind_test_audit.json", "blind_test_audit"),
            ("lifecycle.json", "lifecycle"),
        ):
            manifest.register(relative, role=role)
        manifest.write()
        return cls(
            run_id,
            root,
            spec,
            split_plan,
            dict(blind_audit),
            lifecycle,
            manifest,
            seed_report,
            environment,
        )

    @property
    def split_hash(self) -> str:
        return self.split_plan.stable_hash()

    def _persist_lifecycle(self) -> None:
        atomic_write_json(self.root / "lifecycle.json", self.lifecycle.to_dict())
        self.manifest.register("lifecycle.json", role="lifecycle")
        self.manifest.write()

    def mark_smoke_passed(self, evidence: Mapping[str, Any]) -> None:
        self.lifecycle.advance(ExperimentState.SMOKE_PASSED, evidence)
        self._persist_lifecycle()

    def mark_cv_complete(self, evidence: Mapping[str, Any]) -> None:
        self.lifecycle.advance(ExperimentState.CV_COMPLETE, evidence)
        self._persist_lifecycle()

    def freeze_config(self, config: Mapping[str, Any]) -> str:
        config_hash = hash_payload(config)
        atomic_write_json(self.root / "run_config.json", dict(config))
        self.manifest.register("run_config.json", role="frozen_config", metadata={"sha256": config_hash})
        self.lifecycle.advance(ExperimentState.CONFIG_FROZEN, {"config_hash": config_hash})
        self._persist_lifecycle()
        return config_hash

    def save_complete_checkpoint(
        self,
        *,
        model_state: Any,
        optimizer_state: Any,
        scheduler_state: Any,
        scaler_state: Any,
        trainer_state: TrainerState,
        config_hash: str,
    ) -> tuple[Path, str]:
        path = self.root / "refit" / "checkpoint.pkl"
        save_checkpoint(
            path,
            epoch=max(trainer_state.next_epoch - 1, 0),
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            scaler_state=scaler_state,
            config_hash=config_hash,
            split_hash=self.split_hash,
            trainer_state=trainer_state.to_dict(),
            seed_report=self.seed_report,
            environment=self.environment,
            extra={
                "track_id": "fault",
                "task_id": self.task_spec.task_id,
                "mask_contract": "unknown excluded; proxy separate",
            },
            include_torch_rng=False,
        )
        loaded = load_checkpoint(path)
        recovered = TrainerState.from_dict(loaded["trainer_state"])
        if recovered != trainer_state:
            raise RuntimeError("fault checkpoint TrainerState did not round-trip")
        checkpoint_hash = hash_file(path)
        self.manifest.register("refit/checkpoint.pkl", role="refit_checkpoint")
        self.manifest.write()
        return path, checkpoint_hash

    def mark_refit_complete(self, checkpoint_hash: str) -> None:
        self.lifecycle.advance(
            ExperimentState.REFIT_COMPLETE,
            {"checkpoint_hash": checkpoint_hash},
        )
        self._persist_lifecycle()

    def consume_test_once(self, *, config_hash: str, checkpoint_hash: str) -> None:
        if self.blind_audit.get("status") != "frozen":
            raise RuntimeError("blind test is not frozen; test consumption is forbidden")
        self.lifecycle.consume_test(
            config_hash=config_hash,
            checkpoint_hash=checkpoint_hash,
            split_hash=self.split_hash,
        )
        self._persist_lifecycle()

    def verify_artifacts(self) -> None:
        self.manifest.verify()


def assert_development_interfaces_have_no_test_argument() -> None:
    """Local contract guard against accidental test access."""

    from p4_split import run_buffered_development_cv

    for function in (run_buffered_development_cv, run_fault_fixed_trials):
        forbidden = {name for name in inspect.signature(function).parameters if "test" in name.lower()}
        if forbidden:
            raise AssertionError(f"development API {function.__name__} exposes test inputs: {sorted(forbidden)}")
