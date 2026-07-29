#!/usr/bin/env python3
"""P11 gated MOMENT residual fusion on the immutable lithofacies LOGO4 split.

The strong Stage-3 XGBoost logits remain the main route.  Cached frozen MOMENT
embeddings may contribute only through a bounded, fold-train-fitted residual:

    residual_logits = 2 * tanh(linear(moment_embeddings))
    fused_logits = baseline_logits + sigmoid(gate_logits) * residual_logits

Five development-only ablations are mandatory on every fold/seed pair:
``baseline``, ``direct``, ``pretrained``, ``random``, and ``gate0``.
``gate0`` is the exact zero-gate degeneration of the trained pretrained
residual model and must be bit-identical to ``baseline``.

The two-stage CLI keeps incompatible dependencies separate:

1. ``prepare-baseline`` runs with the approved tabular interpreter and verifies
   the archived XGBoost checkpoints/predictions before creating an ignored
   train/validation-logit bundle.
2. ``run`` uses the approved Torch interpreter and the pinned MOMENT snapshot.

No command accepts or opens frozen-test/known-holdout inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)


def _bootstrap_moment_source() -> None:
    for candidate in (
        Path("/mnt/data/yongan-admin-2/.cache/p8-foundation-deps"),
        Path("/mnt/data/yongan-admin-2/.cache/upstream/moment"),
    ):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))


_bootstrap_moment_source()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from _models.lithofacies.moment_depth import build_model as build_moment  # noqa: E402
from lithofacies_p5_stage3 import (  # noqa: E402
    FOLD_IDS,
    REPEAT_SEEDS,
    _fold_arrays,
    load_stage3_batch,
)
from p4_contract import (  # noqa: E402
    classification_metrics_from_logits,
    lithofacies_task_spec,
)
from p9_moment_effect import _inputs, _randomize_frozen_backbone  # noqa: E402


SCHEMA_VERSION = "lithofacies-p11-residual-fusion/v1"
BASELINE_BUNDLE_SCHEMA = "lithofacies-p11-baseline-logits/v1"
EXPECTED_SPLIT_HASH = (
    "a06375429f9e9cf380fb5cdebd7d0cb7b25d7a13d29522b8e2420f4dae1b4555"
)
EXPECTED_SNAPSHOT_SHA256 = (
    "1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825"
)
BASELINE_MODEL_ID = "xgboost_multisoftprob_window"
ABLATIONS = ("baseline", "direct", "pretrained", "random", "gate0")
UPDATES = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GATE_REGULARIZATION = 0.05
RESIDUAL_REGULARIZATION = 0.001
GATE_INITIAL_LOGIT = -4.0
MAX_RESIDUAL_LOGIT = 2.0
MIN_PROMOTION_DELTA = 0.005
MIN_PROMOTION_PAIR_WIN_FRACTION = 0.75
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p11_residual_fusion"
PRIMARY_METRICS = (
    "fixed_schema_macro_f1",
    "supported_class_macro_f1",
    "accuracy",
    "balanced_accuracy",
    "expected_calibration_error",
    "negative_log_likelihood",
    "multiclass_brier",
)
FORBIDDEN_PATH_MARKERS = ("test", "holdout", "frozen")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    for root in (PROJECT_ROOT, TRACK_DIR):
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return str(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ensure_development_only_paths(paths: Iterable[Path]) -> None:
    """Reject holdout-like path components before opening any input."""
    for raw_path in paths:
        path = Path(raw_path)
        lowered = [part.lower().replace("-", "_") for part in path.parts]
        for part in lowered:
            if any(marker in part for marker in FORBIDDEN_PATH_MARKERS):
                raise ValueError(f"forbidden holdout path: {path}")


def _selection(
    fold_ids: Sequence[int], repeat_ids: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    folds = tuple(int(value) for value in fold_ids)
    repeats = tuple(int(value) for value in repeat_ids)
    if not folds or any(value not in FOLD_IDS for value in folds):
        raise ValueError("fold selection must be a non-empty subset of strict LOGO4")
    valid_repeats = tuple(range(len(REPEAT_SEEDS)))
    if not repeats or any(value not in valid_repeats for value in repeats):
        raise ValueError("repeat selection is outside the three frozen seeds")
    if len(set(folds)) != len(folds) or len(set(repeats)) != len(repeats):
        raise ValueError("fold/repeat selections must not contain duplicates")
    return folds, repeats


def _baseline_rows(stage3_results: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in _read_jsonl(stage3_results):
        if row.get("model_id") != BASELINE_MODEL_ID:
            continue
        key = (int(row["fold_id"]), int(row["repeat_id"]))
        if (
            row.get("status") != "PASS"
            or row.get("rank_eligible") is not True
            or row.get("frozen_test_accessed") is not False
            or row.get("test_metrics_used") is not False
            or row.get("split_hash") != EXPECTED_SPLIT_HASH
        ):
            raise RuntimeError(f"illegal Stage-3 baseline cell: {key}")
        rows[key] = row
    return rows


def prepare_baseline_bundle(
    *,
    development_batch: Path,
    stage3_results: Path,
    checkpoint_dir: Path,
    prediction_dir: Path,
    output_bundle: Path,
    fold_ids: Sequence[int] = FOLD_IDS,
    repeat_ids: Sequence[int] = tuple(range(len(REPEAT_SEEDS))),
) -> dict[str, Any]:
    """Reconstruct and independently verify XGBoost train/validation logits."""
    ensure_development_only_paths(
        (development_batch, stage3_results, checkpoint_dir, prediction_dir)
    )
    folds, repeats = _selection(fold_ids, repeat_ids)
    arrays, batch_manifest = load_stage3_batch(development_batch)
    if (
        batch_manifest.get("split_hash") != EXPECTED_SPLIT_HASH
        or batch_manifest.get("frozen_test_accessed") is not False
    ):
        raise RuntimeError("development batch is not the immutable LOGO4 batch")
    result_rows = _baseline_rows(stage3_results)
    payload_arrays: dict[str, np.ndarray] = {}
    cells: list[dict[str, Any]] = []
    for fold_id in folds:
        fold = _fold_arrays(arrays, fold_id)
        for repeat_id in repeats:
            key = (fold_id, repeat_id)
            if key not in result_rows:
                raise RuntimeError(f"missing Stage-3 baseline cell: {key}")
            row = result_rows[key]
            seed = int(REPEAT_SEEDS[repeat_id])
            checkpoint = (
                checkpoint_dir
                / f"{BASELINE_MODEL_ID}__fold{fold_id}__repeat{repeat_id}.pkl"
            )
            prediction = (
                prediction_dir
                / f"{BASELINE_MODEL_ID}__fold{fold_id}__repeat{repeat_id}.npz"
            )
            if _sha256(checkpoint) != row["checkpoint"]["sha256"]:
                raise RuntimeError(f"baseline checkpoint hash mismatch: {key}")
            if _sha256(prediction) != row["oof_prediction"]["sha256"]:
                raise RuntimeError(f"baseline prediction hash mismatch: {key}")
            with checkpoint.open("rb") as handle:
                checkpoint_payload = pickle.load(handle)
            config = checkpoint_payload.get("config", {})
            if (
                checkpoint_payload.get("model_id") != BASELINE_MODEL_ID
                or int(config.get("rounds", 0)) != 40
                or int(config.get("max_depth", 0)) != 2
                or int(config.get("seed", -1)) != seed
            ):
                raise RuntimeError(f"baseline checkpoint contract changed: {key}")
            model = checkpoint_payload["model"]
            train_logits = np.asarray(
                model.predict_logits(
                    fold["p_train_well"], fold["p_train_seismic"]
                ),
                dtype=np.float32,
            )
            validation_logits = np.asarray(
                model.predict_logits(
                    fold["p_validation_well"], fold["p_validation_seismic"]
                ),
                dtype=np.float32,
            )
            with np.load(prediction, allow_pickle=False) as archived:
                if not np.array_equal(
                    archived["sample_ids"], fold["p_validation_ids"]
                ):
                    raise RuntimeError(f"baseline validation IDs changed: {key}")
                if not np.array_equal(
                    archived["labels"], fold["p_validation_labels"]
                ):
                    raise RuntimeError(f"baseline validation labels changed: {key}")
                if not np.array_equal(archived["logits"], validation_logits):
                    raise RuntimeError(f"baseline validation logits changed: {key}")
            observed = classification_metrics_from_logits(
                fold["p_validation_labels"], validation_logits
            )
            expected = float(
                row["validation_metrics"]["fixed_schema_macro_f1"]
            )
            if not math.isclose(
                float(observed["fixed_schema_macro_f1"]),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"baseline primary metric changed: {key}")
            prefix = f"f{fold_id}_r{repeat_id}"
            payload_arrays[f"{prefix}_train_logits"] = train_logits
            payload_arrays[f"{prefix}_validation_logits"] = validation_logits
            cells.append(
                {
                    "fold_id": fold_id,
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "checkpoint_sha256": row["checkpoint"]["sha256"],
                    "prediction_sha256": row["oof_prediction"]["sha256"],
                    "train_rows": int(len(train_logits)),
                    "validation_rows": int(len(validation_logits)),
                    "fixed_schema_macro_f1": expected,
                }
            )
    manifest = {
        "schema_version": BASELINE_BUNDLE_SCHEMA,
        "split_hash": EXPECTED_SPLIT_HASH,
        "development_batch_sha256": _sha256(development_batch),
        "stage3_results_sha256": _sha256(stage3_results),
        "model_id": BASELINE_MODEL_ID,
        "fold_ids": list(folds),
        "repeat_ids": list(repeats),
        "seeds": [int(REPEAT_SEEDS[index]) for index in repeats],
        "cells": cells,
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
    }
    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_bundle,
        **payload_arrays,
        manifest=np.asarray(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        ),
    )
    return {
        **manifest,
        "output_bundle": str(output_bundle),
        "output_bundle_sha256": _sha256(output_bundle),
    }


def _load_baseline_bundle(
    baseline_bundle: Path,
    *,
    development_batch_sha256: str,
    fold_ids: Sequence[int],
    repeat_ids: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(baseline_bundle, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    manifest = json.loads(str(arrays.pop("manifest").item()))
    if (
        manifest.get("schema_version") != BASELINE_BUNDLE_SCHEMA
        or manifest.get("split_hash") != EXPECTED_SPLIT_HASH
        or manifest.get("development_batch_sha256") != development_batch_sha256
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("known_holdout_accessed") is not False
    ):
        raise RuntimeError("baseline bundle violates the P11 development contract")
    cells = {
        (int(cell["fold_id"]), int(cell["repeat_id"])): cell
        for cell in manifest.get("cells", ())
    }
    for fold_id in fold_ids:
        for repeat_id in repeat_ids:
            key = (fold_id, repeat_id)
            prefix = f"f{fold_id}_r{repeat_id}"
            if (
                key not in cells
                or f"{prefix}_train_logits" not in arrays
                or f"{prefix}_validation_logits" not in arrays
            ):
                raise RuntimeError(f"baseline bundle is missing cell {key}")
    return arrays, manifest


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_moment(
    *,
    snapshot: Path,
    device: str,
    seed: int,
    random_init: bool,
) -> torch.nn.Module:
    _seed_all(seed)
    model = build_moment(
        lithofacies_task_spec(),
        snapshot_path=snapshot,
        device=device,
        freeze_encoder=True,
        freeze_embedder=True,
    )
    if random_init:
        _randomize_frozen_backbone(model, seed=seed)
    return model


class GatedEmbeddingResidual(torch.nn.Module):
    """Conservatively bounded residual head over cached MOMENT embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.normalization = torch.nn.LayerNorm(int(embedding_dim))
        self.residual_head = torch.nn.Linear(int(embedding_dim), 9)
        self.gate_logits = torch.nn.Parameter(
            torch.full((9,), GATE_INITIAL_LOGIT, dtype=torch.float32)
        )

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def forward(
        self,
        embeddings: torch.Tensor,
        baseline_logits: torch.Tensor,
        *,
        force_gate_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_logits = MAX_RESIDUAL_LOGIT * torch.tanh(
            self.residual_head(self.normalization(embeddings))
        )
        gate = (
            torch.zeros_like(self.gate_logits)
            if force_gate_zero
            else self.gate()
        )
        fused = baseline_logits + gate[None, :] * residual_logits
        return fused, gate, residual_logits


def _batch_indices(
    size: int, *, seed: int, updates: int = UPDATES
) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    for _ in range(updates):
        yield rng.choice(size, size=min(BATCH_SIZE, size), replace=False)


def _predict_direct(
    model: torch.nn.Module, values: np.ndarray, *, device: str
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), BATCH_SIZE):
            inputs = torch.as_tensor(
                values[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(inputs).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _extract_embeddings(
    model: torch.nn.Module, values: np.ndarray, *, device: str
) -> np.ndarray:
    """Run the real frozen MOMENT encoder and return mean-reduced embeddings."""
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), BATCH_SIZE):
            inputs = torch.as_tensor(
                values[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            if tuple(inputs.shape[1:]) != (35, 33):
                raise ValueError("MOMENT embedding input must be [B,35,33]")
            interpolated = F.interpolate(
                inputs, size=512, mode="linear", align_corners=False
            )
            mask = torch.ones(
                (inputs.shape[0], 512),
                dtype=torch.long,
                device=device,
            )
            result = model.pipeline.embed(
                x_enc=interpolated, input_mask=mask, reduction="mean"
            )
            embeddings = getattr(result, "embeddings", None)
            if (
                embeddings is None
                or embeddings.ndim != 2
                or embeddings.shape[0] != inputs.shape[0]
                or not bool(torch.isfinite(embeddings).all())
            ):
                shape = (
                    None if embeddings is None else tuple(embeddings.shape)
                )
                raise ValueError(f"invalid MOMENT embeddings: {shape}")
            outputs.append(embeddings.detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _embedding_cache(
    *,
    cache_path: Path,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    snapshot: Path,
    device: str,
    seed: int,
    random_init: bool,
    fold_id: int,
    repeat_id: int | None,
    development_batch_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    contract = {
        "schema_version": "lithofacies-p11-moment-embeddings/v1",
        "snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
        "development_batch_sha256": development_batch_sha256,
        "fold_id": int(fold_id),
        "repeat_id": None if repeat_id is None else int(repeat_id),
        "seed": int(seed),
        "random_init": bool(random_init),
        "reduction": "mean",
        "interpolated_length": 512,
        "train_rows": int(len(train_x)),
        "validation_rows": int(len(validation_x)),
    }
    contract_hash = _stable_hash(contract)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            train_embeddings = cached["train_embeddings"].copy()
            validation_embeddings = cached["validation_embeddings"].copy()
            manifest = json.loads(str(cached["manifest"].item()))
        if manifest.get("contract_hash") != contract_hash:
            raise RuntimeError(f"embedding cache contract changed: {cache_path}")
    else:
        model = _build_moment(
            snapshot=snapshot,
            device=device,
            seed=seed,
            random_init=random_init,
        )
        train_embeddings = _extract_embeddings(
            model, train_x, device=device
        )
        validation_embeddings = _extract_embeddings(
            model, validation_x, device=device
        )
        del model
        torch.cuda.empty_cache()
        manifest = {
            **contract,
            "contract_hash": contract_hash,
            "embedding_dim": int(train_embeddings.shape[1]),
            "real_pretrained_weights_loaded": not random_init,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            train_embeddings=train_embeddings,
            validation_embeddings=validation_embeddings,
            manifest=np.asarray(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True)
            ),
        )
    if (
        train_embeddings.ndim != 2
        or validation_embeddings.ndim != 2
        or train_embeddings.shape[0] != len(train_x)
        or validation_embeddings.shape[0] != len(validation_x)
        or train_embeddings.shape[1] != validation_embeddings.shape[1]
        or not np.isfinite(train_embeddings).all()
        or not np.isfinite(validation_embeddings).all()
    ):
        raise RuntimeError(f"invalid cached MOMENT embedding arrays: {cache_path}")
    manifest = {
        **manifest,
        "cache_path": _portable_path(cache_path),
        "cache_sha256": _sha256(cache_path),
    }
    return train_embeddings, validation_embeddings, manifest


def _train_direct(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    class_weights: np.ndarray,
    snapshot: Path,
    device: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    model = _build_moment(
        snapshot=snapshot, device=device, seed=seed, random_init=False
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    weights = torch.as_tensor(
        class_weights, dtype=torch.float32, device=device
    )
    losses: list[float] = []
    model.train()
    for indices in _batch_indices(len(train_y), seed=seed):
        inputs = torch.as_tensor(
            train_x[indices], dtype=torch.float32, device=device
        )
        labels = torch.as_tensor(
            train_y[indices], dtype=torch.long, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels, weight=weights)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("direct MOMENT loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    logits = _predict_direct(model, validation_x, device=device)
    del model, optimizer
    torch.cuda.empty_cache()
    return logits, {
        "updates": UPDATES,
        "first_train_loss": losses[0],
        "last_train_loss": losses[-1],
        "minimum_train_loss": min(losses),
        "duration_seconds": time.perf_counter() - started,
        "gate_mean": None,
        "gate_max": None,
        "random_init": False,
    }


def _predict_residual(
    model: GatedEmbeddingResidual,
    embeddings: np.ndarray,
    baseline_logits: np.ndarray,
    *,
    device: str,
    force_gate_zero: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs: list[np.ndarray] = []
    contributions: list[np.ndarray] = []
    observed_gate: np.ndarray | None = None
    model.eval()
    with torch.no_grad():
        for start in range(0, len(embeddings), BATCH_SIZE):
            embedding_batch = torch.as_tensor(
                embeddings[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            baseline = torch.as_tensor(
                baseline_logits[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            fused, gate, residual = model(
                embedding_batch,
                baseline,
                force_gate_zero=force_gate_zero,
            )
            outputs.append(fused.detach().cpu().numpy())
            contributions.append(
                (gate[None, :] * residual).detach().cpu().numpy()
            )
            observed_gate = gate.detach().cpu().numpy()
    assert observed_gate is not None
    return (
        np.concatenate(outputs).astype(np.float32, copy=False),
        observed_gate.astype(np.float32, copy=False),
        np.concatenate(contributions).astype(np.float32, copy=False),
    )


def _train_residual(
    *,
    train_embeddings: np.ndarray,
    train_y: np.ndarray,
    train_baseline_logits: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_baseline_logits: np.ndarray,
    class_weights: np.ndarray,
    device: str,
    seed: int,
    random_init: bool,
    embedding_evidence: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    _seed_all(seed)
    model = GatedEmbeddingResidual(train_embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    weights = torch.as_tensor(
        class_weights, dtype=torch.float32, device=device
    )
    losses: list[float] = []
    model.train()
    for indices in _batch_indices(len(train_y), seed=seed):
        embedding_batch = torch.as_tensor(
            train_embeddings[indices], dtype=torch.float32, device=device
        )
        baseline = torch.as_tensor(
            train_baseline_logits[indices],
            dtype=torch.float32,
            device=device,
        )
        labels = torch.as_tensor(
            train_y[indices], dtype=torch.long, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        fused, gate, residual = model(embedding_batch, baseline)
        loss = (
            F.cross_entropy(fused, labels, weight=weights)
            + GATE_REGULARIZATION * gate.mean()
            + RESIDUAL_REGULARIZATION
            * torch.square(gate[None, :] * residual).mean()
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("gated residual loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    logits, gate, contributions = _predict_residual(
        model,
        validation_embeddings,
        validation_baseline_logits,
        device=device,
        force_gate_zero=False,
    )
    gate0_logits, gate0, gate0_contributions = _predict_residual(
        model,
        validation_embeddings,
        validation_baseline_logits,
        device=device,
        force_gate_zero=True,
    )
    gate0_error = float(
        np.max(np.abs(gate0_logits - validation_baseline_logits))
    )
    if (
        gate0_error != 0.0
        or bool(np.any(gate0 != 0.0))
        or bool(np.any(gate0_contributions != 0.0))
    ):
        raise RuntimeError("gate0 is not the exact baseline degeneration")
    contribution_abs = np.abs(contributions)
    baseline_abs_mean = float(np.mean(np.abs(validation_baseline_logits)))
    evidence = {
        "updates": UPDATES,
        "first_train_loss": losses[0],
        "last_train_loss": losses[-1],
        "minimum_train_loss": min(losses),
        "duration_seconds": time.perf_counter() - started,
        "gate_mean": float(gate.mean()),
        "gate_max": float(gate.max()),
        "gate_min": float(gate.min()),
        "gate_values": [float(value) for value in gate],
        "gate0_max_abs_error": gate0_error,
        "residual_contribution_mean_abs": float(contribution_abs.mean()),
        "residual_contribution_max_abs": float(contribution_abs.max()),
        "residual_contribution_to_baseline_abs_ratio": (
            float(contribution_abs.mean() / baseline_abs_mean)
            if baseline_abs_mean > 0.0
            else 0.0
        ),
        "residual_logit_bound": MAX_RESIDUAL_LOGIT,
        "embedding_cache_path": embedding_evidence["cache_path"],
        "embedding_cache_sha256": embedding_evidence["cache_sha256"],
        "embedding_dim": int(train_embeddings.shape[1]),
        "embedding_random_init": bool(
            embedding_evidence["random_init"]
        ),
        "random_init": random_init,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return logits, gate0_logits, evidence


def _metric_subset(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    full = classification_metrics_from_logits(labels, logits)
    return {name: float(full[name]) for name in PRIMARY_METRICS}


def _result_row(
    *,
    fold_id: int,
    repeat_id: int,
    variant: str,
    labels: np.ndarray,
    logits: np.ndarray,
    training: Mapping[str, Any],
    baseline_cell: Mapping[str, Any],
    split_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "track_id": "lithofacies",
        "task_id": "gm09_genetic_facies_9class",
        "lane": "P",
        "variant": variant,
        "fold_id": int(fold_id),
        "repeat_id": int(repeat_id),
        "seed": int(REPEAT_SEEDS[repeat_id]),
        "train_samples": int(baseline_cell["train_rows"]),
        "validation_samples": int(len(labels)),
        "split_hash": split_hash,
        "metrics": _metric_subset(labels, logits),
        "training": dict(training),
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
        "rank_eligible": True,
    }


def _pair_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["fold_id"]), int(row["repeat_id"])


def _validate_pair_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(ABLATIONS):
        raise ValueError("each completed pair must contain all five ablations")
    if {str(row["variant"]) for row in rows} != set(ABLATIONS):
        raise ValueError("completed pair has the wrong ablation set")
    keys = {_pair_key(row) for row in rows}
    if len(keys) != 1:
        raise ValueError("completed pair mixes fold/repeat identities")
    for row in rows:
        if (
            row.get("split_hash") != EXPECTED_SPLIT_HASH
            or row.get("frozen_test_accessed") is not False
            or row.get("known_holdout_accessed") is not False
        ):
            raise ValueError("P11 row violates the development-only contract")


def summarize_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_ids: Sequence[int] = FOLD_IDS,
    repeat_ids: Sequence[int] = tuple(range(len(REPEAT_SEEDS))),
) -> dict[str, Any]:
    folds, repeats = _selection(fold_ids, repeat_ids)
    expected = {
        (fold_id, repeat_id, variant)
        for fold_id in folds
        for repeat_id in repeats
        for variant in ABLATIONS
    }
    observed = {
        (int(row["fold_id"]), int(row["repeat_id"]), str(row["variant"]))
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("result collection is not a complete strict ablation matrix")
    variants: dict[str, Any] = {}
    for variant in ABLATIONS:
        selected = [row for row in rows if row["variant"] == variant]
        metric_summary: dict[str, Any] = {}
        for metric in PRIMARY_METRICS:
            values = np.asarray(
                [row["metrics"][metric] for row in selected], dtype=np.float64
            )
            metric_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        gate_values = [
            float(row["training"]["gate_mean"])
            for row in selected
            if row["training"].get("gate_mean") is not None
        ]
        contribution_values = [
            float(row["training"]["residual_contribution_mean_abs"])
            for row in selected
            if row["training"].get("residual_contribution_mean_abs")
            is not None
        ]
        variants[variant] = {
            "cells": len(selected),
            "metrics": metric_summary,
            "gate_mean": (
                float(np.mean(gate_values)) if gate_values else None
            ),
            "residual_contribution_mean_abs": (
                float(np.mean(contribution_values))
                if contribution_values
                else None
            ),
        }
    baseline = variants["baseline"]["metrics"]["fixed_schema_macro_f1"]["mean"]
    pretrained = variants["pretrained"]["metrics"][
        "fixed_schema_macro_f1"
    ]["mean"]
    random = variants["random"]["metrics"]["fixed_schema_macro_f1"]["mean"]
    direct = variants["direct"]["metrics"]["fixed_schema_macro_f1"]["mean"]
    gate0 = variants["gate0"]["metrics"]["fixed_schema_macro_f1"]["mean"]
    paired = {
        (int(row["fold_id"]), int(row["repeat_id"]), str(row["variant"])): float(
            row["metrics"]["fixed_schema_macro_f1"]
        )
        for row in rows
    }
    pretrained_wins = sum(
        paired[(fold_id, repeat_id, "pretrained")]
        > paired[(fold_id, repeat_id, "baseline")]
        for fold_id in folds
        for repeat_id in repeats
    )
    gate0_errors = [
        float(row["training"].get("gate0_max_abs_error", 0.0))
        for row in rows
        if row["variant"] == "gate0"
    ]
    if not math.isclose(gate0, baseline, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("aggregate gate0 metric differs from baseline")
    paired_comparisons = len(folds) * len(repeats)
    promotion_checks = {
        "baseline_delta_at_least_0_005": (
            pretrained - baseline >= MIN_PROMOTION_DELTA
        ),
        "random_control_delta_at_least_0_005": (
            pretrained - random >= MIN_PROMOTION_DELTA
        ),
        "pair_win_fraction_at_least_0_75": (
            pretrained_wins / paired_comparisons
            >= MIN_PROMOTION_PAIR_WIN_FRACTION
        ),
    }
    promote = all(promotion_checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "evaluation": {
            "protocol": "strict_LOGO4_three_seed_development_only",
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_ids": list(folds),
            "repeat_ids": list(repeats),
            "seeds": [int(REPEAT_SEEDS[index]) for index in repeats],
            "ablation_order": list(ABLATIONS),
            "expected_cells": len(expected),
            "completed_cells": len(rows),
            "updates_per_trainable_lane": UPDATES,
            "batch_size": BATCH_SIZE,
            "frozen_test_accessed": False,
            "known_holdout_accessed": False,
        },
        "variants": variants,
        "comparison": {
            "pretrained_minus_baseline": pretrained - baseline,
            "pretrained_minus_random": pretrained - random,
            "pretrained_minus_direct": pretrained - direct,
            "gate0_minus_baseline": gate0 - baseline,
            "pretrained_pair_wins_over_baseline": int(pretrained_wins),
            "paired_comparisons": paired_comparisons,
            "pretrained_pair_win_fraction": (
                pretrained_wins / paired_comparisons
            ),
            "gate0_max_abs_logit_error": max(gate0_errors, default=0.0),
        },
        "decision": {
            "state": (
                "PROMOTE_RESIDUAL" if promote else "NON_BENEFICIAL_KEEP_BASELINE"
            ),
            "default_enabled": promote,
            "promotion_checks": promotion_checks,
            "minimum_absolute_macro_f1_delta": MIN_PROMOTION_DELTA,
            "minimum_pair_win_fraction": MIN_PROMOTION_PAIR_WIN_FRACTION,
        },
    }


def _write_figure(summary: Mapping[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [
        summary["variants"][variant]["metrics"]["fixed_schema_macro_f1"][
            "mean"
        ]
        for variant in ABLATIONS
    ]
    errors = [
        summary["variants"][variant]["metrics"]["fixed_schema_macro_f1"][
            "std"
        ]
        for variant in ABLATIONS
    ]
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#9D9DAA"]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    positions = np.arange(len(ABLATIONS))
    axis.bar(positions, values, yerr=errors, color=colors, capsize=4)
    axis.set_xticks(positions, ABLATIONS)
    axis.set_ylabel("fixed-schema macro-F1")
    axis.set_title(
        "Lithofacies P11 residual-fusion ablations\n"
        "strict LOGO4 × 3 seeds, development only"
    )
    axis.grid(axis="y", alpha=0.25)
    upper = max(values) + max(errors)
    axis.set_ylim(0.0, upper * 1.25 if upper > 0 else 1.0)
    for position, value in zip(positions, values):
        axis.text(
            position,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_evidence(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    lines = [
        "# P11 lithofacies gated residual-fusion evidence",
        "",
        "## Decision",
        "",
        f"State: `{summary['decision']['state']}`; default enabled: "
        f"`{str(summary['decision']['default_enabled']).lower()}`.",
        "",
        "## Strict LOGO4 ablations",
        "",
        "| variant | fixed-9 macro-F1 mean | std | accuracy | gate mean | mean abs residual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ABLATIONS:
        item = summary["variants"][variant]
        f1 = item["metrics"]["fixed_schema_macro_f1"]
        accuracy = item["metrics"]["accuracy"]["mean"]
        gate = item["gate_mean"]
        contribution = item["residual_contribution_mean_abs"]
        lines.append(
            f"| {variant} | {f1['mean']:.6f} | {f1['std']:.6f} | "
            f"{accuracy:.6f} | "
            f"{'—' if gate is None else f'{gate:.6f}'} | "
            f"{'—' if contribution is None else f'{contribution:.6f}'} |"
        )
    comparison = summary["comparison"]
    lines.extend(
        [
            "",
            "## Paired checks",
            "",
            f"- pretrained − baseline: "
            f"`{comparison['pretrained_minus_baseline']:+.6f}`.",
            f"- pretrained − random residual: "
            f"`{comparison['pretrained_minus_random']:+.6f}`.",
            f"- pretrained − direct MOMENT: "
            f"`{comparison['pretrained_minus_direct']:+.6f}`.",
            f"- pretrained wins over baseline in "
            f"`{comparison['pretrained_pair_wins_over_baseline']}/"
            f"{comparison['paired_comparisons']}` fold/seed pairs.",
            f"- gate0 − baseline: "
            f"`{comparison['gate0_minus_baseline']:+.6f}`; maximum "
            f"logit error `{comparison['gate0_max_abs_logit_error']:.1e}`.",
            "",
            "## Every fold and seed",
            "",
            "| fold | seed | baseline | direct | pretrained | random | gate0 | pretrained gate | mean abs residual |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_cell = {
        (int(row["fold_id"]), int(row["repeat_id"]), str(row["variant"])): row
        for row in rows
    }
    for fold_id in summary["evaluation"]["fold_ids"]:
        for repeat_id in summary["evaluation"]["repeat_ids"]:
            selected = {
                variant: by_cell[(fold_id, repeat_id, variant)]
                for variant in ABLATIONS
            }
            values = {
                variant: selected[variant]["metrics"][
                    "fixed_schema_macro_f1"
                ]
                for variant in ABLATIONS
            }
            training = selected["pretrained"]["training"]
            lines.append(
                f"| {fold_id} | {selected['baseline']['seed']} | "
                f"{values['baseline']:.6f} | {values['direct']:.6f} | "
                f"{values['pretrained']:.6f} | {values['random']:.6f} | "
                f"{values['gate0']:.6f} | {training['gate_mean']:.6f} | "
                f"{training['residual_contribution_mean_abs']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            "- Inputs are the immutable four development mother-family folds; "
            "F-5 and every holdout-like path are rejected before opening.",
            "- All trainable MOMENT lanes use the same 40-update, batch-32, "
            "fold-train-only class-weight contract.",
            "- `pretrained` and `random` have the same bounded embedding-head "
            "residual architecture; only the frozen MOMENT feature extractor "
            "initialization differs.",
            "- `direct` is the pretrained MOMENT classifier without XGBoost "
            "logits; `gate0` is the exact trained-residual degeneration.",
            "- This evidence is development-only and cannot authorize a new "
            "frozen-test read or a competition claim.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_artifacts(
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    evidence_path = output_dir / "evidence.md"
    figure_path = output_dir / "primary_metric.png"
    _write_jsonl(results_path, rows)
    _write_json(summary_path, summary)
    _write_evidence(summary, rows, evidence_path)
    _write_figure(summary, figure_path)
    artifacts = []
    for kind, path in (
        ("result_table", results_path),
        ("summary", summary_path),
        ("evidence_report", evidence_path),
        ("figure", figure_path),
    ):
        artifacts.append(
            {
                "kind": kind,
                "path": _portable_path(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest_path = output_dir / "artifact_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "lithofacies-p11-artifact-manifest/v1",
            "artifacts": artifacts,
            "frozen_test_accessed": False,
            "known_holdout_accessed": False,
        },
    )
    return {
        "results": results_path,
        "summary": summary_path,
        "evidence": evidence_path,
        "figure": figure_path,
        "manifest": manifest_path,
    }


def run_experiment(
    *,
    development_batch: Path,
    baseline_bundle: Path,
    snapshot: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda:0",
    fold_ids: Sequence[int] = FOLD_IDS,
    repeat_ids: Sequence[int] = tuple(range(len(REPEAT_SEEDS))),
    resume: bool = True,
) -> dict[str, Any]:
    ensure_development_only_paths(
        (development_batch, baseline_bundle, snapshot)
    )
    folds, repeats = _selection(fold_ids, repeat_ids)
    snapshot_weights = snapshot / "model.safetensors"
    if _sha256(snapshot_weights) != EXPECTED_SNAPSHOT_SHA256:
        raise RuntimeError("unexpected MOMENT snapshot weights")
    arrays, batch_manifest = load_stage3_batch(development_batch)
    if (
        batch_manifest.get("split_hash") != EXPECTED_SPLIT_HASH
        or batch_manifest.get("frozen_test_accessed") is not False
    ):
        raise RuntimeError("development batch is not strict LOGO4")
    batch_sha256 = _sha256(development_batch)
    baseline_arrays, baseline_manifest = _load_baseline_bundle(
        baseline_bundle,
        development_batch_sha256=batch_sha256,
        fold_ids=folds,
        repeat_ids=repeats,
    )
    run_contract = {
        "schema_version": SCHEMA_VERSION,
        "split_hash": EXPECTED_SPLIT_HASH,
        "development_batch_sha256": batch_sha256,
        "baseline_bundle_sha256": _sha256(baseline_bundle),
        "snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
        "fold_ids": list(folds),
        "repeat_ids": list(repeats),
        "seeds": [int(REPEAT_SEEDS[index]) for index in repeats],
        "ablations": list(ABLATIONS),
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gate_regularization": GATE_REGULARIZATION,
        "residual_regularization": RESIDUAL_REGULARIZATION,
        "max_residual_logit": MAX_RESIDUAL_LOGIT,
        "embedding_reduction": "mean",
        "embedding_cache": True,
    }
    contract_hash = _stable_hash(run_contract)
    runtime_dir = output_dir / "runtime"
    partial_path = runtime_dir / "partial_results.jsonl"
    contract_path = runtime_dir / "run_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("contract_hash") != contract_hash:
            raise RuntimeError("existing partial run has a different contract")
    else:
        _write_json(
            contract_path,
            {**run_contract, "contract_hash": contract_hash},
        )
    existing_rows = _read_jsonl(partial_path) if resume else []
    completed: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in existing_rows:
        completed.setdefault(_pair_key(row), []).append(row)
    for pair_rows in completed.values():
        _validate_pair_rows(pair_rows)
    all_rows = list(existing_rows)
    baseline_cells = {
        (int(cell["fold_id"]), int(cell["repeat_id"])): cell
        for cell in baseline_manifest["cells"]
    }
    started = time.perf_counter()
    for fold_id in folds:
        fold = _fold_arrays(arrays, fold_id)
        train_x = _inputs(fold["p_train_well"], fold["p_train_seismic"])
        validation_x = _inputs(
            fold["p_validation_well"], fold["p_validation_seismic"]
        )
        train_y = np.asarray(fold["p_train_labels"], dtype=np.int64)
        validation_y = np.asarray(
            fold["p_validation_labels"], dtype=np.int64
        )
        class_weights = np.asarray(fold["class_weights"], dtype=np.float32)
        pending_repeats = [
            repeat_id
            for repeat_id in repeats
            if (fold_id, repeat_id) not in completed
        ]
        if not pending_repeats:
            for repeat_id in repeats:
                print(
                    f"resume fold={fold_id} repeat={repeat_id}: "
                    "five ablations already complete",
                    flush=True,
                )
            continue
        (
            pretrained_train_embeddings,
            pretrained_validation_embeddings,
            pretrained_embedding_evidence,
        ) = _embedding_cache(
            cache_path=runtime_dir
            / "embeddings"
            / f"pretrained_fold{fold_id}.npz",
            train_x=train_x,
            validation_x=validation_x,
            snapshot=snapshot,
            device=device,
            seed=2693,
            random_init=False,
            fold_id=fold_id,
            repeat_id=None,
            development_batch_sha256=batch_sha256,
        )
        for repeat_id in repeats:
            key = (fold_id, repeat_id)
            if key in completed:
                print(
                    f"resume fold={fold_id} repeat={repeat_id}: "
                    "five ablations already complete",
                    flush=True,
                )
                continue
            seed = int(REPEAT_SEEDS[repeat_id])
            prefix = f"f{fold_id}_r{repeat_id}"
            train_baseline = baseline_arrays[f"{prefix}_train_logits"]
            validation_baseline = baseline_arrays[
                f"{prefix}_validation_logits"
            ]
            if train_baseline.shape != (len(train_y), 9) or (
                validation_baseline.shape != (len(validation_y), 9)
            ):
                raise RuntimeError(f"baseline logit shape changed: {key}")
            direct_logits, direct_training = _train_direct(
                train_x=train_x,
                train_y=train_y,
                validation_x=validation_x,
                class_weights=class_weights,
                snapshot=snapshot,
                device=device,
                seed=seed,
            )
            (
                random_train_embeddings,
                random_validation_embeddings,
                random_embedding_evidence,
            ) = _embedding_cache(
                cache_path=runtime_dir
                / "embeddings"
                / f"random_fold{fold_id}_repeat{repeat_id}.npz",
                train_x=train_x,
                validation_x=validation_x,
                snapshot=snapshot,
                device=device,
                seed=seed,
                random_init=True,
                fold_id=fold_id,
                repeat_id=repeat_id,
                development_batch_sha256=batch_sha256,
            )
            (
                pretrained_logits,
                gate0_logits,
                pretrained_training,
            ) = _train_residual(
                train_embeddings=pretrained_train_embeddings,
                train_y=train_y,
                train_baseline_logits=train_baseline,
                validation_embeddings=pretrained_validation_embeddings,
                validation_baseline_logits=validation_baseline,
                class_weights=class_weights,
                device=device,
                seed=seed,
                random_init=False,
                embedding_evidence=pretrained_embedding_evidence,
            )
            random_logits, _, random_training = _train_residual(
                train_embeddings=random_train_embeddings,
                train_y=train_y,
                train_baseline_logits=train_baseline,
                validation_embeddings=random_validation_embeddings,
                validation_baseline_logits=validation_baseline,
                class_weights=class_weights,
                device=device,
                seed=seed,
                random_init=True,
                embedding_evidence=random_embedding_evidence,
            )
            baseline_cell = baseline_cells[key]
            no_training = {
                "updates": 0,
                "first_train_loss": None,
                "last_train_loss": None,
                "minimum_train_loss": None,
                "duration_seconds": 0.0,
                "gate_mean": None,
                "gate_max": None,
                "residual_contribution_mean_abs": None,
                "residual_contribution_max_abs": None,
                "random_init": False,
            }
            gate0_training = {
                **pretrained_training,
                "updates": 0,
                "source": "trained_pretrained_residual_forced_to_exact_zero_gate",
                "gate_mean": 0.0,
                "gate_max": 0.0,
                "gate_min": 0.0,
                "gate_values": [0.0] * 9,
                "residual_contribution_mean_abs": 0.0,
                "residual_contribution_max_abs": 0.0,
                "residual_contribution_to_baseline_abs_ratio": 0.0,
            }
            pair_rows = [
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="baseline",
                    labels=validation_y,
                    logits=validation_baseline,
                    training=no_training,
                    baseline_cell=baseline_cell,
                    split_hash=EXPECTED_SPLIT_HASH,
                ),
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="direct",
                    labels=validation_y,
                    logits=direct_logits,
                    training=direct_training,
                    baseline_cell=baseline_cell,
                    split_hash=EXPECTED_SPLIT_HASH,
                ),
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="pretrained",
                    labels=validation_y,
                    logits=pretrained_logits,
                    training=pretrained_training,
                    baseline_cell=baseline_cell,
                    split_hash=EXPECTED_SPLIT_HASH,
                ),
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="random",
                    labels=validation_y,
                    logits=random_logits,
                    training=random_training,
                    baseline_cell=baseline_cell,
                    split_hash=EXPECTED_SPLIT_HASH,
                ),
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="gate0",
                    labels=validation_y,
                    logits=gate0_logits,
                    training=gate0_training,
                    baseline_cell=baseline_cell,
                    split_hash=EXPECTED_SPLIT_HASH,
                ),
            ]
            _validate_pair_rows(pair_rows)
            all_rows.extend(pair_rows)
            _write_jsonl(partial_path, all_rows)
            completed[key] = pair_rows
            print(
                f"complete fold={fold_id} repeat={repeat_id} seed={seed}: "
                f"baseline={pair_rows[0]['metrics']['fixed_schema_macro_f1']:.6f} "
                f"pretrained={pair_rows[2]['metrics']['fixed_schema_macro_f1']:.6f} "
                f"random={pair_rows[3]['metrics']['fixed_schema_macro_f1']:.6f} "
                f"gate={pretrained_training['gate_mean']:.6f}",
                flush=True,
            )
    summary = summarize_results(
        all_rows, fold_ids=folds, repeat_ids=repeats
    )
    summary["model"] = {
        "foundation_model": "AutonLab/MOMENT-1-base",
        "weights_sha256": EXPECTED_SNAPSHOT_SHA256,
        "real_pretrained_weights_loaded": True,
        "baseline_model": BASELINE_MODEL_ID,
        "residual_formula": (
            "baseline_logits + sigmoid(per_class_gate_logits) * "
            "2*tanh(linear(mean_reduced_cached_moment_embeddings))"
        ),
        "pretrained_random_same_architecture": True,
        "moment_embeddings_cached": True,
        "max_residual_logit": MAX_RESIDUAL_LOGIT,
    }
    summary["inputs"] = {
        "development_batch_sha256": batch_sha256,
        "baseline_bundle_sha256": _sha256(baseline_bundle),
        "stage3_results_sha256": baseline_manifest[
            "stage3_results_sha256"
        ],
    }
    summary["runtime"] = {
        "device": device,
        "duration_seconds_this_invocation": time.perf_counter() - started,
        "resumed_pairs": len(existing_rows) // len(ABLATIONS),
        "raw_predictions_persisted": False,
    }
    artifacts = _write_artifacts(
        rows=all_rows, summary=summary, output_dir=output_dir
    )
    return {
        "summary": summary,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }


def verify_artifacts(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "lithofacies-p11-artifact-manifest/v1"
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("known_holdout_accessed") is not False
    ):
        raise RuntimeError("invalid P11 artifact manifest")
    for artifact in manifest.get("artifacts", ()):
        path = PROJECT_ROOT / artifact["path"]
        if (
            not path.is_file()
            or _sha256(path) != artifact["sha256"]
            or path.stat().st_size != int(artifact["bytes"])
        ):
            raise RuntimeError(f"P11 artifact verification failed: {path}")
    rows = _read_jsonl(output_dir / "results.jsonl")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    evaluation = summary["evaluation"]
    recomputed = summarize_results(
        rows,
        fold_ids=evaluation["fold_ids"],
        repeat_ids=evaluation["repeat_ids"],
    )
    if recomputed["comparison"] != summary["comparison"]:
        raise RuntimeError("P11 summary comparison is not reproducible")
    from PIL import Image

    with Image.open(output_dir / "primary_metric.png") as image:
        image.verify()
    return {
        "status": "verified",
        "artifacts": len(manifest["artifacts"]),
        "rows": len(rows),
        "decision": summary["decision"],
    }


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(
        int(item.strip()) for item in value.split(",") if item.strip()
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-baseline")
    prepare.add_argument("--development-batch", type=Path, required=True)
    prepare.add_argument("--stage3-results", type=Path, required=True)
    prepare.add_argument("--checkpoint-dir", type=Path, required=True)
    prepare.add_argument("--prediction-dir", type=Path, required=True)
    prepare.add_argument("--output-bundle", type=Path, required=True)
    prepare.add_argument("--folds", type=_parse_ints, default=FOLD_IDS)
    prepare.add_argument(
        "--repeats",
        type=_parse_ints,
        default=tuple(range(len(REPEAT_SEEDS))),
    )

    run = subparsers.add_parser("run")
    run.add_argument("--development-batch", type=Path, required=True)
    run.add_argument("--baseline-bundle", type=Path, required=True)
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--folds", type=_parse_ints, default=FOLD_IDS)
    run.add_argument(
        "--repeats",
        type=_parse_ints,
        default=tuple(range(len(REPEAT_SEEDS))),
    )
    run.add_argument("--no-resume", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-baseline":
        payload = prepare_baseline_bundle(
            development_batch=args.development_batch,
            stage3_results=args.stage3_results,
            checkpoint_dir=args.checkpoint_dir,
            prediction_dir=args.prediction_dir,
            output_bundle=args.output_bundle,
            fold_ids=args.folds,
            repeat_ids=args.repeats,
        )
    elif args.command == "run":
        payload = run_experiment(
            development_batch=args.development_batch,
            baseline_bundle=args.baseline_bundle,
            snapshot=args.snapshot,
            output_dir=args.output_dir,
            device=args.device,
            fold_ids=args.folds,
            repeat_ids=args.repeats,
            resume=not args.no_resume,
        )
    else:
        payload = verify_artifacts(args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
