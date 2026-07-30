#!/usr/bin/env python3
"""P11 cross-attention fusion of XGBoost leaves and native MOMENT tokens.

The archived Stage-3 XGBoost model remains the main prediction route. Its
40-round by nine-class leaf-index representation supplies the cross-attention
queries. Frozen MOMENT tokens from the 13 real well logs at native length 33
supply keys and values. The learned output is a bounded gated residual over a
train-prior-calibrated copy of the XGBoost logits.

Three development-only variants are reported for every strict LOGO4 fold and
frozen seed:

* ``baseline``: the committed Stage-3 XGBoost logits;
* ``prior_calibrated``: deterministic train-count shrinkage only;
* ``cross_attention``: prior calibration plus the learned fusion residual.

Encoder initialization is a CLI/run-contract parameter with ``pretrained`` and
``random`` choices. This run uses ``pretrained`` only; the switch is retained
for the next attribution ablation. No command accepts or opens frozen-test or
known-holdout inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

import lithofacies_p11_residual_fusion as p11  # noqa: E402

p11._bootstrap_moment_source()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


SCHEMA_VERSION = "lithofacies-p11-cross-attention-fusion/v1"
REPRESENTATION_SCHEMA = "lithofacies-p11-xgboost-leaves/v1"
TOKEN_CACHE_SCHEMA = "lithofacies-p11-moment-token-cache/v1"
ARTIFACT_SCHEMA = "lithofacies-p11-cross-attention-artifacts/v1"
VARIANTS = ("baseline", "prior_calibrated", "cross_attention")
ENCODER_INITS = ("pretrained", "random")
XGB_ROUNDS = 40
NUM_CLASSES = 9
LEAF_VOCAB_SIZE = 7
MOMENT_CHANNELS = 13
MOMENT_PATCHES = 4
MOMENT_TOKENS = MOMENT_CHANNELS * MOMENT_PATCHES
MOMENT_DIM = 768
ATTENTION_DIM = 48
ATTENTION_HEADS = 4
UPDATES = 100
BATCH_SIZE = 32
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-3
GATE_INITIAL_LOGIT = -4.0
GATE_REGULARIZATION = 0.02
CONTRIBUTION_REGULARIZATION = 0.005
MAX_RESIDUAL_LOGIT = 2.0
PRIOR_SHRINKAGE = 0.25
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p11_cross_attention_fusion"

PRESERVED_OUTPUTS = {
    "_outputs/p11_residual_fusion/artifact_manifest.json": (
        "d81c3dc0647020186004c3630825d732e8ad8d18c975131c20ec14be0e97be14"
    ),
    "_outputs/p11_residual_fusion/evidence.md": (
        "eac5c4e31c86dcd080993fc82fee5958b41402edc7dfee666f97b6c80bc2d1f6"
    ),
    "_outputs/p11_residual_fusion/primary_metric.png": (
        "c3a82df5905027027078c9545dc6afe0c74c9dd6174775a59cbc72bc32c739db"
    ),
    "_outputs/p11_residual_fusion/results.jsonl": (
        "9c73c5ead55a4ee3e472368dbfdadc811ce92da5a2ffa321601cc23b550e7e3c"
    ),
    "_outputs/p11_residual_fusion/summary.json": (
        "86d1cda65a9e67e92dff1278fe6bed423fd5d4a8de918547d20eac1b60a75466"
    ),
    "_outputs/p11_clean_well_native33/artifact_manifest.json": (
        "5e1d5b024a78313acd2953edb5b78ccc87c0969d1b706dc7f34b6b1df6c7ca06"
    ),
    "_outputs/p11_clean_well_native33/evidence.md": (
        "420f778568a156b2d0f7c45cafc97d64ae75afeb4217cecb93becb57a1d24e5d"
    ),
    "_outputs/p11_clean_well_native33/primary_metric.png": (
        "5710fad16b433d8a1e58974cb3d7adae361a648bab0b7a7907b25b70941519e1"
    ),
    "_outputs/p11_clean_well_native33/results.jsonl": (
        "b239ed0841a1e09365d1c77deabe912ce8d55bb93904b9e1f27c75fb619b6a59"
    ),
    "_outputs/p11_clean_well_native33/summary.json": (
        "d6058f791cf40f7597e0b820c5379df69e7f1e581f686b71cd497d547cce71ab"
    ),
}
DEVELOPMENT_EXPLORATION = {
    "scope": "development_LOGO_only_not_an_unbiased_holdout_estimate",
    "xgboost_seed_logit_ensemble": {
        "fixed_schema_macro_f1": 0.19493770207563763,
        "delta_from_baseline": 0.0,
        "retained": False,
    },
    "train_prior_shrinkage_candidates": [
        {
            "shrinkage": 0.20,
            "fixed_schema_macro_f1": 0.20137523464754703,
        },
        {
            "shrinkage": 0.25,
            "fixed_schema_macro_f1": 0.20218697566969132,
        },
        {
            "shrinkage": 0.30,
            "fixed_schema_macro_f1": 0.20205804337061503,
        },
    ],
    "depth_logit_smoothing": {
        "radius_candidates": [1, 2, 3, 4, 6, 8],
        "mix_candidates": [0.25, 0.50, 0.75, 1.0],
        "best_standalone_delta_from_baseline": 0.0038213484963086075,
        "retained": False,
        "reason": "unstable_when_combined_with_prior_calibration",
    },
    "cross_attention_freeze_point": {
        "fold_id": 0,
        "repeat_id": 0,
        "baseline_fixed_schema_macro_f1": 0.2135804514277343,
        "prior_calibrated_fixed_schema_macro_f1": 0.22710616915935283,
        "cross_attention_fixed_schema_macro_f1": 0.22821390298494457,
        "action": "froze_architecture_and_training_constants_before_full_matrix",
    },
}


def _verify_preserved_outputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in PRESERVED_OUTPUTS.items():
        path = TRACK_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"committed P11 evidence is missing: {path}")
        digest = p11._sha256(path)
        if digest != expected:
            raise RuntimeError(f"committed P11 evidence changed: {path}")
        observed[relative] = digest
    return observed


def _xgboost_features(well: np.ndarray, seismic: np.ndarray) -> np.ndarray:
    well_values = np.asarray(well, dtype=np.float32)
    seismic_values = np.asarray(seismic, dtype=np.float32)
    if (
        well_values.ndim != 3
        or tuple(well_values.shape[1:]) != (26, 33)
        or seismic_values.ndim != 4
        or tuple(seismic_values.shape[1:]) != (3, 3, 33)
        or len(well_values) != len(seismic_values)
    ):
        raise ValueError("XGBoost representation inputs have changed")
    features = np.concatenate(
        (
            well_values.reshape(len(well_values), -1),
            seismic_values.reshape(len(seismic_values), -1),
        ),
        axis=1,
    )
    if features.shape[1] != 1155 or not np.isfinite(features).all():
        raise ValueError(f"invalid XGBoost feature matrix: {features.shape}")
    return np.ascontiguousarray(features)


def _leaf_ids(booster: Any, features: np.ndarray) -> np.ndarray:
    import xgboost

    raw = np.asarray(
        booster.predict(
            xgboost.DMatrix(features),
            pred_leaf=True,
            strict_shape=True,
        )
    )
    if raw.shape != (len(features), XGB_ROUNDS, NUM_CLASSES, 1):
        raise ValueError(f"unexpected XGBoost leaf tensor: {raw.shape}")
    squeezed = raw[..., 0]
    rounded = np.rint(squeezed)
    if (
        not np.array_equal(squeezed, rounded)
        or rounded.min(initial=0) < 0
        or rounded.max(initial=0) >= LEAF_VOCAB_SIZE
    ):
        raise ValueError("XGBoost leaf IDs exceed the frozen depth-two vocabulary")
    return rounded.astype(np.uint8)


def prepare_representation_bundle(
    *,
    development_batch: Path,
    baseline_bundle: Path,
    stage3_results: Path,
    checkpoint_dir: Path,
    output_bundle: Path,
    fold_ids: Sequence[int] = p11.FOLD_IDS,
    repeat_ids: Sequence[int] = tuple(range(len(p11.REPEAT_SEEDS))),
) -> dict[str, Any]:
    """Verify archived boosters and persist ignored leaf-index representations."""
    p11.ensure_development_only_paths(
        (
            development_batch,
            baseline_bundle,
            stage3_results,
            checkpoint_dir,
        )
    )
    folds, repeats = p11._selection(fold_ids, repeat_ids)
    arrays, batch_manifest = p11.load_stage3_batch(development_batch)
    if (
        batch_manifest.get("split_hash") != p11.EXPECTED_SPLIT_HASH
        or batch_manifest.get("frozen_test_accessed") is not False
    ):
        raise RuntimeError("development batch is not strict LOGO4")
    batch_sha256 = p11._sha256(development_batch)
    _, baseline_manifest = p11._load_baseline_bundle(
        baseline_bundle,
        development_batch_sha256=batch_sha256,
        fold_ids=folds,
        repeat_ids=repeats,
    )
    if baseline_manifest.get("stage3_results_sha256") != p11._sha256(
        stage3_results
    ):
        raise RuntimeError("Stage-3 results do not match the baseline bundle")
    baseline_cells = {
        (int(cell["fold_id"]), int(cell["repeat_id"])): cell
        for cell in baseline_manifest["cells"]
    }
    payload: dict[str, np.ndarray] = {}
    cells: list[dict[str, Any]] = []
    for fold_id in folds:
        fold = p11._fold_arrays(arrays, fold_id)
        train_features = _xgboost_features(
            fold["p_train_well"],
            fold["p_train_seismic"],
        )
        validation_features = _xgboost_features(
            fold["p_validation_well"],
            fold["p_validation_seismic"],
        )
        for repeat_id in repeats:
            key = (fold_id, repeat_id)
            cell = baseline_cells[key]
            checkpoint = (
                checkpoint_dir
                / f"{p11.BASELINE_MODEL_ID}__fold{fold_id}__repeat{repeat_id}.pkl"
            )
            if p11._sha256(checkpoint) != cell["checkpoint_sha256"]:
                raise RuntimeError(f"baseline checkpoint hash mismatch: {key}")
            with checkpoint.open("rb") as handle:
                checkpoint_payload = pickle.load(handle)
            config = checkpoint_payload.get("config", {})
            if (
                checkpoint_payload.get("model_id") != p11.BASELINE_MODEL_ID
                or int(config.get("rounds", 0)) != XGB_ROUNDS
                or int(config.get("max_depth", 0)) != 2
                or int(config.get("seed", -1))
                != int(p11.REPEAT_SEEDS[repeat_id])
            ):
                raise RuntimeError(f"baseline checkpoint contract changed: {key}")
            booster = checkpoint_payload["model"].booster
            prefix = f"f{fold_id}_r{repeat_id}"
            train_leaf_ids = _leaf_ids(booster, train_features)
            validation_leaf_ids = _leaf_ids(booster, validation_features)
            payload[f"{prefix}_train_leaf_ids"] = train_leaf_ids
            payload[f"{prefix}_validation_leaf_ids"] = validation_leaf_ids
            cells.append(
                {
                    "fold_id": fold_id,
                    "repeat_id": repeat_id,
                    "seed": int(p11.REPEAT_SEEDS[repeat_id]),
                    "checkpoint_sha256": cell["checkpoint_sha256"],
                    "train_rows": int(len(train_leaf_ids)),
                    "validation_rows": int(len(validation_leaf_ids)),
                    "leaf_min": int(train_leaf_ids.min(initial=0)),
                    "leaf_max": int(
                        max(
                            train_leaf_ids.max(initial=0),
                            validation_leaf_ids.max(initial=0),
                        )
                    ),
                }
            )
    manifest = {
        "schema_version": REPRESENTATION_SCHEMA,
        "split_hash": p11.EXPECTED_SPLIT_HASH,
        "development_batch_sha256": batch_sha256,
        "baseline_bundle_sha256": p11._sha256(baseline_bundle),
        "stage3_results_sha256": p11._sha256(stage3_results),
        "model_id": p11.BASELINE_MODEL_ID,
        "representation": "booster_leaf_indices",
        "shape_per_row": [XGB_ROUNDS, NUM_CLASSES],
        "leaf_vocab_size": LEAF_VOCAB_SIZE,
        "fold_ids": list(folds),
        "repeat_ids": list(repeats),
        "cells": cells,
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
    }
    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_bundle,
        **payload,
        manifest=np.asarray(json.dumps(manifest, sort_keys=True)),
    )
    return {
        **manifest,
        "output_bundle": p11._portable_path(output_bundle),
        "output_bundle_sha256": p11._sha256(output_bundle),
    }


def _load_representation_bundle(
    path: Path,
    *,
    development_batch_sha256: str,
    baseline_bundle_sha256: str,
    fold_ids: Sequence[int],
    repeat_ids: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    manifest = json.loads(str(arrays.pop("manifest").item()))
    if (
        manifest.get("schema_version") != REPRESENTATION_SCHEMA
        or manifest.get("split_hash") != p11.EXPECTED_SPLIT_HASH
        or manifest.get("development_batch_sha256")
        != development_batch_sha256
        or manifest.get("baseline_bundle_sha256") != baseline_bundle_sha256
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("known_holdout_accessed") is not False
    ):
        raise RuntimeError("XGBoost representation bundle violates the contract")
    for fold_id in fold_ids:
        for repeat_id in repeat_ids:
            prefix = f"f{fold_id}_r{repeat_id}"
            for split in ("train", "validation"):
                key = f"{prefix}_{split}_leaf_ids"
                if key not in arrays:
                    raise RuntimeError(f"representation bundle is missing {key}")
                values = arrays[key]
                if (
                    values.ndim != 3
                    or tuple(values.shape[1:]) != (XGB_ROUNDS, NUM_CLASSES)
                    or values.dtype != np.uint8
                    or values.max(initial=0) >= LEAF_VOCAB_SIZE
                ):
                    raise RuntimeError(f"invalid leaf representation: {key}")
    return arrays, manifest


def _build_encoder(
    *,
    snapshot: Path,
    device: str,
    seed: int,
    encoder_init: str,
) -> Any:
    if encoder_init not in ENCODER_INITS:
        raise ValueError(f"encoder_init must be one of {ENCODER_INITS}")
    import lithofacies_p11_clean_well_native33 as native33

    return native33._build_native_moment(
        snapshot=snapshot,
        device=device,
        seed=seed,
        random_init=encoder_init == "random",
    )


def _extract_moment_tokens(
    model: torch.nn.Module,
    values: np.ndarray,
    *,
    device: str,
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
            if tuple(inputs.shape[1:]) != (MOMENT_CHANNELS, 33):
                raise ValueError("MOMENT token input must be [B,13,33]")
            mask = torch.ones(
                (inputs.shape[0], 33),
                dtype=torch.long,
                device=device,
            )
            result = model.pipeline.embed(
                x_enc=inputs,
                input_mask=mask,
                reduction="none",
            )
            embeddings = getattr(result, "embeddings", None)
            if embeddings is None or tuple(embeddings.shape[1:]) != (
                MOMENT_CHANNELS,
                MOMENT_PATCHES,
                MOMENT_DIM,
            ):
                shape = None if embeddings is None else tuple(embeddings.shape)
                raise ValueError(f"unexpected MOMENT token tensor: {shape}")
            tokens = embeddings.reshape(len(inputs), MOMENT_TOKENS, MOMENT_DIM)
            if not bool(torch.isfinite(tokens).all()):
                raise ValueError("MOMENT token tensor contains non-finite values")
            outputs.append(tokens.detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _clean_log_inputs(well: np.ndarray) -> np.ndarray:
    values = np.asarray(well, dtype=np.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != (26, 33):
        raise ValueError(f"well tensor must be [B,26,33], got {values.shape}")
    clean = values[:, :MOMENT_CHANNELS, :]
    if not np.isfinite(clean).all():
        raise ValueError("clean MOMENT log input contains non-finite values")
    return np.ascontiguousarray(clean)


def _moment_token_cache(
    *,
    cache_path: Path,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    snapshot: Path,
    device: str,
    seed: int,
    encoder_init: str,
    fold_id: int,
    development_batch_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    contract = {
        "schema_version": TOKEN_CACHE_SCHEMA,
        "snapshot_sha256": p11.EXPECTED_SNAPSHOT_SHA256,
        "development_batch_sha256": development_batch_sha256,
        "fold_id": int(fold_id),
        "seed": int(seed),
        "encoder_init": encoder_init,
        "input_shape": [MOMENT_CHANNELS, 33],
        "token_shape": [MOMENT_TOKENS, MOMENT_DIM],
        "reduction": "none",
        "resampling": "none",
        "train_rows": int(len(train_x)),
        "validation_rows": int(len(validation_x)),
    }
    contract_hash = p11._stable_hash(contract)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            train_tokens = cached["train_tokens"].copy()
            validation_tokens = cached["validation_tokens"].copy()
            manifest = json.loads(str(cached["manifest"].item()))
        if manifest.get("contract_hash") != contract_hash:
            raise RuntimeError(f"MOMENT token cache contract changed: {cache_path}")
    else:
        model = _build_encoder(
            snapshot=snapshot,
            device=device,
            seed=seed,
            encoder_init=encoder_init,
        )
        train_tokens = _extract_moment_tokens(model, train_x, device=device)
        validation_tokens = _extract_moment_tokens(
            model,
            validation_x,
            device=device,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        manifest = {
            **contract,
            "contract_hash": contract_hash,
            "real_pretrained_weights_loaded": encoder_init == "pretrained",
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
            manifest=np.asarray(json.dumps(manifest, sort_keys=True)),
        )
    if (
        tuple(train_tokens.shape[1:]) != (MOMENT_TOKENS, MOMENT_DIM)
        or tuple(validation_tokens.shape[1:]) != (MOMENT_TOKENS, MOMENT_DIM)
        or train_tokens.shape[0] != len(train_x)
        or validation_tokens.shape[0] != len(validation_x)
        or not np.isfinite(train_tokens).all()
        or not np.isfinite(validation_tokens).all()
    ):
        raise RuntimeError(f"invalid MOMENT token cache: {cache_path}")
    evidence = {
        **manifest,
        "cache_path": p11._portable_path(cache_path),
        "cache_sha256": p11._sha256(cache_path),
    }
    return train_tokens, validation_tokens, evidence


def _prior_calibrate(
    logits: np.ndarray,
    class_counts: np.ndarray,
    *,
    shrinkage: float = PRIOR_SHRINKAGE,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(logits, dtype=np.float32)
    counts = np.asarray(class_counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != NUM_CLASSES:
        raise ValueError("baseline logits must be [B,9]")
    if counts.shape != (NUM_CLASSES,) or np.any(counts < 0):
        raise ValueError("class counts must be a non-negative fixed-nine vector")
    log_counts = np.log(np.maximum(counts, 1.0))
    bias = float(shrinkage) * (log_counts - log_counts.mean())
    calibrated = values.astype(np.float64) + bias[None, :]
    return calibrated.astype(np.float32), bias.astype(np.float32)


class XGBMomentCrossAttention(torch.nn.Module):
    """Leaf-token queries attending to native MOMENT log tokens."""

    def __init__(
        self,
        *,
        moment_dim: int = MOMENT_DIM,
        attention_dim: int = ATTENTION_DIM,
        heads: int = ATTENTION_HEADS,
    ) -> None:
        super().__init__()
        if attention_dim % heads:
            raise ValueError("attention_dim must be divisible by heads")
        self.leaf_embedding = torch.nn.Embedding(
            LEAF_VOCAB_SIZE,
            attention_dim,
        )
        self.round_embedding = torch.nn.Embedding(XGB_ROUNDS, attention_dim)
        self.class_embedding = torch.nn.Embedding(NUM_CLASSES, attention_dim)
        self.moment_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(moment_dim),
            torch.nn.Linear(moment_dim, attention_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(attention_dim),
        )
        self.logit_projection = torch.nn.Linear(1, attention_dim)
        self.query_norm = torch.nn.LayerNorm(attention_dim)
        self.cross_attention = torch.nn.MultiheadAttention(
            attention_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.fusion_head = torch.nn.Sequential(
            torch.nn.LayerNorm(attention_dim * 4),
            torch.nn.Linear(attention_dim * 4, attention_dim),
            torch.nn.GELU(),
            torch.nn.Linear(attention_dim, 1),
        )
        self.gate_logits = torch.nn.Parameter(
            torch.full((NUM_CLASSES,), GATE_INITIAL_LOGIT)
        )

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def forward(
        self,
        moment_tokens: torch.Tensor,
        leaf_ids: torch.Tensor,
        baseline_logits: torch.Tensor,
        *,
        force_gate_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(moment_tokens)
        if tuple(moment_tokens.shape) != (
            batch_size,
            MOMENT_TOKENS,
            MOMENT_DIM,
        ):
            raise ValueError("moment_tokens must be [B,52,768]")
        if tuple(leaf_ids.shape) != (
            batch_size,
            XGB_ROUNDS,
            NUM_CLASSES,
        ):
            raise ValueError("leaf_ids must be [B,40,9]")
        if tuple(baseline_logits.shape) != (batch_size, NUM_CLASSES):
            raise ValueError("baseline_logits must be [B,9]")
        if torch.any(leaf_ids < 0) or torch.any(
            leaf_ids >= LEAF_VOCAB_SIZE
        ):
            raise ValueError("leaf_ids exceed the depth-two vocabulary")
        class_ids = torch.arange(NUM_CLASSES, device=leaf_ids.device)
        round_ids = torch.arange(XGB_ROUNDS, device=leaf_ids.device)
        leaves = leaf_ids.permute(0, 2, 1)
        queries = (
            self.leaf_embedding(leaves)
            + self.class_embedding(class_ids)[None, :, None, :]
            + self.round_embedding(round_ids)[None, None, :, :]
        )
        queries = self.query_norm(queries)
        flat_queries = queries.reshape(
            batch_size,
            NUM_CLASSES * XGB_ROUNDS,
            -1,
        )
        projected_moment = self.moment_projection(moment_tokens)
        attended, _ = self.cross_attention(
            flat_queries,
            projected_moment,
            projected_moment,
            need_weights=False,
        )
        query_pool = queries.mean(dim=2)
        attended_pool = attended.reshape(
            batch_size,
            NUM_CLASSES,
            XGB_ROUNDS,
            -1,
        ).mean(dim=2)
        logit_token = (
            self.logit_projection(baseline_logits[..., None])
            + self.class_embedding(class_ids)[None, :, :]
        )
        fused_features = torch.cat(
            (
                query_pool,
                attended_pool,
                logit_token,
                query_pool * attended_pool,
            ),
            dim=-1,
        )
        residual = MAX_RESIDUAL_LOGIT * torch.tanh(
            self.fusion_head(fused_features).squeeze(-1)
        )
        gate = (
            torch.zeros_like(self.gate_logits)
            if force_gate_zero
            else self.gate()
        )
        contribution = gate[None, :] * residual
        return baseline_logits + contribution, gate, residual, contribution


def _predict_fusion(
    model: XGBMomentCrossAttention,
    moment_tokens: np.ndarray,
    leaf_ids: np.ndarray,
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
        for start in range(0, len(moment_tokens), BATCH_SIZE):
            moment = torch.as_tensor(
                moment_tokens[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            leaves = torch.as_tensor(
                leaf_ids[start : start + BATCH_SIZE],
                dtype=torch.long,
                device=device,
            )
            baseline = torch.as_tensor(
                baseline_logits[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            fused, gate, _, contribution = model(
                moment,
                leaves,
                baseline,
                force_gate_zero=force_gate_zero,
            )
            outputs.append(fused.detach().cpu().numpy())
            contributions.append(contribution.detach().cpu().numpy())
            observed_gate = gate.detach().cpu().numpy()
    assert observed_gate is not None
    return (
        np.concatenate(outputs).astype(np.float32, copy=False),
        observed_gate.astype(np.float32, copy=False),
        np.concatenate(contributions).astype(np.float32, copy=False),
    )


def _train_fusion(
    *,
    train_moment_tokens: np.ndarray,
    train_leaf_ids: np.ndarray,
    train_baseline_logits: np.ndarray,
    train_y: np.ndarray,
    validation_moment_tokens: np.ndarray,
    validation_leaf_ids: np.ndarray,
    validation_baseline_logits: np.ndarray,
    class_weights: np.ndarray,
    device: str,
    seed: int,
    token_evidence: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    p11._seed_all(seed)
    model = XGBMomentCrossAttention().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    weights = torch.as_tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )
    losses: list[float] = []
    model.train()
    rng = np.random.default_rng(seed)
    for _ in range(UPDATES):
        indices = rng.choice(
            len(train_y),
            size=min(BATCH_SIZE, len(train_y)),
            replace=False,
        )
        moment = torch.as_tensor(
            train_moment_tokens[indices],
            dtype=torch.float32,
            device=device,
        )
        leaves = torch.as_tensor(
            train_leaf_ids[indices],
            dtype=torch.long,
            device=device,
        )
        baseline = torch.as_tensor(
            train_baseline_logits[indices],
            dtype=torch.float32,
            device=device,
        )
        labels = torch.as_tensor(
            train_y[indices],
            dtype=torch.long,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        fused, gate, _, contribution = model(moment, leaves, baseline)
        loss = (
            F.cross_entropy(fused, labels, weight=weights)
            + GATE_REGULARIZATION * gate.mean()
            + CONTRIBUTION_REGULARIZATION * contribution.square().mean()
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("cross-attention training loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    logits, gate, contributions = _predict_fusion(
        model,
        validation_moment_tokens,
        validation_leaf_ids,
        validation_baseline_logits,
        device=device,
        force_gate_zero=False,
    )
    gate0_logits, gate0, gate0_contributions = _predict_fusion(
        model,
        validation_moment_tokens,
        validation_leaf_ids,
        validation_baseline_logits,
        device=device,
        force_gate_zero=True,
    )
    gate0_error = float(
        np.max(np.abs(gate0_logits - validation_baseline_logits))
    )
    if (
        gate0_error != 0.0
        or np.any(gate0 != 0.0)
        or np.any(gate0_contributions != 0.0)
    ):
        raise RuntimeError("cross-attention gate0 is not an exact degeneration")
    evidence = {
        "updates": UPDATES,
        "first_train_loss": losses[0],
        "last_train_loss": losses[-1],
        "minimum_train_loss": min(losses),
        "duration_seconds": time.perf_counter() - started,
        "gate_mean": float(gate.mean()),
        "gate_min": float(gate.min()),
        "gate_max": float(gate.max()),
        "gate_values": [float(value) for value in gate],
        "gate0_max_abs_error": gate0_error,
        "residual_contribution_mean_abs": float(
            np.abs(contributions).mean()
        ),
        "residual_contribution_max_abs": float(np.abs(contributions).max()),
        "moment_token_cache_path": token_evidence["cache_path"],
        "moment_token_cache_sha256": token_evidence["cache_sha256"],
        "encoder_init": token_evidence["encoder_init"],
        "attention_query": "xgboost_leaf_indices_40x9",
        "attention_key_value": "moment_native33_tokens_13x4",
    }
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits, evidence


def _result_row(
    *,
    fold_id: int,
    repeat_id: int,
    variant: str,
    labels: np.ndarray,
    logits: np.ndarray,
    training: Mapping[str, Any],
    baseline_cell: Mapping[str, Any],
    encoder_init: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "track_id": "lithofacies",
        "task_id": "gm09_genetic_facies_9class",
        "lane": "P",
        "variant": variant,
        "fold_id": int(fold_id),
        "repeat_id": int(repeat_id),
        "seed": int(p11.REPEAT_SEEDS[repeat_id]),
        "train_samples": int(baseline_cell["train_rows"]),
        "validation_samples": int(len(labels)),
        "split_hash": p11.EXPECTED_SPLIT_HASH,
        "metrics": p11._metric_subset(labels, logits),
        "training": dict(training),
        "encoder_init": encoder_init,
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
        "rank_eligible": True,
    }


def _validate_pair_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(VARIANTS) or {
        str(row["variant"]) for row in rows
    } != set(VARIANTS):
        raise ValueError("each pair must contain the three fusion variants")
    if len({p11._pair_key(row) for row in rows}) != 1:
        raise ValueError("fusion pair rows mix fold/repeat identities")
    for row in rows:
        if (
            row.get("split_hash") != p11.EXPECTED_SPLIT_HASH
            or row.get("frozen_test_accessed") is not False
            or row.get("known_holdout_accessed") is not False
        ):
            raise ValueError("fusion row violates the development-only contract")


def summarize_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_ids: Sequence[int] = p11.FOLD_IDS,
    repeat_ids: Sequence[int] = tuple(range(len(p11.REPEAT_SEEDS))),
) -> dict[str, Any]:
    folds, repeats = p11._selection(fold_ids, repeat_ids)
    expected = {
        (fold_id, repeat_id, variant)
        for fold_id in folds
        for repeat_id in repeats
        for variant in VARIANTS
    }
    observed = {
        (int(row["fold_id"]), int(row["repeat_id"]), str(row["variant"]))
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("fusion results are not a complete strict matrix")
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        metric_summary: dict[str, Any] = {}
        for metric in p11.PRIMARY_METRICS:
            values = np.asarray(
                [row["metrics"][metric] for row in selected],
                dtype=np.float64,
            )
            metric_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        gates = [
            float(row["training"]["gate_mean"])
            for row in selected
            if row["training"].get("gate_mean") is not None
        ]
        contributions = [
            float(row["training"]["residual_contribution_mean_abs"])
            for row in selected
            if row["training"].get("residual_contribution_mean_abs") is not None
        ]
        variants[variant] = {
            "cells": len(selected),
            "metrics": metric_summary,
            "gate_mean": float(np.mean(gates)) if gates else None,
            "residual_contribution_mean_abs": (
                float(np.mean(contributions)) if contributions else None
            ),
        }
    baseline = variants["baseline"]["metrics"]["fixed_schema_macro_f1"]["mean"]
    calibrated = variants["prior_calibrated"]["metrics"][
        "fixed_schema_macro_f1"
    ]["mean"]
    fusion = variants["cross_attention"]["metrics"][
        "fixed_schema_macro_f1"
    ]["mean"]
    paired = {
        (int(row["fold_id"]), int(row["repeat_id"]), str(row["variant"])): float(
            row["metrics"]["fixed_schema_macro_f1"]
        )
        for row in rows
    }
    fusion_wins = sum(
        paired[(fold_id, repeat_id, "cross_attention")]
        > paired[(fold_id, repeat_id, "baseline")]
        for fold_id in folds
        for repeat_id in repeats
    )
    delta = fusion - baseline
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "evaluation": {
            "protocol": "strict_LOGO4_three_seed_development_only",
            "split_hash": p11.EXPECTED_SPLIT_HASH,
            "fold_ids": list(folds),
            "repeat_ids": list(repeats),
            "seeds": [int(p11.REPEAT_SEEDS[index]) for index in repeats],
            "variant_order": list(VARIANTS),
            "expected_cells": len(expected),
            "completed_cells": len(rows),
            "frozen_test_accessed": False,
            "known_holdout_accessed": False,
        },
        "variants": variants,
        "comparison": {
            "cross_attention_minus_baseline": delta,
            "prior_calibrated_minus_baseline": calibrated - baseline,
            "cross_attention_minus_prior_calibrated": fusion - calibrated,
            "cross_attention_pair_wins_over_baseline": int(fusion_wins),
            "paired_comparisons": len(folds) * len(repeats),
            "cross_attention_pair_win_fraction": (
                fusion_wins / (len(folds) * len(repeats))
            ),
        },
        "decision": {
            "state": (
                "DEVELOPMENT_IMPROVEMENT_ABLATION_PENDING"
                if delta > 0.0
                else "NO_DEVELOPMENT_IMPROVEMENT"
            ),
            "default_enabled": False,
            "large_model_contribution_share": (
                "pending_next_pretrained_vs_random_encoder_ablation"
            ),
            "objective_improved": delta > 0.0,
        },
    }


def _write_figure(summary: Mapping[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [
        summary["variants"][variant]["metrics"]["fixed_schema_macro_f1"]["mean"]
        for variant in VARIANTS
    ]
    errors = [
        summary["variants"][variant]["metrics"]["fixed_schema_macro_f1"]["std"]
        for variant in VARIANTS
    ]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    positions = np.arange(len(VARIANTS))
    axis.bar(positions, values, yerr=errors, color=colors, capsize=4)
    axis.set_xticks(positions, ("baseline", "prior\ncalibrated", "cross\nattention"))
    axis.set_ylabel("fixed-schema macro-F1")
    axis.set_title(
        "Lithofacies P11 XGBoost–MOMENT cross-attention fusion\n"
        "strict LOGO4 × 3 seeds, development only"
    )
    axis.grid(axis="y", alpha=0.25)
    upper = max(values) + max(errors)
    axis.set_ylim(0.0, upper * 1.25 if upper > 0 else 1.0)
    for position, value in zip(positions, values):
        axis.text(position, value, f"{value:.4f}", ha="center", va="bottom")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_evidence(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    comparison = summary["comparison"]
    lines = [
        "# P11 XGBoost–MOMENT cross-attention fusion evidence",
        "",
        "## Objective result",
        "",
        f"Cross-attention fusion minus the strong XGBoost baseline on mean "
        f"fixed-nine Macro-F1: "
        f"`{comparison['cross_attention_minus_baseline']:+.6f}`.",
        "",
        "This is the overall system delta. It is not attributed to MOMENT. "
        "**Large-model contribution share awaits the next pretrained-versus-"
        "random encoder ablation.**",
        "",
        "**大模型贡献占比待下一轮消融确认。**",
        "",
        "## Strict LOGO4 variants",
        "",
        "| variant | fixed-9 macro-F1 mean | std | accuracy | gate mean | "
        "mean abs residual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = summary["variants"][variant]
        f1 = item["metrics"]["fixed_schema_macro_f1"]
        gate = item["gate_mean"]
        contribution = item["residual_contribution_mean_abs"]
        lines.append(
            f"| {variant} | {f1['mean']:.6f} | {f1['std']:.6f} | "
            f"{item['metrics']['accuracy']['mean']:.6f} | "
            f"{'—' if gate is None else f'{gate:.6f}'} | "
            f"{'—' if contribution is None else f'{contribution:.6f}'} |"
        )
    lines.extend(
        [
            "",
            "## Component deltas",
            "",
            f"- Prior calibration − baseline: "
            f"`{comparison['prior_calibrated_minus_baseline']:+.6f}`.",
            f"- Cross-attention − prior calibration: "
            f"`{comparison['cross_attention_minus_prior_calibrated']:+.6f}`.",
            f"- Cross-attention wins over baseline in "
            f"`{comparison['cross_attention_pair_wins_over_baseline']}/"
            f"{comparison['paired_comparisons']}` fold/seed pairs.",
            "",
            "## Architecture and optimization boundary",
            "",
            "- XGBoost query tokens are real archived booster leaf IDs with "
            "shape `40 rounds × 9 classes`; no validation labels enter them.",
            "- MOMENT key/value tokens are `13 log channels × 4 native patches`; "
            "there is no 33→512 interpolation.",
            f"- Prior calibration adds `{PRIOR_SHRINKAGE} × centered log("
            "fold-train class count)` to partially undo inverse-sqrt class "
            "weighting. It is reported as a separate control.",
            f"- Encoder initialization is parameterized as `{ENCODER_INITS}`. "
            f"This evidence ran `{summary['model']['encoder_init']}` only.",
            "- The fused output is a learned bounded residual over calibrated "
            "XGBoost logits, with exact zero-gate degeneration checked per cell.",
            "",
            "## Development exploration log",
            "",
            "- This is an explicitly exploratory LOGO development result, not "
            "an unbiased holdout estimate.",
            "- Averaging the three XGBoost seed logits changed no argmax and "
            "gave `+0.000000`; it was not retained.",
            "- Train-prior shrinkage candidates `0.20/0.25/0.30` produced mean "
            "fixed-nine Macro-F1 `0.201375/0.202187/0.202058`; `0.25` was "
            "frozen for the formal matrix.",
            "- Depth-logit smoothing tested radii `1/2/3/4/6/8` and mixing "
            "`0.25/0.50/0.75/1.0`. Its best standalone delta was `+0.003821`, "
            "but it was unstable with prior calibration and was rejected.",
            "- Cross-attention architecture/training constants were frozen "
            "after fold 0, repeat 0 smoke "
            "(`0.213580 → 0.227106 → 0.228214`) and then run unchanged.",
            "",
            "## Every fold and seed",
            "",
            "| fold | seed | baseline | prior calibrated | cross attention | gate |",
            "|---:|---:|---:|---:|---:|---:|",
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
                for variant in VARIANTS
            }
            metric_name = "fixed_schema_macro_f1"
            baseline_value = selected["baseline"]["metrics"][metric_name]
            calibrated_value = selected["prior_calibrated"]["metrics"][metric_name]
            fusion_value = selected["cross_attention"]["metrics"][metric_name]
            gate_value = selected["cross_attention"]["training"]["gate_mean"]
            lines.append(
                f"| {fold_id} | {selected['baseline']['seed']} | "
                f"{baseline_value:.6f} | "
                f"{calibrated_value:.6f} | "
                f"{fusion_value:.6f} | "
                f"{gate_value:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Leakage and preservation audit",
            "",
            "- Only the immutable development LOGO4 folds were opened. Frozen "
            "holdout and every holdout-like path remain blocked.",
            "- Calibration uses fold-train class counts only. Leaf IDs come from "
            "the archived fold-train-fitted XGBoost booster.",
            "- The two earlier committed P11 evidence sets were verified "
            "byte-for-byte before this run.",
            "- This development result cannot establish how much of the delta "
            "comes from pretrained MOMENT until the next encoder-init ablation.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


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
    p11._write_jsonl(results_path, rows)
    p11._write_json(summary_path, summary)
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
                "path": p11._portable_path(path),
                "sha256": p11._sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest_path = output_dir / "artifact_manifest.json"
    p11._write_json(
        manifest_path,
        {
            "schema_version": ARTIFACT_SCHEMA,
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
    representation_bundle: Path,
    snapshot: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda:0",
    encoder_init: str = "pretrained",
    fold_ids: Sequence[int] = p11.FOLD_IDS,
    repeat_ids: Sequence[int] = tuple(range(len(p11.REPEAT_SEEDS))),
    resume: bool = True,
) -> dict[str, Any]:
    p11.ensure_development_only_paths(
        (development_batch, baseline_bundle, representation_bundle, snapshot)
    )
    if encoder_init not in ENCODER_INITS:
        raise ValueError(f"encoder_init must be one of {ENCODER_INITS}")
    preserved = _verify_preserved_outputs()
    folds, repeats = p11._selection(fold_ids, repeat_ids)
    if (
        p11._sha256(snapshot / "model.safetensors")
        != p11.EXPECTED_SNAPSHOT_SHA256
    ):
        raise RuntimeError("unexpected MOMENT snapshot weights")
    arrays, batch_manifest = p11.load_stage3_batch(development_batch)
    if (
        batch_manifest.get("split_hash") != p11.EXPECTED_SPLIT_HASH
        or batch_manifest.get("frozen_test_accessed") is not False
    ):
        raise RuntimeError("development batch is not strict LOGO4")
    batch_sha256 = p11._sha256(development_batch)
    baseline_sha256 = p11._sha256(baseline_bundle)
    baseline_arrays, baseline_manifest = p11._load_baseline_bundle(
        baseline_bundle,
        development_batch_sha256=batch_sha256,
        fold_ids=folds,
        repeat_ids=repeats,
    )
    leaf_arrays, leaf_manifest = _load_representation_bundle(
        representation_bundle,
        development_batch_sha256=batch_sha256,
        baseline_bundle_sha256=baseline_sha256,
        fold_ids=folds,
        repeat_ids=repeats,
    )
    run_contract = {
        "schema_version": SCHEMA_VERSION,
        "split_hash": p11.EXPECTED_SPLIT_HASH,
        "development_batch_sha256": batch_sha256,
        "baseline_bundle_sha256": baseline_sha256,
        "representation_bundle_sha256": p11._sha256(representation_bundle),
        "snapshot_sha256": p11.EXPECTED_SNAPSHOT_SHA256,
        "encoder_init": encoder_init,
        "fold_ids": list(folds),
        "repeat_ids": list(repeats),
        "seeds": [int(p11.REPEAT_SEEDS[index]) for index in repeats],
        "variants": list(VARIANTS),
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "attention_dim": ATTENTION_DIM,
        "attention_heads": ATTENTION_HEADS,
        "gate_regularization": GATE_REGULARIZATION,
        "contribution_regularization": CONTRIBUTION_REGULARIZATION,
        "max_residual_logit": MAX_RESIDUAL_LOGIT,
        "prior_shrinkage": PRIOR_SHRINKAGE,
        "preserved_outputs": preserved,
    }
    contract_hash = p11._stable_hash(run_contract)
    runtime_dir = output_dir / "runtime"
    contract_path = runtime_dir / "run_contract.json"
    partial_path = runtime_dir / "partial_results.jsonl"
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract.get("contract_hash") != contract_hash:
            raise RuntimeError("existing fusion run has a different contract")
    else:
        p11._write_json(
            contract_path,
            {**run_contract, "contract_hash": contract_hash},
        )
    existing_rows = p11._read_jsonl(partial_path) if resume else []
    completed: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in existing_rows:
        completed.setdefault(p11._pair_key(row), []).append(row)
    for pair_rows in completed.values():
        _validate_pair_rows(pair_rows)
    all_rows = list(existing_rows)
    baseline_cells = {
        (int(cell["fold_id"]), int(cell["repeat_id"])): cell
        for cell in baseline_manifest["cells"]
    }
    started = time.perf_counter()
    for fold_id in folds:
        fold = p11._fold_arrays(arrays, fold_id)
        train_x = _clean_log_inputs(fold["p_train_well"])
        validation_x = _clean_log_inputs(fold["p_validation_well"])
        train_y = np.asarray(fold["p_train_labels"], dtype=np.int64)
        validation_y = np.asarray(
            fold["p_validation_labels"],
            dtype=np.int64,
        )
        class_counts = np.asarray(fold["class_counts"], dtype=np.int64)
        class_weights = np.asarray(fold["class_weights"], dtype=np.float32)
        for repeat_id in repeats:
            key = (fold_id, repeat_id)
            if key in completed:
                print(
                    f"resume fold={fold_id} repeat={repeat_id}: "
                    "three variants already complete",
                    flush=True,
                )
                continue
            seed = int(p11.REPEAT_SEEDS[repeat_id])
            encoder_seed = 2693 if encoder_init == "pretrained" else seed
            cache_name = (
                f"{encoder_init}_fold{fold_id}"
                + ("" if encoder_init == "pretrained" else f"_repeat{repeat_id}")
                + ".npz"
            )
            (
                train_moment_tokens,
                validation_moment_tokens,
                token_evidence,
            ) = _moment_token_cache(
                cache_path=runtime_dir / "moment_tokens" / cache_name,
                train_x=train_x,
                validation_x=validation_x,
                snapshot=snapshot,
                device=device,
                seed=encoder_seed,
                encoder_init=encoder_init,
                fold_id=fold_id,
                development_batch_sha256=batch_sha256,
            )
            prefix = f"f{fold_id}_r{repeat_id}"
            train_baseline = baseline_arrays[f"{prefix}_train_logits"]
            validation_baseline = baseline_arrays[
                f"{prefix}_validation_logits"
            ]
            train_calibrated, prior_bias = _prior_calibrate(
                train_baseline,
                class_counts,
            )
            validation_calibrated, validation_prior_bias = _prior_calibrate(
                validation_baseline,
                class_counts,
            )
            if not np.array_equal(prior_bias, validation_prior_bias):
                raise RuntimeError("train/validation prior calibration diverged")
            train_leaf_ids = leaf_arrays[f"{prefix}_train_leaf_ids"]
            validation_leaf_ids = leaf_arrays[
                f"{prefix}_validation_leaf_ids"
            ]
            fusion_logits, fusion_training = _train_fusion(
                train_moment_tokens=train_moment_tokens,
                train_leaf_ids=train_leaf_ids,
                train_baseline_logits=train_calibrated,
                train_y=train_y,
                validation_moment_tokens=validation_moment_tokens,
                validation_leaf_ids=validation_leaf_ids,
                validation_baseline_logits=validation_calibrated,
                class_weights=class_weights,
                device=device,
                seed=seed,
                token_evidence=token_evidence,
            )
            baseline_cell = baseline_cells[key]
            no_training = {
                "updates": 0,
                "gate_mean": None,
                "residual_contribution_mean_abs": None,
            }
            calibration_training = {
                "updates": 0,
                "gate_mean": None,
                "residual_contribution_mean_abs": None,
                "prior_shrinkage": PRIOR_SHRINKAGE,
                "prior_bias": [float(value) for value in prior_bias],
                "fit_scope": "fold_train_class_counts_only",
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
                    encoder_init=encoder_init,
                ),
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="prior_calibrated",
                    labels=validation_y,
                    logits=validation_calibrated,
                    training=calibration_training,
                    baseline_cell=baseline_cell,
                    encoder_init=encoder_init,
                ),
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="cross_attention",
                    labels=validation_y,
                    logits=fusion_logits,
                    training=fusion_training,
                    baseline_cell=baseline_cell,
                    encoder_init=encoder_init,
                ),
            ]
            _validate_pair_rows(pair_rows)
            all_rows.extend(pair_rows)
            p11._write_jsonl(partial_path, all_rows)
            completed[key] = pair_rows
            print(
                f"complete fold={fold_id} repeat={repeat_id} seed={seed}: "
                f"baseline={pair_rows[0]['metrics']['fixed_schema_macro_f1']:.6f} "
                f"calibrated={pair_rows[1]['metrics']['fixed_schema_macro_f1']:.6f} "
                f"fusion={pair_rows[2]['metrics']['fixed_schema_macro_f1']:.6f} "
                f"gate={fusion_training['gate_mean']:.6f}",
                flush=True,
            )
    summary = summarize_results(
        all_rows,
        fold_ids=folds,
        repeat_ids=repeats,
    )
    summary["model"] = {
        "baseline_model": p11.BASELINE_MODEL_ID,
        "foundation_model": "AutonLab/MOMENT-1-base",
        "encoder_init": encoder_init,
        "encoder_init_choices": list(ENCODER_INITS),
        "random_init_ablation_run": encoder_init == "random",
        "weights_sha256": p11.EXPECTED_SNAPSHOT_SHA256,
        "attention_query": "xgboost_leaf_indices_40_rounds_x_9_classes",
        "attention_key_value": "moment_native33_13_channels_x_4_patches",
        "resampling": "none",
        "residual_formula": (
            "prior_calibrated_xgboost_logits + "
            "sigmoid(per_class_gate)*bounded_cross_attention_residual"
        ),
    }
    summary["optimization"] = {
        "prior_shrinkage": PRIOR_SHRINKAGE,
        "calibration_fit_scope": "fold_train_class_counts_only",
        "large_model_contribution_share": (
            "pending_next_pretrained_vs_random_encoder_ablation"
        ),
    }
    summary["report_note_zh"] = "大模型贡献占比待下一轮消融确认"
    summary["development_exploration"] = DEVELOPMENT_EXPLORATION
    summary["inputs"] = {
        "development_batch_sha256": batch_sha256,
        "baseline_bundle_sha256": baseline_sha256,
        "representation_bundle_sha256": p11._sha256(representation_bundle),
        "stage3_results_sha256": leaf_manifest["stage3_results_sha256"],
    }
    summary["preserved_outputs"] = preserved
    summary["runtime"] = {
        "device": device,
        "duration_seconds_this_invocation": time.perf_counter() - started,
        "resumed_pairs": len(existing_rows) // len(VARIANTS),
        "raw_predictions_persisted": False,
    }
    artifacts = _write_artifacts(
        rows=all_rows,
        summary=summary,
        output_dir=output_dir,
    )
    return {
        "summary": summary,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }


def verify_artifacts(output_dir: Path) -> dict[str, Any]:
    preserved = _verify_preserved_outputs()
    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("known_holdout_accessed") is not False
    ):
        raise RuntimeError("invalid cross-attention artifact manifest")
    for artifact in manifest.get("artifacts", ()):
        path = PROJECT_ROOT / artifact["path"]
        if (
            not path.is_file()
            or p11._sha256(path) != artifact["sha256"]
            or path.stat().st_size != int(artifact["bytes"])
        ):
            raise RuntimeError(
                f"cross-attention artifact verification failed: {path}"
            )
    rows = p11._read_jsonl(output_dir / "results.jsonl")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    evaluation = summary["evaluation"]
    recomputed = summarize_results(
        rows,
        fold_ids=evaluation["fold_ids"],
        repeat_ids=evaluation["repeat_ids"],
    )
    if (
        recomputed["comparison"] != summary["comparison"]
        or recomputed["decision"] != summary["decision"]
        or summary.get("preserved_outputs") != preserved
        or summary.get("model", {}).get("encoder_init") not in ENCODER_INITS
        or summary.get("development_exploration") != DEVELOPMENT_EXPLORATION
        or summary.get("report_note_zh")
        != "大模型贡献占比待下一轮消融确认"
        or summary["decision"].get("large_model_contribution_share")
        != "pending_next_pretrained_vs_random_encoder_ablation"
    ):
        raise RuntimeError("cross-attention summary is not reproducible")
    from PIL import Image

    with Image.open(output_dir / "primary_metric.png") as image:
        image.verify()
    return {
        "status": "verified",
        "artifacts": len(manifest["artifacts"]),
        "rows": len(rows),
        "comparison": summary["comparison"],
        "decision": summary["decision"],
        "preserved_artifacts": len(preserved),
    }


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-representations")
    prepare.add_argument("--development-batch", type=Path, required=True)
    prepare.add_argument("--baseline-bundle", type=Path, required=True)
    prepare.add_argument("--stage3-results", type=Path, required=True)
    prepare.add_argument("--checkpoint-dir", type=Path, required=True)
    prepare.add_argument("--output-bundle", type=Path, required=True)
    prepare.add_argument("--folds", type=_parse_ints, default=p11.FOLD_IDS)
    prepare.add_argument(
        "--repeats",
        type=_parse_ints,
        default=tuple(range(len(p11.REPEAT_SEEDS))),
    )
    run = subparsers.add_parser("run")
    run.add_argument("--development-batch", type=Path, required=True)
    run.add_argument("--baseline-bundle", type=Path, required=True)
    run.add_argument("--representation-bundle", type=Path, required=True)
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--device", default="cuda:0")
    run.add_argument(
        "--encoder-init",
        choices=ENCODER_INITS,
        default="pretrained",
    )
    run.add_argument("--folds", type=_parse_ints, default=p11.FOLD_IDS)
    run.add_argument(
        "--repeats",
        type=_parse_ints,
        default=tuple(range(len(p11.REPEAT_SEEDS))),
    )
    run.add_argument("--no-resume", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-representations":
        payload = prepare_representation_bundle(
            development_batch=args.development_batch,
            baseline_bundle=args.baseline_bundle,
            stage3_results=args.stage3_results,
            checkpoint_dir=args.checkpoint_dir,
            output_bundle=args.output_bundle,
            fold_ids=args.folds,
            repeat_ids=args.repeats,
        )
    elif args.command == "run":
        payload = run_experiment(
            development_batch=args.development_batch,
            baseline_bundle=args.baseline_bundle,
            representation_bundle=args.representation_bundle,
            snapshot=args.snapshot,
            output_dir=args.output_dir,
            device=args.device,
            encoder_init=args.encoder_init,
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
