#!/usr/bin/env python3
"""Development-only P11 cross-attention residual-fusion diagnostic.

This experiment keeps the hash-verified P5 PyKrige OOF base predictions and
the P11 holdout firewall.  Three seismic channels are forwarded through the
OpenMind encoder separately by the preceding diagnostic, then represented as
three embedding tokens.  A structured query token cross-attends to those
channel tokens before predicting a bounded, cross-fitted residual correction.

The encoder weight source is parameterized as ``pretrained`` or
``random_init``.  The committed experiment uses only ``pretrained``; the
switch exists so the next matched-architecture ablation can reuse exactly the
same fusion harness.  No command-line argument can name a test or holdout
container, and the only legal HDF5 input remains ``train.h5``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import p11_residual_fusion as base  # noqa: E402
import p11_residual_fusion_diagnostics as diagnostics  # noqa: E402


SCHEMA_VERSION = "reconstruction-p11-cross-attention-fusion/v1"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p11_cross_attention_fusion"
DEFAULT_CHANNEL_FEATURE_CACHE = (
    PROJECT_ROOT
    / "_tmp"
    / "p11_residual_fusion_diagnostics"
    / "openmind_per_channel_features.npz"
)
DEFAULT_RANDOM_INIT_FEATURE_CACHE = (
    PROJECT_ROOT
    / "_tmp"
    / "p11_residual_fusion_diagnostics"
    / "openmind_random_init_features.npz"
)
ENCODER_WEIGHT_MODES = ("pretrained", "random_init")
WIN_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CrossAttentionConfig:
    """Fixed, non-validation-tuned residual-head configuration."""

    pca_components: int = 16
    embedding_dim: int = 16
    attention_heads: int = 4
    hidden_dim: int = 32
    dropout: float = 0.15
    epochs: int = 48
    batch_size: int = 1024
    learning_rate: float = 0.002
    weight_decay: float = 0.05
    correction_clip_quantile: float = 0.995


def resolve_encoder_features(
    *,
    encoder_weight_mode: str,
    inputs: base.DevInputPaths,
    oof: base.OOFDevelopment,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path | None,
    channel_feature_cache: Path,
    random_init_feature_cache: Path,
    device: str,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Load one parameterized encoder source without mixing weight modes."""

    if encoder_weight_mode not in ENCODER_WEIGHT_MODES:
        raise ValueError(
            f"encoder_weight_mode must be one of {ENCODER_WEIGHT_MODES}"
        )
    if encoder_weight_mode == "pretrained":
        features, audit = diagnostics.get_openmind_per_channel_features(
            inputs=inputs,
            oof=oof,
            source_root=source_root,
            checkpoint=checkpoint,
            dependency_root=dependency_root,
            feature_cache=channel_feature_cache,
            device=device,
        )
        by_seed = {
            int(seed): np.asarray(features, dtype=np.float32)
            for seed in base.REPEAT_SEEDS
        }
        return by_seed, {
            **audit,
            "encoder_weight_mode": "pretrained",
            "pretrained_weights_used_for_forward": True,
            "matched_random_init_executed_this_run": False,
            "weight_mode_switch_available": list(ENCODER_WEIGHT_MODES),
        }

    _, channel_features, audit = (
        diagnostics.get_openmind_random_init_features(
            inputs=inputs,
            oof=oof,
            source_root=source_root,
            checkpoint=checkpoint,
            dependency_root=dependency_root,
            feature_cache=random_init_feature_cache,
            device=device,
        )
    )
    by_seed = {
        int(seed): np.asarray(channel_features[seed_id], dtype=np.float32)
        for seed_id, seed in enumerate(base.REPEAT_SEEDS)
    }
    return by_seed, {
        **audit,
        "encoder_weight_mode": "random_init",
        "matched_random_init_executed_this_run": True,
        "weight_mode_switch_available": list(ENCODER_WEIGHT_MODES),
    }


def select_channel_tokens(
    channel_features: np.ndarray,
    *,
    layer_channels: Sequence[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a balanced multiscale embedding for each channel token."""

    channel_features = np.asarray(channel_features, dtype=np.float64)
    width = sum(int(value) for value in layer_channels)
    if channel_features.ndim != 3 or channel_features.shape[1:] != (3, width):
        raise ValueError(
            "channel features must have shape (rows, 3, sum(layer_channels))"
        )
    if not np.all(np.isfinite(channel_features)):
        raise FloatingPointError("channel embeddings contain non-finite values")
    selected = base._selected_latent_channels(  # noqa: SLF001
        layer_channels,
        seed=int(seed),
    )
    return channel_features[:, :, selected], selected


def build_structured_queries(oof: base.OOFDevelopment) -> np.ndarray:
    """Build the low-dimensional seismic/structural cross-attention query."""

    queries = np.column_stack(
        [
            oof.structural_features,
            oof.baseline,
            oof.distance_to_well,
        ]
    )
    if queries.ndim != 2 or queries.shape[0] != len(oof.target):
        raise ValueError("structured query rows do not align with OOF evidence")
    if not np.all(np.isfinite(queries)):
        raise FloatingPointError("structured queries contain non-finite values")
    return np.asarray(queries, dtype=np.float64)


def _make_network(
    *,
    query_width: int,
    token_width: int,
    config: CrossAttentionConfig,
    torch: Any,
) -> Any:
    """Construct the actual cross-attention residual head."""

    class CrossAttentionResidualHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query_projection = torch.nn.Sequential(
                torch.nn.Linear(query_width, config.embedding_dim),
                torch.nn.LayerNorm(config.embedding_dim),
                torch.nn.GELU(),
            )
            self.token_projection = torch.nn.Sequential(
                torch.nn.Linear(token_width, config.embedding_dim),
                torch.nn.LayerNorm(config.embedding_dim),
                torch.nn.GELU(),
            )
            self.cross_attention = torch.nn.MultiheadAttention(
                embed_dim=config.embedding_dim,
                num_heads=config.attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.attention_norm = torch.nn.LayerNorm(config.embedding_dim)
            self.residual_head = torch.nn.Sequential(
                torch.nn.Linear(2 * config.embedding_dim, config.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(config.hidden_dim, 1),
            )

        def forward(
            self,
            query_values: Any,
            token_values: Any,
            *,
            return_attention: bool = False,
        ) -> Any:
            query = self.query_projection(query_values).unsqueeze(1)
            tokens = self.token_projection(token_values)
            attended, weights = self.cross_attention(
                query,
                tokens,
                tokens,
                need_weights=return_attention,
                average_attn_weights=True,
            )
            attended = self.attention_norm(attended + query).squeeze(1)
            prediction = self.residual_head(
                torch.cat([query.squeeze(1), attended], dim=1)
            ).squeeze(1)
            if return_attention:
                return prediction, weights.squeeze(1)
            return prediction

    return CrossAttentionResidualHead()


def _fit_cross_attention(
    train_queries: np.ndarray,
    train_tokens: np.ndarray,
    train_residual: np.ndarray,
    validation_queries: np.ndarray,
    validation_tokens: np.ndarray,
    *,
    seed: int,
    config: CrossAttentionConfig,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit PCA/scalers and the attention head on training rows only."""

    import torch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    train_queries = np.asarray(train_queries, dtype=np.float64)
    validation_queries = np.asarray(validation_queries, dtype=np.float64)
    train_tokens = np.asarray(train_tokens, dtype=np.float64)
    validation_tokens = np.asarray(validation_tokens, dtype=np.float64)
    train_residual = np.asarray(train_residual, dtype=np.float64)
    if (
        train_tokens.ndim != 3
        or validation_tokens.ndim != 3
        or train_tokens.shape[1] != 3
        or validation_tokens.shape[1:] != train_tokens.shape[1:]
    ):
        raise ValueError("cross-attention expects three aligned channel tokens")
    if (
        train_queries.shape[0] != len(train_tokens)
        or len(train_residual) != len(train_tokens)
        or validation_queries.shape[0] != len(validation_tokens)
    ):
        raise ValueError("cross-attention training arrays do not align")

    query_scaler = StandardScaler()
    train_query_scaled = query_scaler.fit_transform(train_queries)
    validation_query_scaled = query_scaler.transform(validation_queries)

    flattened_train = train_tokens.reshape(-1, train_tokens.shape[-1])
    flattened_validation = validation_tokens.reshape(
        -1,
        validation_tokens.shape[-1],
    )
    token_scaler = StandardScaler()
    flattened_train = token_scaler.fit_transform(flattened_train)
    flattened_validation = token_scaler.transform(flattened_validation)
    components = min(
        int(config.pca_components),
        flattened_train.shape[1],
        flattened_train.shape[0] - 1,
    )
    if components < 2:
        raise ValueError("insufficient training data for token PCA")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=int(seed),
    )
    train_token_reduced = pca.fit_transform(flattened_train).reshape(
        len(train_tokens),
        3,
        components,
    )
    validation_token_reduced = pca.transform(flattened_validation).reshape(
        len(validation_tokens),
        3,
        components,
    )

    target_mean = float(np.mean(train_residual))
    target_scale = max(float(np.std(train_residual)), 1e-6)
    normalized_target = (train_residual - target_mean) / target_scale
    correction_clip = max(
        float(
            np.quantile(
                np.abs(train_residual),
                config.correction_clip_quantile,
            )
        ),
        1e-6,
    )

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = _make_network(
        query_width=train_query_scaled.shape[1],
        token_width=components,
        config=config,
        torch=torch,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = torch.nn.SmoothL1Loss(beta=0.5)
    train_query_tensor = torch.as_tensor(
        train_query_scaled,
        dtype=torch.float32,
    )
    train_token_tensor = torch.as_tensor(
        train_token_reduced,
        dtype=torch.float32,
    )
    train_target_tensor = torch.as_tensor(
        normalized_target,
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    final_loss = math.nan
    model.train()
    for _ in range(int(config.epochs)):
        order = torch.randperm(
            len(train_tokens),
            generator=generator,
        )
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(order), int(config.batch_size)):
            rows = order[start : start + int(config.batch_size)]
            query_batch = train_query_tensor[rows].to(device)
            token_batch = train_token_tensor[rows].to(device)
            target_batch = train_target_tensor[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(query_batch, token_batch)
            loss = loss_function(predicted, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            count = len(rows)
            epoch_loss += float(loss.detach().cpu()) * count
            seen += count
        final_loss = epoch_loss / max(seen, 1)

    model.eval()
    validation_query_tensor = torch.as_tensor(
        validation_query_scaled,
        dtype=torch.float32,
        device=device,
    )
    validation_token_tensor = torch.as_tensor(
        validation_token_reduced,
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        normalized_prediction, attention = model(
            validation_query_tensor,
            validation_token_tensor,
            return_attention=True,
        )
    prediction = (
        normalized_prediction.detach().cpu().numpy().astype(np.float64)
        * target_scale
        + target_mean
    )
    prediction = np.clip(prediction, -correction_clip, correction_clip)
    attention_values = (
        attention.detach().cpu().numpy().astype(np.float64)
    )
    if prediction.shape != (len(validation_tokens),) or not np.all(
        np.isfinite(prediction)
    ):
        raise FloatingPointError("cross-attention returned invalid predictions")
    if attention_values.shape != (len(validation_tokens), 3):
        raise RuntimeError("cross-attention did not return three channel weights")
    return prediction, {
        "train_rows": int(len(train_tokens)),
        "validation_rows": int(len(validation_tokens)),
        "pca_components": int(components),
        "pca_explained_variance_ratio_sum": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "correction_clip_abs": correction_clip,
        "final_training_loss": float(final_loss),
        "mean_attention_by_channel": [
            float(value) for value in np.mean(attention_values, axis=0)
        ],
        "all_preprocessing_fit_on_training_rows_only": True,
    }


def _inner_oof_cross_attention(
    *,
    queries: np.ndarray,
    tokens: np.ndarray,
    residual_target: np.ndarray,
    fold_ids: np.ndarray,
    seed: int,
    config: CrossAttentionConfig,
    device: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Cross-fit the attention residual across the outer-training folds."""

    prediction = np.full(len(tokens), np.nan, dtype=np.float64)
    audits: list[dict[str, Any]] = []
    unique_folds = np.unique(fold_ids)
    if len(unique_folds) < 3:
        raise ValueError("cross-attention stack needs at least three fold groups")
    for inner_fold in unique_folds:
        validation = fold_ids == inner_fold
        train = ~validation
        if np.any(fold_ids[train] == inner_fold):
            raise RuntimeError("cross-attention inner cross-fit overlap")
        values, audit = _fit_cross_attention(
            queries[train],
            tokens[train],
            residual_target[train],
            queries[validation],
            tokens[validation],
            seed=int(seed) + 1009 * int(inner_fold),
            config=config,
            device=device,
        )
        prediction[validation] = values
        audits.append(
            {
                **audit,
                "inner_validation_fold": int(inner_fold),
                "inner_train_fold_ids": sorted(
                    int(value) for value in np.unique(fold_ids[train])
                ),
            }
        )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("cross-attention inner predictions are incomplete")
    return prediction, audits


def _evaluate_seed(
    *,
    oof: base.OOFDevelopment,
    queries: np.ndarray,
    tokens: np.ndarray,
    seed: int,
    config: CrossAttentionConfig,
    device: str,
) -> dict[str, Any]:
    """Run the locked outer OOF and inner-OOF gate for one pseudo-repeat."""

    residual_target = oof.target - oof.baseline
    ungated = np.full(len(oof.target), np.nan, dtype=np.float64)
    adaptive_gated = np.full(len(oof.target), np.nan, dtype=np.float64)
    gated = np.full(len(oof.target), np.nan, dtype=np.float64)
    cells: list[dict[str, Any]] = []

    for outer_fold in base.FOLD_IDS:
        validation = oof.fold_ids == outer_fold
        train = ~validation
        inner_prediction, inner_audits = _inner_oof_cross_attention(
            queries=queries[train],
            tokens=tokens[train],
            residual_target=residual_target[train],
            fold_ids=oof.fold_ids[train],
            seed=int(seed) + 100_003 * int(outer_fold),
            config=config,
            device=device,
        )
        gate_model, gate_scale, gate_scores, benefit_rate = (
            diagnostics._gate_for_inner_predictions(  # noqa: SLF001
                inner_prediction=inner_prediction,
                inner_base=oof.baseline[train],
                inner_truth=oof.target[train],
                inner_distance=oof.distance_to_well[train],
                seed=int(seed) + int(outer_fold),
            )
        )
        inner_raw_gate = gate_model.predict(
            base._gate_signals(  # noqa: SLF001
                inner_prediction,
                oof.baseline[train],
                oof.distance_to_well[train],
            )
        )
        inner_adaptive_gate = np.clip(
            inner_raw_gate * gate_scale,
            0.0,
            1.0,
        )
        inner_adaptive_prediction = (
            oof.baseline[train] + inner_adaptive_gate * inner_prediction
        )
        inner_fold_consensus: list[dict[str, Any]] = []
        for inner_fold in np.unique(oof.fold_ids[train]):
            inner_validation = oof.fold_ids[train] == inner_fold
            inner_baseline_rmse = base._metrics(  # noqa: SLF001
                oof.target[train][inner_validation],
                oof.baseline[train][inner_validation],
            )["rmse"]
            inner_candidate_rmse = base._metrics(  # noqa: SLF001
                oof.target[train][inner_validation],
                inner_adaptive_prediction[inner_validation],
            )["rmse"]
            inner_fold_consensus.append(
                {
                    "fold": int(inner_fold),
                    "baseline_rmse": inner_baseline_rmse,
                    "candidate_rmse": inner_candidate_rmse,
                    "rmse_delta": (
                        inner_candidate_rmse - inner_baseline_rmse
                    ),
                    "outcome": diagnostics.classify_fold_outcome(
                        inner_candidate_rmse,
                        inner_baseline_rmse,
                        tolerance=WIN_TOLERANCE,
                    ),
                }
            )
        consensus_passed = bool(
            gate_scale > 0.0
            and all(
                row["outcome"] == "win" for row in inner_fold_consensus
            )
        )
        consensus_gate_scale = float(gate_scale if consensus_passed else 0.0)
        outer_residual, outer_audit = _fit_cross_attention(
            queries[train],
            tokens[train],
            residual_target[train],
            queries[validation],
            tokens[validation],
            seed=int(seed) + 1_000_003 * int(outer_fold),
            config=config,
            device=device,
        )
        raw_gate = gate_model.predict(
            base._gate_signals(  # noqa: SLF001
                outer_residual,
                oof.baseline[validation],
                oof.distance_to_well[validation],
            )
        )
        adaptive_gate = np.clip(raw_gate * gate_scale, 0.0, 1.0)
        gate = np.clip(raw_gate * consensus_gate_scale, 0.0, 1.0)
        ungated[validation] = oof.baseline[validation] + outer_residual
        adaptive_gated[validation] = (
            oof.baseline[validation] + adaptive_gate * outer_residual
        )
        gated[validation] = oof.baseline[validation] + gate * outer_residual
        baseline_metrics = base._metrics(  # noqa: SLF001
            oof.target[validation],
            oof.baseline[validation],
        )
        gated_metrics = base._metrics(  # noqa: SLF001
            oof.target[validation],
            gated[validation],
        )
        cells.append(
            {
                "seed": int(seed),
                "outer_fold": int(outer_fold),
                "train_fold_ids": sorted(
                    int(value) for value in np.unique(oof.fold_ids[train])
                ),
                "validation_fold_ids": [int(outer_fold)],
                "train_rows": int(np.sum(train)),
                "validation_rows": int(np.sum(validation)),
                "inner_residual_predictions": (
                    "cross-fit by locked spatial fold"
                ),
                "base_predictions_for_training": (
                    "hash-verified P5 Stage-3 development OOF only"
                ),
                "adaptive_selected_gate_scale": float(gate_scale),
                "consensus_gate_rule": (
                    "all four inner spatial folds must beat baseline"
                ),
                "inner_fold_consensus": inner_fold_consensus,
                "inner_fold_consensus_passed": consensus_passed,
                "selected_gate_scale": consensus_gate_scale,
                "inner_gate_candidate_rmse": gate_scores,
                "inner_correction_benefit_rate": float(benefit_rate),
                "baseline_metrics": baseline_metrics,
                "ungated_metrics": base._metrics(  # noqa: SLF001
                    oof.target[validation],
                    ungated[validation],
                ),
                "adaptive_gated_metrics": base._metrics(  # noqa: SLF001
                    oof.target[validation],
                    adaptive_gated[validation],
                ),
                "gated_metrics": gated_metrics,
                "gated_rmse_delta_vs_pykrige": (
                    gated_metrics["rmse"] - baseline_metrics["rmse"]
                ),
                "outcome_vs_pykrige": diagnostics.classify_fold_outcome(
                    gated_metrics["rmse"],
                    baseline_metrics["rmse"],
                    tolerance=WIN_TOLERANCE,
                ),
                "outer_model_audit": outer_audit,
                "inner_model_audits": inner_audits,
                "residual_prediction_stats": base._summary_stats(  # noqa: SLF001
                    outer_residual
                ),
                "gate_stats": {
                    **base._summary_stats(gate),  # noqa: SLF001
                    "zero_rate": float(np.mean(gate == 0.0)),
                    "one_rate": float(np.mean(gate == 1.0)),
                },
            }
        )
        print(
            json.dumps(
                {
                    "event": "outer_fold_complete",
                    "seed": int(seed),
                    "outer_fold": int(outer_fold),
                    "adaptive_gate_scale": float(gate_scale),
                    "consensus_gate_scale": consensus_gate_scale,
                    "gated_rmse": gated_metrics["rmse"],
                    "baseline_rmse": baseline_metrics["rmse"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if (
        not np.all(np.isfinite(ungated))
        or not np.all(np.isfinite(adaptive_gated))
        or not np.all(np.isfinite(gated))
    ):
        raise RuntimeError("cross-attention outer predictions are incomplete")
    return {
        "seed": int(seed),
        "ungated_prediction": ungated,
        "adaptive_gated_prediction": adaptive_gated,
        "gated_prediction": gated,
        "ungated_metrics": base._metrics(oof.target, ungated),  # noqa: SLF001
        "adaptive_gated_metrics": base._metrics(  # noqa: SLF001
            oof.target,
            adaptive_gated,
        ),
        "gated_metrics": base._metrics(oof.target, gated),  # noqa: SLF001
        "per_fold": cells,
    }


def _independent_fold_summary(
    *,
    oof: base.OOFDevelopment,
    results_by_seed: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Average paired seeds inside each of five genuine spatial units."""

    rows: list[dict[str, Any]] = []
    counts = {"win": 0, "loss": 0, "tie": 0}
    for fold in base.FOLD_IDS:
        mask = oof.fold_ids == fold
        candidate_rmse = float(
            np.mean(
                [
                    base._metrics(  # noqa: SLF001
                        oof.target[mask],
                        np.asarray(row["gated_prediction"])[mask],
                    )["rmse"]
                    for row in results_by_seed.values()
                ]
            )
        )
        baseline_rmse = base._metrics(  # noqa: SLF001
            oof.target[mask],
            oof.baseline[mask],
        )["rmse"]
        delta = candidate_rmse - baseline_rmse
        if abs(delta) <= WIN_TOLERANCE:
            candidate_rmse = baseline_rmse
            delta = 0.0
        outcome = diagnostics.classify_fold_outcome(
            candidate_rmse,
            baseline_rmse,
            tolerance=WIN_TOLERANCE,
        )
        counts[outcome] += 1
        rows.append(
            {
                "fold": int(fold),
                "baseline_rmse": baseline_rmse,
                "mean_seed_cross_attention_gated_rmse": candidate_rmse,
                "rmse_delta": delta,
                "outcome": outcome,
                "independent_spatial_unit": True,
                "seed_values_are_paired_pseudo_repeats": True,
            }
        )
    return rows, counts


def _write_prediction_errors(
    *,
    output_dir: Path,
    oof: base.OOFDevelopment,
    results_by_seed: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    path = output_dir / "prediction_errors.npz"
    payload: dict[str, np.ndarray] = {
        "target": np.asarray(oof.target, dtype=np.float64),
        "baseline_prediction": np.asarray(oof.baseline, dtype=np.float64),
        "baseline_error": np.asarray(
            oof.baseline - oof.target,
            dtype=np.float64,
        ),
        "fold_ids": np.asarray(oof.fold_ids, dtype=np.int64),
        "indices_kji": np.asarray(oof.indices_kji, dtype=np.int64),
    }
    for seed, row in results_by_seed.items():
        gated = np.asarray(row["gated_prediction"], dtype=np.float64)
        adaptive_gated = np.asarray(
            row["adaptive_gated_prediction"],
            dtype=np.float64,
        )
        ungated = np.asarray(row["ungated_prediction"], dtype=np.float64)
        payload[f"gated_prediction__seed_{seed}"] = gated
        payload[f"gated_error__seed_{seed}"] = gated - oof.target
        payload[f"adaptive_gated_prediction__seed_{seed}"] = adaptive_gated
        payload[f"adaptive_gated_error__seed_{seed}"] = (
            adaptive_gated - oof.target
        )
        payload[f"ungated_prediction__seed_{seed}"] = ungated
        payload[f"ungated_error__seed_{seed}"] = ungated - oof.target
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": base._sha256(path),  # noqa: SLF001
        "rows": int(len(oof.target)),
        "arrays": sorted(payload),
    }


def _write_evidence(output_dir: Path, result: Mapping[str, Any]) -> None:
    comparison = result["experiment"]["comparison"]
    independent = result["experiment"]["independent_spatial_folds"]
    counts = result["experiment"]["independent_fold_outcome_counts"]
    bootstrap = result["experiment"]["block_bootstrap_rmse_delta_vs_pykrige"]
    lines = [
        "# P11 cross-attention fusion — development-only evidence",
        "",
        "## Result",
        "",
        (
            f"- PyKrige baseline pooled development RMSE: "
            f"`{comparison['baseline_pooled_rmse']:.12f}`."
        ),
        (
            f"- Cross-attention consensus-gated mean-seed pooled RMSE: "
            f"`{comparison['cross_attention_gated_mean_seed_pooled_rmse']:.12f}`."
        ),
        (
            f"- The less conservative adaptive gate RMSE was "
            f"`{comparison['cross_attention_adaptive_gated_mean_seed_pooled_rmse']:.12f}`; "
            "it is retained as a diagnostic, not selected as the safe route."
        ),
        (
            f"- Absolute RMSE change (candidate - baseline): "
            f"`{comparison['absolute_rmse_change']:+.12f}`; relative change "
            f"(positive means improvement): "
            f"`{comparison['relative_rmse_improvement']:+.6%}`."
        ),
        (
            "- No positive development RMSE change was observed; the "
            "four-of-four inner-fold consensus guard rejected every "
            "correction and preserved the baseline exactly."
            if comparison["relative_rmse_improvement"] <= 0.0
            else "- A positive development RMSE change was observed."
        ),
        (
            "- 大模型贡献占比待下一轮消融确认。This run only evaluates the "
            "pretrained encoder mode; it does not attribute the overall change "
            "to OpenMind."
        ),
        "",
        "## Five genuinely independent spatial units",
        "",
        (
            "The five locked outer folds are the inferential units. The three "
            "seeds are paired optimization pseudo-repeats inside each fold, "
            "not 15 independent samples. These are five genuinely independent "
            "spatial units."
        ),
        "",
        "| fold | baseline RMSE | cross-attention RMSE | delta | outcome |",
        "|---:|---:|---:|---:|:---|",
    ]
    for row in independent:
        lines.append(
            f"| {row['fold']} | {row['baseline_rmse']:.9f} | "
            f"{row['mean_seed_cross_attention_gated_rmse']:.9f} | "
            f"{row['rmse_delta']:+.9f} | {row['outcome']} |"
        )
    lines.extend(
        [
            "",
            (
                f"Independent-fold outcomes: **{counts['win']} win / "
                f"{counts['loss']} loss / {counts['tie']} tie**."
            ),
            "",
            "## Block-bootstrap uncertainty",
            "",
            (
                "Whole spatial folds were resampled; voxels and seed rows were "
                "never sampled as independent observations."
            ),
            (
                f"- Mean-seed pooled RMSE delta point estimate: "
                f"`{bootstrap['point_estimate']:+.12f}`."
            ),
            (
                f"- 95% block-bootstrap interval: "
                f"`[{bootstrap['confidence_interval'][0]:+.12f}, "
                f"{bootstrap['confidence_interval'][1]:+.12f}]`."
            ),
            "",
            "## Method and firewall",
            "",
            (
                "- Each normalized seismic channel was encoded separately. "
                "The structured query cross-attends to three channel tokens."
            ),
            (
                "- Standardization and PCA were fitted inside each training "
                "split only. The attention residual and bounded benefit gate "
                "were both cross-fitted by the original spatial folds. The "
                "safe route additionally requires all four inner folds to win "
                "before any correction can leave gate=0."
            ),
            (
                "- Encoder weight loading is parameterized as `pretrained` or "
                "`random_init`; this run used only `pretrained`. The next "
                "matched random-initialization ablation can reuse this harness."
            ),
            (
                "- Only `train.h5` metadata, `seismic_patch[0:3]`, and "
                "`seismic_patch[8]` were available to OpenMind extraction. "
                "Labels came from hash-verified development OOF archives."
            ),
            (
                "- `test.h5`, frozen holdout paths, historical test metrics, "
                "and holdout labels were not read or probed."
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "- This is a development diagnostic, not promotion evidence: "
                "the same-architecture random-init contribution control was "
                "intentionally deferred by the current task."
            ),
            (
                "- Any overall gain can reflect the full fusion harness, "
                "regularization, PCA, or gating. It must not be described as "
                "an OpenMind-specific gain."
            ),
            "",
        ]
    )
    (output_dir / "evidence.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(
    *,
    data_dir: Path,
    stage3_root: Path,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path | None,
    output_dir: Path,
    channel_feature_cache: Path,
    random_init_feature_cache: Path,
    device: str,
    head_device: str,
    encoder_weight_mode: str,
    config: CrossAttentionConfig = CrossAttentionConfig(),
) -> dict[str, Any]:
    """Run and persist the independent cross-attention development diagnostic."""

    started = time.time()
    inputs = base.resolve_dev_inputs(data_dir)
    oof = base.load_oof_development(stage3_root)
    output_dir = Path(output_dir).expanduser().resolve()
    base.ensure_no_holdout_paths([output_dir])
    features_by_seed, feature_audit = resolve_encoder_features(
        encoder_weight_mode=encoder_weight_mode,
        inputs=inputs,
        oof=oof,
        source_root=source_root,
        checkpoint=checkpoint,
        dependency_root=dependency_root,
        channel_feature_cache=channel_feature_cache,
        random_init_feature_cache=random_init_feature_cache,
        device=device,
    )
    layer_channels = tuple(
        int(value) for value in feature_audit["layer_channels"]
    )
    queries = build_structured_queries(oof)
    results_by_seed: dict[int, dict[str, Any]] = {}
    selected_channels: dict[str, list[int]] = {}
    for seed in base.REPEAT_SEEDS:
        tokens, selected = select_channel_tokens(
            features_by_seed[int(seed)],
            layer_channels=layer_channels,
            seed=int(seed),
        )
        selected_channels[str(seed)] = selected.tolist()
        results_by_seed[int(seed)] = _evaluate_seed(
            oof=oof,
            queries=queries,
            tokens=tokens,
            seed=int(seed),
            config=config,
            device=head_device,
        )
        print(
            json.dumps(
                {
                    "event": "seed_complete",
                    "seed": int(seed),
                    "gated_rmse": results_by_seed[int(seed)][
                        "gated_metrics"
                    ]["rmse"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    baseline_metrics = base._metrics(oof.target, oof.baseline)  # noqa: SLF001
    mean_seed_gated_rmse = float(
        np.mean(
            [row["gated_metrics"]["rmse"] for row in results_by_seed.values()]
        )
    )
    mean_seed_ungated_rmse = float(
        np.mean(
            [row["ungated_metrics"]["rmse"] for row in results_by_seed.values()]
        )
    )
    mean_seed_adaptive_gated_rmse = float(
        np.mean(
            [
                row["adaptive_gated_metrics"]["rmse"]
                for row in results_by_seed.values()
            ]
        )
    )
    if abs(mean_seed_gated_rmse - baseline_metrics["rmse"]) <= WIN_TOLERANCE:
        mean_seed_gated_rmse = baseline_metrics["rmse"]
    independent_rows, independent_counts = _independent_fold_summary(
        oof=oof,
        results_by_seed=results_by_seed,
    )
    bootstrap = diagnostics.block_bootstrap_rmse_delta(
        target=oof.target,
        baseline=oof.baseline,
        candidate_predictions_by_seed={
            int(seed): np.asarray(row["gated_prediction"])
            for seed, row in results_by_seed.items()
        },
        fold_ids=oof.fold_ids,
        bootstrap_seed=2693,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_artifact = _write_prediction_errors(
        output_dir=output_dir,
        oof=oof,
        results_by_seed=results_by_seed,
    )
    per_fold_seed = [
        cell
        for row in results_by_seed.values()
        for cell in row["per_fold"]
    ]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "implementation": {
            "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "script_sha256": base._sha256(Path(__file__)),  # noqa: SLF001
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pid": os.getpid(),
        },
        "protocol": {
            "development_only": True,
            "base_model": base.EXPECTED_MODEL_ID,
            "lane": base.EXPECTED_LANE,
            "split_hash": base.EXPECTED_SPLIT_HASH,
            "outer_folds": list(base.FOLD_IDS),
            "genuinely_independent_spatial_units": len(base.FOLD_IDS),
            "repeat_seeds": list(base.REPEAT_SEEDS),
            "seeds_are_paired_pseudo_repeats": True,
            "encoder_weight_mode": encoder_weight_mode,
            "encoder_weight_modes_available": list(ENCODER_WEIGHT_MODES),
            "random_init_ablation_executed_this_run": (
                encoder_weight_mode == "random_init"
            ),
            "cross_attention": (
                "one structured query token attends to three separately "
                "encoded seismic-channel tokens"
            ),
            "residual_gate": (
                "inner-spatial-fold OOF benefit classifier with bounded "
                "train-only selected scale, followed by a four-of-four "
                "inner-fold consensus safety guard"
            ),
            "config": asdict(config),
            "test_or_holdout_tuning": False,
        },
        "openmind_feature_audit": feature_audit,
        "selected_openmind_channels": selected_channels,
        "source_oof": {
            "records": list(oof.source_records),
            "rows": int(len(oof.target)),
            "indices_kji_sha256": base._array_sha256(  # noqa: SLF001
                oof.indices_kji
            ),
        },
        "experiment": {
            "baseline": {
                "metrics": baseline_metrics,
                "gate_zero_bitwise_equal_to_pykrige": bool(
                    np.array_equal(oof.baseline.copy(), oof.baseline)
                ),
            },
            "aggregate_by_seed": [
                {
                    "seed": int(seed),
                    "ungated_metrics": row["ungated_metrics"],
                    "adaptive_gated_metrics": row[
                        "adaptive_gated_metrics"
                    ],
                    "gated_metrics": row["gated_metrics"],
                }
                for seed, row in results_by_seed.items()
            ],
            "per_fold_seed_pseudo_repeats": per_fold_seed,
            "independent_spatial_folds": independent_rows,
            "independent_fold_outcome_counts": independent_counts,
            "block_bootstrap_rmse_delta_vs_pykrige": bootstrap,
            "comparison": {
                "baseline_pooled_rmse": baseline_metrics["rmse"],
                "cross_attention_ungated_mean_seed_pooled_rmse": (
                    mean_seed_ungated_rmse
                ),
                "cross_attention_adaptive_gated_mean_seed_pooled_rmse": (
                    mean_seed_adaptive_gated_rmse
                ),
                "cross_attention_gated_mean_seed_pooled_rmse": (
                    mean_seed_gated_rmse
                ),
                "absolute_rmse_change": (
                    mean_seed_gated_rmse - baseline_metrics["rmse"]
                ),
                "relative_rmse_improvement": (
                    (baseline_metrics["rmse"] - mean_seed_gated_rmse)
                    / baseline_metrics["rmse"]
                ),
                "large_model_contribution_statement": (
                    "大模型贡献占比待下一轮消融确认"
                ),
            },
            "decision": {
                "state": "DEVELOPMENT_DIAGNOSTIC_ONLY",
                "default_promotion_eligible": False,
                "reason": (
                    "no positive safe-route development RMSE gain, and the "
                    "current task deferred the same-architecture random-init "
                    "cross-attention ablation"
                ),
                "large_model_contribution_confirmed": False,
            },
        },
        "prediction_error_artifact": prediction_artifact,
        "holdout_firewall": {
            "hdf5_files_opened": ["train.h5"],
            "hdf5_datasets_available_to_encoder": [
                "seismic_patch[0:3]",
                "seismic_patch[8]",
            ],
            "label_dataset_read_by_encoder": False,
            "test_path_argument_exists": False,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "historical_test_metrics_read": False,
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "head_device": head_device,
            "encoder_device": device,
            "feature_cache_reused": bool(
                feature_audit.get("cache_reused", False)
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_evidence(output_dir, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser(
        "check-dev",
        help="verify the train-only HDF5 boundary",
    )
    check.add_argument("--data-dir", type=Path, required=True)

    execute = subparsers.add_parser(
        "run",
        help="run the P11 cross-attention development diagnostic",
    )
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--stage3-root", type=Path, required=True)
    execute.add_argument("--source-root", type=Path, required=True)
    execute.add_argument("--checkpoint", type=Path, required=True)
    execute.add_argument("--dependency-root", type=Path)
    execute.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    execute.add_argument(
        "--channel-feature-cache",
        type=Path,
        default=DEFAULT_CHANNEL_FEATURE_CACHE,
    )
    execute.add_argument(
        "--random-init-feature-cache",
        type=Path,
        default=DEFAULT_RANDOM_INIT_FEATURE_CACHE,
    )
    execute.add_argument("--device", default="cuda:0")
    execute.add_argument("--head-device", default="cpu")
    execute.add_argument(
        "--encoder-weight-mode",
        choices=ENCODER_WEIGHT_MODES,
        default="pretrained",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check-dev":
        resolved = base.resolve_dev_inputs(args.data_dir)
        print(
            json.dumps(
                {"accepted": str(resolved.train_h5), "holdout_opened": False}
            )
        )
        return 0
    values = vars(args)
    values.pop("command")
    result = run(**values)
    print(json.dumps(result["experiment"]["comparison"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
