#!/usr/bin/env python3
"""P11 diagnostic: native-length MOMENT encoding of 13 real log curves only.

This is a deliberately isolated representation diagnostic built on the
committed P11 gated-residual harness.  It changes exactly one representation
contract:

* MOMENT receives only the first 13 physical well-log curves as ``[B,13,33]``;
* no interpolation or other resampling is performed;
* the 13 observation-mask planes and 9 seismic-patch traces are excluded.

The pinned MOMENT patch length/stride are both eight, so a native 33-point
window produces four real tokens from 32 measured-depth samples.  The final
unpatched point is reported rather than synthetically stretched.

Every legal development cell still evaluates ``baseline``, ``direct``,
``pretrained``, ``random``, and exact ``gate0`` under the same strict LOGO4
four-fold by three-seed protocol and the same 40-update training budget.  No
command accepts or opens a frozen-test or known-holdout input.
"""
from __future__ import annotations

import argparse
import json
import os
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
from momentfm import MOMENTPipeline  # noqa: E402


SCHEMA_VERSION = "lithofacies-p11-clean-well-native33/v1"
EMBEDDING_SCHEMA = "lithofacies-p11-clean-well-embeddings/v1"
ARTIFACT_SCHEMA = "lithofacies-p11-clean-well-artifacts/v1"
INPUT_CHANNELS = 13
INPUT_LENGTH = 33
PATCH_LENGTH = 8
PATCH_STRIDE = 8
PATCH_COUNT = (INPUT_LENGTH - PATCH_LENGTH) // PATCH_STRIDE + 1
EFFECTIVE_CONTEXT_STEPS = PATCH_LENGTH + (PATCH_COUNT - 1) * PATCH_STRIDE
TRAILING_UNPATCHED_STEPS = INPUT_LENGTH - EFFECTIVE_CONTEXT_STEPS
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p11_clean_well_native33"
LEGACY_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p11_residual_fusion"
LEGACY_P11_HASHES = {
    "artifact_manifest.json": (
        "d81c3dc0647020186004c3630825d732e8ad8d18c975131c20ec14be0e97be14"
    ),
    "evidence.md": (
        "eac5c4e31c86dcd080993fc82fee5958b41402edc7dfee666f97b6c80bc2d1f6"
    ),
    "primary_metric.png": (
        "c3a82df5905027027078c9545dc6afe0c74c9dd6174775a59cbc72bc32c739db"
    ),
    "results.jsonl": (
        "9c73c5ead55a4ee3e472368dbfdadc811ce92da5a2ffa321601cc23b550e7e3c"
    ),
    "summary.json": (
        "86d1cda65a9e67e92dff1278fe6bed423fd5d4a8de918547d20eac1b60a75466"
    ),
}


def _clean_log_inputs(well: np.ndarray) -> np.ndarray:
    """Select physical curves without admitting mask planes or seismic data."""
    values = np.asarray(well, dtype=np.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != (26, INPUT_LENGTH):
        raise ValueError(
            f"well tensor must be [B,26,{INPUT_LENGTH}], got {values.shape}"
        )
    clean = values[:, :INPUT_CHANNELS, :]
    if (
        tuple(clean.shape[1:]) != (INPUT_CHANNELS, INPUT_LENGTH)
        or not np.isfinite(clean).all()
    ):
        raise ValueError(f"invalid clean well-log input: {clean.shape}")
    return np.ascontiguousarray(clean)


def _verify_legacy_outputs() -> dict[str, str]:
    """Prove that the committed 60-cell P11 evidence remains byte-identical."""
    observed: dict[str, str] = {}
    for name, expected in LEGACY_P11_HASHES.items():
        path = LEGACY_OUTPUT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"committed P11 evidence is missing: {path}")
        digest = p11._sha256(path)
        if digest != expected:
            raise RuntimeError(f"committed P11 evidence changed: {path}")
        observed[name] = digest
    return observed


class NativeLogMomentClassifier(torch.nn.Module):
    """Strict wrapper around a native 13-channel, 33-point MOMENT pipeline."""

    def __init__(self, pipeline: torch.nn.Module) -> None:
        super().__init__()
        self.pipeline = pipeline

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (
            INPUT_CHANNELS,
            INPUT_LENGTH,
        ):
            raise ValueError(
                f"native MOMENT input must be [B,{INPUT_CHANNELS},{INPUT_LENGTH}]"
            )
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("native MOMENT input contains non-finite values")
        mask = torch.ones(
            (inputs.shape[0], INPUT_LENGTH),
            dtype=torch.long,
            device=inputs.device,
        )
        result = self.pipeline(x_enc=inputs, input_mask=mask)
        logits = getattr(result, "logits", None)
        if logits is None or logits.shape != (inputs.shape[0], 9):
            shape = None if logits is None else tuple(logits.shape)
            raise ValueError(f"unexpected native MOMENT logits: {shape}")
        return logits


def _build_native_moment(
    *,
    snapshot: Path,
    device: str,
    seed: int,
    random_init: bool,
) -> NativeLogMomentClassifier:
    p11._seed_all(seed)
    pipeline = MOMENTPipeline.from_pretrained(
        str(snapshot),
        local_files_only=True,
        model_kwargs={
            "task_name": "classification",
            "n_channels": INPUT_CHANNELS,
            "num_class": 9,
            "seq_len": INPUT_LENGTH,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
    )
    pipeline.init()
    model = NativeLogMomentClassifier(pipeline).to(device)
    if (
        int(model.pipeline.config.seq_len) != INPUT_LENGTH
        or int(model.pipeline.config.patch_len) != PATCH_LENGTH
        or int(model.pipeline.config.patch_stride_len) != PATCH_STRIDE
    ):
        raise RuntimeError("pinned MOMENT native-context contract changed")
    if random_init:
        p11._randomize_frozen_backbone(model, seed=seed)
    return model


def _extract_native_embeddings(
    model: NativeLogMomentClassifier,
    values: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Encode native inputs directly; no resampling operation is permitted."""
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), p11.BATCH_SIZE):
            inputs = torch.as_tensor(
                values[start : start + p11.BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            if tuple(inputs.shape[1:]) != (INPUT_CHANNELS, INPUT_LENGTH):
                raise ValueError(
                    "native embedding input must be "
                    f"[B,{INPUT_CHANNELS},{INPUT_LENGTH}]"
                )
            mask = torch.ones(
                (inputs.shape[0], INPUT_LENGTH),
                dtype=torch.long,
                device=device,
            )
            result = model.pipeline.embed(
                x_enc=inputs,
                input_mask=mask,
                reduction="mean",
            )
            embeddings = getattr(result, "embeddings", None)
            if (
                embeddings is None
                or embeddings.ndim != 2
                or embeddings.shape[0] != inputs.shape[0]
                or not bool(torch.isfinite(embeddings).all())
            ):
                shape = None if embeddings is None else tuple(embeddings.shape)
                raise ValueError(f"invalid native MOMENT embeddings: {shape}")
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
        "schema_version": EMBEDDING_SCHEMA,
        "snapshot_sha256": p11.EXPECTED_SNAPSHOT_SHA256,
        "development_batch_sha256": development_batch_sha256,
        "fold_id": int(fold_id),
        "repeat_id": None if repeat_id is None else int(repeat_id),
        "seed": int(seed),
        "random_init": bool(random_init),
        "reduction": "mean",
        "input_channels": INPUT_CHANNELS,
        "input_length": INPUT_LENGTH,
        "patch_length": PATCH_LENGTH,
        "patch_stride": PATCH_STRIDE,
        "patch_count": PATCH_COUNT,
        "effective_context_steps": EFFECTIVE_CONTEXT_STEPS,
        "trailing_unpatched_steps": TRAILING_UNPATCHED_STEPS,
        "resampling": "none",
        "observation_masks_in_moment": False,
        "seismic_in_moment": False,
        "train_rows": int(len(train_x)),
        "validation_rows": int(len(validation_x)),
    }
    contract_hash = p11._stable_hash(contract)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            train_embeddings = cached["train_embeddings"].copy()
            validation_embeddings = cached["validation_embeddings"].copy()
            manifest = json.loads(str(cached["manifest"].item()))
        if manifest.get("contract_hash") != contract_hash:
            raise RuntimeError(f"native embedding cache contract changed: {cache_path}")
    else:
        model = _build_native_moment(
            snapshot=snapshot,
            device=device,
            seed=seed,
            random_init=random_init,
        )
        train_embeddings = _extract_native_embeddings(
            model,
            train_x,
            device=device,
        )
        validation_embeddings = _extract_native_embeddings(
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
        raise RuntimeError(f"invalid native embedding cache arrays: {cache_path}")
    evidence = {
        **manifest,
        "cache_path": p11._portable_path(cache_path),
        "cache_sha256": p11._sha256(cache_path),
    }
    return train_embeddings, validation_embeddings, evidence


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
    model = _build_native_moment(
        snapshot=snapshot,
        device=device,
        seed=seed,
        random_init=False,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=p11.LEARNING_RATE,
        weight_decay=p11.WEIGHT_DECAY,
    )
    weights = torch.as_tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )
    losses: list[float] = []
    model.train()
    for indices in p11._batch_indices(len(train_y), seed=seed):
        inputs = torch.as_tensor(
            train_x[indices],
            dtype=torch.float32,
            device=device,
        )
        labels = torch.as_tensor(
            train_y[indices],
            dtype=torch.long,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels, weight=weights)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("native direct MOMENT loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(validation_x), p11.BATCH_SIZE):
            inputs = torch.as_tensor(
                validation_x[start : start + p11.BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(inputs).detach().cpu().numpy())
    logits = np.concatenate(outputs).astype(np.float32, copy=False)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits, {
        "updates": p11.UPDATES,
        "first_train_loss": losses[0],
        "last_train_loss": losses[-1],
        "minimum_train_loss": min(losses),
        "duration_seconds": time.perf_counter() - started,
        "gate_mean": None,
        "gate_max": None,
        "residual_contribution_mean_abs": None,
        "residual_contribution_max_abs": None,
        "random_init": False,
        "input_contract": "13_real_logs_native_33_no_resampling",
    }


def _result_row(
    *,
    fold_id: int,
    repeat_id: int,
    variant: str,
    labels: np.ndarray,
    logits: np.ndarray,
    training: Mapping[str, Any],
    baseline_cell: Mapping[str, Any],
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
        "input_contract": {
            "moment_channels": INPUT_CHANNELS,
            "moment_length": INPUT_LENGTH,
            "resampling": "none",
            "observation_masks_in_moment": False,
            "observation_masks_in_residual_head": False,
            "seismic_in_moment": False,
            "seismic_in_residual_head": False,
        },
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
        "rank_eligible": True,
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_ids: Sequence[int],
    repeat_ids: Sequence[int],
) -> dict[str, Any]:
    summary = p11.summarize_results(
        rows,
        fold_ids=fold_ids,
        repeat_ids=repeat_ids,
    )
    pretrained_gate = summary["variants"]["pretrained"]["gate_mean"]
    random_gate = summary["variants"]["random"]["gate_mean"]
    pretrained_random_delta = summary["comparison"]["pretrained_minus_random"]
    material_separation = abs(pretrained_random_delta) >= p11.MIN_PROMOTION_DELTA
    summary["schema_version"] = SCHEMA_VERSION
    summary["representation_diagnostic"] = {
        "question": (
            "Does pretrained MOMENT separate from same-architecture random "
            "initialization after removing synthetic length expansion and "
            "non-log pseudo-channels?"
        ),
        "pretrained_minus_random_fixed_schema_macro_f1": pretrained_random_delta,
        "absolute_materiality_threshold": p11.MIN_PROMOTION_DELTA,
        "material_separation_detected": material_separation,
        "pretrained_gate_mean": pretrained_gate,
        "random_gate_mean": random_gate,
        "pretrained_minus_random_gate_mean": (
            float(pretrained_gate - random_gate)
            if pretrained_gate is not None and random_gate is not None
            else None
        ),
        "state": (
            "MATERIAL_PRETRAINED_RANDOM_SEPARATION"
            if material_separation
            else "NO_MATERIAL_PRETRAINED_RANDOM_SEPARATION"
        ),
    }
    return summary


def _write_figure(summary: Mapping[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [
        summary["variants"][variant]["metrics"]["fixed_schema_macro_f1"]["mean"]
        for variant in p11.ABLATIONS
    ]
    errors = [
        summary["variants"][variant]["metrics"]["fixed_schema_macro_f1"]["std"]
        for variant in p11.ABLATIONS
    ]
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#9D9DAA"]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    positions = np.arange(len(p11.ABLATIONS))
    axis.bar(positions, values, yerr=errors, color=colors, capsize=4)
    axis.set_xticks(positions, p11.ABLATIONS)
    axis.set_ylabel("fixed-schema macro-F1")
    axis.set_title(
        "P11 clean well-log native-context diagnostic\n"
        "13 curves × 33 points, strict LOGO4 × 3 seeds"
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
    diagnostic = summary["representation_diagnostic"]
    lines = [
        "# P11 clean well-log native-context diagnostic evidence",
        "",
        "## Answer",
        "",
        f"Representation-control state: `{diagnostic['state']}`. "
        "Pretrained minus random-init fixed-nine Macro-F1 is "
        f"`{diagnostic['pretrained_minus_random_fixed_schema_macro_f1']:+.6f}` "
        f"against the existing absolute materiality threshold "
        f"`{diagnostic['absolute_materiality_threshold']:.3f}`.",
        "",
        "No larger MOMENT model was run or selected. The separate mask-residual "
        "and seismic-CNN late-fusion phase remains outside this minimal "
        "diagnostic, exactly so this comparison isolates the representation.",
        "",
        "## Native representation contract",
        "",
        f"- MOMENT input: `{INPUT_CHANNELS}` real normalized log curves × "
        f"`{INPUT_LENGTH}` measured-depth samples.",
        "- Resampling/interpolation: `none`.",
        f"- Pinned patch length/stride: `{PATCH_LENGTH}/{PATCH_STRIDE}`; "
        f"`{PATCH_COUNT}` real tokens cover `{EFFECTIVE_CONTEXT_STEPS}` samples; "
        f"`{TRAILING_UNPATCHED_STEPS}` trailing sample is reported as unpatched.",
        "- The 13 binary observation-mask planes and 9 flattened seismic "
        "traces never enter MOMENT or the residual head in this clean-input run.",
        "",
        "## Strict LOGO4 ablations",
        "",
        "| variant | fixed-9 macro-F1 mean | std | accuracy | gate mean "
        "| mean abs residual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in p11.ABLATIONS:
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
    comparison = summary["comparison"]
    lines.extend(
        [
            "",
            "## Paired checks",
            "",
            f"- pretrained − baseline: "
            f"`{comparison['pretrained_minus_baseline']:+.6f}`.",
            f"- pretrained − random: "
            f"`{comparison['pretrained_minus_random']:+.6f}`.",
            f"- pretrained − direct: "
            f"`{comparison['pretrained_minus_direct']:+.6f}`.",
            f"- pretrained wins over baseline in "
            f"`{comparison['pretrained_pair_wins_over_baseline']}/"
            f"{comparison['paired_comparisons']}` fold/seed pairs.",
            f"- gate0 − baseline: `{comparison['gate0_minus_baseline']:+.6f}`; "
            "maximum logit error "
            f"`{comparison['gate0_max_abs_logit_error']:.1e}`.",
            "",
            "## Every fold and seed",
            "",
            "| fold | seed | baseline | direct | pretrained | random | gate0 "
            "| pretrained gate | random gate |",
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
                for variant in p11.ABLATIONS
            }
            values = {
                variant: selected[variant]["metrics"]["fixed_schema_macro_f1"]
                for variant in p11.ABLATIONS
            }
            lines.append(
                f"| {fold_id} | {selected['baseline']['seed']} | "
                f"{values['baseline']:.6f} | {values['direct']:.6f} | "
                f"{values['pretrained']:.6f} | {values['random']:.6f} | "
                f"{values['gate0']:.6f} | "
                f"{selected['pretrained']['training']['gate_mean']:.6f} | "
                f"{selected['random']['training']['gate_mean']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Leakage and preservation audit",
            "",
            "- Inputs are the immutable four development mother-family folds. "
            "Every holdout-like input path is rejected before opening.",
            "- Normalization, class weights, Stage-3 logits, LOGO4 membership, "
            "seeds, metric, batch size, and 40-update budget are unchanged.",
            "- The committed original P11 60-cell artifacts were checked "
            "byte-for-byte before this run:",
        ]
    )
    for name, digest in summary["preserved_p11_artifacts"].items():
        lines.append(f"  - `{name}`: `{digest}`")
    lines.extend(
        [
            "- This evidence is development-only and cannot authorize a frozen "
            "holdout read or competition claim.",
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
    snapshot: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda:0",
    fold_ids: Sequence[int] = p11.FOLD_IDS,
    repeat_ids: Sequence[int] = tuple(range(len(p11.REPEAT_SEEDS))),
    resume: bool = True,
) -> dict[str, Any]:
    p11.ensure_development_only_paths((development_batch, baseline_bundle, snapshot))
    preserved_hashes = _verify_legacy_outputs()
    folds, repeats = p11._selection(fold_ids, repeat_ids)
    snapshot_weights = snapshot / "model.safetensors"
    if p11._sha256(snapshot_weights) != p11.EXPECTED_SNAPSHOT_SHA256:
        raise RuntimeError("unexpected MOMENT snapshot weights")
    arrays, batch_manifest = p11.load_stage3_batch(development_batch)
    if (
        batch_manifest.get("split_hash") != p11.EXPECTED_SPLIT_HASH
        or batch_manifest.get("frozen_test_accessed") is not False
    ):
        raise RuntimeError("development batch is not strict LOGO4")
    batch_sha256 = p11._sha256(development_batch)
    baseline_arrays, baseline_manifest = p11._load_baseline_bundle(
        baseline_bundle,
        development_batch_sha256=batch_sha256,
        fold_ids=folds,
        repeat_ids=repeats,
    )
    run_contract = {
        "schema_version": SCHEMA_VERSION,
        "split_hash": p11.EXPECTED_SPLIT_HASH,
        "development_batch_sha256": batch_sha256,
        "baseline_bundle_sha256": p11._sha256(baseline_bundle),
        "snapshot_sha256": p11.EXPECTED_SNAPSHOT_SHA256,
        "fold_ids": list(folds),
        "repeat_ids": list(repeats),
        "seeds": [int(p11.REPEAT_SEEDS[index]) for index in repeats],
        "ablations": list(p11.ABLATIONS),
        "updates": p11.UPDATES,
        "batch_size": p11.BATCH_SIZE,
        "learning_rate": p11.LEARNING_RATE,
        "weight_decay": p11.WEIGHT_DECAY,
        "gate_regularization": p11.GATE_REGULARIZATION,
        "residual_regularization": p11.RESIDUAL_REGULARIZATION,
        "max_residual_logit": p11.MAX_RESIDUAL_LOGIT,
        "moment_input_channels": INPUT_CHANNELS,
        "moment_input_length": INPUT_LENGTH,
        "patch_count": PATCH_COUNT,
        "effective_context_steps": EFFECTIVE_CONTEXT_STEPS,
        "trailing_unpatched_steps": TRAILING_UNPATCHED_STEPS,
        "resampling": "none",
        "observation_masks_in_model": False,
        "seismic_in_model": False,
        "embedding_reduction": "mean",
        "embedding_cache": True,
        "preserved_p11_artifacts": preserved_hashes,
    }
    contract_hash = p11._stable_hash(run_contract)
    runtime_dir = output_dir / "runtime"
    partial_path = runtime_dir / "partial_results.jsonl"
    contract_path = runtime_dir / "run_contract.json"
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract.get("contract_hash") != contract_hash:
            raise RuntimeError("existing native diagnostic has a different contract")
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
        p11._validate_pair_rows(pair_rows)
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
            seed = int(p11.REPEAT_SEEDS[repeat_id])
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
            ) = p11._train_residual(
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
            random_logits, _, random_training = p11._train_residual(
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
                ),
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="direct",
                    labels=validation_y,
                    logits=direct_logits,
                    training=direct_training,
                    baseline_cell=baseline_cell,
                ),
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="pretrained",
                    labels=validation_y,
                    logits=pretrained_logits,
                    training=pretrained_training,
                    baseline_cell=baseline_cell,
                ),
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="random",
                    labels=validation_y,
                    logits=random_logits,
                    training=random_training,
                    baseline_cell=baseline_cell,
                ),
                _result_row(
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    variant="gate0",
                    labels=validation_y,
                    logits=gate0_logits,
                    training=gate0_training,
                    baseline_cell=baseline_cell,
                ),
            ]
            p11._validate_pair_rows(pair_rows)
            all_rows.extend(pair_rows)
            p11._write_jsonl(partial_path, all_rows)
            completed[key] = pair_rows
            print(
                f"complete fold={fold_id} repeat={repeat_id} seed={seed}: "
                f"baseline={pair_rows[0]['metrics']['fixed_schema_macro_f1']:.6f} "
                f"pretrained={pair_rows[2]['metrics']['fixed_schema_macro_f1']:.6f} "
                f"random={pair_rows[3]['metrics']['fixed_schema_macro_f1']:.6f} "
                f"gate={pretrained_training['gate_mean']:.6f}",
                flush=True,
            )
    summary = _summarize(
        all_rows,
        fold_ids=folds,
        repeat_ids=repeats,
    )
    summary["model"] = {
        "foundation_model": "AutonLab/MOMENT-1-base",
        "weights_sha256": p11.EXPECTED_SNAPSHOT_SHA256,
        "real_pretrained_weights_loaded": True,
        "baseline_model": p11.BASELINE_MODEL_ID,
        "residual_formula": (
            "baseline_logits + sigmoid(per_class_gate_logits) * "
            "2*tanh(linear(mean_reduced_cached_native_log_embeddings))"
        ),
        "pretrained_random_same_architecture": True,
        "moment_embeddings_cached": True,
        "max_residual_logit": p11.MAX_RESIDUAL_LOGIT,
        "larger_moment_model_run": False,
    }
    summary["representation"] = {
        "moment_input": f"[B,{INPUT_CHANNELS},{INPUT_LENGTH}]",
        "real_log_channels": INPUT_CHANNELS,
        "observation_mask_channels": 0,
        "seismic_channels": 0,
        "resampling": "none",
        "configured_context_length": INPUT_LENGTH,
        "patch_length": PATCH_LENGTH,
        "patch_stride": PATCH_STRIDE,
        "patch_count": PATCH_COUNT,
        "effective_context_steps": EFFECTIVE_CONTEXT_STEPS,
        "trailing_unpatched_steps": TRAILING_UNPATCHED_STEPS,
        "next_late_fusion_phase_run": False,
        "larger_model_requires_explicit_user_decision": True,
    }
    summary["inputs"] = {
        "development_batch_sha256": batch_sha256,
        "baseline_bundle_sha256": p11._sha256(baseline_bundle),
        "stage3_results_sha256": baseline_manifest["stage3_results_sha256"],
    }
    summary["preserved_p11_artifacts"] = preserved_hashes
    summary["runtime"] = {
        "device": device,
        "duration_seconds_this_invocation": time.perf_counter() - started,
        "resumed_pairs": len(existing_rows) // len(p11.ABLATIONS),
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
    preserved = _verify_legacy_outputs()
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("known_holdout_accessed") is not False
    ):
        raise RuntimeError("invalid native diagnostic artifact manifest")
    for artifact in manifest.get("artifacts", ()):
        path = PROJECT_ROOT / artifact["path"]
        if (
            not path.is_file()
            or p11._sha256(path) != artifact["sha256"]
            or path.stat().st_size != int(artifact["bytes"])
        ):
            raise RuntimeError(
                f"native diagnostic artifact verification failed: {path}"
            )
    rows = p11._read_jsonl(output_dir / "results.jsonl")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    evaluation = summary["evaluation"]
    recomputed = _summarize(
        rows,
        fold_ids=evaluation["fold_ids"],
        repeat_ids=evaluation["repeat_ids"],
    )
    if (
        recomputed["comparison"] != summary["comparison"]
        or recomputed["representation_diagnostic"]
        != summary["representation_diagnostic"]
    ):
        raise RuntimeError("native diagnostic summary is not reproducible")
    if (
        summary.get("preserved_p11_artifacts") != preserved
        or summary.get("representation", {}).get("resampling") != "none"
        or summary["representation"].get("real_log_channels") != INPUT_CHANNELS
        or summary["representation"].get("observation_mask_channels") != 0
        or summary["representation"].get("seismic_channels") != 0
    ):
        raise RuntimeError("native representation or preservation evidence changed")
    from PIL import Image

    with Image.open(output_dir / "primary_metric.png") as image:
        image.verify()
    return {
        "status": "verified",
        "artifacts": len(manifest["artifacts"]),
        "rows": len(rows),
        "representation_diagnostic": summary["representation_diagnostic"],
        "decision": summary["decision"],
        "legacy_artifacts_preserved": len(preserved),
    }


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--development-batch", type=Path, required=True)
    run.add_argument("--baseline-bundle", type=Path, required=True)
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--device", default="cuda:0")
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
    if args.command == "run":
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
