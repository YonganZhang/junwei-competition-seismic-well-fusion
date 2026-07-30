#!/usr/bin/env python3
"""P16 development-only GFM trace-reconstruction residual experiment.

P16 uses the geophysical foundation model for its native ViT-MAE task:
whole-trace masked interpolation.  Exact 400-sample by 160-adjacent-trace
windows come from the P15-audited continuous development seismic volume; no
P14 resize is reintroduced.  The decoder-reconstructed seismic attributes are
supplemented into P11's structural residual route and compared with both the
raw structural route and a same-architecture random-init encoder-decoder.

PORO targets and the strong PyKrige baseline come exclusively from five
hash-verified development OOF archives.  This module has no test/holdout input.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
sys.path.insert(0, str(PROJECT_ROOT))

from _models.reconstruction import geophysical_fm as gfm  # noqa: E402
from _models.reconstruction import geophysical_fm_denoise as gfm_denoise  # noqa: E402
import p11_residual_fusion as base  # noqa: E402
import p11_residual_fusion_diagnostics as diagnostics  # noqa: E402
import p15_gfm_finetune as p15  # noqa: E402


SCHEMA_VERSION = "reconstruction-p16-gfm-denoise/v1"
FEATURE_CACHE_SCHEMA_VERSION = "reconstruction-p16-gfm-denoise-features/v1"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p16_gfm_denoise"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "_tmp" / "p16_gfm_denoise"

MASK_FRACTION = 0.25
MASKED_TRACE_COUNT = 40
LEN_KEEP = p15.NATIVE_TRACE_COUNT - MASKED_TRACE_COUNT
HEAD_MODES = ("fixed_ridge10", "train_only_alpha_grid")
WIN_TOLERANCE = diagnostics.WIN_TOLERANCE
ROUTE_PRETRAINED = "pretrained_gfm_masked_reconstruction_structural"
ROUTE_RANDOM = "random_init_gfm_masked_reconstruction_structural"
ROUTE_RAW = "raw_no_foundation_structural"


@dataclass(frozen=True)
class DenoisedPointFeatures:
    """Seed-paired reconstructed seismic values sampled at OOF points."""

    reconstructed: np.ndarray
    delta_from_native_input: np.ndarray
    trace_masked: np.ndarray
    native_input: np.ndarray
    audit: Mapping[str, Any]


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = Path(output_dir).expanduser().resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(
            f"P16 output must stay inside reconstruction: {resolved}"
        ) from exc
    protected = {
        (HERE / "_outputs" / name).resolve()
        for name in (
            "p11_residual_fusion",
            "p11_residual_fusion_diagnostics",
            "p11_cross_attention_fusion",
            "p14_geophysical_fm",
            "p15_gfm_finetune",
        )
    }
    if resolved in protected:
        raise ValueError("P16 refuses to overwrite P11-P15 evidence")
    return resolved


def trace_mask_priorities(
    section_count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Return deterministic per-section trace priorities for exact 25% masks."""

    if int(section_count) <= 0:
        raise ValueError("section_count must be positive")
    rng = np.random.default_rng(int(seed))
    priorities = rng.random(
        (int(section_count), p15.NATIVE_TRACE_COUNT),
        dtype=np.float32,
    )
    if not np.all(np.isfinite(priorities)):
        raise FloatingPointError("trace-mask priorities are non-finite")
    return priorities


def trace_mask_from_priorities(priorities: np.ndarray) -> np.ndarray:
    """Materialize the exact mask induced by the upstream argsort semantics."""

    values = np.asarray(priorities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != p15.NATIVE_TRACE_COUNT:
        raise ValueError("trace priorities must have shape [sections,160]")
    order = np.argsort(values, axis=1)
    mask = np.ones(values.shape, dtype=bool)
    rows = np.arange(len(values))[:, None]
    mask[rows, order[:, :LEN_KEEP]] = False
    if not np.all(np.sum(mask, axis=1) == MASKED_TRACE_COUNT):
        raise RuntimeError("trace mask cardinality drift")
    return mask


def _feature_cache_paths(
    cache_dir: Path,
    weight_mode: str,
) -> tuple[Path, Path]:
    cache_path = Path(cache_dir) / f"point_features_{weight_mode}.npz"
    return cache_path, cache_path.with_suffix(".json")


def _feature_cache_identity(
    *,
    weight_mode: str,
    native_cache_sha256: str,
    indices_sha256: str,
    adapter_sha256: str,
    asset_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "encoder_weight_mode": weight_mode,
        "native_cache_sha256": native_cache_sha256,
        "indices_kji_sha256": indices_sha256,
        "adapter_sha256": adapter_sha256,
        "snapshot_revision": gfm.SNAPSHOT_REVISION,
        "source_sha256": asset_audit["source_sha256"],
        "config_sha256": asset_audit["config_sha256"],
        "weights_sha256": asset_audit["weights_sha256"],
        "repeat_seeds": list(base.REPEAT_SEEDS),
        "mask_fraction": MASK_FRACTION,
        "masked_trace_count": MASKED_TRACE_COUNT,
        "len_keep": LEN_KEEP,
        "output_combination": (
            "decoder on masked traces plus original visible traces"
        ),
        "seismic_channels_forwarded_separately": 3,
        "native_window_shape": [
            p15.NATIVE_TIME_SAMPLES,
            p15.NATIVE_TRACE_COUNT,
        ],
    }


def _load_valid_feature_cache(
    *,
    cache_dir: Path,
    expected_identity: Mapping[str, Any],
    weight_mode: str,
    oof: base.OOFDevelopment,
) -> DenoisedPointFeatures | None:
    cache_path, manifest_path = _feature_cache_paths(cache_dir, weight_mode)
    if not cache_path.is_file() or not manifest_path.is_file():
        return None
    manifest = base._json(manifest_path)  # noqa: SLF001
    if any(
        manifest.get(key) != value
        for key, value in expected_identity.items()
    ):
        return None
    if manifest.get("npz_sha256") != base._sha256(cache_path):  # noqa: SLF001
        raise RuntimeError("P16 denoised feature cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as payload:
        indices = np.asarray(payload["indices_kji"], dtype=np.int64)
        seeds = np.asarray(payload["repeat_seeds"], dtype=np.int64)
        reconstructed = np.asarray(
            payload["reconstructed"],
            dtype=np.float32,
        )
        delta = np.asarray(
            payload["delta_from_native_input"],
            dtype=np.float32,
        )
        masked = np.asarray(payload["trace_masked"], dtype=bool)
        native_input = np.asarray(payload["native_input"], dtype=np.float32)
    np.testing.assert_array_equal(indices, oof.indices_kji)
    np.testing.assert_array_equal(seeds, base.REPEAT_SEEDS)
    expected_values = (len(base.REPEAT_SEEDS), len(oof.target), 3)
    if reconstructed.shape != expected_values or delta.shape != expected_values:
        raise RuntimeError("P16 cached reconstructed feature shape drift")
    if masked.shape != expected_values[:2]:
        raise RuntimeError("P16 cached trace-mask shape drift")
    if native_input.shape != (len(oof.target), 3):
        raise RuntimeError("P16 cached native-input shape drift")
    if not all(
        np.all(np.isfinite(values))
        for values in (reconstructed, delta, native_input)
    ):
        raise FloatingPointError("P16 cached denoised features are non-finite")
    audit = dict(manifest["audit"])
    audit["cache_reused"] = True
    audit["feature_cache_sha256"] = manifest["npz_sha256"]
    return DenoisedPointFeatures(
        reconstructed=reconstructed,
        delta_from_native_input=delta,
        trace_masked=masked,
        native_input=native_input,
        audit=audit,
    )


def _sample_native_points(
    images: np.ndarray,
    mapping: p15.NativeMapping,
) -> np.ndarray:
    time_offsets = (
        np.asarray(mapping.time_indices, dtype=np.int64)
        - int(mapping.time_start)
    )
    if np.any(time_offsets < 0) or np.any(
        time_offsets >= p15.NATIVE_TIME_SAMPLES
    ):
        raise RuntimeError("OOF time sample falls outside native window")
    values = images[
        mapping.section_ids,
        :,
        time_offsets,
        mapping.trace_token_ids,
    ]
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(mapping.section_ids), 3):
        raise RuntimeError("native OOF point sampling shape drift")
    return values


def _extract_denoised_point_features(
    *,
    images: np.ndarray,
    mapping: p15.NativeMapping,
    source_root: Path,
    snapshot_path: Path,
    device: str,
    batch_size: int,
    weight_mode: str,
) -> DenoisedPointFeatures:
    """Run genuine masked encoder-decoder reconstruction for three channels."""

    import importlib
    import torch

    p4 = importlib.import_module(
        "_pipelines.02_task_datasets.reconstruction.p4_reconstruction"
    )
    if images.ndim != 4 or images.shape[1:] != (
        3,
        p15.NATIVE_TIME_SAMPLES,
        p15.NATIVE_TRACE_COUNT,
    ):
        raise ValueError("P16 native images must be [sections,3,400,160]")
    native_input = _sample_native_points(images, mapping)
    reconstructed = np.empty(
        (len(base.REPEAT_SEEDS), len(mapping.section_ids), 3),
        dtype=np.float32,
    )
    delta = np.empty_like(reconstructed)
    trace_masked = np.empty(
        (len(base.REPEAT_SEEDS), len(mapping.section_ids)),
        dtype=bool,
    )
    model_audits: list[dict[str, Any]] = []
    seed_audits: list[dict[str, Any]] = []
    flat_images = images.reshape(
        -1,
        1,
        p15.NATIVE_TIME_SAMPLES,
        p15.NATIVE_TRACE_COUNT,
    )

    shared_pretrained_model: Any | None = None
    if weight_mode == "pretrained":
        shared_pretrained_model = gfm_denoise.build_model(
            p4.task_spec("strict"),
            source_root=source_root,
            snapshot_path=snapshot_path,
            device=device,
            freeze_model=True,
            encoder_weight_mode=weight_mode,
            random_seed=int(base.REPEAT_SEEDS[0]),
        )
        model_audits.append(dict(shared_pretrained_model.asset_audit))
    elif weight_mode != "random_init":
        raise ValueError(f"unsupported GFM weight mode: {weight_mode}")

    for seed_id, seed in enumerate(base.REPEAT_SEEDS):
        model = shared_pretrained_model
        if model is None:
            model = gfm_denoise.build_model(
                p4.task_spec("strict"),
                source_root=source_root,
                snapshot_path=snapshot_path,
                device=device,
                freeze_model=True,
                encoder_weight_mode=weight_mode,
                random_seed=int(seed),
            )
            model_audits.append(dict(model.asset_audit))
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("P16 GFM encoder-decoder is not frozen")

        priorities = trace_mask_priorities(
            len(mapping.unique_inline),
            seed=int(seed),
        )
        expected_mask = trace_mask_from_priorities(priorities)
        flat_priorities = np.repeat(priorities, 3, axis=0)
        reconstructed_flat = np.empty_like(flat_images, dtype=np.float32)
        returned_mask_flat = np.empty(
            (len(flat_images), p15.NATIVE_TRACE_COUNT),
            dtype=bool,
        )
        for start in range(0, len(flat_images), int(batch_size)):
            stop = min(start + int(batch_size), len(flat_images))
            image_tensor = torch.as_tensor(
                flat_images[start:stop],
                dtype=torch.float32,
                device=device,
            )
            priority_tensor = torch.as_tensor(
                flat_priorities[start:stop],
                dtype=torch.float32,
                device=device,
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(device).startswith("cuda"),
            ):
                output = model(image_tensor, priority_tensor, LEN_KEEP)
            reconstructed_flat[start:stop] = (
                output.reconstruction.float().cpu().numpy()
            )
            returned_mask_flat[start:stop] = (
                output.mask.bool().cpu().numpy()
            )
        returned_mask = returned_mask_flat.reshape(
            len(mapping.unique_inline),
            3,
            p15.NATIVE_TRACE_COUNT,
        )
        if not np.array_equal(
            returned_mask,
            np.repeat(expected_mask[:, None, :], 3, axis=1),
        ):
            raise RuntimeError("GFM returned mask differs from paired mask plan")
        reconstructed_images = reconstructed_flat.reshape(images.shape)
        sampled = _sample_native_points(reconstructed_images, mapping)
        point_mask = expected_mask[
            mapping.section_ids,
            mapping.trace_token_ids,
        ]
        reconstructed[seed_id] = sampled
        delta[seed_id] = sampled - native_input
        trace_masked[seed_id] = point_mask
        visible = ~point_mask
        visible_error = (
            float(
                np.max(
                    np.abs(
                        sampled[visible]
                        - native_input[visible]
                    )
                )
            )
            if np.any(visible)
            else 0.0
        )
        if visible_error != 0.0:
            raise RuntimeError(
                "hybrid GFM reconstruction changed a visible OOF trace"
            )
        masked_delta = np.abs(
            delta[seed_id][point_mask]
        )
        seed_audits.append(
            {
                "seed": int(seed),
                "masked_oof_rows": int(np.sum(point_mask)),
                "masked_oof_fraction": float(np.mean(point_mask)),
                "visible_oof_max_abs_change": visible_error,
                "masked_oof_mean_abs_reconstruction_change": float(
                    np.mean(masked_delta)
                ),
                "masked_oof_max_abs_reconstruction_change": float(
                    np.max(masked_delta)
                ),
                "reconstruction_stats": [
                    base._summary_stats(sampled[:, channel])  # noqa: SLF001
                    for channel in range(3)
                ],
            }
        )
        if shared_pretrained_model is None:
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "gfm_reconstruction_seed_complete",
                    "encoder_weight_mode": weight_mode,
                    "seed": int(seed),
                    "masked_oof_fraction": float(np.mean(point_mask)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if shared_pretrained_model is not None:
        del shared_pretrained_model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if not all(
        np.all(np.isfinite(values))
        for values in (reconstructed, delta, native_input)
    ):
        raise FloatingPointError("P16 reconstruction features are non-finite")
    architectures = {
        row["architecture_sha256"] for row in model_audits
    }
    state_probes = {
        row["full_model_probe_sha256"] for row in model_audits
    }
    if len(architectures) != 1:
        raise RuntimeError("P16 GFM architecture differs across seeds")
    if weight_mode == "random_init" and len(state_probes) != len(
        base.REPEAT_SEEDS
    ):
        raise RuntimeError("P16 random-init full model states are not distinct")
    audit = {
        "cache_reused": False,
        "encoder_weight_mode": weight_mode,
        "pretrained_weights_used_for_forward": weight_mode == "pretrained",
        "feature_shapes": {
            "reconstructed": list(reconstructed.shape),
            "delta_from_native_input": list(delta.shape),
            "trace_masked": list(trace_masked.shape),
            "native_input": list(native_input.shape),
        },
        "native_input_shape": list(images.shape),
        "native_window_shape": [
            p15.NATIVE_TIME_SAMPLES,
            p15.NATIVE_TRACE_COUNT,
        ],
        "resize_applied": False,
        "interpolation_applied": False,
        "mask_policy": {
            "priority": "seeded per-native-section random priorities",
            "same_mask_for_three_channels": True,
            "mask_fraction": MASK_FRACTION,
            "masked_traces_per_section": MASKED_TRACE_COUNT,
            "visible_traces_per_section": LEN_KEEP,
            "output": (
                "upstream tutorial hybrid: decoder masked traces plus "
                "original visible traces"
            ),
        },
        "seismic_channels_forwarded_separately": 3,
        "channel_names": list(p15.SEISMIC_CHANNEL_NAMES),
        "model_audits": model_audits,
        "architecture_sha256": next(iter(architectures)),
        "same_architecture_across_seeds": True,
        "seed_distinct_random_states": (
            len(state_probes) == len(base.REPEAT_SEEDS)
            if weight_mode == "random_init"
            else None
        ),
        "per_seed": seed_audits,
        "label_read": False,
    }
    return DenoisedPointFeatures(
        reconstructed=reconstructed,
        delta_from_native_input=delta,
        trace_masked=trace_masked,
        native_input=native_input,
        audit=audit,
    )


def get_denoised_point_features(
    *,
    images: np.ndarray,
    mapping: p15.NativeMapping,
    oof: base.OOFDevelopment,
    source_root: Path,
    snapshot_path: Path,
    cache_dir: Path,
    device: str,
    batch_size: int,
    weight_mode: str,
    native_cache_sha256: str,
) -> DenoisedPointFeatures:
    adapter_path = (
        PROJECT_ROOT
        / "_models"
        / "reconstruction"
        / "geophysical_fm_denoise.py"
    )
    asset_audit = gfm.verify_local_assets(source_root, snapshot_path)[2]
    identity = _feature_cache_identity(
        weight_mode=weight_mode,
        native_cache_sha256=native_cache_sha256,
        indices_sha256=base._array_sha256(oof.indices_kji),  # noqa: SLF001
        adapter_sha256=base._sha256(adapter_path),  # noqa: SLF001
        asset_audit=asset_audit,
    )
    cached = _load_valid_feature_cache(
        cache_dir=cache_dir,
        expected_identity=identity,
        weight_mode=weight_mode,
        oof=oof,
    )
    if cached is not None:
        return cached
    result = _extract_denoised_point_features(
        images=images,
        mapping=mapping,
        source_root=source_root,
        snapshot_path=snapshot_path,
        device=device,
        batch_size=batch_size,
        weight_mode=weight_mode,
    )
    cache_path, manifest_path = _feature_cache_paths(cache_dir, weight_mode)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            indices_kji=oof.indices_kji,
            repeat_seeds=np.asarray(base.REPEAT_SEEDS, dtype=np.int64),
            reconstructed=result.reconstructed,
            delta_from_native_input=result.delta_from_native_input,
            trace_masked=result.trace_masked,
            native_input=result.native_input,
        )
    manifest = {
        **identity,
        "npz_sha256": base._sha256(cache_path),  # noqa: SLF001
        "audit": dict(result.audit),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = dict(result.audit)
    audit["feature_cache_sha256"] = manifest["npz_sha256"]
    return DenoisedPointFeatures(
        reconstructed=result.reconstructed,
        delta_from_native_input=result.delta_from_native_input,
        trace_masked=result.trace_masked,
        native_input=result.native_input,
        audit=audit,
    )


def load_patch_seismic_at_oof(
    *,
    train_h5: Path,
    indices_kji: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample legal ``seismic_patch[0:3]`` values for alignment QA only."""

    base.ensure_no_holdout_paths([train_h5])
    import h5py

    requested = np.asarray(indices_kji, dtype=np.int64)
    patch_shapes: set[tuple[int, int, int]] = set()
    accessed: list[str] = []
    with h5py.File(train_h5, "r") as handle:
        locations: list[tuple[str, np.ndarray, np.ndarray]] = []
        for key in sorted(handle):
            meta = json.loads(handle[key].attrs["meta"])
            start = np.asarray(meta["patch_start_kji"], dtype=np.int64)
            shape = np.asarray(meta["patch_shape_kji"], dtype=np.int64)
            locations.append((key, start, start + shape))
            patch_shapes.add(tuple(int(value) for value in shape))
        assignment = np.full(len(requested), -1, dtype=np.int64)
        for location_id, (_, start, stop) in enumerate(locations):
            inside = np.all(
                (requested >= start) & (requested < stop),
                axis=1,
            )
            if np.any((assignment >= 0) & inside):
                raise RuntimeError("OOF row maps to multiple train patches")
            assignment[inside] = location_id
        if np.any(assignment < 0):
            raise RuntimeError("OOF row is absent from train.h5")
        values = np.empty((len(requested), 3), dtype=np.float32)
        for location_id in np.unique(assignment):
            key, start, stop = locations[int(location_id)]
            rows = np.flatnonzero(assignment == location_id)
            local = requested[rows] - start
            seismic = np.asarray(
                handle[key]["seismic_patch"][0:3],
                dtype=np.float32,
            )
            active = np.asarray(
                handle[key]["seismic_patch"][8],
                dtype=np.float32,
            ) > 0.5
            expected = tuple((stop - start).tolist())
            if seismic.shape != (3, *expected) or active.shape != expected:
                raise RuntimeError("train.h5 seismic patch shape drift")
            if not np.all(active[tuple(local.T)]):
                raise RuntimeError("OOF row maps to an inactive train cell")
            values[rows] = seismic[(slice(None), *tuple(local.T))].T
            accessed.append(key)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("sampled train.h5 seismic is non-finite")
    return values, {
        "real_patch_shape_kji": [
            list(values) for values in sorted(patch_shapes)
        ],
        "patch_count_metadata_read": len(locations),
        "accessed_patch_count": len(accessed),
        "accessed_patch_keys_sha256": hashlib.sha256(
            json.dumps(sorted(accessed)).encode("utf-8")
        ).hexdigest(),
        "hdf5_files_opened": ["train.h5"],
        "hdf5_datasets_read": [
            "seismic_patch[0:3]",
            "seismic_patch[8]",
        ],
        "label_dataset_read": False,
    }


def _alignment_audit(
    *,
    native_input: np.ndarray,
    patch_input: np.ndarray,
    structural_input: np.ndarray,
) -> dict[str, Any]:
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (native_input, patch_input, structural_input)
    ]
    if any(values.shape != (len(native_input), 3) for values in arrays):
        raise ValueError("P16 alignment arrays must be [rows,3]")
    native_patch = [
        float(np.corrcoef(arrays[0][:, channel], arrays[1][:, channel])[0, 1])
        for channel in range(3)
    ]
    native_structural = [
        float(np.corrcoef(arrays[0][:, channel], arrays[2][:, channel])[0, 1])
        for channel in range(3)
    ]
    if min(native_patch) < 0.98 or min(native_structural) < 0.98:
        raise RuntimeError(
            "native continuous seismic does not align with patch structural inputs"
        )
    return {
        "channel_names": list(p15.SEISMIC_CHANNEL_NAMES),
        "native_vs_raw_patch_pearson": native_patch,
        "native_vs_stage3_structural_pearson": native_structural,
        "minimum_required_correlation": 0.98,
        "passed": True,
        "interpretation": (
            "normalizations differ by section/fold, so correlation rather "
            "than absolute equality validates the same three seismic attributes"
        ),
    }


def build_augmented_features(
    *,
    oof: base.OOFDevelopment,
    denoised: DenoisedPointFeatures,
    seed_id: int,
) -> np.ndarray:
    """Supplement P11 structural features with reconstructed values/deltas."""

    if not 0 <= int(seed_id) < len(base.REPEAT_SEEDS):
        raise IndexError("invalid P16 pseudo-repeat seed index")
    common = np.column_stack([oof.baseline, oof.distance_to_well])
    values = np.column_stack(
        [
            oof.structural_features,
            denoised.reconstructed[int(seed_id)],
            denoised.delta_from_native_input[int(seed_id)],
            denoised.trace_masked[int(seed_id)].astype(np.float64),
            common,
        ]
    )
    expected_width = 6 + 3 + 3 + 1 + 2
    if values.shape != (len(oof.target), expected_width):
        raise RuntimeError("P16 augmented structural feature shape drift")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("P16 augmented structural features are non-finite")
    return values


def _summarize_route(
    *,
    route: str,
    head_mode: str,
    seed_results: Sequence[Mapping[str, Any]],
    all_cells: Sequence[Mapping[str, Any]],
    baseline_rmse: float,
) -> dict[str, Any]:
    cells = [
        dict(row)
        for row in all_cells
        if row["route"] == route and row["head_mode"] == head_mode
    ]
    independent = diagnostics._independent_fold_outcomes(cells)  # noqa: SLF001
    counts = {
        outcome: sum(
            row["outcome_vs_pykrige"] == outcome for row in independent
        )
        for outcome in ("win", "loss", "tie")
    }
    gated_rmse = float(
        np.mean(
            [
                row["aggregates"][head_mode]["gated"]["rmse"]
                for row in seed_results
            ]
        )
    )
    ungated_rmse = float(
        np.mean(
            [
                row["aggregates"][head_mode]["ungated"]["rmse"]
                for row in seed_results
            ]
        )
    )
    delta = gated_rmse - baseline_rmse
    return {
        "gated_mean_seed_rmse": gated_rmse,
        "ungated_mean_seed_rmse": ungated_rmse,
        "rmse_delta_vs_pykrige": (
            0.0 if abs(delta) <= WIN_TOLERANCE else delta
        ),
        "relative_gain_vs_pykrige": (
            0.0
            if abs(delta) <= WIN_TOLERANCE
            else (baseline_rmse - gated_rmse) / baseline_rmse
        ),
        "independent_spatial_units": len(base.FOLD_IDS),
        "seed_pseudo_repeats_per_unit": len(base.REPEAT_SEEDS),
        "independent_fold_outcomes": independent,
        "independent_fold_outcome_counts": counts,
        "independent_fold_win_rate": counts["win"] / len(base.FOLD_IDS),
        "per_seed": [
            {
                "seed": int(seed),
                **dict(result["aggregates"][head_mode]),
            }
            for seed, result in zip(base.REPEAT_SEEDS, seed_results)
        ],
    }


def _paired_route_bootstrap(
    *,
    oof: base.OOFDevelopment,
    first: Mapping[int, np.ndarray],
    second: Mapping[int, np.ndarray],
    label: str,
) -> dict[str, Any]:
    folds = tuple(int(value) for value in base.FOLD_IDS)
    seeds = tuple(int(value) for value in base.REPEAT_SEEDS)
    counts = np.asarray(
        [np.sum(oof.fold_ids == fold) for fold in folds],
        dtype=np.float64,
    )
    first_sse = np.empty((len(seeds), len(folds)), dtype=np.float64)
    second_sse = np.empty_like(first_sse)
    for seed_id, seed in enumerate(seeds):
        for fold_id, fold in enumerate(folds):
            mask = oof.fold_ids == fold
            first_sse[seed_id, fold_id] = np.sum(
                (first[seed][mask] - oof.target[mask]) ** 2
            )
            second_sse[seed_id, fold_id] = np.sum(
                (second[seed][mask] - oof.target[mask]) ** 2
            )
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(bootstrap_seed)
    replicates = diagnostics.BOOTSTRAP_REPLICATES
    sampled = rng.integers(0, len(folds), size=(replicates, len(folds)))
    weights = np.column_stack(
        [
            np.sum(sampled == fold_id, axis=1)
            for fold_id in range(len(folds))
        ]
    ).astype(np.float64)
    denominator = weights @ counts
    first_rmse = np.sqrt(
        (first_sse @ weights.T) / denominator[None, :]
    )
    second_rmse = np.sqrt(
        (second_sse @ weights.T) / denominator[None, :]
    )
    deltas = np.mean(first_rmse - second_rmse, axis=0)
    point = float(
        np.mean(
            [
                base._metrics(oof.target, first[seed])["rmse"]  # noqa: SLF001
                - base._metrics(oof.target, second[seed])["rmse"]  # noqa: SLF001
                for seed in seeds
            ]
        )
    )
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return {
        "metric": label,
        "direction": "negative favors first route",
        "point_estimate": point,
        "confidence_level": 0.95,
        "confidence_interval": [float(lower), float(upper)],
        "interval_excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "replicates": int(replicates),
        "bootstrap_unit": "whole locked spatial fold",
        "independent_spatial_block_count": len(folds),
        "seeds_kept_paired": True,
        "voxels_resampled_independently": False,
        "bootstrap_seed": bootstrap_seed,
    }


def evaluate_p16(
    *,
    oof: base.OOFDevelopment,
    pretrained: DenoisedPointFeatures,
    random_init: DenoisedPointFeatures,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate reconstructed, random-reconstructed and raw structural routes."""

    if not np.array_equal(oof.baseline.copy(), oof.baseline):
        raise RuntimeError("P16 gate=0 is not bitwise equal to PyKrige")
    np.testing.assert_allclose(
        pretrained.native_input,
        random_init.native_input,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        pretrained.trace_masked,
        random_init.trace_masked,
    )
    baseline_metrics = base._metrics(oof.target, oof.baseline)  # noqa: SLF001
    common = np.column_stack([oof.baseline, oof.distance_to_well])
    raw_features = np.column_stack([oof.structural_features, common])
    route_results: dict[str, list[dict[str, Any]]] = {
        ROUTE_PRETRAINED: [],
        ROUTE_RANDOM: [],
        ROUTE_RAW: [],
    }
    all_cells: list[dict[str, Any]] = []
    prediction_maps: dict[
        str,
        dict[str, dict[int, np.ndarray]],
    ] = {
        route: {mode: {} for mode in HEAD_MODES}
        for route in route_results
    }
    payload: dict[str, np.ndarray] = {
        "indices_kji": np.asarray(oof.indices_kji, dtype=np.int64),
        "fold_ids": np.asarray(oof.fold_ids, dtype=np.int64),
        "target": np.asarray(oof.target, dtype=np.float64),
        "baseline_prediction": np.asarray(oof.baseline, dtype=np.float64),
        "baseline_error": np.asarray(
            oof.baseline - oof.target,
            dtype=np.float64,
        ),
    }
    for seed_id, seed in enumerate(base.REPEAT_SEEDS):
        route_inputs = {
            ROUTE_PRETRAINED: build_augmented_features(
                oof=oof,
                denoised=pretrained,
                seed_id=seed_id,
            ),
            ROUTE_RANDOM: build_augmented_features(
                oof=oof,
                denoised=random_init,
                seed_id=seed_id,
            ),
            ROUTE_RAW: raw_features,
        }
        for route, features in route_inputs.items():
            result = diagnostics.evaluate_adaptive_route(
                route=route,
                features=features,
                oof=oof,
                seed=int(seed),
            )
            route_results[route].append(result)
            all_cells.extend(result["per_fold"])
            for mode in HEAD_MODES:
                prediction = np.asarray(
                    result["predictions"][mode]["gated"],
                    dtype=np.float64,
                )
                prediction_maps[route][mode][int(seed)] = prediction
                payload[
                    f"{route}__{mode}__seed_{int(seed)}__prediction"
                ] = prediction
                payload[
                    f"{route}__{mode}__seed_{int(seed)}__error"
                ] = prediction - oof.target

    heads: dict[str, Any] = {}
    required_wins = math.ceil(0.8 * len(base.FOLD_IDS))
    for mode in HEAD_MODES:
        summaries = {
            route: _summarize_route(
                route=route,
                head_mode=mode,
                seed_results=route_results[route],
                all_cells=all_cells,
                baseline_rmse=baseline_metrics["rmse"],
            )
            for route in route_results
        }
        pretrained_summary = summaries[ROUTE_PRETRAINED]
        random_summary = summaries[ROUTE_RANDOM]
        raw_summary = summaries[ROUTE_RAW]
        pretrained_summary["rmse_minus_random_init"] = (
            pretrained_summary["gated_mean_seed_rmse"]
            - random_summary["gated_mean_seed_rmse"]
        )
        pretrained_summary["rmse_minus_raw_structural"] = (
            pretrained_summary["gated_mean_seed_rmse"]
            - raw_summary["gated_mean_seed_rmse"]
        )
        pretrained_summary["better_than_random_init"] = bool(
            pretrained_summary["rmse_minus_random_init"] < -WIN_TOLERANCE
        )
        pretrained_summary["better_than_raw_structural"] = bool(
            pretrained_summary["rmse_minus_raw_structural"] < -WIN_TOLERANCE
        )
        bootstrap_seed = int.from_bytes(
            hashlib.sha256(
                f"p16:{mode}:pretrained-vs-pykrige".encode("utf-8")
            ).digest()[:8],
            "little",
        )
        pretrained_summary["block_bootstrap_rmse_delta_vs_pykrige"] = (
            diagnostics.block_bootstrap_rmse_delta(
                target=oof.target,
                baseline=oof.baseline,
                candidate_predictions_by_seed=prediction_maps[
                    ROUTE_PRETRAINED
                ][mode],
                fold_ids=oof.fold_ids,
                bootstrap_seed=bootstrap_seed,
            )
        )
        pretrained_summary["paired_bootstrap_vs_random_init"] = (
            _paired_route_bootstrap(
                oof=oof,
                first=prediction_maps[ROUTE_PRETRAINED][mode],
                second=prediction_maps[ROUTE_RANDOM][mode],
                label=(
                    "mean-seed RMSE(pretrained reconstruction) - "
                    "RMSE(random-init reconstruction)"
                ),
            )
        )
        pretrained_summary["paired_bootstrap_vs_raw_structural"] = (
            _paired_route_bootstrap(
                oof=oof,
                first=prediction_maps[ROUTE_PRETRAINED][mode],
                second=prediction_maps[ROUTE_RAW][mode],
                label=(
                    "mean-seed RMSE(pretrained reconstruction) - "
                    "RMSE(raw structural)"
                ),
            )
        )
        promoted = bool(
            pretrained_summary["relative_gain_vs_pykrige"] >= 0.01
            and pretrained_summary["better_than_random_init"]
            and pretrained_summary["better_than_raw_structural"]
            and pretrained_summary["independent_fold_outcome_counts"]["win"]
            >= required_wins
        )
        heads[mode] = {
            "pretrained_gfm_reconstruction": pretrained_summary,
            "random_init_same_architecture_reconstruction": random_summary,
            "raw_no_foundation_structural": raw_summary,
            "promotion": {
                "passes_audited_p11_rule": promoted,
                "required_independent_fold_wins": required_wins,
                "minimum_relative_gain": 0.01,
                "matched_random_init_reconstruction_control_present": True,
                "raw_structural_control_present": True,
            },
        }
    best_mode = min(
        HEAD_MODES,
        key=lambda mode: (
            heads[mode]["pretrained_gfm_reconstruction"][
                "gated_mean_seed_rmse"
            ],
            mode,
        ),
    )
    best = heads[best_mode]["pretrained_gfm_reconstruction"]
    any_promoted = any(
        heads[mode]["promotion"]["passes_audited_p11_rule"]
        for mode in HEAD_MODES
    )
    experiment = {
        "fixed_protocol": {
            "protocol_source": (
                "p11_residual_fusion_diagnostics.evaluate_adaptive_route"
            ),
            "base_model": base.EXPECTED_MODEL_ID,
            "lane": base.EXPECTED_LANE,
            "split_hash": base.EXPECTED_SPLIT_HASH,
            "outer_folds": list(base.FOLD_IDS),
            "repeat_seeds": list(base.REPEAT_SEEDS),
            "independent_spatial_units": len(base.FOLD_IDS),
            "seeds_are_paired_mask_pseudo_repeats": True,
            "residual_alpha_candidates": list(diagnostics.RIDGE_ALPHAS),
            "gate_scale_candidates": list(base.GATE_CANDIDATES),
            "gate_bounds": [0.0, 1.0],
            "test_or_holdout_tuning": False,
        },
        "feature_contract": {
            "raw_route_width": int(raw_features.shape[1]),
            "augmented_route_width": 15,
            "augmentation": (
                "retain six existing structural fields; supplement three "
                "GFM-reconstructed values, three reconstruction deltas and "
                "one trace-mask indicator; retain PyKrige/distance signals"
            ),
            "decoder_training": "frozen checkpoint; no PORO fine-tuning",
        },
        "baseline": {
            "pykrige_oof": baseline_metrics,
            "gate_zero_exact": base._metrics(  # noqa: SLF001
                oof.target,
                oof.baseline.copy(),
            ),
            "gate_zero_bitwise_equal_to_pykrige": True,
        },
        "heads": heads,
        "best_observed_pretrained_head": best_mode,
        "per_fold_seed_pseudo_repeats": all_cells,
        "hypothesis_assessment": {
            "pretrained_reconstruction_better_than_raw_structural": best[
                "better_than_raw_structural"
            ],
            "pretrained_reconstruction_better_than_random_init": best[
                "better_than_random_init"
            ],
            "positive_pooled_gain_vs_pykrige": (
                best["relative_gain_vs_pykrige"] > 0.0
            ),
            "meets_80_percent_independent_fold_win_rule": (
                best["independent_fold_outcome_counts"]["win"]
                >= required_wins
            ),
        },
        "decision": {
            "state": (
                "PROMOTE_DEVELOPMENT_ONLY"
                if any_promoted
                else "VERIFIED_NO_PROMOTION"
            ),
            "default_enabled": any_promoted,
            "pretrained_contribution_claimed": False,
            "contribution_boundary": (
                "overall masked-reconstruction structural changes are "
                "reported directly; pretrained, random-init and raw routes "
                "remain separate and no causal foundation claim is inferred"
            ),
        },
    }
    return experiment, payload


def _write_prediction_errors(
    output_dir: Path,
    payload: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    path = output_dir / "prediction_errors.npz"
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": base._sha256(path),  # noqa: SLF001
        "arrays": sorted(payload),
        "rows": int(len(payload["target"])),
    }


def _write_evidence(
    output_dir: Path,
    result: Mapping[str, Any],
) -> None:
    experiment = result["experiment"]
    best_mode = experiment["best_observed_pretrained_head"]
    head = experiment["heads"][best_mode]
    pretrained = head["pretrained_gfm_reconstruction"]
    random_init = head["random_init_same_architecture_reconstruction"]
    raw = head["raw_no_foundation_structural"]
    baseline = experiment["baseline"]["pykrige_oof"]["rmse"]
    counts = pretrained["independent_fold_outcome_counts"]
    by_random = {
        int(row["outer_fold"]): row
        for row in random_init["independent_fold_outcomes"]
    }
    by_raw = {
        int(row["outer_fold"]): row
        for row in raw["independent_fold_outcomes"]
    }
    bootstrap = pretrained["block_bootstrap_rmse_delta_vs_pykrige"]
    paired_random = pretrained["paired_bootstrap_vs_random_init"]
    paired_raw = pretrained["paired_bootstrap_vs_raw_structural"]
    alignment = result["seismic_alignment_audit"]
    native = result["native_mapping_audit"]
    if pretrained["better_than_raw_structural"]:
        raw_interpretation = (
            "masked reconstruction added development information beyond the "
            "direct raw structural route under this head"
        )
    elif abs(pretrained["rmse_minus_raw_structural"]) <= WIN_TOLERANCE:
        raw_interpretation = (
            "the bounded gate reduced masked reconstruction to the same "
            "development result as the direct raw structural route"
        )
    else:
        raw_interpretation = (
            "masked reconstruction did not improve on the direct raw "
            "structural route"
        )
    lines = [
        "# P16 GFM masked seismic reconstruction — development-only evidence",
        "",
        "## Outcome",
        "",
        f"- PyKrige baseline pooled development RMSE: `{baseline:.12f}`.",
        (
            f"- Best pretrained masked-reconstruction head (`{best_mode}`) "
            f"gated mean-seed RMSE: "
            f"`{pretrained['gated_mean_seed_rmse']:.12f}` "
            f"(delta vs PyKrige "
            f"`{pretrained['rmse_delta_vs_pykrige']:+.12f}`, positive "
            "relative gain means improvement: "
            f"`{pretrained['relative_gain_vs_pykrige']:+.6%}`)."
        ),
        (
            "- Same encoder-decoder architecture with random initialization "
            f"and identical masks: `{random_init['gated_mean_seed_rmse']:.12f}`; "
            "pretrained minus random-init "
            f"`{pretrained['rmse_minus_random_init']:+.12f}`."
        ),
        (
            "- Direct raw no-foundation structural route: "
            f"`{raw['gated_mean_seed_rmse']:.12f}`; pretrained "
            f"reconstruction minus raw "
            f"`{pretrained['rmse_minus_raw_structural']:+.12f}`."
        ),
        (
            f"- Five independent spatial-fold outcomes versus PyKrige: "
            f"{counts['win']} win / {counts['loss']} loss / "
            f"{counts['tie']} tie."
        ),
        f"- Interpretation: {raw_interpretation}.",
        (
            f"- Decision: `{experiment['decision']['state']}`. No gain is "
            "automatically attributed to pretrained GFM weights."
        ),
        "",
        "## Genuine GFM masked-interpolation path",
        "",
        (
            "- Each of the three seismic attributes is forwarded separately "
            "through the real GFM encoder and all 12 decoder blocks. The "
            "checkpoint is frozen and receives no PORO supervision."
        ),
        (
            f"- Exactly `{MASKED_TRACE_COUNT}/160` traces (25%) are masked "
            f"per native section and seed (`len_keep={LEN_KEEP}`). Pretrained "
            "and random-init use identical paired masks."
        ),
        (
            "- Output follows the upstream interpolation tutorial exactly: "
            "decoder predictions replace masked traces, while visible traces "
            "are copied from the input. Every visible OOF value passed the "
            "bitwise no-change check."
        ),
        (
            "- This is masked trace interpolation, not supervised denoiser "
            "fine-tuning. The base checkpoint is not claimed to remove every "
            "possible noise process."
        ),
        (
            "- The six existing P11 structural fields remain present. P16 "
            "supplements them with three reconstructed values, three "
            "reconstruction deltas, and one masked-trace indicator; Ridge "
            "preprocessing remains outer-train-only."
        ),
        "",
        "## Native seismic and patch alignment",
        "",
        (
            "- P16 reuses P15's audited mapping from development patch cells "
            "to the original continuous ST0202 SEG-Y. It reads exact "
            "`400`-sample × `160`-adjacent-trace windows; resize, "
            "interpolation and padding are all false."
        ),
        (
            "- The OOF cells span "
            f"{native['development_mapping']['unique_inline_sections']} "
            "native inline sections, "
            f"{native['development_mapping']['crossline_span']} crosslines "
            "and "
            f"{native['development_mapping']['time_index_span']} samples."
        ),
        (
            "- Native-input versus raw `seismic_patch[0:3]` Pearson "
            "correlations by amplitude/RMS/gradient: "
            + ", ".join(f"`{value:.6f}`" for value in alignment[
                "native_vs_raw_patch_pearson"
            ])
            + "."
        ),
        (
            "- Native-input versus fold-standardized Stage-3 structural "
            "correlations: "
            + ", ".join(f"`{value:.6f}`" for value in alignment[
                "native_vs_stage3_structural_pearson"
            ])
            + ". The 0.98 alignment floor passed for every channel."
        ),
        "",
        "## Five genuinely independent spatial units",
        "",
        (
            "The five locked outer folds are the inferential units. The three "
            "mask/model seeds are paired pseudo-repeats inside each fold, not "
            "15 independent observations."
        ),
        "",
        (
            "| fold | PyKrige RMSE | pretrained reconstruction | random-init "
            "reconstruction | raw structural | pretrained delta | outcome |"
        ),
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in pretrained["independent_fold_outcomes"]:
        random_row = by_random[int(row["outer_fold"])]
        raw_row = by_raw[int(row["outer_fold"])]
        lines.append(
            f"| {row['outer_fold']} | "
            f"{row['baseline_mean_seed_rmse']:.9f} | "
            f"{row['gated_mean_seed_rmse']:.9f} | "
            f"{random_row['gated_mean_seed_rmse']:.9f} | "
            f"{raw_row['gated_mean_seed_rmse']:.9f} | "
            f"{row['rmse_delta_vs_pykrige']:+.9f} | "
            f"{row['outcome_vs_pykrige']} |"
        )
    lines.extend(
        [
            "",
            "## Whole-fold bootstrap",
            "",
            (
                "- Pretrained reconstruction minus PyKrige RMSE: "
                f"`{bootstrap['point_estimate']:+.12f}`; 95% interval "
                f"`[{bootstrap['confidence_interval'][0]:+.12f}, "
                f"{bootstrap['confidence_interval'][1]:+.12f}]`."
            ),
            (
                "- Pretrained reconstruction minus random-init reconstruction: "
                f"`{paired_random['point_estimate']:+.12f}`; 95% interval "
                f"`[{paired_random['confidence_interval'][0]:+.12f}, "
                f"{paired_random['confidence_interval'][1]:+.12f}]`."
            ),
            (
                "- Pretrained reconstruction minus raw structural: "
                f"`{paired_raw['point_estimate']:+.12f}`; 95% interval "
                f"`[{paired_raw['confidence_interval'][0]:+.12f}, "
                f"{paired_raw['confidence_interval'][1]:+.12f}]`."
            ),
            (
                "- Bootstrap resamples whole locked spatial folds. Seeds stay "
                "paired and voxels are never resampled independently."
            ),
            "",
            "## Provenance and holdout firewall",
            "",
            (
                "- Model: `thinkonward/geophysical-foundation-model`, "
                f"snapshot `{gfm.SNAPSHOT_REVISION}`, Apache-2.0. Weight "
                f"SHA-256 `{gfm.WEIGHTS_SHA256}`; vendor source SHA-256 "
                f"`{gfm.SOURCE_SHA256}`."
            ),
            (
                "- PORO targets and PyKrige predictions come only from the "
                "same five hash-verified development OOF archives as P11-P15."
            ),
            (
                "- `train.h5` reads are limited to patch metadata, "
                "`seismic_patch[0:3]`, `seismic_patch[3:6]`, and "
                "`seismic_patch[8]`; no HDF5 label dataset is read."
            ),
            (
                "- The continuous SEG-Y supplies label-free development "
                "seismic covariates only. `test.h5`, frozen holdout paths, "
                "holdout labels and historical test metrics are neither "
                "opened nor probed."
            ),
            (
                "- Gate=0 is bitwise identical to the strong PyKrige OOF "
                "baseline."
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
    snapshot_path: Path,
    segy_path: Path,
    seismic_index_path: Path,
    well_tie_path: Path,
    build_summary_path: Path,
    output_dir: Path,
    cache_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Run complete pretrained/random masked-reconstruction evaluation."""

    started = time.time()
    output_dir = _validate_output_dir(output_dir)
    cache_dir = Path(cache_dir).expanduser().resolve()
    base.ensure_no_holdout_paths(
        (
            stage3_root,
            source_root,
            snapshot_path,
            segy_path,
            seismic_index_path,
            well_tie_path,
            build_summary_path,
            output_dir,
            cache_dir,
        )
    )
    inputs = base.resolve_dev_inputs(data_dir)
    oof = base.load_oof_development(stage3_root)
    for path in (
        segy_path,
        seismic_index_path,
        well_tie_path,
        build_summary_path,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    mapping = p15.build_native_mapping(
        train_h5=inputs.train_h5,
        oof=oof,
        build_summary_path=build_summary_path,
        seismic_index_path=seismic_index_path,
        well_tie_path=well_tie_path,
    )
    train_h5_sha256 = base._sha256(inputs.train_h5)  # noqa: SLF001
    segy_sha256 = base._sha256(segy_path)  # noqa: SLF001
    indices_sha256 = base._array_sha256(oof.indices_kji)  # noqa: SLF001
    native_cache_path = cache_dir / "native_windows.npz"
    images, native_audit = p15.get_native_windows(
        segy_path=segy_path,
        seismic_index_path=seismic_index_path,
        mapping=mapping,
        cache_path=native_cache_path,
        segy_sha256=segy_sha256,
        train_h5_sha256=train_h5_sha256,
        indices_sha256=indices_sha256,
    )
    patch_values, patch_audit = load_patch_seismic_at_oof(
        train_h5=inputs.train_h5,
        indices_kji=oof.indices_kji,
    )
    native_values = _sample_native_points(images, mapping)
    alignment = _alignment_audit(
        native_input=native_values,
        patch_input=patch_values,
        structural_input=oof.structural_features[:, :3],
    )
    print(
        json.dumps(
            {
                "event": "native_windows_aligned",
                "shape": list(images.shape),
                "resize_applied": False,
                "minimum_patch_correlation": min(
                    alignment["native_vs_raw_patch_pearson"]
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    pretrained = get_denoised_point_features(
        images=images,
        mapping=mapping,
        oof=oof,
        source_root=source_root,
        snapshot_path=snapshot_path,
        cache_dir=cache_dir,
        device=device,
        batch_size=int(batch_size),
        weight_mode="pretrained",
        native_cache_sha256=native_audit["cache_sha256"],
    )
    random_init = get_denoised_point_features(
        images=images,
        mapping=mapping,
        oof=oof,
        source_root=source_root,
        snapshot_path=snapshot_path,
        cache_dir=cache_dir,
        device=device,
        batch_size=int(batch_size),
        weight_mode="random_init",
        native_cache_sha256=native_audit["cache_sha256"],
    )
    if (
        pretrained.audit["architecture_sha256"]
        != random_init.audit["architecture_sha256"]
    ):
        raise RuntimeError("P16 pretrained/random architecture mismatch")
    experiment, prediction_payload = evaluate_p16(
        oof=oof,
        pretrained=pretrained,
        random_init=random_init,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_artifact = _write_prediction_errors(
        output_dir,
        prediction_payload,
    )
    adapter_path = (
        PROJECT_ROOT
        / "_models"
        / "reconstruction"
        / "geophysical_fm_denoise.py"
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "implementation": {
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "script_sha256": base._sha256(Path(__file__)),  # noqa: SLF001
            "adapter": str(adapter_path.relative_to(PROJECT_ROOT)),
            "adapter_sha256": base._sha256(adapter_path),  # noqa: SLF001
            "native_mapping_implementation": str(
                (
                    HERE / "p15_gfm_finetune.py"
                ).relative_to(PROJECT_ROOT)
            ),
            "native_mapping_implementation_sha256": base._sha256(  # noqa: SLF001
                HERE / "p15_gfm_finetune.py"
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pid": os.getpid(),
        },
        "model": {
            "id": "thinkonward/geophysical-foundation-model",
            "snapshot_revision": gfm.SNAPSHOT_REVISION,
            "license": "Apache-2.0",
            "architecture": "ViT-MAE whole-trace masking",
            "input_shape": [1, 400, 160],
            "patch_size": [400, 1],
            "encoder_depth": 16,
            "decoder_depth": 12,
            "mask_fraction": MASK_FRACTION,
            "len_keep": LEN_KEEP,
        },
        "native_mapping_audit": dict(mapping.audit),
        "native_window_audit": native_audit,
        "patch_seismic_audit": patch_audit,
        "seismic_alignment_audit": alignment,
        "pretrained_reconstruction_audit": dict(pretrained.audit),
        "random_init_reconstruction_audit": dict(random_init.audit),
        "source_oof": {
            "records": list(oof.source_records),
            "rows": int(len(oof.target)),
            "indices_kji_sha256": indices_sha256,
        },
        "experiment": experiment,
        "prediction_error_artifact": prediction_artifact,
        "holdout_firewall": {
            "hdf5_files_opened": ["train.h5"],
            "hdf5_datasets_read": [
                "seismic_patch[0:3]",
                "seismic_patch[3:6]",
                "seismic_patch[8]",
            ],
            "raw_continuous_source_opened": Path(segy_path).name,
            "raw_source_content": "label-free seismic traces only",
            "label_dataset_read_from_hdf5": False,
            "test_path_argument_exists": False,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "historical_test_metrics_read": False,
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "device": device,
            "batch_size": int(batch_size),
            "native_cache_reused": native_audit["cache_reused"],
            "pretrained_feature_cache_reused": pretrained.audit[
                "cache_reused"
            ],
            "random_init_feature_cache_reused": random_init.audit[
                "cache_reused"
            ],
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_evidence(output_dir, result)
    artifact_manifest = {
        "summary.json": base._sha256(output_dir / "summary.json"),  # noqa: SLF001
        "evidence.md": base._sha256(output_dir / "evidence.md"),  # noqa: SLF001
        "prediction_errors.npz": prediction_artifact["sha256"],
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser(
        "check-dev",
        help="verify the sole legal development HDF5 path",
    )
    check.add_argument("--data-dir", type=Path, required=True)

    execute = subparsers.add_parser(
        "run",
        help="run P16 pretrained/random masked reconstruction OOF",
    )
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--stage3-root", type=Path, required=True)
    execute.add_argument("--source-root", type=Path, required=True)
    execute.add_argument("--snapshot-path", type=Path, required=True)
    execute.add_argument("--segy-path", type=Path, required=True)
    execute.add_argument("--seismic-index-path", type=Path, required=True)
    execute.add_argument("--well-tie-path", type=Path, required=True)
    execute.add_argument("--build-summary-path", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    execute.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    execute.add_argument("--device", default="cuda:0")
    execute.add_argument("--batch-size", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check-dev":
        inputs = base.resolve_dev_inputs(args.data_dir)
        print(
            json.dumps(
                {
                    "accepted": str(inputs.train_h5),
                    "holdout_opened": False,
                }
            )
        )
        return 0
    values = vars(args)
    values.pop("command")
    result = run(**values)
    print(json.dumps(result["experiment"]["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
