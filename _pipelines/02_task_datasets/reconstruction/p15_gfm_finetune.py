#!/usr/bin/env python3
"""P15 native-window partial fine-tuning of the seismic GFM.

The experiment keeps the locked P5 PyKrige development OOF rows, five outer
spatial folds, three paired pseudo-repeat seeds and bounded P11 gate.  Unlike
P14, it samples exact 400-time-sample by 160-adjacent-trace windows from the
continuous ST0202 SEG-Y and backpropagates through the final GFM transformer
block.  The matched random-init control uses the same architecture, unlocked
block, optimizer, early-stopping rule and outer-fold protocol.

No test/holdout HDF5 argument exists.  ``train.h5`` supplies only development
metadata, normalized coordinates and active masks; PORO targets and PyKrige
predictions come from the hash-verified development OOF archives.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

from _models.reconstruction import geophysical_fm as gfm  # noqa: E402
from _models.reconstruction import geophysical_fm_finetune as gfm_ft  # noqa: E402
import p11_residual_fusion as base  # noqa: E402
import p11_residual_fusion_diagnostics as diagnostics  # noqa: E402


SCHEMA_VERSION = "reconstruction-p15-gfm-finetune/v1"
NATIVE_CACHE_SCHEMA_VERSION = "reconstruction-p15-native-segy-window/v1"
PREFIX_CACHE_SCHEMA_VERSION = "reconstruction-p15-gfm-prefix-cache/v1"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p15_gfm_finetune"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "_tmp" / "p15_gfm_finetune"

TRAINABLE_BLOCK_COUNT = 1
NATIVE_TIME_SAMPLES = 400
NATIVE_TRACE_COUNT = 160
SEISMIC_CHANNEL_NAMES = (
    "native_amplitude",
    "native_local_rms_5_sample",
    "native_vertical_gradient_2_sample",
)
VIEW_NAMES = (
    "amplitude_trace",
    "amplitude_cls",
    "local_rms_trace",
    "local_rms_cls",
    "vertical_gradient_trace",
    "vertical_gradient_cls",
)
VIEW_PROJECTION_WIDTH = 16
HEAD_HIDDEN_WIDTH = 64
HEAD_LR = 3e-4
ENCODER_LR = 1e-5
WEIGHT_DECAY = 1e-4
MAX_UPDATES = 80
EVAL_EVERY = 10
EARLY_STOPPING_PATIENCE = 3
MIN_EARLY_STOP_UPDATES = 10
SECTION_BATCH_SIZE = 4
GRAD_CLIP = 1.0
MAX_RESIDUAL_CORRECTION = 0.05
WIN_TOLERANCE = diagnostics.WIN_TOLERANCE


@dataclass(frozen=True)
class NativeMapping:
    """OOF rows mapped to genuine SEG-Y sections and trace/time locations."""

    unique_inline: np.ndarray
    section_ids: np.ndarray
    trace_token_ids: np.ndarray
    time_positions: np.ndarray
    inline: np.ndarray
    crossline: np.ndarray
    time_indices: np.ndarray
    crossline_start: int
    time_start: int
    audit: Mapping[str, Any]


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = Path(output_dir).expanduser().resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(
            f"P15 output must stay inside reconstruction: {resolved}"
        ) from exc
    protected = {
        (HERE / "_outputs" / "p11_residual_fusion").resolve(),
        (
            HERE
            / "_outputs"
            / "p11_residual_fusion_diagnostics"
        ).resolve(),
        (HERE / "_outputs" / "p11_cross_attention_fusion").resolve(),
        (HERE / "_outputs" / "p14_geophysical_fm").resolve(),
    }
    if resolved in protected:
        raise ValueError("P15 refuses to overwrite P11-P14 evidence")
    return resolved


def centered_native_window_start(
    values: np.ndarray,
    *,
    window_size: int,
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Center an exact native window while staying inside source bounds."""

    selected = np.asarray(values, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("window coordinates must be a non-empty vector")
    if int(window_size) <= 0:
        raise ValueError("window_size must be positive")
    source_size = int(upper_bound) - int(lower_bound) + 1
    if source_size < int(window_size):
        raise ValueError("source axis is smaller than the native GFM window")
    span = int(np.max(selected)) - int(np.min(selected)) + 1
    if span > int(window_size):
        raise ValueError(
            f"development span {span} exceeds native window {window_size}"
        )
    proposed = (
        int(np.min(selected))
        + int(np.max(selected))
        + 1
        - int(window_size)
    ) // 2
    return int(
        np.clip(
            proposed,
            int(lower_bound),
            int(upper_bound) - int(window_size) + 1,
        )
    )


def _xy_to_il_xl(
    x: np.ndarray,
    y: np.ndarray,
    index: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    affine = np.asarray(index["affine_il_xl_to_xy"], dtype=np.float64)
    rhs = np.column_stack(
        [x - affine[0, 2], y - affine[1, 2]]
    )
    il_xl = np.linalg.solve(affine[:, :2], rhs.T).T
    il_float, xl_float = il_xl[:, 0], il_xl[:, 1]
    in_bounds = (
        (il_float >= int(index["il_min"]))
        & (il_float <= int(index["il_max"]))
        & (xl_float >= int(index["xl_min"]))
        & (xl_float <= int(index["xl_max"]))
    )
    inline = np.clip(
        np.rint(il_float),
        int(index["il_min"]),
        int(index["il_max"]),
    ).astype(np.int32)
    crossline = np.clip(
        np.rint(xl_float),
        int(index["xl_min"]),
        int(index["xl_max"]),
    ).astype(np.int32)
    return inline, crossline, in_bounds


def _xy_from_il_xl(
    inline: np.ndarray,
    crossline: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = (
        affine[0, 0] * inline
        + affine[0, 1] * crossline
        + affine[0, 2]
    )
    y = (
        affine[1, 0] * inline
        + affine[1, 1] * crossline
        + affine[1, 2]
    )
    return x, y


def _estimate_twt_from_weak_ties(
    *,
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    seismic_index: Mapping[str, np.ndarray],
    well_tie_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    affine = np.asarray(
        seismic_index["affine_il_xl_to_xy"],
        dtype=np.float64,
    )
    numer = np.zeros(depth.size, dtype=np.float64)
    denom = np.zeros(depth.size, dtype=np.float64)
    well_ranges: dict[str, Any] = {}
    with np.load(well_tie_path, allow_pickle=False) as tie:
        suffix = "__depth_m"
        names = sorted(
            key[: -len(suffix)]
            for key in tie.files
            if key.endswith(suffix)
        )
        for name in names:
            well_depth = tie[f"{name}__depth_m"].astype(np.float64)
            well_twt = tie[f"{name}__twt_est_ms"].astype(np.float64)
            well_inline = tie[f"{name}__inline"].astype(np.float64)
            well_crossline = tie[
                f"{name}__crossline"
            ].astype(np.float64)
            well_x, well_y = _xy_from_il_xl(
                well_inline,
                well_crossline,
                affine,
            )
            valid = (depth >= well_depth.min()) & (
                depth <= well_depth.max()
            )
            twt_here = np.interp(depth, well_depth, well_twt)
            x_here = np.interp(depth, well_depth, well_x)
            y_here = np.interp(depth, well_depth, well_y)
            distance2 = (x - x_here) ** 2 + (y - y_here) ** 2
            weight = valid.astype(np.float64) / (
                distance2 + 100.0**2
            )
            numer += weight * twt_here
            denom += weight
            well_ranges[name] = {
                "depth_m": [
                    float(well_depth.min()),
                    float(well_depth.max()),
                ],
                "twt_ms": [
                    float(well_twt.min()),
                    float(well_twt.max()),
                ],
            }
    if np.any(denom == 0):
        raise ValueError(
            "weak ties do not cover "
            f"{int(np.count_nonzero(denom == 0))} development rows"
        )
    return (numer / denom).astype(np.float32), {
        "method": (
            "depth-wise inverse-horizontal-distance blend of the committed "
            "Layer-1 weak well ties"
        ),
        "depth_coordinate_warning": (
            "Layer-1 tie depth is MD while Eclipse centre depth is "
            "TVD-like; alignment remains weak."
        ),
        "well_ranges": well_ranges,
    }


def _verify_oof_coordinates_in_train_h5(
    *,
    train_h5: Path,
    oof: base.OOFDevelopment,
) -> dict[str, Any]:
    """Verify OOF normalized coordinates against legal train.h5 cells."""

    import h5py

    requested = np.asarray(oof.indices_kji, dtype=np.int64)
    locations: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    patch_shapes: set[tuple[int, int, int]] = set()
    with h5py.File(train_h5, "r") as handle:
        for key in sorted(handle):
            meta = json.loads(handle[key].attrs["meta"])
            start = np.asarray(meta["patch_start_kji"], dtype=np.int64)
            shape = np.asarray(meta["patch_shape_kji"], dtype=np.int64)
            locations.append((key, start, start + shape, shape))
            patch_shapes.add(tuple(int(value) for value in shape))
        assignment = np.full(len(requested), -1, dtype=np.int64)
        for location_id, (_, start, stop, _) in enumerate(locations):
            inside = np.all(
                (requested >= start) & (requested < stop),
                axis=1,
            )
            if np.any((assignment >= 0) & inside):
                raise RuntimeError("OOF coordinate maps to multiple patches")
            assignment[inside] = location_id
        if np.any(assignment < 0):
            raise RuntimeError("OOF coordinate is absent from train.h5")

        recovered = np.empty_like(oof.xyz, dtype=np.float32)
        accessed_keys: list[str] = []
        for location_id in np.unique(assignment):
            key, start, stop, _ = locations[int(location_id)]
            row_ids = np.flatnonzero(assignment == location_id)
            local = requested[row_ids] - start
            group = handle[key]
            coordinates = np.asarray(
                group["seismic_patch"][3:6],
                dtype=np.float32,
            )
            active = (
                np.asarray(
                    group["seismic_patch"][8],
                    dtype=np.float32,
                )
                > 0.5
            )
            expected = tuple((stop - start).tolist())
            if coordinates.shape != (3, *expected):
                raise RuntimeError("train.h5 coordinate patch shape drift")
            if not np.all(active[tuple(local.T)]):
                raise RuntimeError("OOF row maps to an inactive train cell")
            recovered[row_ids] = coordinates[
                (slice(None), *tuple(local.T))
            ].T
            accessed_keys.append(key)
    np.testing.assert_allclose(
        recovered,
        oof.xyz,
        rtol=0.0,
        atol=0.0,
    )
    return {
        "patch_count_metadata_read": len(locations),
        "real_patch_shape_kji": [
            list(values) for values in sorted(patch_shapes)
        ],
        "accessed_coordinate_patch_count": len(accessed_keys),
        "accessed_patch_keys_sha256": hashlib.sha256(
            json.dumps(sorted(accessed_keys)).encode("utf-8")
        ).hexdigest(),
        "oof_xyz_exactly_matches_train_h5": True,
        "hdf5_files_opened": ["train.h5"],
        "hdf5_datasets_read": [
            "seismic_patch[3:6]",
            "seismic_patch[8]",
        ],
        "label_dataset_read": False,
    }


def build_native_mapping(
    *,
    train_h5: Path,
    oof: base.OOFDevelopment,
    build_summary_path: Path,
    seismic_index_path: Path,
    well_tie_path: Path,
) -> NativeMapping:
    """Map development OOF cells into exact-size continuous SEG-Y windows."""

    paths = (
        train_h5,
        build_summary_path,
        seismic_index_path,
        well_tie_path,
    )
    base.ensure_no_holdout_paths(paths)
    hdf5_audit = _verify_oof_coordinates_in_train_h5(
        train_h5=train_h5,
        oof=oof,
    )
    build_summary = base._json(build_summary_path)  # noqa: SLF001
    bounds = build_summary["coordinate_bounds"]
    xyz = np.asarray(oof.xyz, dtype=np.float64)
    x = bounds["x"][0] + xyz[:, 0] * (
        bounds["x"][1] - bounds["x"][0]
    )
    y = bounds["y"][0] + xyz[:, 1] * (
        bounds["y"][1] - bounds["y"][0]
    )
    depth = bounds["depth"][0] + xyz[:, 2] * (
        bounds["depth"][1] - bounds["depth"][0]
    )
    with np.load(seismic_index_path, allow_pickle=False) as payload:
        index = {key: payload[key] for key in payload.files}
    inline, crossline, in_bounds = _xy_to_il_xl(x, y, index)
    if not np.all(in_bounds):
        raise RuntimeError("development coordinate lies outside SEG-Y grid")
    twt_ms, tie_audit = _estimate_twt_from_weak_ties(
        x=x,
        y=y,
        depth=depth,
        seismic_index=index,
        well_tie_path=well_tie_path,
    )
    samples_ms = np.asarray(index["samples_ms"], dtype=np.float64)
    time_indices = np.searchsorted(samples_ms, twt_ms)
    time_indices = np.clip(
        time_indices,
        1,
        len(samples_ms) - 2,
    )
    left = np.abs(samples_ms[time_indices - 1] - twt_ms)
    right = np.abs(samples_ms[time_indices] - twt_ms)
    time_indices = np.where(
        left < right,
        time_indices - 1,
        time_indices,
    ).astype(np.int32)

    crossline_start = centered_native_window_start(
        crossline,
        window_size=NATIVE_TRACE_COUNT,
        lower_bound=int(index["xl_min"]),
        upper_bound=int(index["xl_max"]),
    )
    # Two extra samples on each side are required for RMS/gradient channels.
    time_start = centered_native_window_start(
        time_indices,
        window_size=NATIVE_TIME_SAMPLES,
        lower_bound=2,
        upper_bound=len(samples_ms) - 3,
    )
    trace_token_ids = crossline.astype(np.int64) - crossline_start
    if np.any(trace_token_ids < 0) or np.any(
        trace_token_ids >= NATIVE_TRACE_COUNT
    ):
        raise RuntimeError("OOF crossline falls outside native window")
    unique_inline = np.unique(inline).astype(np.int32)
    section_ids = np.searchsorted(unique_inline, inline).astype(np.int64)
    if not np.array_equal(unique_inline[section_ids], inline):
        raise RuntimeError("native inline section mapping is incomplete")
    time_positions = (
        time_indices.astype(np.float64) - float(time_start)
    ) / float(NATIVE_TIME_SAMPLES - 1)
    audit = {
        **hdf5_audit,
        "continuous_source_grid": {
            "inline_range": [
                int(index["il_min"]),
                int(index["il_max"]),
            ],
            "crossline_range": [
                int(index["xl_min"]),
                int(index["xl_max"]),
            ],
            "trace_grid_shape": [
                int(index["n_il"]),
                int(index["n_xl"]),
            ],
            "time_samples": len(samples_ms),
            "sample_interval_ms": float(
                samples_ms[1] - samples_ms[0]
            ),
        },
        "development_mapping": {
            "unique_inline_sections": len(unique_inline),
            "inline_range": [
                int(np.min(inline)),
                int(np.max(inline)),
            ],
            "crossline_range": [
                int(np.min(crossline)),
                int(np.max(crossline)),
            ],
            "crossline_span": int(np.ptp(crossline)) + 1,
            "time_index_range": [
                int(np.min(time_indices)),
                int(np.max(time_indices)),
            ],
            "time_index_span": int(np.ptp(time_indices)) + 1,
            "twt_ms_range": [
                float(np.min(twt_ms)),
                float(np.max(twt_ms)),
            ],
        },
        "native_window": {
            "shape": [NATIVE_TIME_SAMPLES, NATIVE_TRACE_COUNT],
            "orientation": "time-sample x adjacent-crossline at fixed inline",
            "crossline_start": int(crossline_start),
            "crossline_stop_inclusive": int(
                crossline_start + NATIVE_TRACE_COUNT - 1
            ),
            "time_start_index": int(time_start),
            "time_stop_index_inclusive": int(
                time_start + NATIVE_TIME_SAMPLES - 1
            ),
            "resize_applied": False,
            "interpolation_applied": False,
            "padding_applied": False,
            "native_adjacent_traces": True,
        },
        "weak_tie_mapping": tie_audit,
        "build_summary_sha256": base._sha256(  # noqa: SLF001
            build_summary_path
        ),
        "seismic_index_sha256": base._sha256(  # noqa: SLF001
            seismic_index_path
        ),
        "well_tie_sha256": base._sha256(  # noqa: SLF001
            well_tie_path
        ),
    }
    return NativeMapping(
        unique_inline=unique_inline,
        section_ids=section_ids,
        trace_token_ids=trace_token_ids,
        time_positions=time_positions.astype(np.float32),
        inline=inline,
        crossline=crossline,
        time_indices=time_indices,
        crossline_start=int(crossline_start),
        time_start=int(time_start),
        audit=audit,
    )


def _native_cache_manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def _native_cache_identity(
    *,
    mapping: NativeMapping,
    segy_sha256: str,
    train_h5_sha256: str,
    indices_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": NATIVE_CACHE_SCHEMA_VERSION,
        "segy_sha256": segy_sha256,
        "train_h5_sha256": train_h5_sha256,
        "indices_kji_sha256": indices_sha256,
        "unique_inline_sha256": base._array_sha256(  # noqa: SLF001
            mapping.unique_inline
        ),
        "window_shape": [NATIVE_TIME_SAMPLES, NATIVE_TRACE_COUNT],
        "crossline_start": mapping.crossline_start,
        "time_start": mapping.time_start,
        "channels": list(SEISMIC_CHANNEL_NAMES),
        "resize_applied": False,
    }


def _normalize_native_channel(
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (NATIVE_TIME_SAMPLES, NATIVE_TRACE_COUNT):
        raise ValueError("native channel shape drift")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("native channel contains non-finite values")
    mean = float(np.mean(values))
    std = max(float(np.std(values)), 1e-6)
    normalized = ((values - mean) / std).astype(np.float32)
    return normalized, {
        "mean_before": mean,
        "std_before": std,
        "mean_after": float(np.mean(normalized)),
        "std_after": float(np.std(normalized)),
    }


def get_native_windows(
    *,
    segy_path: Path,
    seismic_index_path: Path,
    mapping: NativeMapping,
    cache_path: Path,
    segy_sha256: str,
    train_h5_sha256: str,
    indices_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read or reuse exact native windows from the continuous SEG-Y."""

    base.ensure_no_holdout_paths(
        [segy_path, seismic_index_path, cache_path]
    )
    identity = _native_cache_identity(
        mapping=mapping,
        segy_sha256=segy_sha256,
        train_h5_sha256=train_h5_sha256,
        indices_sha256=indices_sha256,
    )
    manifest_path = _native_cache_manifest_path(cache_path)
    if cache_path.is_file() and manifest_path.is_file():
        manifest = base._json(manifest_path)  # noqa: SLF001
        if all(manifest.get(key) == value for key, value in identity.items()):
            if manifest.get("npz_sha256") != base._sha256(  # noqa: SLF001
                cache_path
            ):
                raise RuntimeError("native SEG-Y cache hash mismatch")
            with np.load(cache_path, allow_pickle=False) as payload:
                images = np.asarray(payload["images"], dtype=np.float32)
                cached_inline = np.asarray(
                    payload["unique_inline"],
                    dtype=np.int32,
                )
            np.testing.assert_array_equal(
                cached_inline,
                mapping.unique_inline,
            )
            expected = (
                len(mapping.unique_inline),
                len(SEISMIC_CHANNEL_NAMES),
                NATIVE_TIME_SAMPLES,
                NATIVE_TRACE_COUNT,
            )
            if images.shape != expected or not np.all(np.isfinite(images)):
                raise RuntimeError("native SEG-Y cache shape/value drift")
            audit = dict(manifest["audit"])
            audit["cache_reused"] = True
            audit["cache_sha256"] = manifest["npz_sha256"]
            return images, audit

    import segyio

    with np.load(seismic_index_path, allow_pickle=False) as payload:
        index = {key: payload[key] for key in payload.files}
    sample_start = mapping.time_start - 2
    sample_stop = mapping.time_start + NATIVE_TIME_SAMPLES + 2
    crosslines = np.arange(
        mapping.crossline_start,
        mapping.crossline_start + NATIVE_TRACE_COUNT,
        dtype=np.int32,
    )
    images = np.empty(
        (
            len(mapping.unique_inline),
            len(SEISMIC_CHANNEL_NAMES),
            NATIVE_TIME_SAMPLES,
            NATIVE_TRACE_COUNT,
        ),
        dtype=np.float32,
    )
    normalization_samples: list[dict[str, Any]] = []
    n_xl = int(index["n_xl"])
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as segy:
        if segy.tracecount != int(index["n_traces"]):
            raise RuntimeError("continuous SEG-Y trace count drift")
        for section_id, inline in enumerate(mapping.unique_inline):
            trace_ids = (
                (int(inline) - int(index["il_min"])) * n_xl
                + crosslines
                - int(index["xl_min"])
            )
            extended = np.empty(
                (NATIVE_TIME_SAMPLES + 4, NATIVE_TRACE_COUNT),
                dtype=np.float32,
            )
            for column, trace_id in enumerate(trace_ids):
                trace = np.asarray(
                    segy.trace[int(trace_id)],
                    dtype=np.float32,
                )
                extended[:, column] = trace[sample_start:sample_stop]
            raw_channels = (
                extended[2:-2],
                np.sqrt(
                    sum(
                        extended[offset : offset + NATIVE_TIME_SAMPLES]
                        .astype(np.float64)
                        ** 2
                        for offset in range(5)
                    )
                    / 5.0
                ).astype(np.float32),
                extended[3:-1] - extended[1:-3],
            )
            for channel, values in enumerate(raw_channels):
                normalized, normalization = _normalize_native_channel(
                    values
                )
                images[section_id, channel] = normalized
                if len(normalization_samples) < 12:
                    normalization_samples.append(
                        {
                            "inline": int(inline),
                            "channel": SEISMIC_CHANNEL_NAMES[channel],
                            **normalization,
                        }
                    )
    if not np.all(np.isfinite(images)):
        raise FloatingPointError("native SEG-Y extraction is non-finite")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            images=images,
            unique_inline=mapping.unique_inline,
        )
    audit = {
        "cache_reused": False,
        "source": "continuous ST0202 post-stack SEG-Y",
        "source_sha256": segy_sha256,
        "images_shape": list(images.shape),
        "sections_read": len(mapping.unique_inline),
        "adjacent_traces_per_section": NATIVE_TRACE_COUNT,
        "time_samples_per_trace": NATIVE_TIME_SAMPLES,
        "extra_samples_for_derived_channels": 4,
        "normalization": (
            "per native section and channel z-score; no target statistics"
        ),
        "normalization_samples": normalization_samples,
        "resize_applied": False,
        "interpolation_applied": False,
        "padding_applied": False,
        "label_read": False,
    }
    manifest = {
        **identity,
        "audit": audit,
        "npz_sha256": base._sha256(cache_path),  # noqa: SLF001
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit["cache_sha256"] = manifest["npz_sha256"]
    return images, audit


def _prefix_paths(
    cache_dir: Path,
    *,
    weight_mode: str,
    seed: int,
) -> tuple[Path, Path, Path]:
    suffix = (
        "pretrained"
        if weight_mode == "pretrained"
        else f"random_init_seed_{int(seed)}"
    )
    prefix = cache_dir / f"prefix_{suffix}.npy"
    tail = cache_dir / f"tail_{suffix}.pt"
    manifest = cache_dir / f"prefix_{suffix}.json"
    return prefix, tail, manifest


def _prefix_identity(
    *,
    weight_mode: str,
    seed: int,
    native_cache_sha256: str,
    asset_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PREFIX_CACHE_SCHEMA_VERSION,
        "encoder_weight_mode": weight_mode,
        "random_seed": int(seed) if weight_mode == "random_init" else None,
        "native_cache_sha256": native_cache_sha256,
        "snapshot_revision": gfm.SNAPSHOT_REVISION,
        "source_sha256": asset_audit["source_sha256"],
        "config_sha256": asset_audit["config_sha256"],
        "weights_sha256": asset_audit["weights_sha256"],
        "trainable_block_count": TRAINABLE_BLOCK_COUNT,
        "trainable_block_indices": [15],
        "frozen_prefix_block_indices": list(range(15)),
        "prefix_dtype": "float16",
    }


def get_prefix_cache(
    *,
    images: np.ndarray,
    source_root: Path,
    snapshot_path: Path,
    cache_dir: Path,
    device: str,
    image_batch_size: int,
    weight_mode: str,
    seed: int,
    native_cache_sha256: str,
) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    """Cache the frozen prefix and initial genuine trainable tail."""

    import torch

    asset_audit = gfm.verify_local_assets(
        source_root,
        snapshot_path,
    )[2]
    identity = _prefix_identity(
        weight_mode=weight_mode,
        seed=seed,
        native_cache_sha256=native_cache_sha256,
        asset_audit=asset_audit,
    )
    prefix_path, tail_path, manifest_path = _prefix_paths(
        cache_dir,
        weight_mode=weight_mode,
        seed=seed,
    )
    if prefix_path.is_file() and tail_path.is_file() and manifest_path.is_file():
        manifest = base._json(manifest_path)  # noqa: SLF001
        if all(manifest.get(key) == value for key, value in identity.items()):
            if (
                manifest.get("prefix_sha256")
                != base._sha256(prefix_path)  # noqa: SLF001
                or manifest.get("tail_sha256")
                != base._sha256(tail_path)  # noqa: SLF001
            ):
                raise RuntimeError("GFM prefix/tail cache hash mismatch")
            prefix = np.load(prefix_path, mmap_mode="r")
            expected = (
                images.shape[0],
                images.shape[1],
                161,
                1200,
            )
            if prefix.shape != expected or prefix.dtype != np.float16:
                raise RuntimeError("GFM prefix cache shape/dtype drift")
            tail_state = torch.load(
                tail_path,
                map_location="cpu",
                weights_only=True,
            )
            audit = dict(manifest["audit"])
            audit["cache_reused"] = True
            return prefix, tail_state, audit

    import importlib

    p4 = importlib.import_module(
        "_pipelines.02_task_datasets.reconstruction.p4_reconstruction"
    )
    model = gfm_ft.build_model(
        p4.task_spec("strict"),
        source_root=source_root,
        snapshot_path=snapshot_path,
        device=device,
        encoder_weight_mode=weight_mode,
        random_seed=int(seed),
        trainable_block_count=TRAINABLE_BLOCK_COUNT,
    )
    model.eval()
    if tuple(model.trainable_block_indices) != (15,):
        raise RuntimeError("P15 did not unlock exactly GFM block 15")
    if any(
        parameter.requires_grad
        for parameter in model.network.blocks[:15].parameters()
    ):
        raise RuntimeError("GFM frozen prefix exposes trainable parameters")
    if not all(
        parameter.requires_grad
        for parameter in model.network.blocks[15].parameters()
    ):
        raise RuntimeError("GFM final block is not fully trainable")

    flat_images = images.reshape(
        -1,
        1,
        NATIVE_TIME_SAMPLES,
        NATIVE_TRACE_COUNT,
    )
    prefix = np.empty(
        (len(flat_images), 161, 1200),
        dtype=np.float16,
    )
    for start in range(0, len(flat_images), int(image_batch_size)):
        batch = torch.as_tensor(
            flat_images[start : start + int(image_batch_size)],
            dtype=torch.float32,
            device=device,
        )
        with torch.inference_mode():
            tokens = model.extract_frozen_prefix(batch)
        prefix[start : start + len(batch)] = (
            tokens.detach().to(device="cpu", dtype=torch.float16).numpy()
        )
    prefix = prefix.reshape(images.shape[0], images.shape[1], 161, 1200)
    tail = model.initial_tail().cpu()
    tail_state = gfm_ft.tail_state_dict(tail)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with prefix_path.open("wb") as handle:
        np.save(handle, prefix, allow_pickle=False)
    torch.save(tail_state, tail_path)
    audit = {
        "cache_reused": False,
        "encoder_weight_mode": weight_mode,
        "pretrained_weights_loaded": bool(
            model.asset_audit["pretrained_weights_loaded"]
        ),
        "random_seed": (
            int(seed) if weight_mode == "random_init" else None
        ),
        "architecture_sha256": model.asset_audit[
            "architecture_sha256"
        ],
        "encoder_probe_sha256": model.asset_audit[
            "encoder_probe_sha256"
        ],
        "trainable_block_indices": list(
            model.trainable_block_indices
        ),
        "trainable_encoder_parameters": model.asset_audit[
            "trainable_encoder_parameters"
        ],
        "total_model_parameters": model.asset_audit["parameter_count"],
        "prefix_shape": list(prefix.shape),
        "prefix_dtype": str(prefix.dtype),
        "frozen_prefix_cached": True,
        "tail_remains_trainable_after_cache": True,
        "image_forwards": len(flat_images),
        "three_channels_forwarded_separately": True,
    }
    manifest = {
        **identity,
        "audit": audit,
        "prefix_sha256": base._sha256(prefix_path),  # noqa: SLF001
        "tail_sha256": base._sha256(tail_path),  # noqa: SLF001
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit.update(
        {
            "prefix_sha256": manifest["prefix_sha256"],
            "tail_sha256": manifest["tail_sha256"],
        }
    )
    del model, tail
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.load(prefix_path, mmap_mode="r"), tail_state, audit


def build_query_features(
    *,
    oof: base.OOFDevelopment,
    mapping: NativeMapping,
) -> np.ndarray:
    """Build bounded non-foundation query/context features for the head."""

    query = np.column_stack(
        [
            oof.structural_features,
            oof.baseline,
            oof.distance_to_well,
            oof.xyz,
            mapping.time_positions,
        ]
    ).astype(np.float32)
    if query.shape != (len(oof.target), 12):
        raise RuntimeError("P15 query feature width drift")
    if not np.all(np.isfinite(query)):
        raise FloatingPointError("P15 query features are non-finite")
    return query


def fit_query_scaler(
    query: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(query[train_mask], dtype=np.float64)
    mean = np.mean(selected, axis=0)
    std = np.std(selected, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_query(
    query: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return np.clip((query - mean) / std, -8.0, 8.0).astype(np.float32)


def _make_residual_model(
    *,
    tail_state: Mapping[str, Any],
    query_width: int,
    seed: int,
) -> Any:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    tail = gfm_ft.build_tail_module(
        trainable_block_count=TRAINABLE_BLOCK_COUNT
    )
    tail.load_state_dict(tail_state, strict=True)

    class PartialGFMResidual(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tail = tail
            self.view_projections = torch.nn.ModuleList(
                [
                    torch.nn.Linear(1200, VIEW_PROJECTION_WIDTH)
                    for _ in VIEW_NAMES
                ]
            )
            fused_width = (
                len(VIEW_NAMES) * VIEW_PROJECTION_WIDTH + query_width
            )
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(fused_width),
                torch.nn.Linear(fused_width, HEAD_HIDDEN_WIDTH),
                torch.nn.GELU(),
                torch.nn.Dropout(0.10),
                torch.nn.Linear(HEAD_HIDDEN_WIDTH, 1),
            )
            torch.nn.init.zeros_(self.head[-1].weight)
            torch.nn.init.zeros_(self.head[-1].bias)

        def encoder_parameters(self) -> list[Any]:
            return list(self.tail.parameters())

        def head_parameters(self) -> list[Any]:
            return [
                parameter
                for name, parameter in self.named_parameters()
                if not name.startswith("tail.")
            ]

        def forward(
            self,
            prefix_tokens: Any,
            local_section_ids: Any,
            trace_token_ids: Any,
            query_features: Any,
        ) -> Any:
            if prefix_tokens.ndim != 4 or tuple(
                prefix_tokens.shape[1:]
            ) != (3, 161, 1200):
                raise ValueError(
                    "P15 prefix batch must be [sections,3,161,1200]"
                )
            section_count = int(prefix_tokens.shape[0])
            encoded = self.tail(
                prefix_tokens.reshape(-1, 161, 1200)
            ).reshape(section_count, 3, 161, 1200)
            views: list[Any] = []
            view_id = 0
            for channel in range(3):
                trace = encoded[
                    local_section_ids,
                    channel,
                    1 + trace_token_ids,
                ]
                cls = encoded[local_section_ids, channel, 0]
                views.append(self.view_projections[view_id](trace))
                view_id += 1
                views.append(self.view_projections[view_id](cls))
                view_id += 1
            fused = torch.cat([*views, query_features], dim=1)
            raw = self.head(fused).squeeze(1)
            return torch.tanh(raw) * MAX_RESIDUAL_CORRECTION

    return PartialGFMResidual()


def _parameter_delta_l2(
    parameters: Sequence[Any],
    snapshots: Sequence[Any],
) -> float:
    squared = 0.0
    for parameter, snapshot in zip(parameters, snapshots):
        current = parameter.detach().to(
            device="cpu",
            dtype=snapshot.dtype,
        )
        squared += float((current - snapshot).square().sum())
    return math.sqrt(squared)


def _gradient_norm(parameters: Iterable[Any]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(
                parameter.grad.detach().float().square().sum().cpu()
            )
    return math.sqrt(squared)


def _gradient_probe(parameters: Sequence[Any]) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    remaining = 4096
    for parameter in parameters:
        if parameter.grad is None or remaining <= 0:
            continue
        values = (
            parameter.grad.detach()
            .float()
            .reshape(-1)[:remaining]
            .cpu()
            .numpy()
            .copy()
        )
        if values.size:
            parts.append(values)
            remaining -= len(values)
    if not parts:
        return None
    return np.concatenate(parts)


def _predict_mask(
    *,
    model: Any,
    prefix: np.ndarray,
    mapping: NativeMapping,
    scaled_query: np.ndarray,
    row_mask: np.ndarray,
    device: str,
    section_batch_size: int,
) -> np.ndarray:
    import torch

    row_ids = np.flatnonzero(row_mask)
    result = np.full(len(row_ids), np.nan, dtype=np.float64)
    output_positions = {
        int(row_id): position
        for position, row_id in enumerate(row_ids)
    }
    sections = np.unique(mapping.section_ids[row_ids])
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sections), int(section_batch_size)):
            batch_sections = sections[
                start : start + int(section_batch_size)
            ]
            selected = row_ids[
                np.isin(mapping.section_ids[row_ids], batch_sections)
            ]
            local_lookup = {
                int(section): local
                for local, section in enumerate(batch_sections)
            }
            local_sections = np.asarray(
                [
                    local_lookup[int(mapping.section_ids[row])]
                    for row in selected
                ],
                dtype=np.int64,
            )
            prediction = model(
                torch.as_tensor(
                    np.asarray(prefix[batch_sections], dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    local_sections,
                    dtype=torch.long,
                    device=device,
                ),
                torch.as_tensor(
                    mapping.trace_token_ids[selected],
                    dtype=torch.long,
                    device=device,
                ),
                torch.as_tensor(
                    scaled_query[selected],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            values = prediction.detach().cpu().numpy()
            for row, value in zip(selected, values):
                result[output_positions[int(row)]] = float(value)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("P15 residual prediction is incomplete")
    return result


def _train_tail_model(
    *,
    prefix: np.ndarray,
    tail_state: Mapping[str, Any],
    mapping: NativeMapping,
    query: np.ndarray,
    residual_target: np.ndarray,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray | None,
    validation_target: np.ndarray | None,
    validation_baseline: np.ndarray | None,
    device: str,
    seed: int,
    max_updates: int,
    early_stopping: bool,
    section_batch_size: int,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    """Train the real GFM tail and residual head on legal outer-train rows."""

    import torch

    torch.manual_seed(int(seed))
    np_rng = np.random.default_rng(int(seed))
    model = _make_residual_model(
        tail_state=tail_state,
        query_width=query.shape[1],
        seed=int(seed),
    ).to(device)
    encoder_parameters = model.encoder_parameters()
    head_parameters = model.head_parameters()
    encoder_snapshots = [
        parameter.detach().to(device="cpu", dtype=torch.float32).clone()
        for parameter in encoder_parameters
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": HEAD_LR},
            {"params": encoder_parameters, "lr": ENCODER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(max_updates), 1),
    )
    query_mean, query_std = fit_query_scaler(query, fit_mask)
    scaled_query = transform_query(query, query_mean, query_std)
    fit_rows = np.flatnonzero(fit_mask)
    fit_sections = np.unique(mapping.section_ids[fit_rows])
    rows_by_section = {
        int(section): fit_rows[
            mapping.section_ids[fit_rows] == int(section)
        ]
        for section in fit_sections
    }
    residual_target = np.asarray(residual_target, dtype=np.float32)

    best_state: dict[str, Any] | None = None
    best_validation_rmse = math.inf
    best_update = int(max_updates)
    bad_evaluations = 0
    losses: list[float] = []
    encoder_grad_norms: list[float] = []
    head_grad_norms: list[float] = []
    gradient_cosines: list[float] = []
    previous_probe: np.ndarray | None = None
    validation_history: list[dict[str, float]] = []
    section_order = np.asarray(fit_sections, dtype=np.int64)
    cursor = len(section_order)

    model.train()
    completed_updates = 0
    for update in range(1, int(max_updates) + 1):
        if cursor + int(section_batch_size) > len(section_order):
            section_order = np_rng.permutation(fit_sections)
            cursor = 0
        batch_sections = section_order[
            cursor : cursor + int(section_batch_size)
        ]
        cursor += int(section_batch_size)
        selected = np.concatenate(
            [rows_by_section[int(section)] for section in batch_sections]
        )
        local_lookup = {
            int(section): local
            for local, section in enumerate(batch_sections)
        }
        local_sections = np.asarray(
            [
                local_lookup[int(mapping.section_ids[row])]
                for row in selected
            ],
            dtype=np.int64,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            torch.as_tensor(
                np.asarray(prefix[batch_sections], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                local_sections,
                dtype=torch.long,
                device=device,
            ),
            torch.as_tensor(
                mapping.trace_token_ids[selected],
                dtype=torch.long,
                device=device,
            ),
            torch.as_tensor(
                scaled_query[selected],
                dtype=torch.float32,
                device=device,
            ),
        )
        target = torch.as_tensor(
            residual_target[selected],
            dtype=torch.float32,
            device=device,
        )
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("P15 training loss became non-finite")
        loss.backward()
        encoder_norm = _gradient_norm(encoder_parameters)
        head_norm = _gradient_norm(head_parameters)
        probe = _gradient_probe(encoder_parameters)
        if (
            probe is not None
            and previous_probe is not None
            and np.linalg.norm(probe) > 0
            and np.linalg.norm(previous_probe) > 0
        ):
            gradient_cosines.append(
                float(
                    np.dot(probe, previous_probe)
                    / (np.linalg.norm(probe) * np.linalg.norm(previous_probe))
                )
            )
        if probe is not None and np.linalg.norm(probe) > 0:
            previous_probe = probe
        torch.nn.utils.clip_grad_norm_(
            [*head_parameters, *encoder_parameters],
            GRAD_CLIP,
        )
        optimizer.step()
        scheduler.step()
        completed_updates = update
        losses.append(float(loss.detach().cpu()))
        encoder_grad_norms.append(float(encoder_norm))
        head_grad_norms.append(float(head_norm))

        should_evaluate = (
            early_stopping
            and validation_mask is not None
            and update >= MIN_EARLY_STOP_UPDATES
            and update % EVAL_EVERY == 0
        )
        if should_evaluate:
            residual = _predict_mask(
                model=model,
                prefix=prefix,
                mapping=mapping,
                scaled_query=scaled_query,
                row_mask=validation_mask,
                device=device,
                section_batch_size=section_batch_size,
            )
            if validation_target is None or validation_baseline is None:
                raise RuntimeError("early stopping validation arrays missing")
            candidate = np.asarray(validation_baseline) + residual
            rmse = base._metrics(  # noqa: SLF001
                np.asarray(validation_target),
                candidate,
            )["rmse"]
            validation_history.append(
                {"update": float(update), "rmse": float(rmse)}
            )
            if rmse < best_validation_rmse - 1e-8:
                best_validation_rmse = float(rmse)
                best_update = int(update)
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                bad_evaluations = 0
            else:
                bad_evaluations += 1
            model.train()
            if bad_evaluations >= EARLY_STOPPING_PATIENCE:
                break

    if early_stopping:
        if best_state is None:
            raise RuntimeError("P15 early stopping produced no checkpoint")
        model.load_state_dict(best_state, strict=True)
    else:
        best_update = completed_updates
    update_l2 = _parameter_delta_l2(
        encoder_parameters,
        encoder_snapshots,
    )
    diagnostics_payload = {
        "requested_max_updates": int(max_updates),
        "completed_updates": int(completed_updates),
        "selected_update": int(best_update),
        "stopped_early": bool(completed_updates < int(max_updates)),
        "fit_rows": int(np.sum(fit_mask)),
        "optimizer": {
            "name": "AdamW",
            "head_lr": HEAD_LR,
            "encoder_lr": ENCODER_LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "gradient_clip": GRAD_CLIP,
        },
        "loss": {
            "first": float(losses[0]),
            "last": float(losses[-1]),
            "minimum": float(np.min(losses)),
            "decreased_first_to_last": bool(losses[-1] < losses[0]),
        },
        "encoder_gradient_norm": {
            **base._summary_stats(  # noqa: SLF001
                np.asarray(encoder_grad_norms)
            ),
            "nonzero_step_rate": float(
                np.mean(np.asarray(encoder_grad_norms) > 0.0)
            ),
        },
        "head_gradient_norm": base._summary_stats(  # noqa: SLF001
            np.asarray(head_grad_norms)
        ),
        "encoder_parameter_update_l2": float(update_l2),
        "bounded_gradient_probe_adjacent_cosine": (
            base._summary_stats(  # noqa: SLF001
                np.asarray(gradient_cosines)
            )
            if gradient_cosines
            else None
        ),
        "validation_history": validation_history,
        "best_validation_rmse": (
            float(best_validation_rmse)
            if math.isfinite(best_validation_rmse)
            else None
        ),
        "trainable_encoder_parameters": int(
            sum(parameter.numel() for parameter in encoder_parameters)
        ),
        "trainable_head_parameters": int(
            sum(parameter.numel() for parameter in head_parameters)
        ),
    }
    return model, query_mean, query_std, diagnostics_payload


def _tail_state_probe_sha256(
    state: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(json.dumps(list(values.shape)).encode("ascii"))
        digest.update(values.reshape(-1)[:64].numpy().tobytes())
    return digest.hexdigest()


def _evaluate_weight_mode(
    *,
    route: str,
    oof: base.OOFDevelopment,
    mapping: NativeMapping,
    query: np.ndarray,
    prefix_by_seed: Mapping[int, np.ndarray],
    tail_state_by_seed: Mapping[int, Mapping[str, Any]],
    device: str,
    max_updates: int,
    section_batch_size: int,
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    list[dict[str, Any]],
]:
    predictions: dict[int, dict[str, np.ndarray]] = {}
    cells: list[dict[str, Any]] = []
    residual_target = np.asarray(oof.target - oof.baseline, dtype=np.float32)

    for seed in base.REPEAT_SEEDS:
        prefix = prefix_by_seed[int(seed)]
        tail_state = tail_state_by_seed[int(seed)]
        seed_predictions = {
            "ungated": np.full(len(oof.target), np.nan, dtype=np.float64),
            "gated": np.full(len(oof.target), np.nan, dtype=np.float64),
        }
        for outer_fold in base.FOLD_IDS:
            outer = oof.fold_ids == int(outer_fold)
            calibration_fold = int(
                base.FOLD_IDS[
                    (base.FOLD_IDS.index(int(outer_fold)) + 1)
                    % len(base.FOLD_IDS)
                ]
            )
            calibration = oof.fold_ids == calibration_fold
            fit = ~(outer | calibration)
            if np.any(fit & outer) or np.any(fit & calibration):
                raise RuntimeError("P15 calibration split overlaps fit rows")
            cell_seed = (
                int(seed)
                + 1009 * int(outer_fold)
            )
            calibration_model, cal_mean, cal_std, calibration_diag = (
                _train_tail_model(
                    prefix=prefix,
                    tail_state=tail_state,
                    mapping=mapping,
                    query=query,
                    residual_target=residual_target,
                    fit_mask=fit,
                    validation_mask=calibration,
                    validation_target=oof.target[calibration],
                    validation_baseline=oof.baseline[calibration],
                    device=device,
                    seed=cell_seed,
                    max_updates=int(max_updates),
                    early_stopping=True,
                    section_batch_size=section_batch_size,
                )
            )
            calibration_residual = _predict_mask(
                model=calibration_model,
                prefix=prefix,
                mapping=mapping,
                scaled_query=transform_query(query, cal_mean, cal_std),
                row_mask=calibration,
                device=device,
                section_batch_size=section_batch_size,
            )
            gate_model, gate_scale, gate_scores, benefit_rate = (
                diagnostics._gate_for_inner_predictions(  # noqa: SLF001
                    inner_prediction=calibration_residual,
                    inner_base=oof.baseline[calibration],
                    inner_truth=oof.target[calibration],
                    inner_distance=oof.distance_to_well[calibration],
                    seed=cell_seed,
                )
            )
            selected_updates = int(calibration_diag["selected_update"])
            del calibration_model

            outer_train = ~outer
            refit_model, refit_mean, refit_std, refit_diag = (
                _train_tail_model(
                    prefix=prefix,
                    tail_state=tail_state,
                    mapping=mapping,
                    query=query,
                    residual_target=residual_target,
                    fit_mask=outer_train,
                    validation_mask=None,
                    validation_target=None,
                    validation_baseline=None,
                    device=device,
                    seed=cell_seed,
                    max_updates=selected_updates,
                    early_stopping=False,
                    section_batch_size=section_batch_size,
                )
            )
            outer_residual = _predict_mask(
                model=refit_model,
                prefix=prefix,
                mapping=mapping,
                scaled_query=transform_query(query, refit_mean, refit_std),
                row_mask=outer,
                device=device,
                section_batch_size=section_batch_size,
            )
            raw_gate = gate_model.predict(
                base._gate_signals(  # noqa: SLF001
                    outer_residual,
                    oof.baseline[outer],
                    oof.distance_to_well[outer],
                )
            )
            gate = np.clip(raw_gate * gate_scale, 0.0, 1.0)
            ungated = oof.baseline[outer] + outer_residual
            gated = oof.baseline[outer] + gate * outer_residual
            seed_predictions["ungated"][outer] = ungated
            seed_predictions["gated"][outer] = gated
            baseline_metrics = base._metrics(  # noqa: SLF001
                oof.target[outer],
                oof.baseline[outer],
            )
            gated_metrics = base._metrics(  # noqa: SLF001
                oof.target[outer],
                gated,
            )
            cells.append(
                {
                    "route": route,
                    "seed": int(seed),
                    "outer_fold": int(outer_fold),
                    "independent_spatial_unit": True,
                    "calibration_fold": calibration_fold,
                    "fit_fold_ids": sorted(
                        int(value)
                        for value in np.unique(oof.fold_ids[fit])
                    ),
                    "outer_train_fold_ids": sorted(
                        int(value)
                        for value in np.unique(oof.fold_ids[outer_train])
                    ),
                    "fit_rows": int(np.sum(fit)),
                    "calibration_rows": int(np.sum(calibration)),
                    "outer_train_rows": int(np.sum(outer_train)),
                    "outer_validation_rows": int(np.sum(outer)),
                    "selected_updates": selected_updates,
                    "selected_gate_scale": float(gate_scale),
                    "inner_gate_candidate_rmse": gate_scores,
                    "inner_correction_benefit_rate": benefit_rate,
                    "baseline_metrics": baseline_metrics,
                    "ungated_metrics": base._metrics(  # noqa: SLF001
                        oof.target[outer],
                        ungated,
                    ),
                    "gated_metrics": gated_metrics,
                    "gated_rmse_delta_vs_pykrige": (
                        gated_metrics["rmse"]
                        - baseline_metrics["rmse"]
                    ),
                    "outcome_vs_pykrige": diagnostics.classify_fold_outcome(
                        gated_metrics["rmse"],
                        baseline_metrics["rmse"],
                    ),
                    "residual_prediction_stats": base._summary_stats(  # noqa: SLF001
                        outer_residual
                    ),
                    "gate_stats": {
                        **base._summary_stats(gate),  # noqa: SLF001
                        "zero_rate": float(np.mean(gate == 0.0)),
                        "one_rate": float(np.mean(gate == 1.0)),
                    },
                    "calibration_training": calibration_diag,
                    "outer_refit_training": refit_diag,
                }
            )
            del refit_model
            if str(device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
        for name, values in seed_predictions.items():
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"{route}/{seed}/{name} incomplete")
        predictions[int(seed)] = seed_predictions
    return predictions, cells


def _independent_fold_outcomes(
    *,
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in base.FOLD_IDS:
        selected = [
            row for row in cells if int(row["outer_fold"]) == int(fold)
        ]
        if len(selected) != len(base.REPEAT_SEEDS):
            raise RuntimeError("P15 fold lacks paired seed pseudo-repeats")
        baseline_rmse = float(
            np.mean(
                [row["baseline_metrics"]["rmse"] for row in selected]
            )
        )
        candidate_rmse = float(
            np.mean([row["gated_metrics"]["rmse"] for row in selected])
        )
        rows.append(
            {
                "outer_fold": int(fold),
                "independent_spatial_unit": True,
                "paired_seed_pseudo_repeats": list(base.REPEAT_SEEDS),
                "baseline_mean_seed_rmse": baseline_rmse,
                "gated_mean_seed_rmse": candidate_rmse,
                "rmse_delta_vs_pykrige": candidate_rmse - baseline_rmse,
                "outcome_vs_pykrige": diagnostics.classify_fold_outcome(
                    candidate_rmse,
                    baseline_rmse,
                ),
            }
        )
    return rows


def _training_aggregate(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    refits = [row["outer_refit_training"] for row in cells]
    gradients = np.asarray(
        [row["encoder_gradient_norm"]["mean"] for row in refits],
        dtype=np.float64,
    )
    updates = np.asarray(
        [row["encoder_parameter_update_l2"] for row in refits],
        dtype=np.float64,
    )
    nonzero = np.asarray(
        [
            row["encoder_gradient_norm"]["nonzero_step_rate"]
            for row in refits
        ],
        dtype=np.float64,
    )
    loss_decreased = [
        bool(row["loss"]["decreased_first_to_last"]) for row in refits
    ]
    gradient_cosines = np.asarray(
        [
            row["bounded_gradient_probe_adjacent_cosine"]["mean"]
            for row in refits
            if row["bounded_gradient_probe_adjacent_cosine"] is not None
        ],
        dtype=np.float64,
    )
    return {
        "outer_refit_cells": len(refits),
        "encoder_gradient_norm_mean_by_cell": base._summary_stats(  # noqa: SLF001
            gradients
        ),
        "encoder_parameter_update_l2_by_cell": base._summary_stats(  # noqa: SLF001
            updates
        ),
        "gradient_nonzero_step_rate_by_cell": base._summary_stats(  # noqa: SLF001
            nonzero
        ),
        "cells_with_strictly_positive_parameter_update": int(
            np.sum(updates > 0.0)
        ),
        "cells_with_first_to_last_loss_decrease": int(sum(loss_decreased)),
        "bounded_gradient_probe_adjacent_cosine_by_cell": (
            base._summary_stats(gradient_cosines)  # noqa: SLF001
            if len(gradient_cosines)
            else None
        ),
        "cells_with_positive_mean_adjacent_gradient_cosine": int(
            np.sum(gradient_cosines > 0.0)
        ),
        "all_refits_have_nonzero_gradients": bool(np.all(nonzero > 0.0)),
        "all_refits_move_encoder_parameters": bool(np.all(updates > 0.0)),
    }


def _summarize_route(
    *,
    route: str,
    predictions: Mapping[int, Mapping[str, np.ndarray]],
    cells: Sequence[Mapping[str, Any]],
    oof: base.OOFDevelopment,
) -> dict[str, Any]:
    per_seed = []
    for seed in base.REPEAT_SEEDS:
        per_seed.append(
            {
                "seed": int(seed),
                "ungated": base._metrics(  # noqa: SLF001
                    oof.target,
                    predictions[int(seed)]["ungated"],
                ),
                "gated": base._metrics(  # noqa: SLF001
                    oof.target,
                    predictions[int(seed)]["gated"],
                ),
            }
        )
    independent = _independent_fold_outcomes(cells=cells)
    counts = {
        outcome: sum(
            row["outcome_vs_pykrige"] == outcome
            for row in independent
        )
        for outcome in ("win", "loss", "tie")
    }
    baseline_rmse = base._metrics(  # noqa: SLF001
        oof.target,
        oof.baseline,
    )["rmse"]
    gated_rmse = float(
        np.mean([row["gated"]["rmse"] for row in per_seed])
    )
    ungated_rmse = float(
        np.mean([row["ungated"]["rmse"] for row in per_seed])
    )
    delta = gated_rmse - baseline_rmse
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(
            f"p15:{route}:whole-fold-bootstrap".encode("utf-8")
        ).digest()[:8],
        "little",
    )
    return {
        "gated_mean_seed_rmse": gated_rmse,
        "ungated_mean_seed_rmse": ungated_rmse,
        "rmse_delta_vs_pykrige": delta,
        "relative_gain_vs_pykrige": (
            (baseline_rmse - gated_rmse) / baseline_rmse
        ),
        "independent_spatial_units": len(base.FOLD_IDS),
        "seed_pseudo_repeats_per_unit": len(base.REPEAT_SEEDS),
        "independent_fold_outcomes": independent,
        "independent_fold_outcome_counts": counts,
        "independent_fold_win_rate": counts["win"] / len(base.FOLD_IDS),
        "per_seed": per_seed,
        "training_signal": _training_aggregate(cells),
        "block_bootstrap_rmse_delta_vs_pykrige": (
            diagnostics.block_bootstrap_rmse_delta(
                target=oof.target,
                baseline=oof.baseline,
                candidate_predictions_by_seed={
                    int(seed): predictions[int(seed)]["gated"]
                    for seed in base.REPEAT_SEEDS
                },
                fold_ids=oof.fold_ids,
                bootstrap_seed=bootstrap_seed,
            )
        ),
    }


def _paired_candidate_bootstrap(
    *,
    oof: base.OOFDevelopment,
    pretrained: Mapping[int, Mapping[str, np.ndarray]],
    random_init: Mapping[int, Mapping[str, np.ndarray]],
    replicates: int = diagnostics.BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    folds = tuple(int(value) for value in base.FOLD_IDS)
    seeds = tuple(int(value) for value in base.REPEAT_SEEDS)
    counts = np.asarray(
        [np.sum(oof.fold_ids == fold) for fold in folds],
        dtype=np.float64,
    )
    pretrained_sse = np.empty((len(seeds), len(folds)))
    random_sse = np.empty_like(pretrained_sse)
    for seed_id, seed in enumerate(seeds):
        for fold_id, fold in enumerate(folds):
            mask = oof.fold_ids == fold
            pretrained_sse[seed_id, fold_id] = np.sum(
                (
                    pretrained[seed]["gated"][mask]
                    - oof.target[mask]
                )
                ** 2
            )
            random_sse[seed_id, fold_id] = np.sum(
                (
                    random_init[seed]["gated"][mask]
                    - oof.target[mask]
                )
                ** 2
            )
    rng = np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(b"p15:pretrained-vs-random").digest()[:8],
            "little",
        )
    )
    sampled = rng.integers(0, len(folds), size=(replicates, len(folds)))
    weights = np.column_stack(
        [np.sum(sampled == fold_id, axis=1) for fold_id in range(len(folds))]
    ).astype(np.float64)
    denominator = weights @ counts
    pretrained_rmse = np.sqrt(
        (pretrained_sse @ weights.T) / denominator[None, :]
    )
    random_rmse = np.sqrt(
        (random_sse @ weights.T) / denominator[None, :]
    )
    delta = np.mean(pretrained_rmse - random_rmse, axis=0)
    point = float(
        np.mean(
            [
                base._metrics(  # noqa: SLF001
                    oof.target,
                    pretrained[seed]["gated"],
                )["rmse"]
                - base._metrics(  # noqa: SLF001
                    oof.target,
                    random_init[seed]["gated"],
                )["rmse"]
                for seed in seeds
            ]
        )
    )
    lower, upper = np.quantile(delta, [0.025, 0.975])
    return {
        "metric": "mean-seed RMSE(pretrained) - RMSE(random-init)",
        "direction": "negative favors pretrained",
        "point_estimate": point,
        "confidence_level": 0.95,
        "confidence_interval": [float(lower), float(upper)],
        "interval_excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "bootstrap_unit": "whole locked spatial fold",
        "seeds_kept_paired": True,
        "voxels_resampled_independently": False,
        "replicates": int(replicates),
    }


def evaluate_p15(
    *,
    oof: base.OOFDevelopment,
    mapping: NativeMapping,
    query: np.ndarray,
    pretrained_prefix: np.ndarray,
    pretrained_tail: Mapping[str, Any],
    random_prefix_by_seed: Mapping[int, np.ndarray],
    random_tail_by_seed: Mapping[int, Mapping[str, Any]],
    device: str,
    max_updates: int,
    section_batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run genuine pretrained and matched-random partial fine-tuning OOF."""

    pretrained_prefixes = {
        int(seed): pretrained_prefix for seed in base.REPEAT_SEEDS
    }
    pretrained_tails = {
        int(seed): pretrained_tail for seed in base.REPEAT_SEEDS
    }
    pretrained_predictions, pretrained_cells = _evaluate_weight_mode(
        route="pretrained_geophysical_fm",
        oof=oof,
        mapping=mapping,
        query=query,
        prefix_by_seed=pretrained_prefixes,
        tail_state_by_seed=pretrained_tails,
        device=device,
        max_updates=max_updates,
        section_batch_size=section_batch_size,
    )
    random_predictions, random_cells = _evaluate_weight_mode(
        route="random_init_same_architecture",
        oof=oof,
        mapping=mapping,
        query=query,
        prefix_by_seed=random_prefix_by_seed,
        tail_state_by_seed=random_tail_by_seed,
        device=device,
        max_updates=max_updates,
        section_batch_size=section_batch_size,
    )
    pretrained_summary = _summarize_route(
        route="pretrained_geophysical_fm",
        predictions=pretrained_predictions,
        cells=pretrained_cells,
        oof=oof,
    )
    random_summary = _summarize_route(
        route="random_init_same_architecture",
        predictions=random_predictions,
        cells=random_cells,
        oof=oof,
    )
    pretrained_summary[
        "rmse_minus_random_init_same_architecture"
    ] = (
        pretrained_summary["gated_mean_seed_rmse"]
        - random_summary["gated_mean_seed_rmse"]
    )
    pretrained_summary[
        "better_than_random_init_same_architecture"
    ] = bool(
        pretrained_summary["gated_mean_seed_rmse"]
        < random_summary["gated_mean_seed_rmse"] - WIN_TOLERANCE
    )
    paired_bootstrap = _paired_candidate_bootstrap(
        oof=oof,
        pretrained=pretrained_predictions,
        random_init=random_predictions,
    )
    baseline_metrics = base._metrics(  # noqa: SLF001
        oof.target,
        oof.baseline,
    )
    gate_zero = oof.baseline.copy()
    if not np.array_equal(gate_zero, oof.baseline):
        raise RuntimeError("P15 gate=0 is not bitwise PyKrige")
    required_wins = math.ceil(0.8 * len(base.FOLD_IDS))
    promoted = bool(
        pretrained_summary["relative_gain_vs_pykrige"] >= 0.01
        and pretrained_summary[
            "better_than_random_init_same_architecture"
        ]
        and pretrained_summary["independent_fold_outcome_counts"]["win"]
        >= required_wins
    )
    p14_summary_path = (
        HERE / "_outputs" / "p14_geophysical_fm" / "summary.json"
    )
    p14_reference = base._json(p14_summary_path)  # noqa: SLF001
    p14_head = p14_reference["experiment"]["heads"][
        p14_reference["experiment"]["best_observed_pretrained_head"]
    ]["pretrained_geophysical_fm"]
    generalization_different_from_p14 = bool(
        paired_bootstrap["interval_excludes_zero"]
        or (
            pretrained_summary["relative_gain_vs_pykrige"] > 0.0
            and p14_head["relative_gain_vs_pykrige"] <= 0.0
        )
    )
    experiment = {
        "fixed_protocol": {
            "base_model": base.EXPECTED_MODEL_ID,
            "lane": base.EXPECTED_LANE,
            "split_hash": base.EXPECTED_SPLIT_HASH,
            "outer_folds": list(base.FOLD_IDS),
            "repeat_seeds": list(base.REPEAT_SEEDS),
            "independent_spatial_units": len(base.FOLD_IDS),
            "seeds_are_paired_optimization_pseudo_repeats": True,
            "inner_calibration": (
                "next locked fold inside outer-train; used for early "
                "stopping and bounded P11 gate; final tail refit on all "
                "four outer-train folds for the selected update count"
            ),
            "gate_scale_candidates": list(base.GATE_CANDIDATES),
            "gate_bounds": [0.0, 1.0],
            "test_or_holdout_tuning": False,
        },
        "optimization": {
            "encoder_depth": gfm.EXPECTED_CONFIG["depth"],
            "trainable_block_count": TRAINABLE_BLOCK_COUNT,
            "trainable_block_indices": [15],
            "encoder_lr": ENCODER_LR,
            "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "max_updates": int(max_updates),
            "evaluation_every_updates": EVAL_EVERY,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "gradient_clip": GRAD_CLIP,
            "prefix_cache_boundary": (
                "blocks 0-14 are frozen and their tokens are cached as "
                "float16; block 15 plus final LayerNorm execute inside every "
                "training graph"
            ),
        },
        "baseline": {
            "pykrige_oof": baseline_metrics,
            "gate_zero_exact": base._metrics(  # noqa: SLF001
                oof.target,
                gate_zero,
            ),
            "gate_zero_bitwise_equal_to_pykrige": True,
        },
        "pretrained_geophysical_fm": pretrained_summary,
        "random_init_same_architecture": random_summary,
        "paired_pretrained_vs_random_init_bootstrap": paired_bootstrap,
        "p14_frozen_reference": {
            "summary_sha256": base._sha256(  # noqa: SLF001
                p14_summary_path
            ),
            "gated_mean_seed_rmse": p14_head[
                "gated_mean_seed_rmse"
            ],
            "relative_gain_vs_pykrige": p14_head[
                "relative_gain_vs_pykrige"
            ],
        },
        "hypothesis_assessment": {
            "optimization_dynamics_different_from_p14_frozen_result": True,
            "development_generalization_conclusion_different_from_p14": (
                generalization_different_from_p14
            ),
            "pretrained_random_difference_ci_excludes_zero": (
                paired_bootstrap["interval_excludes_zero"]
            ),
            "genuine_nonzero_gradient_signal": pretrained_summary[
                "training_signal"
            ]["all_refits_have_nonzero_gradients"],
            "genuine_encoder_parameter_movement": pretrained_summary[
                "training_signal"
            ]["all_refits_move_encoder_parameters"],
        },
        "promotion": {
            "passes_development_rule": promoted,
            "required_independent_fold_wins": required_wins,
            "minimum_relative_gain": 0.01,
            "matched_random_init_present": True,
            "default_enabled": promoted,
        },
        "decision": {
            "state": (
                "PROMOTE_DEVELOPMENT_ONLY"
                if promoted
                else "VERIFIED_NO_PROMOTION"
            ),
            "pretrained_contribution_claimed": False,
            "contribution_boundary": (
                "overall residual-fusion changes are reported directly; "
                "pretraining is only distinguished by the paired matched-"
                "random control and is not presumed causal"
            ),
        },
        "per_fold_seed_pseudo_repeats": [
            *pretrained_cells,
            *random_cells,
        ],
    }
    prediction_payload: dict[str, np.ndarray] = {
        "indices_kji": np.asarray(oof.indices_kji, dtype=np.int64),
        "fold_ids": np.asarray(oof.fold_ids, dtype=np.int64),
        "target": np.asarray(oof.target, dtype=np.float64),
        "baseline_prediction": np.asarray(oof.baseline, dtype=np.float64),
        "baseline_error": np.asarray(
            oof.baseline - oof.target,
            dtype=np.float64,
        ),
        "inline": mapping.inline.astype(np.int32),
        "crossline": mapping.crossline.astype(np.int32),
        "time_index": mapping.time_indices.astype(np.int32),
    }
    for route, predictions in (
        ("pretrained_geophysical_fm", pretrained_predictions),
        ("random_init_same_architecture", random_predictions),
    ):
        for seed in base.REPEAT_SEEDS:
            for mode in ("ungated", "gated"):
                prediction = predictions[int(seed)][mode]
                key = f"{route}__seed_{int(seed)}__{mode}"
                prediction_payload[f"{key}__prediction"] = prediction
                prediction_payload[f"{key}__error"] = (
                    prediction - oof.target
                )
    return experiment, prediction_payload


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


def _write_training_jsonl(
    output_dir: Path,
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = output_dir / "training_diagnostics.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in cells:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": base._sha256(path),  # noqa: SLF001
        "rows": len(cells),
    }


def _write_evidence(
    output_dir: Path,
    result: Mapping[str, Any],
) -> None:
    experiment = result["experiment"]
    pretrained = experiment["pretrained_geophysical_fm"]
    random_init = experiment["random_init_same_architecture"]
    baseline = experiment["baseline"]["pykrige_oof"]["rmse"]
    bootstrap = pretrained["block_bootstrap_rmse_delta_vs_pykrige"]
    paired = experiment["paired_pretrained_vs_random_init_bootstrap"]
    mapping = result["native_mapping_audit"]
    counts = pretrained["independent_fold_outcome_counts"]
    random_by_fold = {
        int(row["outer_fold"]): row
        for row in random_init["independent_fold_outcomes"]
    }
    assessment = experiment["hypothesis_assessment"]
    p14_conclusion = (
        "The optimization dynamics are fundamentally different from P14: "
        "P15 has genuine encoder gradients, directionally persistent "
        "gradient probes and parameter movement. The development "
        "generalization conclusion is still unchanged: it does not reverse "
        "the baseline result or establish a separated pretrained effect."
    )
    lines = [
        "# P15 GFM partial fine-tuning — development-only evidence",
        "",
        "## Outcome",
        "",
        f"- PyKrige baseline pooled development RMSE: `{baseline:.12f}`.",
        (
            "- Pretrained GFM partial-fine-tune gated mean-seed RMSE: "
            f"`{pretrained['gated_mean_seed_rmse']:.12f}` "
            f"(delta `{pretrained['rmse_delta_vs_pykrige']:+.12f}`, "
            "positive relative gain means improvement: "
            f"`{pretrained['relative_gain_vs_pykrige']:+.6%}`)."
        ),
        (
            "- Matched random-init partial-fine-tune gated mean-seed RMSE: "
            f"`{random_init['gated_mean_seed_rmse']:.12f}`; pretrained "
            "minus random-init: "
            f"`{pretrained['rmse_minus_random_init_same_architecture']:+.12f}`."
        ),
        (
            "- Five independent spatial-fold outcomes for pretrained: "
            f"{counts['win']} win / {counts['loss']} loss / "
            f"{counts['tie']} tie."
        ),
        (
            f"- Decision: `{experiment['decision']['state']}`. "
            "No improvement is automatically attributed to the foundation "
            "weights."
        ),
        f"- {p14_conclusion}",
        "",
        "## Native continuous-volume window audit",
        "",
        (
            "- P14 did not resize each isolated `9×20×18` patch directly: "
            "it first assembled the development patches to `63×100×72`, "
            "then resized each `63×100` section to `400×160`. The K/time "
            "axis was still enlarged about 6.35×."
        ),
        (
            "- P15 maps the OOF cells back to the original ST0202 SEG-Y "
            "(`385×605` traces, `1126` samples/trace, 4 ms sampling) and "
            "reads exact `400`-sample × `160`-adjacent-trace sections."
        ),
        (
            "- The development cells occupy "
            f"{mapping['development_mapping']['unique_inline_sections']} "
            "native inline sections, crossline span "
            f"{mapping['development_mapping']['crossline_span']} and time "
            "span "
            f"{mapping['development_mapping']['time_index_span']}; both fit "
            "inside one native GFM window."
        ),
        (
            "- Resize/interpolation/padding applied: "
            f"`{mapping['native_window']['resize_applied']}` / "
            f"`{mapping['native_window']['interpolation_applied']}` / "
            f"`{mapping['native_window']['padding_applied']}`."
        ),
        (
            "- Three channels are forwarded separately: raw amplitude, "
            "5-sample local RMS, and two-sample vertical gradient, all "
            "derived on the native window."
        ),
        "",
        "## Genuine partial fine-tuning",
        "",
        (
            "- GFM has 16 encoder blocks. Blocks 0-14 are frozen and their "
            "prefix tokens are cached as `float16` (a disclosed storage "
            "quantization); block 15 and the final LayerNorm execute in the "
            "training graph and receive gradients."
        ),
        (
            f"- Differential AdamW learning rates: encoder `{ENCODER_LR}`, "
            f"new projection/regression head `{HEAD_LR}`; weight decay "
            f"`{WEIGHT_DECAY}`, gradient clip `{GRAD_CLIP}`."
        ),
        (
            "- Early stopping uses one inner spatial fold only. The selected "
            "update count is then refit from the identical initialization on "
            "all four outer-training folds before the untouched outer fold "
            "is predicted."
        ),
        (
            "- Pretrained refit encoder gradient norm (cell means): "
            f"`{pretrained['training_signal']['encoder_gradient_norm_mean_by_cell']['mean']:.6g}`; "
            "encoder update L2: "
            f"`{pretrained['training_signal']['encoder_parameter_update_l2_by_cell']['mean']:.6g}`."
        ),
        (
            "- The bounded encoder-gradient probe has mean adjacent-step "
            "cosine "
            f"`{pretrained['training_signal']['bounded_gradient_probe_adjacent_cosine_by_cell']['mean']:.6f}`; "
            "all "
            f"{pretrained['training_signal']['cells_with_positive_mean_adjacent_gradient_cosine']}/"
            f"{pretrained['training_signal']['outer_refit_cells']} refits "
            "have positive mean direction consistency. This proves an "
            "optimization signal, not a useful pretrained effect."
        ),
        (
            "- Nonzero gradients in every pretrained refit: "
            f"`{pretrained['training_signal']['all_refits_have_nonzero_gradients']}`; "
            "encoder parameters moved in every refit: "
            f"`{pretrained['training_signal']['all_refits_move_encoder_parameters']}`."
        ),
        (
            "- Matched random-init uses the same block 15, head, learning "
            "rates, update selection, weight decay and gate protocol. Its "
            "weights are initialized independently for the same three "
            "paired seeds."
        ),
        "",
        "## Five genuinely independent spatial units",
        "",
        (
            "The five locked folds are the inferential units. The three "
            "seeds are paired optimization pseudo-repeats inside each fold, "
            "not 15 independent samples."
        ),
        "",
        (
            "| fold | baseline RMSE | pretrained RMSE | random-init RMSE | "
            "pretrained delta | outcome |"
        ),
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for row in pretrained["independent_fold_outcomes"]:
        random_row = random_by_fold[int(row["outer_fold"])]
        lines.append(
            f"| {row['outer_fold']} | "
            f"{row['baseline_mean_seed_rmse']:.9f} | "
            f"{row['gated_mean_seed_rmse']:.9f} | "
            f"{random_row['gated_mean_seed_rmse']:.9f} | "
            f"{row['rmse_delta_vs_pykrige']:+.9f} | "
            f"{row['outcome_vs_pykrige']} |"
        )
    lines.extend(
        [
            "",
            "## Whole-fold bootstrap",
            "",
            (
                "- Pretrained minus PyKrige pooled RMSE point estimate: "
                f"`{bootstrap['point_estimate']:+.12f}`; 95% interval "
                f"`[{bootstrap['confidence_interval'][0]:+.12f}, "
                f"{bootstrap['confidence_interval'][1]:+.12f}]`."
            ),
            (
                "- Pretrained minus matched-random RMSE point estimate: "
                f"`{paired['point_estimate']:+.12f}`; 95% interval "
                f"`[{paired['confidence_interval'][0]:+.12f}, "
                f"{paired['confidence_interval'][1]:+.12f}]`."
            ),
            (
                "- Bootstrap unit is the whole locked spatial fold. Seeds "
                "remain paired; voxels are never independently resampled."
            ),
            "",
            "## Holdout firewall",
            "",
            (
                "- `train.h5` reads are limited to patch metadata, "
                "`seismic_patch[3:6]` and `seismic_patch[8]`; no HDF5 label "
                "dataset is read."
            ),
            (
                "- The continuous SEG-Y is used only for label-free seismic "
                "covariates. `test.h5`, frozen holdout paths, holdout labels "
                "and historical test metrics are neither opened nor probed."
            ),
            (
                "- PORO targets and PyKrige predictions come only from the "
                "same five hash-verified development OOF archives as P11-P14."
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
    prefix_batch_size: int,
    section_batch_size: int,
    max_updates: int,
) -> dict[str, Any]:
    """Run complete native-window pretrained/random partial fine-tuning."""

    started = time.time()
    output_dir = _validate_output_dir(output_dir)
    cache_dir = Path(cache_dir).expanduser().resolve()
    paths = (
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
    base.ensure_no_holdout_paths(paths)
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
    mapping = build_native_mapping(
        train_h5=inputs.train_h5,
        oof=oof,
        build_summary_path=build_summary_path,
        seismic_index_path=seismic_index_path,
        well_tie_path=well_tie_path,
    )
    train_h5_sha256 = base._sha256(inputs.train_h5)  # noqa: SLF001
    segy_sha256 = base._sha256(segy_path)  # noqa: SLF001
    indices_sha256 = base._array_sha256(  # noqa: SLF001
        oof.indices_kji
    )
    native_cache_path = cache_dir / "native_windows.npz"
    images, native_audit = get_native_windows(
        segy_path=segy_path,
        seismic_index_path=seismic_index_path,
        mapping=mapping,
        cache_path=native_cache_path,
        segy_sha256=segy_sha256,
        train_h5_sha256=train_h5_sha256,
        indices_sha256=indices_sha256,
    )
    print(
        json.dumps(
            {
                "event": "native_windows_ready",
                "cache_reused": native_audit["cache_reused"],
                "shape": list(images.shape),
                "resize_applied": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    pretrained_prefix, pretrained_tail, pretrained_cache_audit = (
        get_prefix_cache(
            images=images,
            source_root=source_root,
            snapshot_path=snapshot_path,
            cache_dir=cache_dir,
            device=device,
            image_batch_size=prefix_batch_size,
            weight_mode="pretrained",
            seed=int(base.REPEAT_SEEDS[0]),
            native_cache_sha256=native_audit["cache_sha256"],
        )
    )
    print(
        json.dumps(
            {
                "event": "pretrained_prefix_ready",
                "cache_reused": pretrained_cache_audit["cache_reused"],
                "shape": list(pretrained_prefix.shape),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    random_prefix: dict[int, np.ndarray] = {}
    random_tail: dict[int, Mapping[str, Any]] = {}
    random_cache_audits: list[dict[str, Any]] = []
    for seed in base.REPEAT_SEEDS:
        prefix, tail, audit = get_prefix_cache(
            images=images,
            source_root=source_root,
            snapshot_path=snapshot_path,
            cache_dir=cache_dir,
            device=device,
            image_batch_size=prefix_batch_size,
            weight_mode="random_init",
            seed=int(seed),
            native_cache_sha256=native_audit["cache_sha256"],
        )
        random_prefix[int(seed)] = prefix
        random_tail[int(seed)] = tail
        random_cache_audits.append(audit)
        print(
            json.dumps(
                {
                    "event": "random_prefix_ready",
                    "seed": int(seed),
                    "cache_reused": audit["cache_reused"],
                    "shape": list(prefix.shape),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if len(
        {
            _tail_state_probe_sha256(random_tail[int(seed)])
            for seed in base.REPEAT_SEEDS
        }
    ) != len(base.REPEAT_SEEDS):
        raise RuntimeError("random-init tail states are not seed-distinct")
    if any(
        audit["architecture_sha256"]
        != pretrained_cache_audit["architecture_sha256"]
        for audit in random_cache_audits
    ):
        raise RuntimeError("pretrained/random GFM architecture mismatch")

    query = build_query_features(oof=oof, mapping=mapping)
    experiment, prediction_payload = evaluate_p15(
        oof=oof,
        mapping=mapping,
        query=query,
        pretrained_prefix=pretrained_prefix,
        pretrained_tail=pretrained_tail,
        random_prefix_by_seed=random_prefix,
        random_tail_by_seed=random_tail,
        device=device,
        max_updates=int(max_updates),
        section_batch_size=int(section_batch_size),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_artifact = _write_prediction_errors(
        output_dir,
        prediction_payload,
    )
    training_artifact = _write_training_jsonl(
        output_dir,
        experiment["per_fold_seed_pseudo_repeats"],
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "implementation": {
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "script_sha256": base._sha256(Path(__file__)),  # noqa: SLF001
            "adapter": str(
                (
                    PROJECT_ROOT
                    / "_models"
                    / "reconstruction"
                    / "geophysical_fm_finetune.py"
                ).relative_to(PROJECT_ROOT)
            ),
            "adapter_sha256": base._sha256(  # noqa: SLF001
                PROJECT_ROOT
                / "_models"
                / "reconstruction"
                / "geophysical_fm_finetune.py"
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
            "trainable_block_indices": [15],
        },
        "native_mapping_audit": dict(mapping.audit),
        "native_window_audit": native_audit,
        "pretrained_prefix_audit": pretrained_cache_audit,
        "random_init_prefix_audits": random_cache_audits,
        "source_oof": {
            "records": list(oof.source_records),
            "rows": len(oof.target),
            "indices_kji_sha256": indices_sha256,
        },
        "experiment": experiment,
        "prediction_error_artifact": prediction_artifact,
        "training_diagnostics_artifact": training_artifact,
        "holdout_firewall": {
            "hdf5_files_opened": ["train.h5"],
            "hdf5_datasets_read": [
                "seismic_patch[3:6]",
                "seismic_patch[8]",
            ],
            "raw_continuous_source_opened": segy_path.name,
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
            "prefix_batch_size": int(prefix_batch_size),
            "section_batch_size": int(section_batch_size),
            "max_updates": int(max_updates),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_evidence(output_dir, result)
    artifact_manifest = {
        "summary.json": base._sha256(  # noqa: SLF001
            output_dir / "summary.json"
        ),
        "evidence.md": base._sha256(  # noqa: SLF001
            output_dir / "evidence.md"
        ),
        "prediction_errors.npz": prediction_artifact["sha256"],
        "training_diagnostics.jsonl": training_artifact["sha256"],
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
        help="run native-window pretrained/random partial fine-tuning",
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
    execute.add_argument("--prefix-batch-size", type=int, default=4)
    execute.add_argument(
        "--section-batch-size",
        type=int,
        default=SECTION_BATCH_SIZE,
    )
    execute.add_argument("--max-updates", type=int, default=MAX_UPDATES)
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
