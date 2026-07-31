#!/usr/bin/env python3
"""P20 strict PEFT and staged-unfreezing search for reconstruction.

This diagnostic experiment keeps the P17/P19 outer-fold contract: exactly 512
legal labels train each fold and the unchanged 2,048 development rows form its
validation fold.  It never exposes a holdout CLI argument.  Genuine pretrained
GFM prefix tokens are extracted from exact 400 x 160 continuous SEG-Y windows.

The trainable routes are deliberately small and auditable:

* nonzero_head: frozen GFM tail with a nonzero-initialized prediction head;
* lora_r4: LoRA on qkv/proj/fc1/fc2 of the genuine final transformer block;
* staged_adapter: head warm-up, bottleneck adapter, then terminal LayerNorm;
* staged_lora_r4: head warm-up, LoRA, terminal LayerNorm, then a very-low-LR
  final-block unfreeze.

Within each outer fold, a coordinate-only farthest-point calibration subset
selects the update count.  The model is then restarted and refit on all 512
labels.  Learned hidden features deform the same anisotropic local-kernel
metric used by P18/P19.  No target or validation statistic fits any scaler/PCA.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
PIPELINE_DIR = HERE
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import p11_residual_fusion as base  # noqa: E402
import p15_gfm_finetune as p15  # noqa: E402
import p17_foundation_geostatistics as p17  # noqa: E402
from _models.reconstruction import geophysical_fm_finetune as gfm_ft  # noqa: E402


SCHEMA_VERSION = "reconstruction-p20-peft-staged-unfreeze/v1"
SEED = 2693
CALIBRATION_ROWS = 64
MAX_UPDATES = 32
EVAL_UPDATES = tuple(range(8, 81, 8))
SECTION_BATCH_SIZE = 8
GRAD_CLIP = 1.0
HEAD_LR = 3e-4
PEFT_LR = 1e-4
NORM_LR = 1e-5
FULL_TAIL_LR = 5e-7
WEIGHT_DECAY = 1e-4
LORA_RANK = 4
LORA_ALPHA = 8.0
ADAPTER_WIDTH = 24
HIDDEN_WIDTH = 64
VIEW_WIDTH = 16
PCA_COMPONENTS = 8
VERTICAL_WEIGHT = 4.0
LEARNED_WEIGHT = 0.10
SEISMIC_WEIGHT = 0.10
NEIGHBOURS = 64
DISTANCE_POWER = 1.5
KERNEL_BLEND = 0.75
ROUTES = (
    "nonzero_head",
    "lora_r4",
    "staged_adapter",
    "staged_lora_r4",
)


@dataclass(frozen=True)
class WindowedMapping:
    requested_indices: np.ndarray
    section_keys: tuple[tuple[int, int], ...]
    section_ids: np.ndarray
    trace_token_ids: np.ndarray
    time_positions: np.ndarray
    inline: np.ndarray
    crossline: np.ndarray
    time_indices: np.ndarray
    time_start: int
    audit: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "rows": int(len(error)),
    }


def _index_lookup(requested: np.ndarray) -> dict[tuple[int, int, int], int]:
    return {
        tuple(int(value) for value in row): position
        for position, row in enumerate(requested.tolist())
    }


def _row_ids(
    lookup: Mapping[tuple[int, int, int], int], indices: np.ndarray
) -> np.ndarray:
    try:
        return np.asarray(
            [lookup[tuple(int(value) for value in row)] for row in indices],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RuntimeError("fold KJI row is absent from the native GFM map") from exc


def _farthest_calibration(raw: np.ndarray, count: int) -> np.ndarray:
    """Choose calibration rows using geometry only, never labels."""

    xyz = np.asarray(raw[:, 3:6], dtype=np.float64)
    scale = np.std(xyz, axis=0)
    scale[scale < 1e-8] = 1.0
    points = (xyz - np.mean(xyz, axis=0)) / scale
    chosen = [int(np.argmin(np.sum(points, axis=1)))]
    minimum = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    for _ in range(1, int(count)):
        minimum[np.asarray(chosen, dtype=np.int64)] = -1.0
        next_row = int(np.argmax(minimum))
        chosen.append(next_row)
        distance = np.sum((points - points[next_row]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance)
    selected = np.asarray(sorted(chosen), dtype=np.int64)
    if len(np.unique(selected)) != int(count):
        raise RuntimeError("coordinate-only calibration selection duplicated rows")
    return selected


def build_windowed_mapping(
    *,
    train_h5: Path,
    requested_indices: np.ndarray,
    build_summary_path: Path,
    seismic_index_path: Path,
    well_tie_path: Path,
) -> WindowedMapping:
    """Map all fold train/validation cells to overlapping native windows."""

    coordinates, coordinate_indices, coordinate_audit = p17._coordinate_index(  # noqa: SLF001
        train_h5
    )
    coordinate_lookup = _index_lookup(coordinate_indices)
    positions = _row_ids(coordinate_lookup, requested_indices)
    xyz = np.asarray(coordinates[positions], dtype=np.float64)
    build_summary = base._json(build_summary_path)  # noqa: SLF001
    bounds = build_summary["coordinate_bounds"]
    x = bounds["x"][0] + xyz[:, 0] * (bounds["x"][1] - bounds["x"][0])
    y = bounds["y"][0] + xyz[:, 1] * (bounds["y"][1] - bounds["y"][0])
    depth = bounds["depth"][0] + xyz[:, 2] * (
        bounds["depth"][1] - bounds["depth"][0]
    )
    with np.load(seismic_index_path, allow_pickle=False) as payload:
        seismic_index = {key: payload[key] for key in payload.files}
    inline, crossline, in_bounds = p15._xy_to_il_xl(x, y, seismic_index)  # noqa: SLF001
    if not bool(np.all(in_bounds)):
        raise RuntimeError("a legal development point lies outside the SEG-Y grid")
    twt_ms, tie_audit = p15._estimate_twt_from_weak_ties(  # noqa: SLF001
        x=x,
        y=y,
        depth=depth,
        seismic_index=seismic_index,
        well_tie_path=well_tie_path,
    )
    samples_ms = np.asarray(seismic_index["samples_ms"], dtype=np.float64)
    time_indices = np.searchsorted(samples_ms, twt_ms)
    time_indices = np.clip(time_indices, 1, len(samples_ms) - 2)
    left = np.abs(samples_ms[time_indices - 1] - twt_ms)
    right = np.abs(samples_ms[time_indices] - twt_ms)
    time_indices = np.where(left < right, time_indices - 1, time_indices).astype(
        np.int32
    )
    time_start = p15.centered_native_window_start(
        time_indices,
        window_size=p15.NATIVE_TIME_SAMPLES,
        lower_bound=2,
        upper_bound=len(samples_ms) - 3,
    )

    lower = int(seismic_index["xl_min"])
    upper_start = int(seismic_index["xl_max"]) - p15.NATIVE_TRACE_COUNT + 1
    minimum_crossline = int(np.min(crossline))
    maximum_crossline = int(np.max(crossline))
    span = maximum_crossline - minimum_crossline + 1
    window_count = int(math.ceil(span / p15.NATIVE_TRACE_COUNT))
    first_start = int(np.clip(minimum_crossline, lower, upper_start))
    last_start = int(
        np.clip(
            maximum_crossline - p15.NATIVE_TRACE_COUNT + 1,
            lower,
            upper_start,
        )
    )
    starts = sorted(
        {
            int(round(value))
            for value in np.linspace(first_start, last_start, window_count)
        }
    )
    if not starts:
        raise RuntimeError("no native crossline window was created")
    assigned_starts = np.empty(len(crossline), dtype=np.int32)
    for position, value in enumerate(crossline):
        valid = [
            start
            for start in starts
            if start <= int(value) < start + p15.NATIVE_TRACE_COUNT
        ]
        if not valid:
            raise RuntimeError("native windows do not cover every requested crossline")
        assigned_starts[position] = min(
            valid,
            key=lambda start: (abs(int(value) - (start + 79.5)), start),
        )
    keys = tuple(
        sorted(
            {
                (int(il), int(start))
                for il, start in zip(inline.tolist(), assigned_starts.tolist())
            }
        )
    )
    key_to_id = {key: position for position, key in enumerate(keys)}
    section_ids = np.asarray(
        [key_to_id[(int(il), int(start))] for il, start in zip(inline, assigned_starts)],
        dtype=np.int64,
    )
    trace_token_ids = crossline.astype(np.int64) - assigned_starts.astype(np.int64)
    if np.any(trace_token_ids < 0) or np.any(
        trace_token_ids >= p15.NATIVE_TRACE_COUNT
    ):
        raise RuntimeError("native trace token mapping is out of bounds")
    time_positions = (
        time_indices.astype(np.float64) - float(time_start)
    ) / float(p15.NATIVE_TIME_SAMPLES - 1)
    return WindowedMapping(
        requested_indices=np.asarray(requested_indices, dtype=np.int64),
        section_keys=keys,
        section_ids=section_ids,
        trace_token_ids=trace_token_ids,
        time_positions=time_positions.astype(np.float32),
        inline=inline,
        crossline=crossline,
        time_indices=time_indices,
        time_start=int(time_start),
        audit={
            **coordinate_audit,
            "requested_rows": int(len(requested_indices)),
            "requested_indices_sha256": _array_sha256(requested_indices),
            "native_section_count": int(len(keys)),
            "native_crossline_window_starts": starts,
            "native_crossline_window_count": int(len(starts)),
            "native_window_shape": [
                p15.NATIVE_TIME_SAMPLES,
                p15.NATIVE_TRACE_COUNT,
            ],
            "inline_range": [int(np.min(inline)), int(np.max(inline))],
            "crossline_range": [
                int(np.min(crossline)),
                int(np.max(crossline)),
            ],
            "crossline_span": int(np.ptp(crossline)) + 1,
            "time_index_range": [
                int(np.min(time_indices)),
                int(np.max(time_indices)),
            ],
            "time_start": int(time_start),
            "resize_applied": False,
            "interpolation_applied": False,
            "padding_applied": False,
            "deterministic_overlap_assignment": True,
            "weak_tie_mapping": tie_audit,
        },
    )


def get_native_images(
    *,
    segy_path: Path,
    seismic_index_path: Path,
    mapping: WindowedMapping,
    cache_path: Path,
    source_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract/cache exact native images for every (inline, window) key."""

    manifest_path = cache_path.with_suffix(".json")
    identity = {
        "schema_version": SCHEMA_VERSION + "/native-v1",
        "source_sha256": source_sha256,
        "requested_indices_sha256": _array_sha256(mapping.requested_indices),
        "section_keys_sha256": _array_sha256(
            np.asarray(mapping.section_keys, dtype=np.int32)
        ),
        "time_start": int(mapping.time_start),
        "channels": list(p15.SEISMIC_CHANNEL_NAMES),
    }
    if cache_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in identity.items()):
            if manifest.get("npy_sha256") != _sha256(cache_path):
                raise RuntimeError("P20 native image cache hash mismatch")
            images = np.load(cache_path, mmap_mode="r")
            expected = (
                len(mapping.section_keys),
                3,
                p15.NATIVE_TIME_SAMPLES,
                p15.NATIVE_TRACE_COUNT,
            )
            if images.shape != expected or images.dtype != np.float32:
                raise RuntimeError("P20 native image cache shape/dtype drift")
            return images, {**manifest["audit"], "cache_reused": True}

    import segyio

    with np.load(seismic_index_path, allow_pickle=False) as payload:
        index = {key: payload[key] for key in payload.files}
    sample_start = mapping.time_start - 2
    sample_stop = mapping.time_start + p15.NATIVE_TIME_SAMPLES + 2
    images = np.empty(
        (
            len(mapping.section_keys),
            3,
            p15.NATIVE_TIME_SAMPLES,
            p15.NATIVE_TRACE_COUNT,
        ),
        dtype=np.float32,
    )
    normalization_samples: list[dict[str, Any]] = []
    n_xl = int(index["n_xl"])
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as segy:
        if int(segy.tracecount) != int(index["n_traces"]):
            raise RuntimeError("continuous SEG-Y trace count drift")
        for section_id, (inline, crossline_start) in enumerate(mapping.section_keys):
            crosslines = np.arange(
                crossline_start,
                crossline_start + p15.NATIVE_TRACE_COUNT,
                dtype=np.int32,
            )
            trace_ids = (
                (int(inline) - int(index["il_min"])) * n_xl
                + crosslines
                - int(index["xl_min"])
            )
            extended = np.empty(
                (p15.NATIVE_TIME_SAMPLES + 4, p15.NATIVE_TRACE_COUNT),
                dtype=np.float32,
            )
            for column, trace_id in enumerate(trace_ids):
                trace = np.asarray(segy.trace[int(trace_id)], dtype=np.float32)
                extended[:, column] = trace[sample_start:sample_stop]
            raw_channels = (
                extended[2:-2],
                np.sqrt(
                    sum(
                        extended[offset : offset + p15.NATIVE_TIME_SAMPLES]
                        .astype(np.float64)
                        ** 2
                        for offset in range(5)
                    )
                    / 5.0
                ).astype(np.float32),
                extended[3:-1] - extended[1:-3],
            )
            for channel, values in enumerate(raw_channels):
                normalized, audit = p15._normalize_native_channel(values)  # noqa: SLF001
                images[section_id, channel] = normalized
                if len(normalization_samples) < 12:
                    normalization_samples.append(
                        {
                            "section_id": int(section_id),
                            "inline": int(inline),
                            "crossline_start": int(crossline_start),
                            "channel": p15.SEISMIC_CHANNEL_NAMES[channel],
                            **audit,
                        }
                    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.save(handle, images, allow_pickle=False)
    audit = {
        "cache_reused": False,
        "images_shape": list(images.shape),
        "sections_read": int(len(mapping.section_keys)),
        "normalization": "per-section per-channel z-score",
        "normalization_samples": normalization_samples,
        "resize_applied": False,
        "interpolation_applied": False,
        "padding_applied": False,
        "label_read": False,
    }
    manifest = {**identity, "npy_sha256": _sha256(cache_path), "audit": audit}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return np.load(cache_path, mmap_mode="r"), {
        **audit,
        "npy_sha256": manifest["npy_sha256"],
    }


def _set_linear(parent: Any, name: str, replacement: Any) -> None:
    setattr(parent, name, replacement)


def _make_model(
    *,
    tail_state: Mapping[str, Any],
    query_width: int,
    route: str,
    seed: int,
) -> Any:
    import torch

    if route not in ROUTES:
        raise ValueError(f"unknown P20 route: {route}")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    class LoRALinear(torch.nn.Module):
        def __init__(self, base_linear: Any) -> None:
            super().__init__()
            self.base = base_linear
            self.rank = LORA_RANK
            self.scale = LORA_ALPHA / LORA_RANK
            self.lora_a = torch.nn.Linear(
                base_linear.in_features, LORA_RANK, bias=False
            )
            self.lora_b = torch.nn.Linear(
                LORA_RANK, base_linear.out_features, bias=False
            )
            torch.nn.init.normal_(self.lora_a.weight, mean=0.0, std=1e-4)
            torch.nn.init.normal_(self.lora_b.weight, mean=0.0, std=1e-4)

        def forward(self, values: Any) -> Any:
            return self.base(values) + self.scale * self.lora_b(
                self.lora_a(values)
            )

    tail = gfm_ft.build_tail_module(trainable_block_count=1)
    tail.load_state_dict(tail_state, strict=True)
    uses_lora = route in {"lora_r4", "staged_lora_r4"}
    if uses_lora:
        block = tail.blocks[0]
        _set_linear(block.attn, "qkv", LoRALinear(block.attn.qkv))
        _set_linear(block.attn, "proj", LoRALinear(block.attn.proj))
        _set_linear(block.mlp, "fc1", LoRALinear(block.mlp.fc1))
        _set_linear(block.mlp, "fc2", LoRALinear(block.mlp.fc2))

    class P20Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tail = tail
            self.route = route
            self.view_projections = torch.nn.ModuleList(
                [torch.nn.Linear(1200, VIEW_WIDTH) for _ in range(6)]
            )
            self.adapter_norm = torch.nn.LayerNorm(1200)
            self.adapter_down = torch.nn.Linear(1200, ADAPTER_WIDTH, bias=False)
            self.adapter_up = torch.nn.Linear(ADAPTER_WIDTH, 1200, bias=False)
            torch.nn.init.normal_(self.adapter_down.weight, mean=0.0, std=1e-3)
            torch.nn.init.normal_(self.adapter_up.weight, mean=0.0, std=1e-4)
            fused_width = 6 * VIEW_WIDTH + int(query_width)
            self.head_body = torch.nn.Sequential(
                torch.nn.LayerNorm(fused_width),
                torch.nn.Linear(fused_width, HIDDEN_WIDTH),
                torch.nn.GELU(),
                torch.nn.Dropout(0.10),
            )
            self.head_out = torch.nn.Linear(HIDDEN_WIDTH, 1)
            torch.nn.init.normal_(self.head_out.weight, mean=0.0, std=1e-4)
            torch.nn.init.zeros_(self.head_out.bias)

        def forward(
            self,
            prefix_tokens: Any,
            local_section_ids: Any,
            trace_token_ids: Any,
            query_features: Any,
            *,
            return_hidden: bool = False,
        ) -> Any:
            if prefix_tokens.ndim != 4 or tuple(prefix_tokens.shape[1:]) != (
                3,
                161,
                1200,
            ):
                raise ValueError("P20 prefix batch must be [sections,3,161,1200]")
            section_count = int(prefix_tokens.shape[0])
            encoded = self.tail(prefix_tokens.reshape(-1, 161, 1200)).reshape(
                section_count, 3, 161, 1200
            )
            if self.route == "staged_adapter":
                encoded = encoded + self.adapter_up(
                    torch.nn.functional.gelu(
                        self.adapter_down(self.adapter_norm(encoded))
                    )
                )
            views: list[Any] = []
            projection = 0
            for channel in range(3):
                trace = encoded[
                    local_section_ids, channel, 1 + trace_token_ids
                ]
                cls = encoded[local_section_ids, channel, 0]
                views.append(self.view_projections[projection](trace))
                projection += 1
                views.append(self.view_projections[projection](cls))
                projection += 1
            hidden = self.head_body(torch.cat([*views, query_features], dim=1))
            prediction = self.head_out(hidden).squeeze(1)
            return (prediction, hidden) if return_hidden else prediction

    return P20Model()


def _parameter_groups(model: Any) -> dict[str, list[Any]]:
    all_peft = [
        parameter
        for name, parameter in model.named_parameters()
        if any(token in name for token in ("lora_a", "lora_b", "adapter_"))
    ]
    if model.route in {"lora_r4", "staged_lora_r4"}:
        peft = [
            parameter
            for name, parameter in model.named_parameters()
            if "lora_a" in name or "lora_b" in name
        ]
    elif model.route == "staged_adapter":
        peft = [
            parameter
            for name, parameter in model.named_parameters()
            if "adapter_" in name
        ]
    else:
        peft = []
    active_peft_ids = {id(parameter) for parameter in peft}
    inactive = [
        parameter for parameter in all_peft if id(parameter) not in active_peft_ids
    ]
    norm = list(model.tail.norm.parameters())
    peft_ids = {id(parameter) for parameter in peft}
    norm_ids = {id(parameter) for parameter in norm}
    head = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("view_projections")
        or name.startswith("head_body")
        or name.startswith("head_out")
    ]
    head_ids = {id(parameter) for parameter in head}
    base_tail = [
        parameter
        for parameter in model.tail.parameters()
        if id(parameter) not in peft_ids
        and id(parameter) not in norm_ids
        and id(parameter) not in head_ids
    ]
    return {
        "head": head,
        "peft": peft,
        "norm": norm,
        "base_tail": base_tail,
        "inactive": inactive,
    }


def _set_phase(
    groups: Mapping[str, Sequence[Any]], route: str, update: int
) -> str:
    for parameters in groups.values():
        for parameter in parameters:
            parameter.requires_grad = False
    for parameter in groups["head"]:
        parameter.requires_grad = True
    if route == "nonzero_head":
        return "head"
    if route == "lora_r4":
        for parameter in groups["peft"]:
            parameter.requires_grad = True
        return "head+lora"
    if update <= 8:
        return "head_warmup"
    for parameter in groups["peft"]:
        parameter.requires_grad = True
    if update <= 16:
        return "head+peft"
    for parameter in groups["norm"]:
        parameter.requires_grad = True
    if route == "staged_adapter" or update <= 24:
        return "head+peft+terminal_norm"
    for parameter in groups["base_tail"]:
        parameter.requires_grad = True
    return "head+lora+terminal_norm+full_final_block"


def _grad_norm(parameters: Iterable[Any]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(squared)


def _snapshot(parameters: Sequence[Any]) -> list[Any]:
    return [
        parameter.detach().to(device="cpu", dtype=parameter.dtype).clone()
        for parameter in parameters
    ]


def _delta_l2(parameters: Sequence[Any], snapshots: Sequence[Any]) -> float:
    total = 0.0
    for parameter, before in zip(parameters, snapshots):
        after = parameter.detach().to(device="cpu", dtype=before.dtype)
        total += float((after - before).float().square().sum())
    return math.sqrt(total)


def _scaled_query(
    train_raw: np.ndarray,
    validation_raw: np.ndarray,
    train_time: np.ndarray,
    validation_time: np.ndarray,
    fit_local_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_query = np.column_stack([train_raw, train_time])
    validation_query = np.column_stack([validation_raw, validation_time])
    scaler = StandardScaler().fit(train_query[fit_local_rows])
    return (
        np.clip(scaler.transform(train_query), -8.0, 8.0).astype(np.float32),
        np.clip(scaler.transform(validation_query), -8.0, 8.0).astype(np.float32),
        {
            "query_width": int(train_query.shape[1]),
            "fit_rows": int(len(fit_local_rows)),
            "fit_on_outer_train_only": True,
            "validation_statistics_used": False,
        },
    )


def _batches(
    *,
    row_ids: np.ndarray,
    mapping: WindowedMapping,
    rng: np.random.Generator,
    section_batch_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    sections = np.unique(mapping.section_ids[row_ids])
    order = rng.permutation(sections)
    for start in range(0, len(order), int(section_batch_size)):
        batch_sections = np.asarray(
            order[start : start + int(section_batch_size)], dtype=np.int64
        )
        selected = row_ids[np.isin(mapping.section_ids[row_ids], batch_sections)]
        yield batch_sections, selected


def _forward_rows(
    *,
    model: Any,
    prefix: np.ndarray,
    mapping: WindowedMapping,
    global_row_ids: np.ndarray,
    query: np.ndarray,
    device: str,
    return_hidden: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    predictions = np.full(len(global_row_ids), np.nan, dtype=np.float64)
    hidden = (
        np.full((len(global_row_ids), HIDDEN_WIDTH), np.nan, dtype=np.float64)
        if return_hidden
        else None
    )
    output_lookup = {int(row): pos for pos, row in enumerate(global_row_ids)}
    sections = np.unique(mapping.section_ids[global_row_ids])
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sections), SECTION_BATCH_SIZE):
            batch_sections = np.asarray(
                sections[start : start + SECTION_BATCH_SIZE], dtype=np.int64
            )
            selected = global_row_ids[
                np.isin(mapping.section_ids[global_row_ids], batch_sections)
            ]
            local_lookup = {
                int(section): local
                for local, section in enumerate(batch_sections.tolist())
            }
            local_sections = np.asarray(
                [local_lookup[int(mapping.section_ids[row])] for row in selected],
                dtype=np.int64,
            )
            output = model(
                torch.as_tensor(
                    np.asarray(prefix[batch_sections], dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(local_sections, dtype=torch.long, device=device),
                torch.as_tensor(
                    mapping.trace_token_ids[selected], dtype=torch.long, device=device
                ),
                torch.as_tensor(query[selected], dtype=torch.float32, device=device),
                return_hidden=return_hidden,
            )
            if return_hidden:
                values, representation = output
                representation_np = representation.detach().cpu().numpy()
            else:
                values = output
                representation_np = None
            values_np = values.detach().cpu().numpy()
            for local, row in enumerate(selected.tolist()):
                position = output_lookup[int(row)]
                predictions[position] = float(values_np[local])
                if hidden is not None and representation_np is not None:
                    hidden[position] = representation_np[local]
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("P20 prediction is incomplete")
    if hidden is not None and not np.all(np.isfinite(hidden)):
        raise RuntimeError("P20 hidden feature is incomplete")
    return predictions, hidden


def _train_model(
    *,
    prefix: np.ndarray,
    tail_state: Mapping[str, Any],
    mapping: WindowedMapping,
    fit_global_rows: np.ndarray,
    fit_query: np.ndarray,
    fit_target: np.ndarray,
    route: str,
    seed: int,
    device: str,
    max_updates: int,
    calibration_global_rows: np.ndarray | None = None,
    calibration_query: np.ndarray | None = None,
    calibration_target: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))
    model = _make_model(
        tail_state=tail_state,
        query_width=fit_query.shape[1],
        route=route,
        seed=seed,
    ).to(device)
    groups = _parameter_groups(model)
    snapshots = {name: _snapshot(parameters) for name, parameters in groups.items()}
    optimizer_groups = [
        {"params": groups["head"], "lr": HEAD_LR},
        {"params": groups["norm"], "lr": NORM_LR},
        {"params": groups["base_tail"], "lr": FULL_TAIL_LR},
    ]
    if groups["peft"]:
        optimizer_groups.insert(1, {"params": groups["peft"], "lr": PEFT_LR})
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=WEIGHT_DECAY,
    )
    target_mean = float(np.mean(fit_target))
    target_std = max(float(np.std(fit_target)), 1e-6)
    target_by_global = {
        int(row): float((target - target_mean) / target_std)
        for row, target in zip(fit_global_rows.tolist(), fit_target.tolist())
    }
    query_by_global = {
        int(row): np.asarray(value, dtype=np.float32)
        for row, value in zip(fit_global_rows.tolist(), fit_query)
    }
    if calibration_global_rows is not None and calibration_query is not None:
        query_by_global.update(
            {
                int(row): np.asarray(value, dtype=np.float32)
                for row, value in zip(
                    calibration_global_rows.tolist(), calibration_query
                )
            }
        )
    global_query = np.zeros(
        (len(mapping.requested_indices), fit_query.shape[1]), dtype=np.float32
    )
    for row, value in query_by_global.items():
        global_query[int(row)] = value

    best_state: dict[str, Any] | None = None
    best_update = int(max_updates)
    best_rmse = math.inf
    history: list[dict[str, Any]] = []
    gradient_history: list[dict[str, Any]] = []
    losses: list[float] = []
    tensor_shapes: dict[str, list[int]] | None = None
    completed_updates = 0

    batch_iterator: Iterable[tuple[np.ndarray, np.ndarray]] = ()
    iterator = iter(batch_iterator)
    for update in range(1, int(max_updates) + 1):
        phase = _set_phase(groups, route, update)
        try:
            batch_sections, selected = next(iterator)
        except StopIteration:
            iterator = iter(
                _batches(
                    row_ids=fit_global_rows,
                    mapping=mapping,
                    rng=rng,
                    section_batch_size=SECTION_BATCH_SIZE,
                )
            )
            batch_sections, selected = next(iterator)
        local_lookup = {
            int(section): local
            for local, section in enumerate(batch_sections.tolist())
        }
        local_sections = np.asarray(
            [local_lookup[int(mapping.section_ids[row])] for row in selected],
            dtype=np.int64,
        )
        optimizer.zero_grad(set_to_none=True)
        prefix_tensor = torch.as_tensor(
            np.asarray(prefix[batch_sections], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        query_tensor = torch.as_tensor(
            global_query[selected], dtype=torch.float32, device=device
        )
        prediction = model(
            prefix_tensor,
            torch.as_tensor(local_sections, dtype=torch.long, device=device),
            torch.as_tensor(
                mapping.trace_token_ids[selected], dtype=torch.long, device=device
            ),
            query_tensor,
        )
        target_tensor = torch.as_tensor(
            [target_by_global[int(row)] for row in selected.tolist()],
            dtype=torch.float32,
            device=device,
        )
        loss = torch.nn.functional.smooth_l1_loss(
            prediction, target_tensor, beta=0.25
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("P20 training loss became non-finite")
        loss.backward()
        gradient_row = {
            "update": int(update),
            "phase": phase,
            **{
                f"{name}_gradient_l2": _grad_norm(parameters)
                for name, parameters in groups.items()
            },
        }
        gradient_history.append(gradient_row)
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        completed_updates = update
        losses.append(float(loss.detach().cpu()))
        if tensor_shapes is None:
            tensor_shapes = {
                "prefix_batch": list(prefix_tensor.shape),
                "local_section_ids": list(local_sections.shape),
                "trace_token_ids": list(mapping.trace_token_ids[selected].shape),
                "query_batch": list(query_tensor.shape),
                "prediction": list(prediction.shape),
                "target": list(target_tensor.shape),
            }
        if (
            calibration_global_rows is not None
            and calibration_query is not None
            and calibration_target is not None
            and update in EVAL_UPDATES
        ):
            standardized, _ = _forward_rows(
                model=model,
                prefix=prefix,
                mapping=mapping,
                global_row_ids=calibration_global_rows,
                query=global_query,
                device=device,
                return_hidden=False,
            )
            absolute = target_mean + target_std * standardized
            metrics = _metrics(calibration_target, absolute)
            history.append(
                {
                    "update": int(update),
                    "phase": phase,
                    "calibration": metrics,
                }
            )
            minimum_update = 1
            if route in {"staged_adapter", "staged_lora_r4"}:
                minimum_update = 12
            if update >= minimum_update and metrics["rmse"] < best_rmse - 1e-10:
                best_rmse = float(metrics["rmse"])
                best_update = int(update)
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            model.train()
    if calibration_global_rows is not None:
        if best_state is None:
            raise RuntimeError("P20 calibration did not produce a checkpoint")
        model.load_state_dict(best_state, strict=True)
    else:
        best_update = completed_updates
    diagnostics = {
        "route": route,
        "seed": int(seed),
        "fit_rows": int(len(fit_global_rows)),
        "requested_max_updates": int(max_updates),
        "completed_updates": int(completed_updates),
        "selected_update": int(best_update),
        "target_mean": target_mean,
        "target_std": target_std,
        "tensor_shapes": tensor_shapes,
        "parameter_counts": {
            name: int(sum(parameter.numel() for parameter in parameters))
            for name, parameters in groups.items()
        },
        "loss": {
            "first": float(losses[0]),
            "last": float(losses[-1]),
            "minimum": float(np.min(losses)),
        },
        "calibration_history": history,
        "gradient_history": gradient_history,
        "parameter_update_l2": {
            name: _delta_l2(parameters, snapshots[name])
            for name, parameters in groups.items()
        },
        "optimizer": {
            "name": "AdamW",
            "head_lr": HEAD_LR,
            "peft_lr": PEFT_LR,
            "terminal_norm_lr": NORM_LR,
            "full_tail_lr": FULL_TAIL_LR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRAD_CLIP,
        },
    }
    return model, diagnostics


def _kernel_from_learned_features(
    *,
    fold: p17.FoldSamples,
    train_hidden: np.ndarray,
    validation_hidden: np.ndarray,
    baseline_validation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(train_hidden)
    train_scaled = scaler.transform(train_hidden)
    validation_scaled = scaler.transform(validation_hidden)
    components = min(PCA_COMPONENTS, len(train_hidden) - 1, train_hidden.shape[1])
    pca = PCA(n_components=components, random_state=SEED).fit(train_scaled)
    train_latent = pca.transform(train_scaled)
    validation_latent = pca.transform(validation_scaled)
    latent_std = np.std(train_latent, axis=0)
    latent_std[latent_std < 1e-8] = 1.0
    train_latent /= latent_std
    validation_latent /= latent_std
    coordinate_std = np.std(fold.train_raw_features[:, 3:6], axis=0)
    coordinate_std[coordinate_std < 1e-8] = 1.0
    seismic_std = np.std(fold.train_raw_features[:, 0:3], axis=0)
    seismic_std[seismic_std < 1e-8] = 1.0
    train_coordinate = fold.train_raw_features[:, 3:6] / coordinate_std
    validation_coordinate = fold.validation_raw_features[:, 3:6] / coordinate_std
    train_coordinate[:, 2] *= VERTICAL_WEIGHT
    validation_coordinate[:, 2] *= VERTICAL_WEIGHT
    train_metric = np.column_stack(
        [
            train_coordinate,
            SEISMIC_WEIGHT * fold.train_raw_features[:, 0:3] / seismic_std,
            LEARNED_WEIGHT * train_latent,
        ]
    )
    validation_metric = np.column_stack(
        [
            validation_coordinate,
            SEISMIC_WEIGHT
            * fold.validation_raw_features[:, 0:3]
            / seismic_std,
            LEARNED_WEIGHT * validation_latent,
        ]
    )
    neighbours = min(NEIGHBOURS, len(fold.train_target))
    model = NearestNeighbors(n_neighbors=neighbours, n_jobs=-1).fit(train_metric)
    distance, rows = model.kneighbors(validation_metric)
    weights = 1.0 / np.maximum(distance, 1e-8) ** DISTANCE_POWER
    kernel = np.sum(weights * fold.train_target[rows], axis=1) / np.sum(
        weights, axis=1
    )
    prediction = (1.0 - KERNEL_BLEND) * baseline_validation + KERNEL_BLEND * kernel
    return prediction, {
        "fit_rows": int(len(fold.train_target)),
        "pca_components": int(components),
        "pca_explained_variance_ratio_sum": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "vertical_weight": VERTICAL_WEIGHT,
        "learned_weight": LEARNED_WEIGHT,
        "seismic_weight": SEISMIC_WEIGHT,
        "neighbours": int(neighbours),
        "distance_power": DISTANCE_POWER,
        "kernel_blend": KERNEL_BLEND,
        "all_transforms_fit_on_outer_train_only": True,
        "validation_labels_used": False,
    }


def evaluate_route(
    *,
    route: str,
    folds: Sequence[p17.FoldSamples],
    oof: base.OOFDevelopment,
    prefix: np.ndarray,
    tail_state: Mapping[str, Any],
    mapping: WindowedMapping,
    requested_lookup: Mapping[tuple[int, int, int], int],
    device: str,
    seed: int,
    max_updates: int,
) -> tuple[dict[str, Any], np.ndarray]:
    prediction = np.full(len(oof.target), np.nan, dtype=np.float64)
    fold_audits: list[dict[str, Any]] = []
    for fold in folds:
        train_global = _row_ids(requested_lookup, fold.train_indices_kji)
        validation_global = _row_ids(requested_lookup, fold.validation_indices_kji)
        calibration_local = _farthest_calibration(
            fold.train_raw_features, CALIBRATION_ROWS
        )
        fit_local = np.setdiff1d(
            np.arange(len(fold.train_target), dtype=np.int64),
            calibration_local,
            assume_unique=True,
        )
        train_query_inner, _, query_audit_inner = _scaled_query(
            fold.train_raw_features,
            fold.validation_raw_features,
            mapping.time_positions[train_global],
            mapping.time_positions[validation_global],
            fit_local,
        )
        _, calibration_audit = _train_model(
            prefix=prefix,
            tail_state=tail_state,
            mapping=mapping,
            fit_global_rows=train_global[fit_local],
            fit_query=train_query_inner[fit_local],
            fit_target=fold.train_target[fit_local],
            route=route,
            seed=seed + 1000 * fold.fold_id,
            device=device,
            max_updates=max_updates,
            calibration_global_rows=train_global[calibration_local],
            calibration_query=train_query_inner[calibration_local],
            calibration_target=fold.train_target[calibration_local],
        )
        selected_updates = int(calibration_audit["selected_update"])
        all_local = np.arange(len(fold.train_target), dtype=np.int64)
        train_query, validation_query, query_audit_refit = _scaled_query(
            fold.train_raw_features,
            fold.validation_raw_features,
            mapping.time_positions[train_global],
            mapping.time_positions[validation_global],
            all_local,
        )
        model, refit_audit = _train_model(
            prefix=prefix,
            tail_state=tail_state,
            mapping=mapping,
            fit_global_rows=train_global,
            fit_query=train_query,
            fit_target=fold.train_target,
            route=route,
            seed=seed + 1000 * fold.fold_id,
            device=device,
            max_updates=selected_updates,
        )
        global_query = np.zeros(
            (len(mapping.requested_indices), train_query.shape[1]), dtype=np.float32
        )
        global_query[train_global] = train_query
        global_query[validation_global] = validation_query
        standardized_train, train_hidden = _forward_rows(
            model=model,
            prefix=prefix,
            mapping=mapping,
            global_row_ids=train_global,
            query=global_query,
            device=device,
            return_hidden=True,
        )
        standardized_validation, validation_hidden = _forward_rows(
            model=model,
            prefix=prefix,
            mapping=mapping,
            global_row_ids=validation_global,
            query=global_query,
            device=device,
            return_hidden=True,
        )
        if train_hidden is None or validation_hidden is None:
            raise RuntimeError("P20 hidden feature extraction failed")
        validation_mask = oof.fold_ids == fold.fold_id
        fold_prediction, kernel_audit = _kernel_from_learned_features(
            fold=fold,
            train_hidden=train_hidden,
            validation_hidden=validation_hidden,
            baseline_validation=oof.baseline[validation_mask],
        )
        prediction[validation_mask] = fold_prediction
        target_mean = float(refit_audit["target_mean"])
        target_std = float(refit_audit["target_std"])
        direct_validation = target_mean + target_std * standardized_validation
        fold_audits.append(
            {
                "fold_id": int(fold.fold_id),
                "train_labels": int(len(fold.train_target)),
                "inner_fit_labels": int(len(fit_local)),
                "inner_calibration_labels": int(len(calibration_local)),
                "calibration_selection_uses_coordinates_only": True,
                "calibration_indices_sha256": _array_sha256(calibration_local),
                "selected_updates": selected_updates,
                "query_inner": query_audit_inner,
                "query_refit": query_audit_refit,
                "calibration_training": calibration_audit,
                "refit_training": refit_audit,
                "kernel": kernel_audit,
                "direct_regression_validation": _metrics(
                    oof.target[validation_mask],
                    direct_validation,
                ),
                "train_standardized_prediction": {
                    "mean": float(np.mean(standardized_train)),
                    "std": float(np.std(standardized_train)),
                },
            }
        )
        del model
        if str(device).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"P20 route {route} produced incomplete OOF predictions")
    metrics = _metrics(oof.target, prediction)
    per_fold: list[dict[str, Any]] = []
    outcomes = {"win": 0, "loss": 0, "tie": 0}
    for fold_id in base.FOLD_IDS:
        mask = oof.fold_ids == fold_id
        baseline_metrics = _metrics(oof.target[mask], oof.baseline[mask])
        candidate_metrics = _metrics(oof.target[mask], prediction[mask])
        delta = float(candidate_metrics["rmse"]) - float(baseline_metrics["rmse"])
        outcome = "win" if delta < -1e-12 else "loss" if delta > 1e-12 else "tie"
        outcomes[outcome] += 1
        per_fold.append(
            {
                "fold_id": int(fold_id),
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "rmse_delta_candidate_minus_baseline": delta,
                "outcome": outcome,
            }
        )
    return (
        {
            "route": route,
            "seed": int(seed),
            "metrics": metrics,
            "per_fold": per_fold,
            "outcomes_vs_pykrige": outcomes,
            "fold_audits": fold_audits,
        },
        prediction,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    paths = [
        args.data_dir,
        args.stage3_root,
        args.segy_path,
        args.build_summary,
        args.seismic_index,
        args.well_tie,
        args.source_root,
        args.snapshot_path,
        args.output_dir,
        args.cache_dir,
    ]
    base.ensure_no_holdout_paths(paths)
    inputs = base.resolve_dev_inputs(args.data_dir)
    oof = base.load_oof_development(args.stage3_root)
    folds, fold_loading_audit = p17.load_fold_samples(
        stage3_root=args.stage3_root,
        train_h5=inputs.train_h5,
        oof=oof,
    )
    requested = p17._unique_indices(folds)  # noqa: SLF001
    mapping = build_windowed_mapping(
        train_h5=inputs.train_h5,
        requested_indices=requested,
        build_summary_path=args.build_summary,
        seismic_index_path=args.seismic_index,
        well_tie_path=args.well_tie,
    )
    segy_sha256 = _sha256(args.segy_path)
    images, native_audit = get_native_images(
        segy_path=args.segy_path,
        seismic_index_path=args.seismic_index,
        mapping=mapping,
        cache_path=args.cache_dir / "native_windows.npy",
        source_sha256=segy_sha256,
    )
    native_cache_sha256 = _sha256(args.cache_dir / "native_windows.npy")
    prefix, tail_state, prefix_audit = p15.get_prefix_cache(
        images=images,
        source_root=args.source_root,
        snapshot_path=args.snapshot_path,
        cache_dir=args.cache_dir / "gfm_prefix",
        device=args.device,
        image_batch_size=args.image_batch_size,
        weight_mode="pretrained",
        seed=SEED,
        native_cache_sha256=native_cache_sha256,
    )
    del images
    requested_lookup = _index_lookup(requested)
    route_results: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for route in args.routes:
        route_result, route_prediction = evaluate_route(
            route=route,
            folds=folds,
            oof=oof,
            prefix=prefix,
            tail_state=tail_state,
            mapping=mapping,
            requested_lookup=requested_lookup,
            device=args.device,
            seed=SEED,
            max_updates=args.max_updates,
        )
        route_results[route] = route_result
        predictions[route] = route_prediction
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"partial_{route}.json").write_text(
            json.dumps(route_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (args.output_dir / f"partial_{route}.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                indices_kji=oof.indices_kji,
                fold_ids=oof.fold_ids,
                target=oof.target,
                baseline_prediction=oof.baseline,
                candidate_prediction=route_prediction,
            )
    baseline_metrics = _metrics(oof.target, oof.baseline)
    best_route = min(
        route_results,
        key=lambda name: float(route_results[name]["metrics"]["rmse"]),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "outer_spatial_folds": list(base.FOLD_IDS),
            "train_labels_per_fold": 512,
            "validation_rows_per_fold": 2048,
            "inner_calibration_rows": CALIBRATION_ROWS,
            "maximum_updates": int(args.max_updates),
            "inner_calibration_selection": "coordinate-only farthest point",
            "primary_metric": "pooled development OOF RMSE",
            "holdout_opened": False,
            "test_h5_opened": False,
            "foundation_random_init_ablation": "deferred_by_user",
        },
        "pre_registered_primary_route": "staged_lora_r4",
        "routes": route_results,
        "baseline": {"pykrige_ok3d_repeat_0": baseline_metrics},
        "best_route_descriptive_only": best_route,
        "best_route_metrics": route_results[best_route]["metrics"],
        "best_delta_vs_pykrige": float(
            route_results[best_route]["metrics"]["rmse"]
        )
        - float(baseline_metrics["rmse"]),
        "decision": {
            "default_enabled": False,
            "promotion_requires_strict_p19_improvement": True,
            "p19_reference_rmse": 0.027751397627827728,
            "state": (
                "P20_STRICT_WINNER_CANDIDATE"
                if float(route_results[best_route]["metrics"]["rmse"])
                < 0.027751397627827728
                else "VERIFIED_NO_PROMOTION"
            ),
            "pretrained_contribution_claimed": False,
        },
        "inputs": {
            "train_h5_sha256": _sha256(inputs.train_h5),
            "segy_sha256": segy_sha256,
            "requested_indices_sha256": _array_sha256(requested),
        },
        "fold_loading_audit": fold_loading_audit,
        "window_mapping_audit": mapping.audit,
        "native_audit": native_audit,
        "prefix_audit": prefix_audit,
        "runtime": {
            "seconds": float(time.time() - started),
            "device": args.device,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            indices_kji=oof.indices_kji,
            fold_ids=oof.fold_ids,
            target=oof.target,
            baseline_prediction=oof.baseline,
            **{
                f"{route}_prediction": prediction
                for route, prediction in predictions.items()
            },
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--segy-path", type=Path, required=True)
    parser.add_argument("--build-summary", type=Path, required=True)
    parser.add_argument("--seismic-index", type=Path, required=True)
    parser.add_argument("--well-tie", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "_outputs" / "p20_peft_staged_unfreeze",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "_tmp" / "p20_peft_staged_unfreeze",
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    args = parser.parse_args()
    for name in (
        "data_dir",
        "stage3_root",
        "segy_path",
        "build_summary",
        "seismic_index",
        "well_tie",
        "source_root",
        "snapshot_path",
        "output_dir",
        "cache_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    outcome = run(parse_args())
    print(json.dumps(outcome, indent=2, sort_keys=True))
