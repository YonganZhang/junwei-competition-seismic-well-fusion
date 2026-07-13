#!/usr/bin/env python3
"""Build real GM09 lithofacies samples from Volve LAS, picks, and ST0202.

The split is frozen by mother-well family before resampling, windowing, or
normalization. Labels come only from explicit GM09/GENETIC FACIES intervals.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZipInfo

import lasio
import numpy as np
import openpyxl
import segyio

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _code.dataset_io import save_split  # noqa: E402
from _code.ml_framework.preprocess import (  # noqa: E402
    NormStats,
    denoise_identity,
    denormalize,
    fit_zscore,
    normalize,
)
from pipeline_contract import (  # noqa: E402
    CLASS_NAMES,
    FAMILY_PARTITIONS,
    LOG_ALIASES,
    LOG_CHANNELS,
    PIPELINE_VERSION,
    TARGET_CURVE_TYPE,
    TARGET_SOURCE,
    LabelInterval,
    assert_family_isolation,
    mother_family,
    normalize_well_id,
    partition_for_well,
    validate_class_name,
)

VOLVE_DIR = PROJECT_ROOT / "_sandbox" / "volve_data"
WELL_LOG_ZIP = VOLVE_DIR / "Volve_Well_logs.zip"
PICKS_PATH = (
    VOLVE_DIR
    / "_extracted_interp/Geophysical_Interpretations/Wells/Well_picks_Volve_v1.dat"
)
SEGY_PATH = (
    VOLVE_DIR
    / "_extracted_seismic/ST0202/Stacks/"
    "ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME.MIG_FIN.POST_STACK.3D.JS-017534.segy"
)
SEISMIC_INDEX_PATH = (
    PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/seismic_index.npz"
)
OUTPUT_DIR = TRACK_DIR / "_outputs"


def project_relative(path: Path) -> str:
    """Serialize project-owned paths without leaking a host/worktree prefix."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


@dataclass
class LogTrack:
    well_id: str
    members: list[str]
    channels: dict[str, tuple[np.ndarray, np.ndarray]]
    member_reports: list[dict[str, Any]]


@dataclass
class TieCurve:
    md: np.ndarray
    twt_ms: np.ndarray
    easting: np.ndarray
    northing: np.ndarray
    surfaces: list[str]


def parse_label_intervals(zip_path: Path = WELL_LOG_ZIP) -> list[LabelInterval]:
    """Read the eleven workbooks in memory and retain only fixed GM09 rows."""
    intervals: list[LabelInterval] = []
    workbook_count = 0
    with ZipFile(zip_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if "05.PETROPHYSICAL INTERPRETATION/" in name
            and name.lower().endswith("facies.xlsx")
        )
        for member in members:
            workbook_count += 1
            with archive.open(member) as stream:
                workbook = openpyxl.load_workbook(stream, read_only=True, data_only=True)
                sheet = workbook.active
                rows = list(sheet.iter_rows(values_only=True))
            expected_header = (
                "* Well UWI",
                "Common Well Name",
                "* Litho Crv Type",
                "* Source",
                "* Top Depth (meters)",
                "* Base Depth (meters)",
                "Litho Class",
            )
            if tuple(rows[0][:7]) != expected_header:
                raise ValueError(f"标签工作簿header变化: {member}: {rows[0][:7]}")
            for excel_row, row in enumerate(rows[1:], start=2):
                curve_type = str(row[2]).strip() if row[2] is not None else ""
                source = str(row[3]).strip() if row[3] is not None else ""
                if curve_type != TARGET_CURVE_TYPE or source != TARGET_SOURCE:
                    continue
                well_id = normalize_well_id(str(row[1]))
                class_name = str(row[6]).strip()
                class_id = validate_class_name(class_name)
                top = float(row[4])
                base = float(row[5])
                if not np.isfinite([top, base]).all() or base <= top:
                    raise ValueError(f"非法GM09区间 {member}:{excel_row}: {top}..{base}")
                family = mother_family(well_id)
                partition_for_well(well_id)
                intervals.append(
                    LabelInterval(
                        well_id=well_id,
                        family_id=family,
                        top_md_m=top,
                        base_md_m=base,
                        class_name=class_name,
                        class_id=class_id,
                        source_member=member,
                        source_row=excel_row,
                    )
                )
    observed = {interval.class_name for interval in intervals}
    wells = {interval.well_id for interval in intervals}
    if workbook_count != 11 or len(wells) != 11 or len(intervals) != 139:
        raise ValueError(
            "GM09权威基线变化: "
            f"workbooks={workbook_count}, wells={len(wells)}, intervals={len(intervals)}"
        )
    if observed != set(CLASS_NAMES):
        raise ValueError(f"GM09类别变化: observed={sorted(observed)} expected={list(CLASS_NAMES)}")
    return intervals


def _is_allowed_las(member: str) -> bool:
    """Whitelist only depth-domain raw/basic measurements, never interpretations."""
    upper = member.upper()
    basename = Path(member).name.upper()
    if not upper.endswith(".LAS"):
        return False
    if "/02.LWD_EWL/" in upper:
        if "_TIME_" in basename or "INTERPRETATION" in basename or "COMPUTED" in basename:
            return False
        return "_RAW_" in basename or "PREPLUS_BASIC_LOGS" in basename
    if "/04.COMPOSITE/" in upper:
        return True
    if "/06.LFP/" in upper:
        return True
    return False


def select_las_members(archive: ZipFile, well_id: str) -> list[ZipInfo]:
    archive_well = normalize_well_id(well_id).replace("/", "_", 1)
    prefixes = (
        f"Well_logs/02.LWD_EWL/{archive_well}/",
        f"Well_logs/04.COMPOSITE/{archive_well}/",
        f"Well_logs/06.LFP/{archive_well}/",
    )
    selected = [
        info
        for info in archive.infolist()
        if info.filename.startswith(prefixes) and _is_allowed_las(info.filename)
    ]
    # The SR track has a genuine composite LAS; do not use its LFP package.
    if any("/04.COMPOSITE/" in info.filename for info in selected):
        selected = [info for info in selected if "/06.LFP/" not in info.filename]
    return sorted(selected, key=lambda info: info.filename)


def _deduplicate_curve(depth: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(depth) & np.isfinite(values) & (np.abs(values) < 1e30)
    depth = np.asarray(depth[mask], dtype=np.float64)
    values = np.asarray(values[mask], dtype=np.float64)
    if depth.size == 0:
        return depth, values
    order = np.argsort(depth, kind="stable")
    depth = depth[order]
    values = values[order]
    rounded = np.round(depth, 4)
    unique, inverse = np.unique(rounded, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return unique.astype(np.float64), (sums / counts).astype(np.float64)


def load_log_track(archive: ZipFile, well_id: str) -> LogTrack:
    members = select_las_members(archive, well_id)
    chunks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    reports: list[dict[str, Any]] = []
    for info in members:
        report: dict[str, Any] = {
            "member": info.filename,
            "size_bytes": info.file_size,
            "status": "pending",
            "selected_curves": {},
        }
        try:
            with archive.open(info) as raw:
                text = TextIOWrapper(raw, encoding="latin1", errors="replace")
                las = lasio.read(text, engine="normal", ignore_header_errors=True)
            depth = np.asarray(las.index, dtype=np.float64)
            curve_lookup = {curve.mnemonic.upper(): curve.mnemonic for curve in las.curves[1:]}
            for channel in LOG_CHANNELS:
                selected_alias = next(
                    (alias for alias in LOG_ALIASES[channel] if alias in curve_lookup), None
                )
                if selected_alias is None:
                    continue
                values = np.asarray(las[curve_lookup[selected_alias]], dtype=np.float64)
                clean_depth, clean_values = _deduplicate_curve(depth, values)
                if clean_depth.size:
                    chunks[channel].append((clean_depth, clean_values))
                    report["selected_curves"][channel] = {
                        "mnemonic": curve_lookup[selected_alias],
                        "points": int(clean_depth.size),
                        "depth_range_m": [float(clean_depth[0]), float(clean_depth[-1])],
                    }
            report["status"] = "loaded"
            report["index_points"] = int(depth.size)
        except Exception as exc:  # fail is retained in manifest; other members may still be valid
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
        reports.append(report)

    channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for channel, channel_chunks in chunks.items():
        depth = np.concatenate([item[0] for item in channel_chunks])
        values = np.concatenate([item[1] for item in channel_chunks])
        channels[channel] = _deduplicate_curve(depth, values)
    return LogTrack(
        well_id=normalize_well_id(well_id),
        members=[info.filename for info in members],
        channels=channels,
        member_reports=reports,
    )


def parse_official_picks(path: Path = PICKS_PATH) -> dict[str, TieCurve]:
    """Parse official MD/TWT/XY picks and retain only fully observed points."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    spans: list[tuple[int, int]] | None = None
    points: dict[str, list[tuple[float, float, float, float, str]]] = defaultdict(list)
    for line in lines:
        if re.match(r"^\s*-{5,}", line):
            spans = [(match.start(), match.end()) for match in re.finditer(r"-+", line)]
            continue
        if spans is None or not line.strip() or line.startswith("Well NO"):
            continue
        columns = [line[start:end].strip() for start, end in spans]
        if len(columns) < 12:
            continue
        well, surface, _, _, md, _, _, twt, _, _, easting, northing = columns[:12]
        if not well or not md or not twt or not easting or not northing:
            continue
        try:
            point = (float(md), float(twt), float(easting), float(northing), surface)
        except ValueError:
            continue
        points[normalize_well_id(well)].append(point)

    curves: dict[str, TieCurve] = {}
    for well_id, well_points in points.items():
        well_points.sort(key=lambda item: item[0])
        deduplicated: list[tuple[float, float, float, float, str]] = []
        seen_md: set[float] = set()
        for point in well_points:
            if point[0] in seen_md:
                continue
            seen_md.add(point[0])
            deduplicated.append(point)
        if len(deduplicated) < 2:
            continue
        curves[well_id] = TieCurve(
            md=np.asarray([item[0] for item in deduplicated], dtype=np.float64),
            twt_ms=np.asarray([item[1] for item in deduplicated], dtype=np.float64),
            easting=np.asarray([item[2] for item in deduplicated], dtype=np.float64),
            northing=np.asarray([item[3] for item in deduplicated], dtype=np.float64),
            surfaces=[item[4] for item in deduplicated],
        )
    return curves


class SeismicReader:
    """Read true ST0202 traces lazily; cache only traces actually used by samples."""

    def __init__(self, segy_path: Path = SEGY_PATH, index_path: Path = SEISMIC_INDEX_PATH):
        if not segy_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"缺少ST0202或索引: {segy_path}, {index_path}")
        index = np.load(index_path)
        self.index = {key: index[key] for key in index.files}
        self.samples_ms = np.asarray(self.index["samples_ms"], dtype=np.float64)
        self.handle = segyio.open(str(segy_path), "r", ignore_geometry=True)
        self.cache: dict[tuple[int, int], np.ndarray] = {}

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "SeismicReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def utm_to_il_xl(self, easting: float, northing: float) -> tuple[int, int]:
        affine = np.asarray(self.index["affine_il_xl_to_xy"], dtype=np.float64)
        matrix = affine[:, :2]
        rhs = np.array([easting - affine[0, 2], northing - affine[1, 2]])
        inline_float, crossline_float = np.linalg.solve(matrix, rhs)
        il_min, il_max = int(self.index["il_min"]), int(self.index["il_max"])
        xl_min, xl_max = int(self.index["xl_min"]), int(self.index["xl_max"])
        if not (il_min - 0.5 <= inline_float <= il_max + 0.5):
            raise ValueError(f"官方XY反解inline越界: {inline_float}")
        if not (xl_min - 0.5 <= crossline_float <= xl_max + 0.5):
            raise ValueError(f"官方XY反解crossline越界: {crossline_float}")
        return int(round(inline_float)), int(round(crossline_float))

    def trace(self, inline: int, crossline: int) -> np.ndarray:
        key = (inline, crossline)
        if key not in self.cache:
            il_min, xl_min = int(self.index["il_min"]), int(self.index["xl_min"])
            n_xl = int(self.index["n_xl"])
            trace_index = (inline - il_min) * n_xl + (crossline - xl_min)
            self.cache[key] = np.asarray(self.handle.trace[trace_index], dtype=np.float32)
        return self.cache[key]

    def patch(
        self,
        inline: int,
        crossline: int,
        twt_ms: float,
        spatial_size: int,
        time_samples: int,
    ) -> np.ndarray:
        if spatial_size % 2 != 1 or time_samples % 2 != 1:
            raise ValueError("地震空间和时间窗口都必须是奇数")
        il_min, il_max = int(self.index["il_min"]), int(self.index["il_max"])
        xl_min, xl_max = int(self.index["xl_min"]), int(self.index["xl_max"])
        radius = spatial_size // 2
        if not (
            il_min + radius <= inline <= il_max - radius
            and xl_min + radius <= crossline <= xl_max - radius
        ):
            raise ValueError("井旁地震空间patch越过ST0202边界")
        time_index = int(np.argmin(np.abs(self.samples_ms - twt_ms)))
        half_time = time_samples // 2
        if time_index - half_time < 0 or time_index + half_time >= len(self.samples_ms):
            raise ValueError("井旁地震时间patch越过ST0202边界")
        patch = np.empty((spatial_size, spatial_size, time_samples), dtype=np.float32)
        for row, il in enumerate(range(inline - radius, inline + radius + 1)):
            for col, xl in enumerate(range(crossline - radius, crossline + radius + 1)):
                trace = self.trace(il, xl)
                patch[row, col] = trace[
                    time_index - half_time : time_index + half_time + 1
                ]
        if not np.isfinite(patch).all():
            raise ValueError("真实地震patch含NaN/Inf")
        return patch


def interval_centers(interval: LabelInterval, sample_step_m: float) -> np.ndarray:
    length = interval.base_md_m - interval.top_md_m
    if length <= sample_step_m:
        return np.array([(interval.top_md_m + interval.base_md_m) / 2], dtype=np.float64)
    centers = np.arange(
        interval.top_md_m + sample_step_m / 2,
        interval.base_md_m,
        sample_step_m,
        dtype=np.float64,
    )
    return centers if centers.size else np.array([(interval.top_md_m + interval.base_md_m) / 2])


def interpolate_log_window(
    track: LogTrack,
    center_md_m: float,
    offsets_m: np.ndarray,
    max_nearest_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    query = center_md_m + offsets_m
    values = np.full((len(LOG_CHANNELS), len(query)), np.nan, dtype=np.float32)
    mask = np.zeros_like(values, dtype=np.float32)
    for channel_index, channel in enumerate(LOG_CHANNELS):
        if channel not in track.channels:
            continue
        depth, curve = track.channels[channel]
        if depth.size < 2:
            continue
        insertion = np.searchsorted(depth, query)
        left_index = np.clip(insertion - 1, 0, len(depth) - 1)
        right_index = np.clip(insertion, 0, len(depth) - 1)
        nearest = np.minimum(np.abs(query - depth[left_index]), np.abs(query - depth[right_index]))
        valid = (
            (query >= depth[0])
            & (query <= depth[-1])
            & (nearest <= max_nearest_distance_m)
        )
        interpolated = np.interp(query, depth, curve)
        values[channel_index, valid] = interpolated[valid].astype(np.float32)
        mask[channel_index, valid] = 1.0
    return values, mask


def _fit_normalization(raw_samples: list[dict[str, Any]]) -> dict[str, Any]:
    training = [sample for sample in raw_samples if sample["partition"] == "train"]
    if not training:
        raise ValueError("严格井族划分后没有训练样本")
    log_stats: dict[str, NormStats] = {}
    for channel_index, channel in enumerate(LOG_CHANNELS):
        observed = [
            sample["raw_log"][channel_index][sample["log_mask"][channel_index] > 0]
            for sample in training
        ]
        values = np.concatenate([array for array in observed if array.size])
        if values.size < 2:
            raise ValueError(f"训练井在原始通道 {channel} 上没有足够观测值")
        log_stats[channel] = fit_zscore(denoise_identity(values.astype(np.float64)))
    seismic_values = np.concatenate(
        [denoise_identity(sample["raw_seismic"].astype(np.float64)).reshape(-1) for sample in training]
    )
    seismic_stats = fit_zscore(seismic_values)
    return {"logs": log_stats, "seismic": seismic_stats}


def _normalize_samples(
    raw_samples: list[dict[str, Any]], stats: dict[str, Any]
) -> tuple[list[dict[str, Any]], float]:
    samples: list[dict[str, Any]] = []
    max_round_trip_error = 0.0
    stats_json = {
        "logs": {name: value.to_dict() for name, value in stats["logs"].items()},
        "seismic": stats["seismic"].to_dict(),
    }
    for raw in raw_samples:
        normalized_log = np.zeros_like(raw["raw_log"], dtype=np.float32)
        for channel_index, channel in enumerate(LOG_CHANNELS):
            observed = raw["log_mask"][channel_index] > 0
            if not observed.any():
                continue
            physical = denoise_identity(raw["raw_log"][channel_index, observed].astype(np.float64))
            normalized = normalize(physical, stats["logs"][channel])
            reconstructed = denormalize(normalized, stats["logs"][channel])
            max_round_trip_error = max(
                max_round_trip_error,
                float(np.max(np.abs(reconstructed - physical))),
            )
            normalized_log[channel_index, observed] = normalized.astype(np.float32)
        seismic_physical = denoise_identity(raw["raw_seismic"].astype(np.float64))
        seismic_normalized = normalize(seismic_physical, stats["seismic"])
        seismic_reconstructed = denormalize(seismic_normalized, stats["seismic"])
        max_round_trip_error = max(
            max_round_trip_error,
            float(np.max(np.abs(seismic_reconstructed - seismic_physical))),
        )
        well_log_seq = np.concatenate((normalized_log, raw["log_mask"]), axis=0)
        samples.append(
            {
                "seismic_patch": seismic_normalized.astype(np.float32),
                "well_log_seq": well_log_seq.astype(np.float32),
                "position": {
                    "inline": raw["inline"],
                    "crossline": raw["crossline"],
                    "time_ms": raw["twt_ms"],
                    "well_name": raw["well_id"],
                    # Preserve the actual sampling center for P4 depth tracks.
                    # The interval midpoint is not an acceptable reconstruction.
                    "center_md_m": raw["center_md_m"],
                },
                "label": np.int64(raw["class_id"]),
                "meta": {
                    "pipeline_version": PIPELINE_VERSION,
                    "partition": raw["partition"],
                    "family_id": raw["family_id"],
                    "normalization_fit_scope": "train_mother_well_families_only",
                    "normalization_stats": stats_json,
                    "denoise": "denoise_identity",
                    "log_channels": list(LOG_CHANNELS),
                    "well_log_layout": "first C normalized values, next C observed masks",
                    "label_trace": {
                        "archive": "_sandbox/volve_data/Volve_Well_logs.zip",
                        "member": raw["label_member"],
                        "excel_row": raw["label_row"],
                        "source": TARGET_SOURCE,
                        "curve_type": TARGET_CURVE_TYPE,
                        "field": "Litho Class",
                        "class_name": raw["class_name"],
                        "top_md_m": raw["top_md_m"],
                        "base_md_m": raw["base_md_m"],
                    },
                    "weak_tie": {
                        "source": "Geophysical_Interpretations/Wells/Well_picks_Volve_v1.dat",
                        "method": "bracketed linear MD-to-TWT/XY; no extrapolation",
                    },
                    "raw_log_provenance": raw["well_id"],
                },
            }
        )
    if not all(
        np.isfinite(sample["seismic_patch"]).all()
        and np.isfinite(sample["well_log_seq"]).all()
        for sample in samples
    ):
        raise ValueError("归一化后样本含NaN/Inf")
    return samples, max_round_trip_error


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    intervals = parse_label_intervals()
    intervals_by_well: dict[str, list[LabelInterval]] = defaultdict(list)
    for interval in intervals:
        intervals_by_well[interval.well_id].append(interval)
    picks = parse_official_picks()
    offsets = (
        np.arange(args.log_window_samples, dtype=np.float64)
        - args.log_window_samples // 2
    ) * args.log_step_m
    raw_samples: list[dict[str, Any]] = []
    per_well: dict[str, dict[str, Any]] = {}

    with ZipFile(WELL_LOG_ZIP) as archive, SeismicReader() as seismic:
        for well_id in sorted(intervals_by_well):
            family_id = mother_family(well_id)
            partition = partition_for_well(well_id)  # frozen before all sample operations
            track = load_log_track(archive, well_id)
            report: dict[str, Any] = {
                "well_id": well_id,
                "family_id": family_id,
                "partition": partition,
                "label_intervals": len(intervals_by_well[well_id]),
                "requested_centers": 0,
                "kept_samples": 0,
                "dropped": Counter(),
                "selected_las_members": track.members,
                "las_member_reports": track.member_reports,
                "channel_points": {
                    channel: int(track.channels[channel][0].size)
                    if channel in track.channels
                    else 0
                    for channel in LOG_CHANNELS
                },
            }
            tie = picks.get(well_id)
            if tie is None:
                report["status"] = "excluded_no_official_tie"
                per_well[well_id] = report
                continue
            report["official_tie"] = {
                "points": int(tie.md.size),
                "md_range_m": [float(tie.md[0]), float(tie.md[-1])],
                "surfaces": tie.surfaces,
            }
            if not track.channels:
                report["status"] = "excluded_no_allowed_log_channels"
                per_well[well_id] = report
                continue

            for interval in intervals_by_well[well_id]:
                for center_md_m in interval_centers(interval, args.sample_step_m):
                    report["requested_centers"] += 1
                    if not (tie.md[0] <= center_md_m <= tie.md[-1]):
                        report["dropped"]["outside_bracketed_official_picks"] += 1
                        continue
                    raw_log, log_mask = interpolate_log_window(
                        track,
                        float(center_md_m),
                        offsets,
                        args.max_log_distance_m,
                    )
                    if int(log_mask[:, args.log_window_samples // 2].sum()) < 1:
                        report["dropped"]["no_log_at_label_center"] += 1
                        continue
                    if float(log_mask.mean()) < args.min_log_mask_fraction:
                        report["dropped"]["insufficient_log_window_coverage"] += 1
                        continue
                    twt_ms = float(np.interp(center_md_m, tie.md, tie.twt_ms))
                    easting = float(np.interp(center_md_m, tie.md, tie.easting))
                    northing = float(np.interp(center_md_m, tie.md, tie.northing))
                    try:
                        inline, crossline = seismic.utm_to_il_xl(easting, northing)
                        seismic_patch = seismic.patch(
                            inline,
                            crossline,
                            twt_ms,
                            args.seismic_spatial_size,
                            args.seismic_time_samples,
                        )
                    except ValueError as exc:
                        report["dropped"][f"seismic_alignment: {exc}"] += 1
                        continue
                    raw_samples.append(
                        {
                            "well_id": well_id,
                            "family_id": family_id,
                            "partition": partition,
                            "raw_log": raw_log,
                            "log_mask": log_mask,
                            "raw_seismic": seismic_patch,
                            "center_md_m": float(center_md_m),
                            "inline": inline,
                            "crossline": crossline,
                            "twt_ms": twt_ms,
                            "class_id": interval.class_id,
                            "class_name": interval.class_name,
                            "label_member": interval.source_member,
                            "label_row": interval.source_row,
                            "top_md_m": interval.top_md_m,
                            "base_md_m": interval.base_md_m,
                        }
                    )
                    report["kept_samples"] += 1
            report["dropped"] = dict(report["dropped"])
            report["status"] = "usable" if report["kept_samples"] else "excluded_no_joint_samples"
            per_well[well_id] = report

    isolation = assert_family_isolation(raw_samples)
    stats = _fit_normalization(raw_samples)
    samples, max_round_trip_error = _normalize_samples(raw_samples, stats)
    train_container = [
        sample for sample in samples if sample["meta"]["partition"] in ("train", "guard")
    ]
    test_samples = [sample for sample in samples if sample["meta"]["partition"] == "test"]
    train_path = save_split("lithofacies", "train", train_container)
    test_path = save_split("lithofacies", "test", test_samples)

    sample_counts = Counter(sample["meta"]["partition"] for sample in samples)
    class_counts = {
        partition: dict(
            Counter(
                CLASS_NAMES[int(sample["label"])]
                for sample in samples
                if sample["meta"]["partition"] == partition
            )
        )
        for partition in ("train", "guard", "test")
    }
    normalization_json = {
        "fit_scope": "train_mother_well_families_only",
        "denoise": "denoise_identity",
        "logs": {name: value.to_dict() for name, value in stats["logs"].items()},
        "seismic": stats["seismic"].to_dict(),
        "max_round_trip_error": max_round_trip_error,
    }
    manifest: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "label_contract": {
            "archive": "_sandbox/volve_data/Volve_Well_logs.zip",
            "workbooks": 11,
            "wells": 11,
            "intervals": 139,
            "source": TARGET_SOURCE,
            "curve_type": TARGET_CURVE_TYPE,
            "field": "Litho Class",
            "class_names": list(CLASS_NAMES),
            "excluded": ["LITH", "UNKNOWN", "UNDEFINED", "out-of-interval"],
        },
        "split_contract": {
            "unit": "mother well family including sidetracks",
            "assignment_before": ["resampling", "windowing", "normalization"],
            "frozen_family_partitions": FAMILY_PARTITIONS,
            "usable_families": isolation,
            "guard_storage": "guard samples are stored in lithofacies/train with meta.partition=guard; optimizer filters them out",
        },
        "input_contract": {
            "log_channels": list(LOG_CHANNELS),
            "leakage_policy": "strict observed/basic measurement whitelist; no facies/petrophysical target derivatives",
            "log_window_samples": args.log_window_samples,
            "log_step_m": args.log_step_m,
            "sample_step_m": args.sample_step_m,
            "max_log_distance_m": args.max_log_distance_m,
            "min_log_mask_fraction": args.min_log_mask_fraction,
            "seismic_source": "ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME",
            "seismic_shape": [
                args.seismic_spatial_size,
                args.seismic_spatial_size,
                args.seismic_time_samples,
            ],
            "weak_tie": "official MD/TWT/XY picks, bracketed interpolation only, no extrapolation",
        },
        "sample_counts": dict(sample_counts),
        "class_counts": class_counts,
        "per_well": per_well,
        "normalization": normalization_json,
        "outputs": {
            "train": project_relative(train_path),
            "test": project_relative(test_path),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "normalization_stats.json").write_text(
        json.dumps(normalization_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-step-m", type=float, default=2.0)
    parser.add_argument("--log-window-samples", type=int, default=33)
    parser.add_argument("--log-step-m", type=float, default=0.5)
    parser.add_argument("--max-log-distance-m", type=float, default=0.6)
    parser.add_argument("--min-log-mask-fraction", type=float, default=0.05)
    parser.add_argument("--seismic-spatial-size", type=int, default=3)
    parser.add_argument("--seismic-time-samples", type=int, default=33)
    args = parser.parse_args()
    if args.log_window_samples % 2 != 1 or args.log_window_samples < 3:
        parser.error("--log-window-samples must be an odd integer >=3")
    if args.sample_step_m <= 0 or args.log_step_m <= 0 or args.max_log_distance_m <= 0:
        parser.error("distance/step arguments must be positive")
    if not 0 < args.min_log_mask_fraction <= 1:
        parser.error("--min-log-mask-fraction must be in (0,1]")
    return args


if __name__ == "__main__":
    result = build_dataset(parse_args())
    print(json.dumps({
        "sample_counts": result["sample_counts"],
        "usable_families": result["split_contract"]["usable_families"],
        "outputs": result["outputs"],
    }, ensure_ascii=False, indent=2))
