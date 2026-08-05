#!/usr/bin/env python3
"""P40 attributable well/seismic dual-foundation lithofacies pilot.

The command surface is intentionally split by interpreter:

* ``phase0`` uses the system geoscience environment to reconstruct and audit
  the 447 development-only alignments against native ST0202;
* ``baseline-worker`` uses a local XGBoost environment;
* ``extract-features`` and ``run`` use the pinned foundation environment;
* ``--verify-only`` reads only the portable committed evidence.

No command accepts a test/known-holdout path.  All learned preprocessing,
PCA, head fitting and model selection are fitted inside the active train
families.  The one R0 seed is 2693; outer LOGO4 and inner LOGO3 are fixed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for _root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from lithofacies_p5_stage3 import load_stage3_batch  # noqa: E402
from p4_contract import classification_metrics_from_logits  # noqa: E402
from pipeline_contract import normalize_well_id  # noqa: E402


SCHEMA = "lithofacies-p40-crossmodal-foundation-pilot/v1"
SPLIT_HASH = "a06375429f9e9cf380fb5cdebd7d0cb7b25d7a13d29522b8e2420f4dae1b4555"
DEVELOPMENT_BATCH_SHA256 = (
    "b6817ae218dee12bc72a1551e76423a61eb4ff11faf70dcb0444b01ba422f51c"
)
P4_MANIFEST_SHA256 = (
    "c895e411b10df6b57d4e76713cbf9bd1e96f4c14c22d1e9d13e8917285b1a785"
)
EXPECTED_BASELINE_MEAN = 0.2133487970485067
LOCKED_BASELINE_SUMMARY_SHA256 = (
    "025e189ee2e6c097193be91dee56e348558fb01b0a5b019fe22581433ea9c50b"
)
BASELINE_CONFIG = {"max_depth": 3, "eta": 0.1, "rounds": 60}
SEED = 2693
NUM_CLASSES = 9
PCA_DIM = 24
HEAD_UPDATES = 120
RESIDUAL_UPDATES = 160
DEFAULT_OUTPUT = TRACK_DIR / "_outputs" / "p40_crossmodal_foundation_pilot"
FORBIDDEN_MARKERS = ("known_holdout", "frozen_holdout", "test.h5", "/test/")
VARIANTS = ("B0", "B2", "B4", "F1", "A5", "A6", "A7")
FAMILIES = ("15/9-19", "15/9-F-14", "15/9-F-15", "15/9-F-4")
CLASS_NAMES = (
    "F-MARSH",
    "F-MOUTHBAR",
    "F-OFFSHORE",
    "F-LOWER SHOREFACE",
    "F-UPPER SHOREFACE",
    "F-TIDAL BAR",
    "F-TIDAL CHANNEL",
    "F-TIDAL FLAT MUDDY",
    "F-TIDAL FLAT SANDY",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        text = str(Path(path).expanduser().resolve()).lower().replace("\\", "/")
        if any(marker in text for marker in FORBIDDEN_MARKERS):
            raise ValueError(f"P40 forbids test/frozen-holdout path: {path}")


def _output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(TRACK_DIR.resolve())
    except ValueError as exc:
        raise ValueError("P40 output must remain below the lithofacies track") from exc
    return resolved


def _canonical_rows(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        manifest.get("split_hash") != SPLIT_HASH
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("test_metrics_used") is not False
        or manifest.get("loaded_files") != ["train.h5"]
    ):
        raise RuntimeError("P40 development asset violates the frozen firewall")
    records: dict[str, dict[str, Any]] = {}
    for fold in range(4):
        prefix = f"f{fold}_validation"
        for row, sample_id in enumerate(arrays[f"{prefix}_ids"].tolist()):
            if sample_id in records:
                raise RuntimeError(f"sample appears in two outer-held folds: {sample_id}")
            records[sample_id] = {
                "sample_id": str(sample_id),
                "well": str(arrays[f"{prefix}_well_id"][row]),
                "family": str(arrays[f"{prefix}_family_id"][row]),
                "twt_ms": float(arrays[f"{prefix}_twt_ms"][row]),
                "label": int(arrays[f"{prefix}_labels"][row]),
                "outer_fold": fold,
            }
    rows = [records[key] for key in sorted(records)]
    wells = sorted({row["well"] for row in rows})
    families = sorted({row["family"] for row in rows})
    if len(rows) != 447 or len(wells) != 9 or tuple(families) != FAMILIES:
        raise RuntimeError(
            f"P40 expected 447 rows/9 wells/4 families, got {len(rows)}/{len(wells)}/{families}"
        )
    return rows, {"samples": len(rows), "wells": wells, "families": families}


def _recover_physical_inputs(
    arrays: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    row_by_id = {row["sample_id"]: index for index, row in enumerate(rows)}
    physical_logs = np.full((len(rows), 13, 33), np.nan, dtype=np.float32)
    masks = np.zeros((len(rows), 13, 33), dtype=np.uint8)
    raw_seismic = np.full((len(rows), 3, 3, 33), np.nan, dtype=np.float32)
    log_replay_error = 0.0
    seismic_replay_error = 0.0
    seen_seismic = np.zeros(len(rows), dtype=bool)
    for fold in range(4):
        preprocessor = json.loads(str(arrays[f"f{fold}_preprocessor"].item()))
        log_stats = preprocessor["log_stats"]
        seismic_stats = preprocessor["seismic_stats"]
        for split in ("train", "validation"):
            prefix = f"f{fold}_{split}"
            for position, sample_id in enumerate(arrays[f"{prefix}_ids"].tolist()):
                target = row_by_id[str(sample_id)]
                well = np.asarray(arrays[f"{prefix}_well"][position], dtype=np.float32)
                observed = well[13:] > 0.5
                for channel, stats in enumerate(log_stats):
                    if stats is None or not observed[channel].any():
                        continue
                    reconstructed = (
                        well[channel, observed[channel]] * float(stats["std"])
                        + float(stats["mean"])
                    ).astype(np.float32)
                    already = masks[target, channel, observed[channel]] > 0
                    if already.any():
                        prior = physical_logs[target, channel, observed[channel]][already]
                        log_replay_error = max(
                            log_replay_error,
                            float(np.max(np.abs(prior - reconstructed[already]))),
                        )
                    values = physical_logs[target, channel]
                    values[observed[channel]] = reconstructed
                    physical_logs[target, channel] = values
                    masks[target, channel, observed[channel]] = 1
                seismic = (
                    np.asarray(arrays[f"{prefix}_seismic"][position], dtype=np.float32)
                    * float(seismic_stats["std"])
                    + float(seismic_stats["mean"])
                ).astype(np.float32)
                if seen_seismic[target]:
                    seismic_replay_error = max(
                        seismic_replay_error,
                        float(np.max(np.abs(raw_seismic[target] - seismic))),
                    )
                else:
                    raw_seismic[target] = seismic
                    seen_seismic[target] = True
    physical_logs[~masks.astype(bool)] = 0.0
    if not np.isfinite(physical_logs).all() or not np.isfinite(raw_seismic).all():
        raise RuntimeError("P40 could not invert every development input")
    return physical_logs, masks, raw_seismic, {
        "method": "invert every fold-train fitted z-score; cross-occurrence replay",
        "maximum_log_replay_error": log_replay_error,
        "maximum_seismic_replay_error": seismic_replay_error,
        "target_statistics_used": False,
    }


def _parse_alignment_curves(picks_path: Path, wells: set[str]) -> dict[str, np.ndarray]:
    spans: list[tuple[int, int]] | None = None
    points: dict[str, list[tuple[float, float, float, float, float, float]]] = defaultdict(list)
    for line in Path(picks_path).read_text(encoding="utf-8", errors="replace").splitlines(
        keepends=True
    ):
        if re.match(r"^\s*-{5,}", line):
            spans = [(match.start(), match.end()) for match in re.finditer(r"-+", line)]
            continue
        if spans is None or not line.strip() or line.startswith("Well NO"):
            continue
        columns = [line[start:stop].strip() for start, stop in spans]
        if len(columns) < 12:
            continue
        well = normalize_well_id(columns[0])
        if well not in wells:
            continue
        try:
            md, tvd, tvdss, twt, easting, northing = (
                float(columns[index]) for index in (4, 5, 6, 7, 10, 11)
            )
        except ValueError:
            continue
        points[well].append((md, tvd, tvdss, twt, easting, northing))
    result: dict[str, np.ndarray] = {}
    for well in sorted(wells):
        values = np.asarray(points[well], dtype=np.float64)
        if len(values) < 2:
            raise RuntimeError(f"P40 alignment has fewer than two official points: {well}")
        values = values[np.argsort(values[:, 3])]
        _, unique = np.unique(values[:, 3], return_index=True)
        values = values[np.sort(unique)]
        if len(values) < 2 or np.any(np.diff(values[:, 3]) <= 0):
            raise RuntimeError(f"P40 TWT tie is not invertible: {well}")
        result[well] = values
    return result


def _window_start(low: int, high: int, width: int, minimum: int, maximum: int) -> int:
    if high - low + 1 > width:
        raise RuntimeError(f"native window cannot cover [{low},{high}] in width {width}")
    start = low - (width - (high - low + 1)) // 2
    return max(minimum, min(start, maximum - width + 1))


def _alignment_and_sections(
    *,
    rows: Sequence[Mapping[str, Any]],
    raw_seismic: np.ndarray,
    picks_path: Path,
    seismic_index_path: Path,
    segy_path: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import segyio

    index_archive = np.load(seismic_index_path, allow_pickle=False)
    index = {key: index_archive[key] for key in index_archive.files}
    curves = _parse_alignment_curves(picks_path, {str(row["well"]) for row in rows})
    affine = np.asarray(index["affine_il_xl_to_xy"], dtype=np.float64)
    inverse = np.linalg.inv(affine[:, :2])
    samples_ms = np.asarray(index["samples_ms"], dtype=np.float64)
    aligned: list[dict[str, Any]] = []
    for row in rows:
        well = str(row["well"])
        twt = float(row["twt_ms"])
        curve = curves[well]
        if twt < curve[:, 3].min() or twt > curve[:, 3].max():
            raise RuntimeError(f"P40 TWT outside official bracket: {row['sample_id']}")
        md = float(np.interp(twt, curve[:, 3], curve[:, 0]))
        tvd = float(np.interp(md, curve[:, 0], curve[:, 1]))
        tvdss = float(np.interp(md, curve[:, 0], curve[:, 2]))
        easting = float(np.interp(md, curve[:, 0], curve[:, 4]))
        northing = float(np.interp(md, curve[:, 0], curve[:, 5]))
        fractional = inverse @ np.asarray(
            [easting - affine[0, 2], northing - affine[1, 2]], dtype=np.float64
        )
        inline, crossline = np.rint(fractional).astype(np.int32).tolist()
        time_index = int(np.argmin(np.abs(samples_ms - twt)))
        replay_twt = float(np.interp(md, curve[:, 0], curve[:, 3]))
        aligned.append(
            {
                **dict(row),
                "md_m": md,
                "tvd_m": tvd,
                "tvdss_m": tvdss,
                "twt_ms": twt,
                "easting_m": easting,
                "northing_m": northing,
                "inline": int(inline),
                "crossline": int(crossline),
                "time_index": time_index,
                "twt_roundtrip_error_ms": abs(replay_twt - twt),
            }
        )
    il_min, il_max = int(index["il_min"]), int(index["il_max"])
    xl_min, xl_max = int(index["xl_min"]), int(index["xl_max"])
    per_well: dict[str, dict[str, int]] = {}
    for well in sorted({row["well"] for row in aligned}):
        subset = [row for row in aligned if row["well"] == well]
        per_well[well] = {
            "crossline_start": _window_start(
                min(row["crossline"] for row in subset),
                max(row["crossline"] for row in subset),
                160,
                xl_min,
                xl_max,
            ),
            "time_start": _window_start(
                min(row["time_index"] for row in subset),
                max(row["time_index"] for row in subset),
                400,
                0,
                len(samples_ms) - 1,
            ),
        }
    section_keys = sorted({(str(row["well"]), int(row["inline"])) for row in aligned})
    section_lookup = {key: index for index, key in enumerate(section_keys)}
    sections = np.empty((len(section_keys), 400, 160), dtype=np.float32)
    section_ids = np.empty(len(aligned), dtype=np.int32)
    trace_ids = np.empty(len(aligned), dtype=np.int32)
    patch_error = 0.0
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as handle:
        if handle.tracecount != int(index["n_traces"]):
            raise RuntimeError("P40 ST0202 trace count drift")
        trace_cache: dict[tuple[int, int], np.ndarray] = {}

        def trace(inline: int, crossline: int) -> np.ndarray:
            key = (inline, crossline)
            if key not in trace_cache:
                trace_id = (inline - il_min) * int(index["n_xl"]) + crossline - xl_min
                trace_cache[key] = np.asarray(handle.trace[int(trace_id)], dtype=np.float32)
            return trace_cache[key]

        for section_row, (well, inline) in enumerate(section_keys):
            window = per_well[well]
            for column, crossline in enumerate(
                range(window["crossline_start"], window["crossline_start"] + 160)
            ):
                sections[section_row, :, column] = trace(inline, crossline)[
                    window["time_start"] : window["time_start"] + 400
                ]
        for row_id, row in enumerate(aligned):
            section_ids[row_id] = section_lookup[(str(row["well"]), int(row["inline"]))]
            trace_ids[row_id] = int(row["crossline"]) - per_well[str(row["well"])][
                "crossline_start"
            ]
            half = 16
            replay = np.empty((3, 3, 33), dtype=np.float32)
            for il_offset in range(-1, 2):
                for xl_offset in range(-1, 2):
                    replay[il_offset + 1, xl_offset + 1] = trace(
                        int(row["inline"]) + il_offset,
                        int(row["crossline"]) + xl_offset,
                    )[int(row["time_index"]) - half : int(row["time_index"]) + half + 1]
            row_error = float(np.max(np.abs(replay - raw_seismic[row_id])))
            patch_error = max(patch_error, row_error)
            row["alignment_quality"] = {
                "official_tie_bracketed": True,
                "native_st0202_patch_replayed": True,
                "patch_max_abs_error": row_error,
                "twt_roundtrip_error_ms": row["twt_roundtrip_error_ms"],
                "interpolation_or_padding": False,
            }
            row["split_hash"] = SPLIT_HASH
            row["sample_hash"] = stable_hash(
                {
                    key: row[key]
                    for key in (
                        "sample_id",
                        "well",
                        "family",
                        "md_m",
                        "tvdss_m",
                        "twt_ms",
                        "inline",
                        "crossline",
                        "outer_fold",
                    )
                }
            )
    if not np.isfinite(sections).all():
        raise FloatingPointError("P40 native GFM sections contain non-finite values")
    audit = {
        "chain": [
            "frozen sample TWT",
            "inverse official bracketed TWT-to-MD tie",
            "official MD-to-TVD/TVDSS/XY interpolation",
            "locked UTM-to-ILXL affine",
            "nearest native 4ms ST0202 sample",
        ],
        "source_rows_filtered_before_parse": sorted(curves),
        "native_sections": len(sections),
        "native_section_shape": [400, 160],
        "interpolation_applied_to_seismic": False,
        "padding_applied_to_seismic": False,
        "maximum_twt_roundtrip_error_ms": max(
            float(row["twt_roundtrip_error_ms"]) for row in aligned
        ),
        "maximum_native_patch_replay_error": patch_error,
        "seismic_index_sha256": sha256(seismic_index_path),
        "official_picks_sha256": sha256(picks_path),
        "segy_path": str(segy_path),
        "segy_bytes": segy_path.stat().st_size,
        "test_or_frozen_rows_parsed": False,
    }
    return aligned, sections, section_ids, trace_ids, audit


def phase0(
    *,
    development_batch: Path,
    raw_project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _safe_paths((development_batch,))
    output_dir = _output_path(output_dir)
    locked_summary = TRACK_DIR / "_outputs" / "default_baseline" / "summary.json"
    if sha256(locked_summary) != LOCKED_BASELINE_SUMMARY_SHA256:
        raise RuntimeError("P40 committed default-baseline summary SHA drift")
    locked_payload = json.loads(locked_summary.read_text(encoding="utf-8"))
    locked_mean = locked_payload["variants"]["default_depth3_eta01_rounds60"][
        "mean_fixed_schema_macro_f1"
    ]
    if locked_mean != EXPECTED_BASELINE_MEAN:
        raise RuntimeError("P40 committed default-baseline value drift")
    if sha256(development_batch) != DEVELOPMENT_BATCH_SHA256:
        raise RuntimeError("P40 development batch SHA drift")
    arrays, manifest = load_stage3_batch(development_batch)
    rows, census = _canonical_rows(arrays, manifest)
    physical_logs, masks, raw_seismic, replay = _recover_physical_inputs(arrays, rows)
    raw_root = Path(raw_project_root).resolve()
    picks = raw_root / (
        "_sandbox/volve_data/_extracted_interp/Geophysical_Interpretations/Wells/"
        "Well_picks_Volve_v1.dat"
    )
    seismic_index = raw_root / "_pipelines/01_common_preprocess/outputs/seismic_index.npz"
    segy = raw_root / (
        "_sandbox/volve_data/_extracted_seismic/ST0202/Stacks/"
        "ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME.MIG_FIN.POST_STACK.3D.JS-017534.segy"
    )
    _safe_paths((picks, seismic_index, segy))
    aligned, sections, section_ids, trace_ids, alignment_audit = _alignment_and_sections(
        rows=rows,
        raw_seismic=raw_seismic,
        picks_path=picks,
        seismic_index_path=seismic_index,
        segy_path=segy,
    )
    ids = np.asarray([row["sample_id"] for row in aligned])
    lookup = {sample_id: index for index, sample_id in enumerate(ids.tolist())}
    cache: dict[str, np.ndarray] = {
        "sample_ids": ids,
        "wells": np.asarray([row["well"] for row in aligned]),
        "families": np.asarray([row["family"] for row in aligned]),
        "labels": np.asarray([row["label"] for row in aligned], dtype=np.int64),
        "physical_logs": physical_logs,
        "log_masks": masks,
        "raw_seismic": raw_seismic,
        "sections": sections,
        "section_ids": section_ids,
        "trace_ids": trace_ids,
    }
    split_records = []
    for outer in range(4):
        train_ids = [str(value) for value in arrays[f"f{outer}_train_ids"].tolist()]
        validation_ids = [str(value) for value in arrays[f"f{outer}_validation_ids"].tolist()]
        cache[f"outer{outer}_train"] = np.asarray([lookup[value] for value in train_ids], np.int32)
        cache[f"outer{outer}_validation"] = np.asarray(
            [lookup[value] for value in validation_ids], np.int32
        )
        # Preserve the frozen Stage-3 fold tensors byte-for-byte for B0.  Their
        # normalization was fitted by the original development-only builder on
        # each outer training partition; replaying them avoids numerical drift
        # from inverting and then reapplying that transform.
        cache[f"outer{outer}_frozen_train_well"] = np.asarray(
            arrays[f"f{outer}_train_well"], dtype=np.float32
        )
        cache[f"outer{outer}_frozen_train_seismic"] = np.asarray(
            arrays[f"f{outer}_train_seismic"], dtype=np.float32
        )
        cache[f"outer{outer}_frozen_validation_well"] = np.asarray(
            arrays[f"f{outer}_validation_well"], dtype=np.float32
        )
        cache[f"outer{outer}_frozen_validation_seismic"] = np.asarray(
            arrays[f"f{outer}_validation_seismic"], dtype=np.float32
        )
        cache[f"outer{outer}_frozen_class_counts"] = np.asarray(
            arrays[f"f{outer}_class_counts"], dtype=np.int64
        )
        outer_families = sorted({aligned[lookup[value]]["family"] for value in train_ids})
        for inner, held_family in enumerate(outer_families):
            inner_validation = [
                lookup[value]
                for value in train_ids
                if aligned[lookup[value]]["family"] == held_family
            ]
            inner_train = [
                lookup[value]
                for value in train_ids
                if aligned[lookup[value]]["family"] != held_family
            ]
            cache[f"outer{outer}_inner{inner}_train"] = np.asarray(inner_train, np.int32)
            cache[f"outer{outer}_inner{inner}_validation"] = np.asarray(
                inner_validation, np.int32
            )
            split_records.append(
                {
                    "context": f"outer{outer}_inner{inner}",
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "held_family": held_family,
                    "train_families": sorted(
                        {aligned[index]["family"] for index in inner_train}
                    ),
                    "train_rows": len(inner_train),
                    "validation_rows": len(inner_validation),
                    "normalization_fit_scope": "inner-train only",
                    "pca_fit_scope": "inner-train only",
                    "early_stopping": "disabled; fixed budget",
                }
            )
    phase0_dir = output_dir / "runtime"
    phase0_dir.mkdir(parents=True, exist_ok=True)
    cache_path = phase0_dir / "phase0_cache.npz"
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, **cache)
    _write_jsonl(output_dir / "aligned_pair_manifest.jsonl", aligned)
    freeze = {
        "schema_version": SCHEMA,
        "state": "PHASE0_FROZEN",
        "development_only": True,
        "development_batch": {
            "external_read_only": True,
            "sha256": sha256(development_batch),
            "bytes": development_batch.stat().st_size,
            "split_hash": SPLIT_HASH,
        },
        "baseline": {
            "model_id": "xgboost_multisoftprob_window",
            "config": BASELINE_CONFIG,
            "expected_logo4_three_seed_mean": EXPECTED_BASELINE_MEAN,
            "locked_summary_sha256": LOCKED_BASELINE_SUMMARY_SHA256,
            "locked_evidence_verified": True,
            "source": "committed default_baseline/summary.json; historical three-seed evidence",
        },
        "census": census,
        "input_replay": replay,
        "alignment": alignment_audit,
        "outer_folds": 4,
        "inner_folds_per_outer": 3,
        "split_records": split_records,
        "firewall": {
            "test_h5_opened": False,
            "known_holdout_opened": False,
            "frozen_test_family": "not parsed or materialized",
            "outer_held_used_for_fit": False,
        },
        "phase0_cache_sha256": sha256(cache_path),
    }
    _write_json(output_dir / "phase0_freeze.json", freeze)
    return freeze


def _normalize_inputs(
    cache: Mapping[str, np.ndarray], train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    logs = np.asarray(cache["physical_logs"], dtype=np.float32)
    masks = np.asarray(cache["log_masks"], dtype=bool)
    transformed = np.zeros_like(logs)
    log_stats = []
    for channel in range(13):
        observed = masks[train, channel]
        values = logs[train, channel][observed]
        if len(values) == 0:
            mean, scale = 0.0, 1.0
        else:
            mean = float(values.mean())
            scale = max(float(values.std()), 1e-6)
        channel_observed = masks[:, channel]
        transformed[:, channel][channel_observed] = (
            logs[:, channel][channel_observed] - mean
        ) / scale
        log_stats.append({"mean": mean, "std": scale, "observed": int(len(values))})
    well = np.concatenate([transformed, masks.astype(np.float32)], axis=1)
    raw_seismic = np.asarray(cache["raw_seismic"], dtype=np.float32)
    mean = float(raw_seismic[train].mean())
    scale = max(float(raw_seismic[train].std()), 1e-6)
    seismic = (raw_seismic - mean) / scale
    return well.astype(np.float32), seismic.astype(np.float32), {
        "fit_rows": int(len(train)),
        "log_stats": log_stats,
        "seismic_stats": {"mean": mean, "std": scale},
        "held_rows_used": False,
    }


def _contexts(cache: Mapping[str, np.ndarray]) -> list[str]:
    result = [f"outer{outer}" for outer in range(4)]
    result.extend(
        f"outer{outer}_inner{inner}" for outer in range(4) for inner in range(3)
    )
    for context in result:
        if f"{context}_train" not in cache or f"{context}_validation" not in cache:
            raise RuntimeError(f"P40 split cache missing context: {context}")
    return result


def baseline_worker(*, phase0_cache: Path, output: Path) -> dict[str, Any]:
    import xgboost

    from _models.lithofacies.xgboost_multisoftprob_window import XGBoostWindowAdapter

    _safe_paths((phase0_cache,))
    cache_archive = np.load(phase0_cache, allow_pickle=False)
    cache = {key: cache_archive[key] for key in cache_archive.files}
    labels = np.asarray(cache["labels"], dtype=np.int64)
    arrays: dict[str, np.ndarray] = {}
    cells = []
    phase0_three_seed = []
    for context in _contexts(cache):
        train = np.asarray(cache[f"{context}_train"], dtype=np.int64)
        validation = np.asarray(cache[f"{context}_validation"], dtype=np.int64)
        if "_inner" not in context:
            outer = int(context.removeprefix("outer"))
            train_well = np.asarray(
                cache[f"outer{outer}_frozen_train_well"], dtype=np.float32
            )
            train_seismic = np.asarray(
                cache[f"outer{outer}_frozen_train_seismic"], dtype=np.float32
            )
            validation_well = np.asarray(
                cache[f"outer{outer}_frozen_validation_well"], dtype=np.float32
            )
            validation_seismic = np.asarray(
                cache[f"outer{outer}_frozen_validation_seismic"], dtype=np.float32
            )
            counts = np.asarray(
                cache[f"outer{outer}_frozen_class_counts"], dtype=np.int64
            )
            normalization = {
                "source": "frozen_stage3_outer_fold_tensors",
                "fit_scope": "original outer-train only",
                "fit_rows": int(len(train)),
                "held_rows_used": False,
            }
        else:
            well, seismic, normalization = _normalize_inputs(cache, train)
            train_well = well[train]
            train_seismic = seismic[train]
            validation_well = well[validation]
            validation_seismic = seismic[validation]
            counts = np.bincount(labels[train], minlength=NUM_CLASSES)
        historical_seeds = (1867973658, 2137841944, 3902865753)
        seeds = historical_seeds + (SEED,) if "_inner" not in context else (SEED,)
        chosen_train = chosen_validation = None
        for seed in seeds:
            model = XGBoostWindowAdapter(seed=int(seed))
            model.fit_stage1(
                train_well, train_seismic, labels[train], class_counts=counts
            )
            train_logits = model.predict_logits(train_well, train_seismic).astype(
                np.float32
            )
            validation_logits = model.predict_logits(
                validation_well, validation_seismic
            ).astype(np.float32)
            metrics = classification_metrics_from_logits(
                labels[validation].tolist(), validation_logits
            )
            if "_inner" not in context and int(seed) in historical_seeds:
                phase0_three_seed.append(float(metrics["fixed_schema_macro_f1"]))
            if int(seed) == SEED or len(seeds) == 1:
                chosen_train, chosen_validation = train_logits, validation_logits
        assert chosen_train is not None and chosen_validation is not None
        arrays[f"{context}_train_logits"] = chosen_train
        arrays[f"{context}_validation_logits"] = chosen_validation
        arrays[f"{context}_train_indices"] = train.astype(np.int32)
        arrays[f"{context}_validation_indices"] = validation.astype(np.int32)
        metrics = classification_metrics_from_logits(
            labels[validation].tolist(), chosen_validation
        )
        cells.append(
            {
                "context": context,
                "seed": SEED,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "metrics": metrics,
                "normalization": normalization,
                "outer_held_used_for_fit": False,
            }
        )
    live_mean = float(np.mean(phase0_three_seed))
    r0_outer_mean = float(
        np.mean(
            [
                cell["metrics"]["fixed_schema_macro_f1"]
                for cell in cells
                if "_inner" not in cell["context"]
            ]
        )
    )
    metadata = {
        "schema_version": SCHEMA,
        "model_id": "xgboost_multisoftprob_window",
        "config": BASELINE_CONFIG,
        "seed": SEED,
        "cells": cells,
        "phase0_logo4_three_seed_mean": live_mean,
        "locked_logo4_three_seed_mean": EXPECTED_BASELINE_MEAN,
        "expected_logo4_three_seed_mean": EXPECTED_BASELINE_MEAN,
        "r0_seed2693_outer_mean": r0_outer_mean,
        "r0_seed2693_minus_locked": r0_outer_mean - EXPECTED_BASELINE_MEAN,
        "locked_summary_sha256": LOCKED_BASELINE_SUMMARY_SHA256,
        "locked_evidence_verified": True,
        "xgboost_version": xgboost.__version__,
        "comparison_contract": (
            "R0 variants are paired only against seed2693 B0 from this same "
            "XGBoost environment and split; the locked three-seed value is audit-only"
        ),
        "development_only": True,
        "known_holdout_accessed": False,
        "frozen_test_accessed": False,
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return {**metadata, "output_sha256": sha256(output)}


def _instance_normalize_logs(logs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    result = np.zeros_like(logs, dtype=np.float32)
    for row in range(len(logs)):
        for channel in range(13):
            observed = masks[row, channel].astype(bool)
            if not observed.any():
                continue
            values = logs[row, channel, observed]
            mean = float(values.mean())
            scale = max(float(values.std()), 1e-6)
            result[row, channel, observed] = (values - mean) / scale
    return result


def extract_features(
    *,
    phase0_cache: Path,
    output: Path,
    moment_snapshot: Path,
    moment_dependency_root: Path,
    moment_source_root: Path,
    gfm_snapshot: Path,
    gfm_source_root: Path,
    device: str,
) -> dict[str, Any]:
    import lithofacies_p40_foundation_features as foundation

    cache_archive = np.load(phase0_cache, allow_pickle=False)
    cache = {key: cache_archive[key] for key in cache_archive.files}
    asset_audit = foundation.verify_assets(
        moment_snapshot=moment_snapshot,
        moment_dependency_root=moment_dependency_root,
        moment_source_root=moment_source_root,
        gfm_snapshot=gfm_snapshot,
        gfm_source_root=gfm_source_root,
    )
    moment_input = _instance_normalize_logs(
        np.asarray(cache["physical_logs"], dtype=np.float32),
        np.asarray(cache["log_masks"], dtype=np.uint8),
    )
    arrays: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {"assets": asset_audit, "runs": {}}
    for mode in ("pretrained", "random_init"):
        random_init = mode == "random_init"
        moment, moment_audit = foundation.extract_moment_tokens(
            moment_input,
            snapshot=moment_snapshot,
            dependency_root=moment_dependency_root,
            source_root=moment_source_root,
            device=device,
            random_init=random_init,
            seed=SEED,
        )
        arrays[f"moment_{mode}"] = moment
        audits["runs"][f"moment_{mode}"] = moment_audit
        gfm, gfm_audit = foundation.extract_gfm_sample_tokens(
            np.asarray(cache["sections"], dtype=np.float32),
            np.asarray(cache["section_ids"], dtype=np.int32),
            np.asarray(cache["trace_ids"], dtype=np.int32),
            snapshot=gfm_snapshot,
            source_root=gfm_source_root,
            device=device,
            random_init=random_init,
            seed=SEED,
        )
        arrays[f"gfm_{mode}"] = gfm
        audits["runs"][f"gfm_{mode}"] = gfm_audit
    audits["pretrained_random_max_abs_difference"] = {
        "moment": float(
            np.max(np.abs(arrays["moment_pretrained"] - arrays["moment_random_init"]))
        ),
        "gfm": float(np.max(np.abs(arrays["gfm_pretrained"] - arrays["gfm_random_init"]))),
    }
    if min(audits["pretrained_random_max_abs_difference"].values()) <= 0:
        raise RuntimeError("P40 pretrained/random foundation tokens are identical")
    arrays["audit"] = np.asarray(json.dumps(audits, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return {**audits, "output_sha256": sha256(output)}


def _pca_features(
    tokens: np.ndarray, train: np.ndarray, *, stream: int
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA

    values = np.asarray(tokens, dtype=np.float32).reshape(len(tokens), -1)
    dimension = min(PCA_DIM, len(train) - 1, values.shape[1])
    pca = PCA(n_components=dimension, svd_solver="randomized", random_state=SEED + stream)
    pca.fit(values[train])
    transformed = pca.transform(values).astype(np.float32)
    return transformed, {
        "source_shape": list(tokens.shape),
        "flattened_width": int(values.shape[1]),
        "output_dimension": int(dimension),
        "fit_rows": int(len(train)),
        "fit_indices_sha256": stable_hash(train.tolist()),
        "held_rows_used": False,
        "method": "centered PCA via deterministic randomized SVD; not random projection",
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def _class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    observed = counts > 0
    weights[observed] = np.sqrt(counts[observed].sum() / counts[observed])
    weights[observed] /= weights[observed].mean()
    return weights


def _seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_linear(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    device: str,
    stream: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    _seed_torch(SEED + stream)
    model = torch.nn.Linear(features.shape[1], NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    weights = torch.as_tensor(_class_weights(labels[train]), dtype=torch.float32, device=device)
    x = torch.as_tensor(features[train], dtype=torch.float32, device=device)
    y = torch.as_tensor(labels[train], dtype=torch.long, device=device)
    losses = []
    for _ in range(HEAD_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y, weight=weights)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        logits = model(
            torch.as_tensor(features[validation], dtype=torch.float32, device=device)
        ).cpu().numpy().astype(np.float32)
    audit = {
        "updates": HEAD_UPDATES,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "early_stopping": "disabled_fixed_budget",
        "validation_used_for_fit_or_selection": False,
    }
    del model, optimizer
    return logits, audit


def _shuffle_within_family(
    indices: np.ndarray, families: np.ndarray, *, stream: int
) -> np.ndarray:
    result = np.asarray(indices, dtype=np.int64).copy()
    rng = np.random.default_rng(SEED + stream)
    for family in sorted(set(families[result].tolist())):
        positions = np.flatnonzero(families[result] == family)
        if len(positions) > 1:
            result[positions] = result[positions][rng.permutation(len(positions))]
    if np.array_equal(result, indices):
        raise RuntimeError("P40 shuffled-pair control did not change any pair")
    return result


def _train_residual(
    moment: np.ndarray,
    gfm: np.ndarray,
    baseline_train: np.ndarray,
    baseline_validation: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    families: np.ndarray,
    *,
    device: str,
    stream: int,
    shuffle_pair: bool,
) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    _seed_torch(SEED + stream)

    class DualResidual(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = 32
            self.moment = torch.nn.Linear(moment.shape[1], hidden)
            self.gfm = torch.nn.Linear(gfm.shape[1], hidden)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden * 3, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, NUM_CLASSES),
            )
            self.gate_raw = torch.nn.Parameter(torch.zeros(NUM_CLASSES))

        def forward(
            self, m: Any, g: Any, baseline: Any, *, force_off: bool = False
        ) -> tuple[Any, Any, Any]:
            hm = F.gelu(self.moment(m))
            hg = F.gelu(self.gfm(g))
            residual = 2.0 * torch.tanh(self.head(torch.cat([hm, hg, hm * hg], dim=1)))
            gate = torch.zeros_like(self.gate_raw) if force_off else torch.tanh(self.gate_raw)
            contribution = gate[None] * residual
            return baseline + contribution, gate, contribution

    model = DualResidual().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    paired = (
        _shuffle_within_family(train, families, stream=stream)
        if shuffle_pair
        else np.asarray(train, dtype=np.int64)
    )
    m = torch.as_tensor(moment[train], dtype=torch.float32, device=device)
    g = torch.as_tensor(gfm[paired], dtype=torch.float32, device=device)
    base = torch.as_tensor(baseline_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels[train], dtype=torch.long, device=device)
    weights = torch.as_tensor(_class_weights(labels[train]), dtype=torch.float32, device=device)
    with torch.no_grad():
        initial, initial_gate, _ = model(m, g, base)
    gate0_error = float(torch.max(torch.abs(initial - base)).cpu())
    if gate0_error != 0.0 or bool(torch.any(initial_gate != 0)):
        raise RuntimeError("P40 residual is not exact at zero initialization")
    losses = []
    first_gate_gradient = None
    for update in range(RESIDUAL_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        logits, gate, contribution = model(m, g, base)
        loss = (
            F.cross_entropy(logits, y, weight=weights)
            + 0.002 * contribution.square().mean()
            + 0.001 * gate.square().mean()
        )
        loss.backward()
        if update == 0:
            first_gate_gradient = float(model.gate_raw.grad.norm().detach().cpu())
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    vm = torch.as_tensor(moment[validation], dtype=torch.float32, device=device)
    vg = torch.as_tensor(gfm[validation], dtype=torch.float32, device=device)
    vb = torch.as_tensor(baseline_validation, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, gate, contribution = model(vm, vg, vb)
        off, off_gate, off_contribution = model(vm, vg, vb, force_off=True)
    result = logits.cpu().numpy().astype(np.float32)
    off_result = off.cpu().numpy().astype(np.float32)
    exact_error = float(np.max(np.abs(off_result - baseline_validation)))
    if exact_error != 0.0 or bool(torch.any(off_gate != 0)) or bool(
        torch.any(off_contribution != 0)
    ):
        raise RuntimeError("P40 A7 fusion_off is not elementwise exact")
    changed_labels = int(
        np.count_nonzero(np.argmax(result, axis=1) != np.argmax(baseline_validation, axis=1))
    )
    audit = {
        "updates": RESIDUAL_UPDATES,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "zero_init_max_abs_error": gate0_error,
        "fusion_off_max_abs_error": exact_error,
        "first_gate_gradient_norm": first_gate_gradient,
        "gate_values": [float(value) for value in gate.cpu().numpy()],
        "gate_mean_abs": float(np.abs(gate.cpu().numpy()).mean()),
        "contribution_mean_abs": float(np.abs(contribution.cpu().numpy()).mean()),
        "contribution_max_abs": float(np.abs(contribution.cpu().numpy()).max()),
        "changed_class_predictions": changed_labels,
        "changed_logit_elements": int(np.count_nonzero(result != baseline_validation)),
        "shuffled_pair_training": shuffle_pair,
        "pair_map_sha256": stable_hash(paired.tolist()),
        "early_stopping": "disabled_fixed_budget",
        "validation_used_for_fit_or_selection": False,
    }
    return result, off_result, model, audit


def _predict_residual(
    model: Any,
    moment: np.ndarray,
    gfm: np.ndarray,
    baseline: np.ndarray,
    indices: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        logits, _, _ = model(
            torch.as_tensor(moment[indices], dtype=torch.float32, device=device),
            torch.as_tensor(gfm[indices], dtype=torch.float32, device=device),
            torch.as_tensor(baseline, dtype=torch.float32, device=device),
        )
    return logits.cpu().numpy().astype(np.float32)


def _row(
    *,
    context: str,
    variant: str,
    labels: np.ndarray,
    logits: np.ndarray,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "context": context,
        "level": "inner" if "_inner" in context else "outer",
        "variant": variant,
        "seed": SEED,
        "metrics": classification_metrics_from_logits(labels.tolist(), logits),
        "training": dict(training),
        "development_only": True,
        "outer_held_used_for_fit": False,
        "known_holdout_accessed": False,
        "frozen_test_accessed": False,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], level: str) -> dict[str, Any]:
    result = {}
    for variant in VARIANTS:
        cells = [row for row in rows if row["level"] == level and row["variant"] == variant]
        result[variant] = {
            "cells": len(cells),
            "mean_macro_f1": float(
                np.mean([row["metrics"]["fixed_schema_macro_f1"] for row in cells])
            ),
            "mean_nll": float(np.mean([row["metrics"]["negative_log_likelihood"] for row in cells])),
            "mean_ece": float(
                np.mean([row["metrics"]["expected_calibration_error"] for row in cells])
            ),
        }
    baseline = result["B0"]["mean_macro_f1"]
    for variant in VARIANTS:
        result[variant]["macro_f1_delta_vs_B0"] = result[variant]["mean_macro_f1"] - baseline
    return result


def _bootstrap(
    predictions: Mapping[str, np.ndarray], labels: np.ndarray, families: np.ndarray
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    family_indices = {family: np.flatnonzero(families == family) for family in FAMILIES}
    result = {}
    for variant in VARIANTS[1:]:
        deltas = []
        for _ in range(2000):
            family_delta = []
            for family in FAMILIES:
                indices = family_indices[family]
                chosen = rng.choice(indices, size=len(indices), replace=True)
                candidate = classification_metrics_from_logits(
                    labels[chosen].tolist(), predictions[variant][chosen]
                )["fixed_schema_macro_f1"]
                baseline = classification_metrics_from_logits(
                    labels[chosen].tolist(), predictions["B0"][chosen]
                )["fixed_schema_macro_f1"]
                family_delta.append(candidate - baseline)
            deltas.append(float(np.mean(family_delta)))
        values = np.asarray(deltas, dtype=np.float64)
        result[variant] = {
            "replicates": len(values),
            "unit": "within-family sample bootstrap then equal-family macro",
            "mean_delta": float(values.mean()),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
    return result


def run_pilot(
    *,
    phase0_cache: Path,
    baseline_bundle: Path,
    feature_bundle: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    output_dir = _output_path(output_dir)
    cache_npz = np.load(phase0_cache, allow_pickle=False)
    cache = {key: cache_npz[key] for key in cache_npz.files}
    baseline_npz = np.load(baseline_bundle, allow_pickle=False)
    baseline = {key: baseline_npz[key] for key in baseline_npz.files}
    baseline_meta = json.loads(str(baseline["metadata"].item()))
    features_npz = np.load(feature_bundle, allow_pickle=False)
    features = {key: features_npz[key] for key in features_npz.files}
    feature_audit = json.loads(str(features["audit"].item()))
    labels = np.asarray(cache["labels"], dtype=np.int64)
    families = np.asarray(cache["families"])
    rows = []
    outer_predictions = {variant: np.full((len(labels), NUM_CLASSES), np.nan, np.float32) for variant in VARIANTS}
    pca_audits: dict[str, Any] = {}
    interventions: dict[str, Any] = {}
    for context_index, context in enumerate(_contexts(cache)):
        train = np.asarray(baseline[f"{context}_train_indices"], dtype=np.int64)
        validation = np.asarray(baseline[f"{context}_validation_indices"], dtype=np.int64)
        base_train = np.asarray(baseline[f"{context}_train_logits"], dtype=np.float32)
        base_validation = np.asarray(baseline[f"{context}_validation_logits"], dtype=np.float32)
        projections = {}
        context_audit = {}
        for mode_index, mode in enumerate(("pretrained", "random_init")):
            projections[f"moment_{mode}"], context_audit[f"moment_{mode}"] = _pca_features(
                features[f"moment_{mode}"], train, stream=100 * context_index + mode_index
            )
            projections[f"gfm_{mode}"], context_audit[f"gfm_{mode}"] = _pca_features(
                features[f"gfm_{mode}"], train, stream=100 * context_index + 10 + mode_index
            )
        pca_audits[context] = context_audit
        b0_training = {"updates": 0, "model": "frozen default XGBoost", "config": BASELINE_CONFIG}
        b2, b2_audit = _train_linear(
            projections["moment_pretrained"], labels, train, validation, device=device, stream=2000 + context_index
        )
        b4, b4_audit = _train_linear(
            projections["gfm_pretrained"], labels, train, validation, device=device, stream=3000 + context_index
        )
        f1, a7, f1_model, f1_audit = _train_residual(
            projections["moment_pretrained"],
            projections["gfm_pretrained"],
            base_train,
            base_validation,
            labels,
            train,
            validation,
            families,
            device=device,
            stream=4000 + context_index,
            shuffle_pair=False,
        )
        a5, _, _, a5_audit = _train_residual(
            projections["moment_random_init"],
            projections["gfm_random_init"],
            base_train,
            base_validation,
            labels,
            train,
            validation,
            families,
            device=device,
            stream=5000 + context_index,
            shuffle_pair=False,
        )
        a6, _, _, a6_audit = _train_residual(
            projections["moment_pretrained"],
            projections["gfm_pretrained"],
            base_train,
            base_validation,
            labels,
            train,
            validation,
            families,
            device=device,
            stream=6000 + context_index,
            shuffle_pair=True,
        )
        values = {"B0": base_validation, "B2": b2, "B4": b4, "F1": f1, "A5": a5, "A6": a6, "A7": a7}
        training = {"B0": b0_training, "B2": b2_audit, "B4": b4_audit, "F1": f1_audit, "A5": a5_audit, "A6": a6_audit, "A7": {"fusion_off_max_abs_error": float(np.max(np.abs(a7 - base_validation))), "updates": 0}}
        for variant in VARIANTS:
            rows.append(
                _row(
                    context=context,
                    variant=variant,
                    labels=labels[validation],
                    logits=values[variant],
                    training=training[variant],
                )
            )
        if "_inner" not in context:
            for variant in VARIANTS:
                outer_predictions[variant][validation] = values[variant]
            zeros_m = np.zeros_like(projections["moment_pretrained"])
            zeros_g = np.zeros_like(projections["gfm_pretrained"])
            moment_off = _predict_residual(
                f1_model, zeros_m, projections["gfm_pretrained"], base_validation, validation, device=device
            )
            gfm_off = _predict_residual(
                f1_model, projections["moment_pretrained"], zeros_g, base_validation, validation, device=device
            )
            cyclic = validation[np.roll(np.arange(len(validation)), 1)]
            cyclic_gfm = projections["gfm_pretrained"].copy()
            cyclic_gfm[validation] = projections["gfm_pretrained"][cyclic]
            misaligned = _predict_residual(
                f1_model,
                projections["moment_pretrained"],
                cyclic_gfm,
                base_validation,
                validation,
                device=device,
            )
            interventions[context] = {
                "moment_off_metrics": classification_metrics_from_logits(labels[validation].tolist(), moment_off),
                "gfm_off_metrics": classification_metrics_from_logits(labels[validation].tolist(), gfm_off),
                "cyclic_pair_misalignment_metrics": classification_metrics_from_logits(labels[validation].tolist(), misaligned),
                "cyclic_pair_prediction_max_abs_delta": float(np.max(np.abs(misaligned - f1))),
                "cyclic_pair_changed_predictions": int(np.count_nonzero(np.argmax(misaligned, axis=1) != np.argmax(f1, axis=1))),
            }
    if any(not np.isfinite(value).all() for value in outer_predictions.values()):
        raise RuntimeError("P40 outer OOF predictions are incomplete")
    outer = _aggregate(rows, "outer")
    inner = _aggregate(rows, "inner")
    outer_cells = [row for row in rows if row["level"] == "outer"]
    per_family = {}
    for family in FAMILIES:
        per_family[family] = {}
        for variant in VARIANTS:
            cell = next(
                row
                for row in outer_cells
                if row["variant"] == variant
                and set(cache["families"][baseline[f"{row['context']}_validation_indices"]].tolist()) == {family}
            )
            per_family[family][variant] = cell["metrics"]
    for variant in VARIANTS:
        outer[variant]["family_wins_vs_B0"] = sum(
            per_family[family][variant]["fixed_schema_macro_f1"]
            > per_family[family]["B0"]["fixed_schema_macro_f1"]
            for family in FAMILIES
        )
    best_single = max(("B2", "B4"), key=lambda name: outer[name]["mean_macro_f1"])
    inner_by_outer = {}
    for outer_id in range(4):
        selected = [row for row in rows if row["level"] == "inner" and row["context"].startswith(f"outer{outer_id}_")]
        b0 = {row["context"]: row for row in selected if row["variant"] == "B0"}
        f1 = {row["context"]: row for row in selected if row["variant"] == "F1"}
        deltas = [f1[key]["metrics"]["fixed_schema_macro_f1"] - b0[key]["metrics"]["fixed_schema_macro_f1"] for key in sorted(b0)]
        inner_by_outer[str(outer_id)] = {
            "mean_delta": float(np.mean(deltas)),
            "family_wins": int(sum(value > 0 for value in deltas)),
            "passes": float(np.mean(deltas)) >= 0.005 and sum(value > 0 for value in deltas) >= 2,
        }
    inner_signal = {
        "overall_delta": inner["F1"]["macro_f1_delta_vs_B0"],
        "overall_wins": sum(
            row["metrics"]["fixed_schema_macro_f1"]
            > next(
                base["metrics"]["fixed_schema_macro_f1"]
                for base in rows
                if base["context"] == row["context"] and base["variant"] == "B0"
            )
            for row in rows
            if row["level"] == "inner" and row["variant"] == "F1"
        ),
        "by_outer": inner_by_outer,
    }
    inner_signal["continue"] = (
        inner_signal["overall_delta"] >= 0.005
        and inner_signal["overall_wins"] >= 8
        and all(item["passes"] for item in inner_by_outer.values())
    )
    promotion = {
        "f1_minus_b0_at_least_0p005": outer["F1"]["macro_f1_delta_vs_B0"] >= 0.005,
        "f1_family_wins_at_least_3_of_4": outer["F1"]["family_wins_vs_B0"] >= 3,
        "f1_minus_best_single_at_least_0p003": outer["F1"]["mean_macro_f1"] - outer[best_single]["mean_macro_f1"] >= 0.003,
        "f1_minus_a5_at_least_0p005": outer["F1"]["mean_macro_f1"] - outer["A5"]["mean_macro_f1"] >= 0.005,
        "f1_minus_a6_at_least_0p005": outer["F1"]["mean_macro_f1"] - outer["A6"]["mean_macro_f1"] >= 0.005,
        "fusion_off_elementwise_exact": all(row["training"].get("fusion_off_max_abs_error") == 0.0 for row in rows if row["variant"] == "A7"),
    }
    promotion["all_pass"] = all(promotion.values())
    oof_metrics = {
        variant: classification_metrics_from_logits(labels.tolist(), outer_predictions[variant])
        for variant in VARIANTS
    }
    summary = {
        "schema_version": SCHEMA,
        "status": "complete",
        "decision": (
            "R0_PROMOTION_GATE_PASS_NEXT_STAGE_ADVICE_ONLY"
            if promotion["all_pass"] and inner_signal["continue"]
            else "R0_STOP_NO_ATTRIBUTABLE_SIGNAL"
        ),
        "protocol": "seed2693_outer_LOGO4_inner_LOGO3_development_only",
        "baseline_phase0": baseline_meta,
        "outer_family_macro": outer,
        "inner_family_macro": inner,
        "inner_continue_signal": inner_signal,
        "per_family": per_family,
        "oof_metrics": oof_metrics,
        "bootstrap": _bootstrap(outer_predictions, labels, families),
        "best_single_foundation": best_single,
        "promotion_gate": promotion,
        "interventions": interventions,
        "pca_audit": pca_audits,
        "foundation_audit": feature_audit,
        "firewall": {
            "development_only": True,
            "known_holdout_accessed": False,
            "frozen_test_accessed": False,
            "outer_held_used_for_normalization_pca_early_stopping_selection": False,
            "early_stopping": "disabled_fixed_budget",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": device,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "results.jsonl", rows)
    _write_json(output_dir / "summary.json", summary)
    with (output_dir / "predictions.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=cache["sample_ids"],
            labels=labels,
            families=families,
            **{f"logits_{variant}": outer_predictions[variant] for variant in VARIANTS},
        )
    evidence = _evidence(summary)
    (output_dir / "evidence.md").write_text(evidence, encoding="utf-8")
    _write_json(output_dir / "foundation_audit.json", feature_audit)
    _write_json(output_dir / "pca_audit.json", pca_audits)
    commands = {
        "phase0": "see CLI --help; development-only external asset is SHA locked",
        "baseline_worker": "run with the local XGBoost interpreter",
        "extract_features": "run with the pinned foundation interpreter and local snapshots",
        "run": "run with the pinned foundation interpreter; --device cuda:5",
        "verify_only": f"{sys.executable} {Path(__file__).relative_to(PROJECT_ROOT)} --verify-only --output-dir {output_dir.relative_to(PROJECT_ROOT)}",
    }
    _write_json(output_dir / "rerun_commands.json", commands)
    _write_artifact_manifest(output_dir)
    verify_only(output_dir)
    return summary


def _evidence(summary: Mapping[str, Any]) -> str:
    outer = summary["outer_family_macro"]
    lines = [
        "# P40 岩相井震双基础模型可归因小试",
        "",
        f"结论：`{summary['decision']}`。本结果仅来自真实 development 9 井、4 井族；未读取 test/frozen holdout。",
        "",
        "| 变体 | Macro-F1 | 相对 B0 | 家族胜场 | NLL | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = outer[variant]
        lines.append(
            f"| {variant} | {item['mean_macro_f1']:.12f} | {item['macro_f1_delta_vs_B0']:+.12f} | {item['family_wins_vs_B0']}/4 | {item['mean_nll']:.12f} | {item['mean_ece']:.12f} |"
        )
    lines.extend(
        [
            "",
            "## 归因与停止线",
            "",
            f"- 内层继续信号：`{summary['inner_continue_signal']['continue']}`，overall delta `{summary['inner_continue_signal']['overall_delta']:+.12f}`，胜场 `{summary['inner_continue_signal']['overall_wins']}/12`。",
            f"- 晋级门全部通过：`{summary['promotion_gate']['all_pass']}`。",
            "- MOMENT 保留 `[13,4,768]` native tokens；GFM 保留 `[CLS, aligned-trace, 1200]` native tokens。所有 PCA 均只在当前 train families 拟合；没有使用 P38 的 768/1200→16 随机投影。",
            "- A7 从同一 F1 结构强制关闭 gate，逐元素复现 B0；门值、梯度、特征消融和配对错位干预均在机读证据中。",
            "- 若 decision 为 STOP，本 Goal 在 R0 结束，不启动 LoRA/Adapter；若通过，也只给下一阶段建议，不在本 Goal 扩展。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifact_manifest(output_dir: Path) -> None:
    names = (
        "aligned_pair_manifest.jsonl",
        "phase0_freeze.json",
        "results.jsonl",
        "summary.json",
        "predictions.npz",
        "evidence.md",
        "foundation_audit.json",
        "pca_audit.json",
        "rerun_commands.json",
    )
    artifacts = []
    for name in names:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"P40 portable artifact missing: {path}")
        artifacts.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    _write_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": SCHEMA,
            "artifacts": artifacts,
            "development_batch_sha256": DEVELOPMENT_BATCH_SHA256,
            "split_hash": SPLIT_HASH,
            "known_holdout_accessed": False,
            "frozen_test_accessed": False,
        },
    )


def verify_only(output_dir: Path) -> dict[str, Any]:
    output_dir = _output_path(output_dir)
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "schema": manifest.get("schema_version") == SCHEMA,
        "split_hash": manifest.get("split_hash") == SPLIT_HASH,
        "firewall": manifest.get("known_holdout_accessed") is False
        and manifest.get("frozen_test_accessed") is False,
        "hashes": all(sha256(output_dir / item["path"]) == item["sha256"] for item in manifest["artifacts"]),
    }
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    checks.update(
        {
            "fusion_off_exact": summary["promotion_gate"]["fusion_off_elementwise_exact"] is True,
            "seven_variants": set(summary["outer_family_macro"]) == set(VARIANTS),
            "four_families": set(summary["per_family"]) == set(FAMILIES),
            "baseline_config": summary["baseline_phase0"]["config"] == BASELINE_CONFIG,
            "locked_baseline_evidence": summary["baseline_phase0"][
                "locked_evidence_verified"
            ]
            is True
            and summary["baseline_phase0"]["locked_summary_sha256"]
            == LOCKED_BASELINE_SUMMARY_SHA256,
            "no_holdout": summary["firewall"]["known_holdout_accessed"] is False
            and summary["firewall"]["frozen_test_accessed"] is False,
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"P40 verify-only failed: {checks}")
    return {"status": "PASS", "checks": checks, "decision": summary["decision"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command")
    phase = sub.add_parser("phase0")
    phase.add_argument("--development-batch", type=Path, required=True)
    phase.add_argument("--raw-project-root", type=Path, required=True)
    baseline = sub.add_parser("baseline-worker")
    baseline.add_argument("--phase0-cache", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    features = sub.add_parser("extract-features")
    features.add_argument("--phase0-cache", type=Path, required=True)
    features.add_argument("--output", type=Path, required=True)
    features.add_argument("--moment-snapshot", type=Path, required=True)
    features.add_argument("--moment-dependency-root", type=Path, required=True)
    features.add_argument("--moment-source-root", type=Path, required=True)
    features.add_argument("--gfm-snapshot", type=Path, required=True)
    features.add_argument("--gfm-source-root", type=Path, required=True)
    features.add_argument("--device", default="cuda:5")
    run = sub.add_parser("run")
    run.add_argument("--phase0-cache", type=Path, required=True)
    run.add_argument("--baseline-bundle", type=Path, required=True)
    run.add_argument("--feature-bundle", type=Path, required=True)
    run.add_argument("--device", default="cuda:5")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_only:
        payload = verify_only(args.output_dir)
    elif args.command == "phase0":
        payload = phase0(
            development_batch=args.development_batch,
            raw_project_root=args.raw_project_root,
            output_dir=args.output_dir,
        )
    elif args.command == "baseline-worker":
        payload = baseline_worker(phase0_cache=args.phase0_cache, output=args.output)
    elif args.command == "extract-features":
        payload = extract_features(
            phase0_cache=args.phase0_cache,
            output=args.output,
            moment_snapshot=args.moment_snapshot,
            moment_dependency_root=args.moment_dependency_root,
            moment_source_root=args.moment_source_root,
            gfm_snapshot=args.gfm_snapshot,
            gfm_source_root=args.gfm_source_root,
            device=args.device,
        )
    elif args.command == "run":
        payload = run_pilot(
            phase0_cache=args.phase0_cache,
            baseline_bundle=args.baseline_bundle,
            feature_bundle=args.feature_bundle,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        raise SystemExit("choose a P40 command or --verify-only")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
