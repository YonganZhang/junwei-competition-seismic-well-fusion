#!/usr/bin/env python3
"""P23: independently validate a checkshot-based Volve well tie."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence
from zipfile import ZipFile

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RAW_PROJECT_ROOT = Path(
    os.environ.get("VOLVE_RAW_PROJECT_ROOT", str(PROJECT_ROOT))
)
VSP_ZIP = RAW_PROJECT_ROOT / "_sandbox/volve_data/Volve_Seismic_VSP.zip"
WELL_PICKS = (
    RAW_PROJECT_ROOT
    / "_sandbox/volve_data/_extracted_interp/Geophysical_Interpretations/"
    "Wells/Well_picks_Volve_v1.dat"
)
WEAK_TIE = PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/well_tie_weak.npz"
SEISMIC_INDEX = PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/seismic_index.npz"
BUILD_SUMMARY = PROJECT_ROOT / "_pipelines/02_task_datasets/reconstruction/build_summary.json"
OUTPUT = HERE / "_outputs/p23_checkshot_calibration/summary.json"
FIT_WELLS = ("19A", "19BT2", "19SR")
VALIDATION_WELLS = ("F11T2", "F15A")
PICK_NAMES = {
    "19A": "NO 15/9-19 A",
    "19BT2": "NO 15/9-19 BT2",
    "19SR": "NO 15/9-19 SR",
    "F11T2": "NO 15/9-F-11 T2",
    "F15A": "NO 15/9-F-15 A",
}
CHECKSHOT_MEMBERS = {
    "19A": "VSP/Checkshots/checkshot_15_9_19A.txt",
    "19BT2": "VSP/Checkshots/checkshot_15_9_19BT2.txt",
    "19SR": "VSP/Checkshots/checkshot_15_9_19_SR.txt",
    "F11T2": "VSP/Checkshots/checkshot_15_9_F_11T2.ASC",
    "F15A": "VSP/Checkshots/checkshot_15_9_F_15A.txt",
}


@dataclass(frozen=True)
class Curve:
    depth_msl: np.ndarray
    twt_ms: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    depth_msl: np.ndarray
    x: np.ndarray
    y: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_sorted(x: Sequence[float], *ys: Sequence[float]) -> tuple[np.ndarray, ...]:
    order = np.argsort(np.asarray(x, dtype=np.float64))
    sorted_x = np.asarray(x, dtype=np.float64)[order]
    sorted_ys = [np.asarray(y, dtype=np.float64)[order] for y in ys]
    unique_x, positions = np.unique(sorted_x, return_index=True)
    return (unique_x, *(y[positions] for y in sorted_ys))


def parse_checkshots(path: Path) -> dict[str, Curve]:
    curves: dict[str, Curve] = {}
    number = re.compile(r"[-+]?\d+(?:\.\d+)?")
    with ZipFile(path) as archive:
        for well, member in CHECKSHOT_MEMBERS.items():
            lines = archive.read(member).decode("latin1", errors="replace").splitlines()
            depth: list[float] = []
            twt: list[float] = []
            for line in lines:
                values = [float(value) for value in number.findall(line)]
                if well in {"19A", "19BT2", "19SR"}:
                    if not line.lstrip().startswith("TIME-CKS") or len(values) < 4:
                        continue
                    depth.append(abs(values[2]))  # positive TVDSS below MSL
                    twt.append(values[3])
                else:
                    if len(values) != 3 or values[0] < 500.0:
                        continue
                    depth.append(values[1])  # published TVD-MSL / TVDMSL
                    twt.append(values[2])
            depth_array, twt_array = deduplicate_sorted(depth, twt)
            if len(depth_array) < 20 or np.any(np.diff(depth_array) <= 0):
                raise RuntimeError(f"invalid checkshot curve for {well}")
            curves[well] = Curve(depth_array, twt_array)
    return curves


def parse_pick_trajectories(path: Path) -> dict[str, Trajectory]:
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    spans: list[tuple[int, int]] | None = None
    rows: dict[str, list[tuple[float, float, float]]] = {}
    wanted = set(PICK_NAMES.values())
    for line in lines:
        if re.match(r"^\s*-{5,}", line):
            spans = [(match.start(), match.end()) for match in re.finditer(r"-+", line)]
            continue
        if spans is None or not line.strip():
            continue
        columns = [line[start:end].strip() for start, end in spans]
        if len(columns) < 12 or columns[0] not in wanted:
            continue
        try:
            tvdss = abs(float(columns[6]))
            x = float(columns[10])
            y = float(columns[11])
        except (ValueError, IndexError):
            continue
        rows.setdefault(columns[0], []).append((tvdss, x, y))
    trajectories: dict[str, Trajectory] = {}
    for well, pick_name in PICK_NAMES.items():
        values = rows.get(pick_name, [])
        depth, x, y = deduplicate_sorted(
            [row[0] for row in values],
            [row[1] for row in values],
            [row[2] for row in values],
        )
        if len(depth) < 4:
            raise RuntimeError(f"insufficient trajectory picks for {well}")
        trajectories[well] = Trajectory(depth, x, y)
    return trajectories


def interp_bounded(query: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = (query >= x.min()) & (query <= x.max())
    return np.interp(query, x, y), valid


def load_current_weak_curves() -> tuple[dict[str, Curve], dict[str, Trajectory]]:
    with np.load(SEISMIC_INDEX, allow_pickle=False) as index:
        affine = np.asarray(index["affine_il_xl_to_xy"], dtype=np.float64)
    curves: dict[str, Curve] = {}
    trajectories: dict[str, Trajectory] = {}
    weak_prefix = {"19A": "15_9-19_A", "19BT2": "15_9-19_BT2", "19SR": "15_9-19_SR"}
    with np.load(WEAK_TIE, allow_pickle=False) as tie:
        for well, prefix in weak_prefix.items():
            depth = np.asarray(tie[f"{prefix}__depth_m"], dtype=np.float64)
            twt = np.asarray(tie[f"{prefix}__twt_est_ms"], dtype=np.float64)
            inline = np.asarray(tie[f"{prefix}__inline"], dtype=np.float64)
            crossline = np.asarray(tie[f"{prefix}__crossline"], dtype=np.float64)
            x = affine[0, 0] * inline + affine[0, 1] * crossline + affine[0, 2]
            y = affine[1, 0] * inline + affine[1, 1] * crossline + affine[1, 2]
            curves[well] = Curve(depth, twt)
            trajectories[well] = Trajectory(depth, x, y)
    return curves, trajectories


def spatial_prediction(
    *,
    query_depth: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    curves: Mapping[str, Curve],
    trajectories: Mapping[str, Trajectory],
    wells: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.zeros(len(query_depth), dtype=np.float64)
    denominator = np.zeros(len(query_depth), dtype=np.float64)
    contributors = np.zeros(len(query_depth), dtype=np.int64)
    for well in wells:
        curve = curves[well]
        trajectory = trajectories[well]
        twt, curve_valid = interp_bounded(query_depth, curve.depth_msl, curve.twt_ms)
        well_x, x_valid = interp_bounded(query_depth, trajectory.depth_msl, trajectory.x)
        well_y, y_valid = interp_bounded(query_depth, trajectory.depth_msl, trajectory.y)
        valid = curve_valid & x_valid & y_valid
        distance2 = (query_x - well_x) ** 2 + (query_y - well_y) ** 2
        weight = valid.astype(np.float64) / (distance2 + 100.0**2)
        numerator += weight * twt
        denominator += weight
        contributors += valid.astype(np.int64)
    valid = denominator > 0.0
    prediction = np.full(len(query_depth), np.nan, dtype=np.float64)
    prediction[valid] = numerator[valid] / denominator[valid]
    return prediction, contributors


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "rows": int(len(error)),
        "mae_ms": float(np.mean(np.abs(error))),
        "rmse_ms": float(np.sqrt(np.mean(error**2))),
        "bias_ms": float(np.mean(error)),
        "median_absolute_error_ms": float(np.median(np.abs(error))),
        "equivalent_4ms_sample_mae": float(np.mean(np.abs(error)) / 4.0),
    }


def main() -> int:
    build = json.loads(BUILD_SUMMARY.read_text(encoding="utf-8"))
    depth_min, depth_max = map(float, build["coordinate_bounds"]["depth"])
    checkshots = parse_checkshots(VSP_ZIP)
    pick_trajectories = parse_pick_trajectories(WELL_PICKS)
    weak_curves, weak_trajectories = load_current_weak_curves()

    rows: dict[str, dict[str, object]] = {}
    pooled_target: list[np.ndarray] = []
    pooled_weak: list[np.ndarray] = []
    pooled_candidate: list[np.ndarray] = []
    for well in VALIDATION_WELLS:
        truth = checkshots[well]
        trajectory = pick_trajectories[well]
        target_depth_mask = (truth.depth_msl >= depth_min) & (truth.depth_msl <= depth_max)
        depth = truth.depth_msl[target_depth_mask]
        target_twt = truth.twt_ms[target_depth_mask]
        query_x, x_valid = interp_bounded(depth, trajectory.depth_msl, trajectory.x)
        query_y, y_valid = interp_bounded(depth, trajectory.depth_msl, trajectory.y)
        query_valid = x_valid & y_valid
        depth = depth[query_valid]
        target_twt = target_twt[query_valid]
        query_x = query_x[query_valid]
        query_y = query_y[query_valid]
        weak, weak_contributors = spatial_prediction(
            query_depth=depth,
            query_x=query_x,
            query_y=query_y,
            curves=weak_curves,
            trajectories=weak_trajectories,
            wells=FIT_WELLS,
        )
        candidate, candidate_contributors = spatial_prediction(
            query_depth=depth,
            query_x=query_x,
            query_y=query_y,
            curves=checkshots,
            trajectories=pick_trajectories,
            wells=FIT_WELLS,
        )
        valid = np.isfinite(weak) & np.isfinite(candidate)
        if np.count_nonzero(valid) < 10:
            raise RuntimeError(f"too few comparable rows for {well}")
        target_twt = target_twt[valid]
        weak = weak[valid]
        candidate = candidate[valid]
        depth = depth[valid]
        rows[well] = {
            "depth_range_msl_m": [float(depth.min()), float(depth.max())],
            "weak": metrics(target_twt, weak),
            "checkshot_candidate": metrics(target_twt, candidate),
            "candidate_mae_delta_ms": (
                metrics(target_twt, candidate)["mae_ms"] - metrics(target_twt, weak)["mae_ms"]
            ),
            "weak_contributors_range": [
                int(weak_contributors[valid].min()),
                int(weak_contributors[valid].max()),
            ],
            "candidate_contributors_range": [
                int(candidate_contributors[valid].min()),
                int(candidate_contributors[valid].max()),
            ],
        }
        pooled_target.append(target_twt)
        pooled_weak.append(weak)
        pooled_candidate.append(candidate)

    target = np.concatenate(pooled_target)
    weak = np.concatenate(pooled_weak)
    candidate = np.concatenate(pooled_candidate)
    weak_metrics = metrics(target, weak)
    candidate_metrics = metrics(target, candidate)
    well_wins = sum(
        rows[well]["checkshot_candidate"]["mae_ms"] < rows[well]["weak"]["mae_ms"]
        for well in VALIDATION_WELLS
    )
    result = {
        "schema_version": "reconstruction-p23-checkshot-calibration/v1",
        "status": "VALIDATED_CALIBRATION_ONLY",
        "protocol": {
            "fit_wells": list(FIT_WELLS),
            "independent_validation_wells": list(VALIDATION_WELLS),
            "target_depth_range_msl_m": [depth_min, depth_max],
            "primary_metric": "pooled independent-well TWT MAE",
            "porosity_labels_used": False,
            "reconstruction_hdf5_opened": [],
            "holdout_opened": False,
        },
        "curve_ranges": {
            well: {
                "rows": int(len(curve.depth_msl)),
                "depth_msl_m": [float(curve.depth_msl.min()), float(curve.depth_msl.max())],
                "twt_ms": [float(curve.twt_ms.min()), float(curve.twt_ms.max())],
            }
            for well, curve in checkshots.items()
        },
        "per_validation_well": rows,
        "pooled": {
            "weak": weak_metrics,
            "checkshot_candidate": candidate_metrics,
            "candidate_mae_delta_ms": candidate_metrics["mae_ms"] - weak_metrics["mae_ms"],
        },
        "decision": {
            "validation_well_wins": int(well_wins),
            "strict_pooled_mae_improvement": bool(
                candidate_metrics["mae_ms"] < weak_metrics["mae_ms"]
            ),
            "eligible_for_downstream_alignment_evaluation": bool(
                well_wins == len(VALIDATION_WELLS)
                and candidate_metrics["mae_ms"] < weak_metrics["mae_ms"]
            ),
            "porosity_blind_test_claimed": False,
        },
        "inputs": {
            "vsp_zip_sha256": sha256(VSP_ZIP),
            "well_picks_sha256": sha256(WELL_PICKS),
            "weak_tie_sha256": sha256(WEAK_TIE),
            "seismic_index_sha256": sha256(SEISMIC_INDEX),
            "build_summary_sha256": sha256(BUILD_SUMMARY),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
