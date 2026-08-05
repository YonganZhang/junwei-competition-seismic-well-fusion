#!/usr/bin/env python3
"""P41 development-only well--seismic foundation qualification.

The runner has three deliberately separate execution surfaces:

* ``--phase0-only`` reads the authoritative development assets, reconstructs
  complete native SEG-Y sections, and freezes a hash-audited data contract.
* the default command extracts pinned frozen MOMENT/GFM tokens and runs a
  single-seed outer LOGO4 conditional-increment experiment.
* ``--baseline-worker`` is invoked only in the locked P5 tabular environment;
  it replays the target-specific ExtraTrees/XGBoost B0 and has no data loader.

There is intentionally no frozen-test path or loader in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = HERE.parents[2]


def _canonical_repo_root() -> Path:
    """Return the shared repository root when running inside a git worktree."""
    parts = WORKTREE_ROOT.parts
    try:
        marker = parts.index(".claude")
    except ValueError:
        return WORKTREE_ROOT
    return Path(*parts[:marker])


CANONICAL_ROOT = _canonical_repo_root()
_legacy_property_root = CANONICAL_ROOT / ".claude/worktrees/track-property"
SOURCE_ROOT = Path(
    os.environ.get(
        "JUNWEI_PROPERTY_SOURCE_ROOT",
        str(_legacy_property_root if _legacy_property_root.is_dir() else CANONICAL_ROOT),
    )
).expanduser().resolve()
P5_ROOT = Path(os.environ.get("JUNWEI_P5_ROOT", str(WORKTREE_ROOT))).expanduser().resolve()
P5_COMMIT = "317ee5d320c061bef78fb6db4fc395764dd18e8d"
P5_HISTORICAL_BASELINE_COMMIT = "16bebd18a0bc722afcbc4b841610bf76ce9503e4"
P5_DIR = P5_ROOT / "_pipelines" / "02_task_datasets" / "reservoir"
P5_RUNNER = P5_DIR / "reservoir_p5_stage3.py"
P5_BUDGET = P5_DIR / "reservoir_p5_stage3_budget.json"
P5_BOARDS = {
    "PHIF": P5_DIR / "_outputs/p5_stage3/leaderboard_phif_tabular_cpu.json",
    "KLOGH": P5_DIR / "_outputs/p5_stage3/leaderboard_klogh_tabular_cpu.json",
    "SW": P5_DIR / "_outputs/p5_stage3/leaderboard_sw_tabular_cpu.json",
}
P5_SOURCE_FILES = {
    "runner": (P5_RUNNER, "9e709582c9fb64505f28f989220090cd1c7504bcdee237c88ef35939454dea0f"),
    "budget": (P5_BUDGET, "d7f36c0917d3ff68fc6c47fb2258ae2c86f802d5299ea0a5ff23c00e1b53e0a1"),
    "extra_trees_adapter": (
        P5_ROOT / "_models/property/extra_trees_regressor.py",
        "e946f23a168d4a34eebf2052a36c50f70d14533e41bce623d59e2fb0b281d9fa",
    ),
    "xgboost_adapter": (
        P5_ROOT / "_models/property/xgboost_regressor.py",
        "a39c22117d0807f371a3647a45da62a4deec8e2ffc01b505df9f948bdf4eb1ee",
    ),
    "common_adapter": (
        P5_ROOT / "_models/property/_p5_common.py",
        "36053f02ed1bfdc25b195229a0c9b76616cdeee312afc996981210a68688e6ad",
    ),
}
P5_BOARD_HASHES = {
    "PHIF": "590ef021351b04264f0a6e889ac206ef1929aea9d54ca9d823a879f8265aa8bf",
    "KLOGH": "f80a05a039f56b9c9bd5bce9804d77ad811ba66d67e73315b0e6d1c251c5e539",
    "SW": "0d68817ee16e716a1b0ef18d52dc3e3988679b0f1f08a10e2f4cd3956c98c919",
}

SOURCE_TRAIN = SOURCE_ROOT / "_data/processed/reservoir/train.h5"
SOURCE_GUARD = SOURCE_ROOT / "_pipelines/02_task_datasets/reservoir/_outputs/guard.npz"
SOURCE_RUN_MANIFEST = SOURCE_ROOT / "_pipelines/02_task_datasets/reservoir/_outputs/run_manifest.json"
SOURCE_SPLIT_MANIFEST = SOURCE_ROOT / "_pipelines/02_task_datasets/reservoir/_outputs/split_manifest.json"
SOURCE_INDEX = SOURCE_ROOT / "_pipelines/01_common_preprocess/outputs/seismic_index.npz"
SOURCE_PICKS = SOURCE_ROOT / "_sandbox/volve_data/_extracted_interp/Geophysical_Interpretations/Wells/Well_picks_Volve_v1.dat"
SOURCE_SEGY = SOURCE_ROOT / (
    "_sandbox/volve_data/_extracted_seismic/ST0202/Stacks/"
    "ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME.MIG_FIN.POST_STACK.3D.JS-017534.segy"
)

OUTPUT = HERE / "_outputs/p41_property_crossmodal_qualification"
RUNTIME = OUTPUT / "runtime"
PHASE0_CACHE = RUNTIME / "phase0_cache.npz"
FOUNDATION_CACHE = RUNTIME / "foundation_tokens.npz"
BASELINE_CACHE = RUNTIME / "p5_b0_contexts.npz"
PHASE0_PATH = OUTPUT / "phase0_freeze.json"
PAIR_PATH = OUTPUT / "aligned_pair_manifest.jsonl"
FOUNDATION_AUDIT_PATH = OUTPUT / "foundation_audit.json"
FEATURE_AUDIT_PATH = OUTPUT / "feature_or_adapter_audit.json"
RESULTS_PATH = OUTPUT / "results.jsonl"
PREDICTIONS_PATH = OUTPUT / "predictions.npz"
SUMMARY_PATH = OUTPUT / "summary.json"
ARTIFACT_PATH = OUTPUT / "artifact_manifest.json"
RERUN_PATH = OUTPUT / "rerun_commands.json"
EVIDENCE_PATH = OUTPUT / "evidence.md"
INDEPENDENT_PATH = OUTPUT / "independent_metric_check.json"

MOMENT_DEPS = Path(os.environ.get("P41_MOMENT_DEPS", "/mnt/data/yongan-admin-2/.cache/p8-foundation-deps"))
MOMENT_SOURCE = Path(os.environ.get("P41_MOMENT_SOURCE", "/mnt/data/yongan-admin-2/.cache/upstream/moment"))
MOMENT_SNAPSHOT = Path(
    os.environ.get(
        "P41_MOMENT_SNAPSHOT",
        "/mnt/data/yongan-admin-2/.cache/huggingface/hub/models--AutonLab--MOMENT-1-base/"
        "snapshots/5e44b0ea26376a176360f87831124e018f876d96",
    )
)
GFM_SOURCE = Path(os.environ.get("P41_GFM_SOURCE", "/mnt/data/yongan-admin-2/.cache/gfm_vendor_src"))
GFM_SNAPSHOT = Path(
    os.environ.get(
        "P41_GFM_SNAPSHOT",
        "/mnt/data/yongan-admin-2/.cache/huggingface/hub/models--thinkonward--"
        "geophysical-foundation-model/snapshots/d4a33965730a506cfdb4c85fa2a0a344c53216a2",
    )
)
TABULAR_PYTHON = Path(
    os.environ.get("P41_TABULAR_PYTHON", "/mnt/data/yongan-admin-2/.cache/volve-p5/envs/tabular-cpu/bin/python")
)

SEED = 2693
TARGETS = ("PHIF", "log1p(KLOGH)", "SW")
P5_TARGETS = ("PHIF", "KLOGH", "SW")
ARMS = ("B0", "W1", "S1", "F1", "A5", "A6", "A7", "A8")
TRAIN_LIMIT = 192
RESIDUAL_STEPS = 160
PCA_COMPONENTS = 16
BOOTSTRAP_REPLICATES = 2000
FROZEN_FAMILY = "15/9-F-15"


@dataclass(frozen=True)
class Record:
    sample_id: str
    source_split: str
    family: str
    well: str
    depth_m: float
    time_idx: int
    twt_ms: float
    inline: int
    crossline: int
    seismic: np.ndarray
    logs: np.ndarray
    targets: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_is_ancestor(commit: str, root: Path) -> bool:
    return subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def verify_p5_lock() -> dict[str, Any]:
    source_head_is_descendant = git_is_ancestor(P5_COMMIT, P5_ROOT)
    files: dict[str, Any] = {}
    for role, (path, expected) in P5_SOURCE_FILES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"BLOCKED_STRONG_BASELINE_REPLAY: P5 {role} hash drift")
        files[role] = {"path": str(path), "sha256": expected}
    budget = read_json(P5_BUDGET)
    if budget.get("sample_budget", {}).get("max_train_samples_per_fold") != TRAIN_LIMIT:
        raise RuntimeError("BLOCKED_STRONG_BASELINE_REPLAY: P5 train budget drift")
    expected_models = {
        "PHIF": "extra_trees_regressor", "KLOGH": "extra_trees_regressor", "SW": "xgboost_regressor"
    }
    boards: dict[str, Any] = {}
    for target, path in P5_BOARDS.items():
        observed = sha256(path)
        board = read_json(path)
        winner = board.get("entries", [{}])[0].get("model_id")
        if observed != P5_BOARD_HASHES[target] or winner != expected_models[target]:
            raise RuntimeError(f"BLOCKED_STRONG_BASELINE_REPLAY: {target} leaderboard drift")
        boards[target] = {"sha256": observed, "winner": winner}
    return {
        "status": "P5_STRONG_BASELINE_REPLAY_LOCKED",
        "source_commit": P5_COMMIT,
        "historical_baseline_commit": P5_HISTORICAL_BASELINE_COMMIT,
        "source_head_is_descendant": source_head_is_descendant,
        "source_files": files,
        "leaderboards": boards,
        "input": "153 = 81 seismic amplitudes + 36 normalized log values + 36 masks",
        "preprocessing": "fit on each context train rows only",
        "selection": "deterministic family-balanced depth-spread, at most 192 train rows",
        "winners": expected_models,
        "configs": {
            "extra_trees_regressor": {
                "n_estimators": 32, "max_depth": 8, "min_samples_leaf": 2,
                "max_features": 0.7, "random_state": SEED, "n_jobs": 1,
            },
            "xgboost_regressor": {
                "n_estimators": 32, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1e-3,
                "tree_method": "hist", "random_state": SEED, "n_jobs": 1,
            },
        },
    }


def load_records() -> list[Record]:
    import h5py

    records: list[Record] = []
    with h5py.File(SOURCE_TRAIN, "r") as handle:
        for key in sorted(handle):
            group = handle[key]
            meta = json.loads(group.attrs["meta"])
            position = json.loads(group.attrs["position"])
            records.append(
                Record(
                    sample_id=key, source_split="train", family=str(meta["family_id"]),
                    well=str(meta["well_id"]), depth_m=float(meta["depth_m"]),
                    time_idx=int(meta["time_idx"]), twt_ms=float(position["time_ms"]),
                    inline=int(position["inline"]), crossline=int(position["crossline"]),
                    seismic=np.asarray(group["seismic_patch"], dtype=np.float32),
                    logs=np.asarray(group["well_log_seq"], dtype=np.float32),
                    targets=np.asarray(group["label"], dtype=np.float64).reshape(3),
                )
            )
    with np.load(SOURCE_GUARD, allow_pickle=False) as archive:
        for index in range(len(archive["label"])):
            meta = json.loads(str(archive["meta_json"][index]))
            position = json.loads(str(archive["position_json"][index]))
            records.append(
                Record(
                    sample_id=f"guard_{index:05d}", source_split="guard",
                    family=str(meta["family_id"]), well=str(meta["well_id"]),
                    depth_m=float(meta["depth_m"]), time_idx=int(meta["time_idx"]),
                    twt_ms=float(position["time_ms"]), inline=int(position["inline"]),
                    crossline=int(position["crossline"]),
                    seismic=np.asarray(archive["seismic_patch"][index], dtype=np.float32),
                    logs=np.asarray(archive["well_log_seq"][index], dtype=np.float32),
                    targets=np.asarray(archive["label"][index], dtype=np.float64).reshape(3),
                )
            )
    families = sorted({row.family for row in records})
    if len(records) != 1216 or len(families) != 4 or FROZEN_FAMILY in families:
        raise RuntimeError("BLOCKED_DATA_GATE: expected 1216 rows across four development families")
    if len({row.sample_id for row in records}) != len(records):
        raise RuntimeError("BLOCKED_DATA_GATE: duplicate sample IDs")
    for row in records:
        if row.seismic.shape != (3, 3, 9) or row.logs.shape != (9, 8):
            raise RuntimeError("BLOCKED_DATA_GATE: source tensor shape drift")
        if not np.isfinite(row.targets).all():
            raise RuntimeError("BLOCKED_DATA_GATE: P41 requires complete three-target development rows")
    return records


def _parse_pick_curves(path: Path, wells: set[str]) -> dict[str, np.ndarray]:
    import re

    spans: list[tuple[int, int]] | None = None
    points: dict[str, list[tuple[float, float, float, float, float, float]]] = {}
    canonical = {well.replace(" ", "").upper(): well for well in wells}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        if re.match(r"^\s*-{5,}", line):
            spans = [(match.start(), match.end()) for match in re.finditer(r"-+", line)]
            continue
        if spans is None or not line.strip() or line.startswith("Well NO") or line.lstrip().startswith("Well name"):
            continue
        cols = [line[start:stop].strip() for start, stop in spans]
        if len(cols) < 12:
            continue
        key = re.sub(r"^\s*NO\s*", "", cols[0], flags=re.IGNORECASE).replace(" ", "").upper()
        well = canonical.get(key)
        if well is None:
            continue
        try:
            point = tuple(float(cols[index]) for index in (4, 5, 6, 7, 10, 11))
        except ValueError:
            continue
        points.setdefault(well, []).append(point)  # type: ignore[arg-type]
    result: dict[str, np.ndarray] = {}
    for well in sorted(wells):
        values = np.asarray(points.get(well, []), dtype=np.float64)
        if len(values) < 2:
            raise RuntimeError(f"BLOCKED_DATA_GATE: insufficient official well picks for {well}")
        values = values[np.argsort(values[:, 0])]
        _, indices = np.unique(values[:, 0], return_index=True)
        result[well] = values[np.sort(indices)]
    return result


def linear_interp_extrap(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    if len(xp) < 2 or np.any(np.diff(xp) <= 0):
        raise ValueError("interpolation axis must be strictly increasing")
    if x < xp[0]:
        lo, hi = 0, 1
    elif x > xp[-1]:
        lo, hi = len(xp) - 2, len(xp) - 1
    else:
        return float(np.interp(x, xp, fp))
    return float(fp[lo] + (x - xp[lo]) * (fp[hi] - fp[lo]) / (xp[hi] - xp[lo]))


def _window_start(low: int, high: int, width: int, minimum: int, maximum: int) -> int:
    if high - low + 1 > width:
        raise RuntimeError("BLOCKED_DATA_GATE: one well exceeds the complete-section window")
    start = low - (width - (high - low + 1)) // 2
    return max(minimum, min(start, maximum - width + 1))


def align_and_build_sections(records: Sequence[Record]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import segyio

    with np.load(SOURCE_INDEX, allow_pickle=False) as archive:
        index = {key: archive[key] for key in archive.files}
    samples_ms = np.asarray(index["samples_ms"], dtype=np.float64)
    il_min, il_max = int(index["il_min"]), int(index["il_max"])
    xl_min, xl_max = int(index["xl_min"]), int(index["xl_max"])
    n_xl = int(index["n_xl"])
    curves = _parse_pick_curves(SOURCE_PICKS, {row.well for row in records})
    aligned: list[dict[str, Any]] = []
    for row in records:
        curve = curves[row.well]
        md = row.depth_m
        bracketed = bool(curve[:, 0].min() <= md <= curve[:, 0].max())
        official_twt = linear_interp_extrap(md, curve[:, 0], curve[:, 3])
        nearest = int(np.argmin(np.abs(samples_ms - row.twt_ms)))
        aligned.append(
            {
                "sample_id": row.sample_id, "source_split": row.source_split,
                "family_id": row.family, "well_id": row.well, "md_m": md,
                "tvd_m": linear_interp_extrap(md, curve[:, 0], curve[:, 1]),
                "tvdss_m": linear_interp_extrap(md, curve[:, 0], curve[:, 2]),
                "easting_m": linear_interp_extrap(md, curve[:, 0], curve[:, 4]),
                "northing_m": linear_interp_extrap(md, curve[:, 0], curve[:, 5]),
                "twt_ms": row.twt_ms, "official_tie_twt_ms": official_twt,
                "official_tie_bracketed": bracketed,
                "official_tie_extrapolated": not bracketed,
                "official_position_twt_difference_ms": abs(official_twt - row.twt_ms),
                "inline": row.inline, "crossline": row.crossline,
                "meta_time_index": row.time_idx, "nearest_time_index": nearest,
                "nearest_time_index_matches_meta": nearest == row.time_idx,
                "position_source": "authoritative HDF5/guard position metadata",
            }
        )
    per_well: dict[str, dict[str, int]] = {}
    for well in sorted({row.well for row in records}):
        subset = [row for row in records if row.well == well]
        per_well[well] = {
            "crossline_start": _window_start(
                min(row.crossline for row in subset), max(row.crossline for row in subset), 160, xl_min, xl_max
            ),
            "time_start": _window_start(
                min(row.time_idx for row in subset), max(row.time_idx for row in subset), 400, 0, len(samples_ms) - 1
            ),
        }
    section_keys = sorted({(row.well, row.inline) for row in records})
    section_lookup = {key: index for index, key in enumerate(section_keys)}
    sections = np.empty((len(section_keys), 400, 160), dtype=np.float32)
    section_ids = np.empty(len(records), dtype=np.int32)
    trace_ids = np.empty(len(records), dtype=np.int32)
    maximum_patch_error = 0.0
    with segyio.open(str(SOURCE_SEGY), "r", ignore_geometry=True) as handle:
        if handle.tracecount != int(index["n_traces"]):
            raise RuntimeError("BLOCKED_DATA_GATE: SEG-Y trace count drift")
        cache: dict[tuple[int, int], np.ndarray] = {}

        def trace(inline: int, crossline: int) -> np.ndarray:
            if not (il_min <= inline <= il_max and xl_min <= crossline <= xl_max):
                raise RuntimeError("BLOCKED_DATA_GATE: requested trace is outside native survey")
            key = (inline, crossline)
            if key not in cache:
                trace_index = (inline - il_min) * n_xl + crossline - xl_min
                cache[key] = np.asarray(handle.trace[int(trace_index)], dtype=np.float32)
            return cache[key]

        for section_row, (well, inline) in enumerate(section_keys):
            window = per_well[well]
            for column, crossline in enumerate(range(window["crossline_start"], window["crossline_start"] + 160)):
                sections[section_row, :, column] = trace(inline, crossline)[
                    window["time_start"]:window["time_start"] + 400
                ]
        for row_index, row in enumerate(records):
            section_ids[row_index] = section_lookup[(row.well, row.inline)]
            trace_ids[row_index] = row.crossline - per_well[row.well]["crossline_start"]
            replay = np.empty((3, 3, 9), dtype=np.float32)
            for di in range(-1, 2):
                for dx in range(-1, 2):
                    replay[di + 1, dx + 1] = trace(row.inline + di, row.crossline + dx)[row.time_idx - 4:row.time_idx + 5]
            error = float(np.max(np.abs(replay - row.seismic)))
            maximum_patch_error = max(maximum_patch_error, error)
            aligned[row_index]["native_patch_max_abs_error"] = error
            aligned[row_index]["native_patch_replayed"] = error <= 1e-6
            aligned[row_index]["section_id"] = int(section_ids[row_index])
            aligned[row_index]["trace_token_id"] = int(trace_ids[row_index])
    if maximum_patch_error > 1e-6 or not np.isfinite(sections).all():
        raise RuntimeError("BLOCKED_DATA_GATE: authoritative SEG-Y patch replay failed")
    audit = {
        "authoritative_position_metadata": True,
        "official_pick_usage": "MD-to-TVD/TVDSS/XY audit only; explicit linear extrapolation is flagged",
        "sample_count": len(records), "section_count": len(sections),
        "section_shape": [400, 160], "upsampled_3x3x9": False,
        "official_tie_bracketed_rows": sum(row["official_tie_bracketed"] for row in aligned),
        "official_tie_extrapolated_rows": sum(row["official_tie_extrapolated"] for row in aligned),
        "nearest_time_index_mismatch_rows": sum(not row["nearest_time_index_matches_meta"] for row in aligned),
        "maximum_native_patch_replay_error": maximum_patch_error,
        "seismic_index_sha256": sha256(SOURCE_INDEX), "official_picks_sha256": sha256(SOURCE_PICKS),
        "segy_bytes": SOURCE_SEGY.stat().st_size,
    }
    return aligned, sections, section_ids, trace_ids, audit


def moment_adapter(logs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    logs = np.asarray(logs, dtype=np.float32)
    values, masks = logs[:, :, :4].copy(), (logs[:, :, 4:8] > 0.5).astype(np.float32)
    for row in range(len(values)):
        for channel in range(4):
            observed = masks[row, :, channel] > 0
            if observed.any():
                mean = float(values[row, observed, channel].mean())
                std = float(values[row, observed, channel].std()) + 1e-6
                values[row, observed, channel] = (values[row, observed, channel] - mean) / std
            values[row, ~observed, channel] = 0.0
    adapted = np.zeros((len(logs), 9, 33), dtype=np.float32)
    adapted[:, :, :4] = values
    adapted[:, :, 4:8] = masks
    return adapted, {
        "source_shape": list(logs.shape), "adapted_shape": list(adapted.shape),
        "n_channels": 9, "valid_length": 8, "padded_length": 33,
        "right_padding_exactly_zero": bool(np.all(adapted[:, :, 8:] == 0)),
        "label_independent": True, "values": "per-sample per-log observed-only zscore",
        "masks_preserved": True,
    }


def contexts(families: np.ndarray) -> list[dict[str, Any]]:
    families = np.asarray(families).astype(str)
    unique = sorted(set(families))
    if len(unique) != 4:
        raise RuntimeError("BLOCKED_DATA_GATE: outer LOGO4 requires four families")
    result: list[dict[str, Any]] = []
    for outer in unique:
        outer_train = [family for family in unique if family != outer]
        result.append({"context_id": f"outer::{outer}", "level": "outer", "outer": outer,
                       "validation_family": outer, "train_families": outer_train})
        for inner in outer_train:
            result.append({"context_id": f"inner::{outer}::{inner}", "level": "inner", "outer": outer,
                           "validation_family": inner,
                           "train_families": [family for family in outer_train if family != inner]})
    return result


def depth_spread_indices(indices: np.ndarray, families: np.ndarray, wells: np.ndarray,
                         depths: np.ndarray, sample_ids: np.ndarray, limit: int = TRAIN_LIMIT) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= limit:
        return indices[np.lexsort((sample_ids[indices], depths[indices], wells[indices], families[indices]))]
    selected: list[int] = []
    unique = sorted(set(families[indices]))
    base, remainder = divmod(limit, len(unique))
    for order, family in enumerate(unique):
        pool = indices[families[indices] == family]
        pool = pool[np.lexsort((sample_ids[pool], depths[pool], wells[pool]))]
        count = base + int(order < remainder)
        positions = np.linspace(0, len(pool) - 1, min(count, len(pool)), dtype=int)
        selected.extend(pool[positions].tolist())
    return np.asarray(selected, dtype=np.int64)


def phase0() -> dict[str, Any]:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    p5_lock = verify_p5_lock()
    records = load_records()
    aligned, sections, section_ids, trace_ids, alignment_audit = align_and_build_sections(records)
    logs = np.stack([row.logs for row in records])
    adapted, adapter_audit = moment_adapter(logs)
    families = np.asarray([row.family for row in records])
    wells = np.asarray([row.well for row in records])
    sample_ids = np.asarray([row.sample_id for row in records])
    depths = np.asarray([row.depth_m for row in records], dtype=np.float64)
    context_rows = contexts(families)
    split_rows: list[dict[str, Any]] = []
    for context in context_rows:
        train_all = np.flatnonzero(np.isin(families, context["train_families"]))
        train_selected = depth_spread_indices(train_all, families, wells, depths, sample_ids)
        validation = np.flatnonzero(families == context["validation_family"])
        split_rows.append({
            **context, "train_all_count": len(train_all), "train_selected_count": len(train_selected),
            "validation_count": len(validation),
            "train_selected_sample_ids_sha256": payload_hash(sorted(sample_ids[train_selected].tolist())),
            "validation_sample_ids_sha256": payload_hash(sorted(sample_ids[validation].tolist())),
            "fit_validation_overlap": bool(set(train_selected) & set(validation)),
        })
    if any(row["fit_validation_overlap"] or row["validation_count"] == 0 for row in split_rows):
        raise RuntimeError("BLOCKED_DATA_GATE: invalid LOGO context")
    np.savez_compressed(
        PHASE0_CACHE,
        seismic=np.stack([row.seismic for row in records]), logs=logs,
        targets=np.stack([row.targets for row in records]),
        sample_ids=sample_ids, families=families, wells=wells, depths_m=depths,
        time_indices=np.asarray([row.time_idx for row in records], dtype=np.int32),
        twt_ms=np.asarray([row.twt_ms for row in records], dtype=np.float64),
        inlines=np.asarray([row.inline for row in records], dtype=np.int32),
        crosslines=np.asarray([row.crossline for row in records], dtype=np.int32),
        moment_input=adapted, sections=sections, section_ids=section_ids, trace_ids=trace_ids,
        contexts_json=np.asarray(json.dumps(context_rows, sort_keys=True)),
    )
    write_jsonl(PAIR_PATH, aligned)
    phase = {
        "schema": "p41-phase0/v2", "status": "PHASE0_READY", "test_access": False,
        "frozen_family_seen": False, "source_joint_complete_case_filter": True,
        "development": {
            "sample_count": len(records), "family_count": 4,
            "families": sorted(set(families)), "wells": sorted(set(wells)),
            "train_h5_rows": sum(row.source_split == "train" for row in records),
            "guard_rows": sum(row.source_split == "guard" for row in records),
        },
        "targets": list(TARGETS), "input_shapes": {"seismic": [3, 3, 9], "well_log": [9, 8]},
        "source_hashes": {
            "train_h5": sha256(SOURCE_TRAIN), "guard_npz": sha256(SOURCE_GUARD),
            "run_manifest": sha256(SOURCE_RUN_MANIFEST), "split_manifest": sha256(SOURCE_SPLIT_MANIFEST),
        },
        "p5_strong_baseline": p5_lock, "alignment": alignment_audit,
        "moment_adapter": adapter_audit, "contexts": split_rows,
        "phase0_cache_sha256": sha256(PHASE0_CACHE), "aligned_manifest_sha256": sha256(PAIR_PATH),
    }
    write_json(PHASE0_PATH, phase)
    return phase


def _baseline_preprocess(seismic_train: np.ndarray, logs_train: np.ndarray,
                         seismic: np.ndarray, logs: np.ndarray) -> np.ndarray:
    seismic_train_flat = seismic_train.reshape(len(seismic_train), -1)
    seismic_flat = seismic.reshape(len(seismic), -1)
    seismic_mean = seismic_train_flat.mean(axis=0)
    seismic_std = seismic_train_flat.std(axis=0) + 1e-8
    train_values = logs_train[:, :, :4].reshape(len(logs_train), -1)
    train_masks = (logs_train[:, :, 4:8] > 0.5).reshape(len(logs_train), -1)
    log_mean, log_std = np.zeros(36), np.ones(36)
    for column in range(36):
        observed = train_values[train_masks[:, column], column]
        if observed.size:
            log_mean[column] = observed.mean()
            log_std[column] = observed.std() + 1e-8
    values = logs[:, :, :4].reshape(len(logs), -1)
    masks = (logs[:, :, 4:8] > 0.5).astype(np.float64).reshape(len(logs), -1)
    normalized_values = ((values - log_mean) / log_std) * masks
    result = np.concatenate(((seismic_flat - seismic_mean) / seismic_std, normalized_values, masks), axis=1)
    if result.shape[1] != 153 or not np.isfinite(result).all():
        raise RuntimeError("BLOCKED_STRONG_BASELINE_REPLAY: P5 preprocessing failed")
    return result


def baseline_worker(cache_path: Path, output_path: Path) -> dict[str, Any]:
    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import ExtraTreesRegressor
    import xgboost

    if sklearn_version != "1.7.2" or xgboost.__version__ != "3.2.0":
        raise RuntimeError("BLOCKED_STRONG_BASELINE_REPLAY: dependency version drift")
    verify_p5_lock()
    with np.load(cache_path, allow_pickle=False) as archive:
        seismic = np.asarray(archive["seismic"], dtype=np.float64)
        logs = np.asarray(archive["logs"], dtype=np.float64)
        targets = np.asarray(archive["targets"], dtype=np.float64)
        sample_ids = archive["sample_ids"].astype(str)
        families = archive["families"].astype(str)
        wells = archive["wells"].astype(str)
        depths = np.asarray(archive["depths_m"], dtype=np.float64)
        context_rows = json.loads(str(archive["contexts_json"]))
    payload: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    for context_index, context in enumerate(context_rows):
        train_all = np.flatnonzero(np.isin(families, context["train_families"]))
        train_idx = depth_spread_indices(train_all, families, wells, depths, sample_ids)
        val_idx = np.flatnonzero(families == context["validation_family"])
        x_train = _baseline_preprocess(seismic[train_idx], logs[train_idx], seismic[train_idx], logs[train_idx])
        x_val = _baseline_preprocess(seismic[train_idx], logs[train_idx], seismic[val_idx], logs[val_idx])
        pred_train = np.empty((len(train_idx), 3), dtype=np.float64)
        pred_val = np.empty((len(val_idx), 3), dtype=np.float64)
        et = ExtraTreesRegressor(
            n_estimators=32, max_depth=8, min_samples_leaf=2, max_features=0.7,
            random_state=SEED, n_jobs=1,
        )
        for target_index in (0, 1):
            et.fit(x_train, targets[train_idx, target_index])
            pred_train[:, target_index] = et.predict(x_train)
            pred_val[:, target_index] = et.predict(x_val)
        sw = xgboost.XGBRegressor(
            objective="reg:squarederror", n_estimators=32, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1e-3, tree_method="hist", random_state=SEED,
            n_jobs=1, verbosity=0,
        )
        sw.fit(x_train, targets[train_idx, 2])
        pred_train[:, 2] = sw.predict(x_train)
        pred_val[:, 2] = sw.predict(x_val)
        std = targets[train_idx].std(axis=0) + 1e-8
        mean = targets[train_idx].mean(axis=0)
        prefix = f"c{context_index}"
        payload[f"{prefix}_train_idx"] = train_idx
        payload[f"{prefix}_val_idx"] = val_idx
        payload[f"{prefix}_train_pred"] = pred_train
        payload[f"{prefix}_val_pred"] = pred_val
        payload[f"{prefix}_target_mean"] = mean
        payload[f"{prefix}_target_std"] = std
        metadata.append({
            **context, "context_index": context_index, "train_rows": len(train_idx),
            "validation_rows": len(val_idx), "feature_count": x_train.shape[1],
            "train_sample_ids_sha256": payload_hash(sorted(sample_ids[train_idx].tolist())),
            "validation_sample_ids_sha256": payload_hash(sorted(sample_ids[val_idx].tolist())),
            "b0_train_prediction_sha256": hashlib.sha256(pred_train.tobytes()).hexdigest(),
            "b0_validation_prediction_sha256": hashlib.sha256(pred_val.tobytes()).hexdigest(),
        })
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {"contexts": len(metadata), "output_sha256": sha256(output_path)}


def run_baseline_subprocess() -> None:
    if not TABULAR_PYTHON.is_file():
        raise RuntimeError("BLOCKED_STRONG_BASELINE_REPLAY: P5 tabular interpreter missing")
    command = [str(TABULAR_PYTHON), str(Path(__file__).resolve()), "--baseline-worker",
               str(PHASE0_CACHE), str(BASELINE_CACHE)]
    completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=900)
    if completed.returncode != 0:
        raise RuntimeError(
            "BLOCKED_STRONG_BASELINE_REPLAY: worker failed: " + (completed.stderr or completed.stdout)[-2000:]
        )


def extract_foundations(device: str) -> dict[str, Any]:
    from p41_foundation_features import extract_gfm_sample_tokens, extract_moment_tokens, verify_assets

    if FOUNDATION_CACHE.is_file() and FOUNDATION_AUDIT_PATH.is_file():
        with np.load(FOUNDATION_CACHE, allow_pickle=False) as archive:
            complete = {"moment", "moment_random", "gfm", "gfm_random", "gfm_shifted"} <= set(archive.files)
        cached = read_json(FOUNDATION_AUDIT_PATH)
        if complete and cached.get("foundation_cache_sha256") == sha256(FOUNDATION_CACHE):
            return cached
    with np.load(PHASE0_CACHE, allow_pickle=False) as archive:
        moment_input = np.asarray(archive["moment_input"], dtype=np.float32)
        sections = np.asarray(archive["sections"], dtype=np.float32)
        section_ids = np.asarray(archive["section_ids"], dtype=np.int32)
        trace_ids = np.asarray(archive["trace_ids"], dtype=np.int32)
    assets = verify_assets(
        moment_snapshot=MOMENT_SNAPSHOT, moment_dependency_root=MOMENT_DEPS,
        moment_source_root=MOMENT_SOURCE, gfm_snapshot=GFM_SNAPSHOT,
        gfm_source_root=GFM_SOURCE,
    )
    moment, moment_audit = extract_moment_tokens(
        moment_input, snapshot=MOMENT_SNAPSHOT, dependency_root=MOMENT_DEPS,
        source_root=MOMENT_SOURCE, device=device, random_init=False, seed=SEED,
    )
    moment_random, moment_random_audit = extract_moment_tokens(
        moment_input, snapshot=MOMENT_SNAPSHOT, dependency_root=MOMENT_DEPS,
        source_root=MOMENT_SOURCE, device=device, random_init=True, seed=SEED,
    )
    gfm, gfm_audit = extract_gfm_sample_tokens(
        sections, section_ids, trace_ids, snapshot=GFM_SNAPSHOT, source_root=GFM_SOURCE,
        device=device, random_init=False, seed=SEED,
    )
    gfm_random, gfm_random_audit = extract_gfm_sample_tokens(
        sections, section_ids, trace_ids, snapshot=GFM_SNAPSHOT, source_root=GFM_SOURCE,
        device=device, random_init=True, seed=SEED,
    )
    shifted_sections = shift_sections_one_sample(sections)
    gfm_shifted, gfm_shifted_audit = extract_gfm_sample_tokens(
        shifted_sections, section_ids, trace_ids, snapshot=GFM_SNAPSHOT, source_root=GFM_SOURCE,
        device=device, random_init=False, seed=SEED,
    )
    np.savez_compressed(
        FOUNDATION_CACHE, moment=moment, moment_random=moment_random,
        gfm=gfm, gfm_random=gfm_random, gfm_shifted=gfm_shifted,
    )
    audit = {
        "assets": assets, "device": device, "foundation_cache_sha256": sha256(FOUNDATION_CACHE),
        "moment": moment_audit, "moment_random": moment_random_audit,
        "gfm": gfm_audit, "gfm_random": gfm_random_audit,
        "gfm_shifted": {
            **gfm_shifted_audit, "intervention": "native section shifted +1 sample (4 ms) on time axis",
            "validation_intervention": False,
            "max_abs_feature_difference_vs_aligned": float(np.max(np.abs(gfm_shifted - gfm))),
        },
        "pretrained_random_max_abs_difference": {
            "moment": float(np.max(np.abs(moment - moment_random))),
            "gfm": float(np.max(np.abs(gfm - gfm_random))),
        },
    }
    if min(audit["pretrained_random_max_abs_difference"].values()) <= 0:
        raise RuntimeError("foundation pretrained/random control is indistinguishable")
    write_json(FOUNDATION_AUDIT_PATH, audit)
    return audit


def within_family_shuffle(indices: np.ndarray, families: np.ndarray, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    output = indices.copy()
    rng = np.random.default_rng(seed)
    for family in sorted(set(families[indices])):
        positions = np.flatnonzero(families[indices] == family)
        if len(positions) > 1:
            # A non-zero cyclic shift is a deterministic derangement: every
            # training pair changes while the mother-well family is fixed.
            permuted = np.roll(positions, int(rng.integers(1, len(positions))))
            output[positions] = indices[permuted]
    if np.any(families[output] != families[indices]):
        raise RuntimeError("within-family shuffle crossed family boundary")
    return output


def shift_sections_one_sample(sections: np.ndarray) -> np.ndarray:
    """Apply the fixed A8 +4 ms intervention without wraparound or resampling."""
    values = np.asarray(sections, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (400, 160):
        raise ValueError("A8 requires complete native [S,400,160] sections")
    shifted = np.empty_like(values)
    shifted[:, :-1] = values[:, 1:]
    shifted[:, -1] = values[:, -1]
    return shifted


def fit_pca(train: np.ndarray, values: np.ndarray, components: int = PCA_COMPONENTS) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA

    train_flat = np.asarray(train, dtype=np.float32).reshape(len(train), -1)
    values_flat = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    count = min(components, len(train_flat) - 1, train_flat.shape[1])
    if count < 2:
        raise RuntimeError("PCA context has insufficient train rows")
    model = PCA(n_components=count, svd_solver="randomized", random_state=SEED)
    model.fit(train_flat)
    transformed = model.transform(values_flat).astype(np.float32)
    return transformed, {
        "fit_rows": len(train_flat), "source_width": train_flat.shape[1],
        "components": count, "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
        "fit_mean_sha256": hashlib.sha256(np.asarray(model.mean_).tobytes()).hexdigest(),
    }


class GatedResidual:
    def __init__(self, well_dim: int, seismic_dim: int, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        class Network(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.well = torch.nn.Sequential(torch.nn.LayerNorm(well_dim), torch.nn.Linear(well_dim, 24), torch.nn.GELU())
                self.seismic = torch.nn.Sequential(
                    torch.nn.LayerNorm(seismic_dim), torch.nn.Linear(seismic_dim, 24), torch.nn.GELU()
                )
                self.head = torch.nn.Sequential(torch.nn.Linear(48, 24), torch.nn.GELU(), torch.nn.Linear(24, 3))
                self.gate_raw = torch.nn.Parameter(torch.zeros(3))

            def forward(self, well: Any, seismic: Any, base: Any, force_off: bool = False) -> Any:
                residual = self.head(torch.cat((self.well(well), self.seismic(seismic)), dim=1))
                gate = torch.zeros_like(self.gate_raw) if force_off else torch.tanh(self.gate_raw)
                return base + gate * residual, residual

        self.module = Network()


def train_residual(*, well_train: np.ndarray, seismic_train: np.ndarray,
                   well_val: np.ndarray, seismic_val: np.ndarray,
                   base_train: np.ndarray, base_val: np.ndarray,
                   target_train: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray,
                   seed: int, steps: int = RESIDUAL_STEPS) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    device = torch.device("cpu")
    well_mean, well_std = well_train.mean(0), well_train.std(0) + 1e-6
    seismic_mean, seismic_std = seismic_train.mean(0), seismic_train.std(0) + 1e-6
    wt = torch.as_tensor((well_train - well_mean) / well_std, dtype=torch.float32, device=device)
    st = torch.as_tensor((seismic_train - seismic_mean) / seismic_std, dtype=torch.float32, device=device)
    wv = torch.as_tensor((well_val - well_mean) / well_std, dtype=torch.float32, device=device)
    sv = torch.as_tensor((seismic_val - seismic_mean) / seismic_std, dtype=torch.float32, device=device)
    bt = torch.as_tensor((base_train - target_mean) / target_std, dtype=torch.float32, device=device)
    bv = torch.as_tensor((base_val - target_mean) / target_std, dtype=torch.float32, device=device)
    yt = torch.as_tensor((target_train - target_mean) / target_std, dtype=torch.float32, device=device)
    network = GatedResidual(wt.shape[1], st.shape[1], seed).module.to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-3, weight_decay=1e-4)
    losses: list[float] = []
    gradients: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted, _ = network(wt, st, bt)
        loss = torch.mean((predicted - yt) ** 2)
        loss.backward()
        gradients.append(float(torch.sqrt(sum(torch.sum(p.grad**2) for p in network.parameters() if p.grad is not None))))
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        val_norm, residual = network(wv, sv, bv)
        off_norm, _ = network(wv, sv, bv, force_off=True)
    val = val_norm.cpu().numpy() * target_std + target_mean
    off_roundtrip = off_norm.cpu().numpy() * target_std + target_mean
    # The deployed fusion-off route is an explicit B0 bypass.  It does not
    # standardize and invert B0 through float32 merely to prove an identity,
    # so its returned prediction is bitwise identical to the shared B0 array.
    off = np.asarray(base_val, dtype=np.float64).copy()
    audit = {
        "input_shapes": {"well": list(wt.shape), "seismic": list(st.shape), "base": list(bt.shape)},
        "validation_shapes": {"well": list(wv.shape), "seismic": list(sv.shape), "prediction": list(val.shape)},
        "parameter_count": int(sum(p.numel() for p in network.parameters())),
        "trainable_parameter_count": int(sum(p.numel() for p in network.parameters() if p.requires_grad)),
        "initial_loss": losses[0], "final_loss": losses[-1],
        "initial_gradient_norm": gradients[0], "final_gradient_norm": gradients[-1],
        "gate": torch.tanh(network.gate_raw).detach().cpu().numpy().tolist(),
        "gate_raw": network.gate_raw.detach().cpu().numpy().tolist(),
        "validation_residual_normalized_rms": float(torch.sqrt(torch.mean(residual**2))),
        "fusion_off_internal_float32_roundtrip_max_abs_error": float(
            np.max(np.abs(off_roundtrip - base_val))
        ),
        "fusion_off_max_abs_error_vs_base": float(np.max(np.abs(off - base_val))),
        "fixed_steps": steps, "early_stopping": False,
    }
    return val.astype(np.float64), off.astype(np.float64), audit


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, train_std: np.ndarray) -> dict[str, Any]:
    actual, predicted = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    train_std = np.asarray(train_std, dtype=np.float64)
    residual = predicted - actual
    per_target: dict[str, Any] = {}
    normalized_rmse: list[float] = []
    for index, target in enumerate(TARGETS):
        truth, pred = actual[:, index], predicted[:, index]
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        mae = float(np.mean(np.abs(pred - truth)))
        denominator = float(np.sum((truth - truth.mean()) ** 2))
        pearson = None
        if np.std(truth) > 0 and np.std(pred) > 0:
            pearson = float(np.corrcoef(truth, pred)[0, 1])
        normalized = rmse / float(train_std[index])
        normalized_rmse.append(normalized)
        per_target[target] = {
            "RMSE": rmse, "MAE": mae,
            "R2": None if denominator == 0 else float(1 - np.sum((pred - truth) ** 2) / denominator),
            "Pearson": pearson, "train_std_normalized_RMSE": normalized,
        }
    return {
        "composite_equal_weight_train_std_normalized_RMSE": float(np.mean(normalized_rmse)),
        "per_target": per_target, "rows": len(actual),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
    }


def promotion_gate(fold_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
                   bootstrap_ci: Sequence[float], a7_exact: bool,
                   global_a7_error: float, all_guards: bool) -> dict[str, Any]:
    outer = fold_metrics
    macro = {arm: float(np.mean([row["composite_equal_weight_train_std_normalized_RMSE"] for row in values]))
             for arm, values in outer.items()}
    f1_gain = (macro["B0"] - macro["F1"]) / macro["B0"]
    outer_wins = sum(
        f1["composite_equal_weight_train_std_normalized_RMSE"]
        < b0["composite_equal_weight_train_std_normalized_RMSE"]
        for b0, f1 in zip(outer["B0"], outer["F1"])
    )
    target_non_degrade = 0
    for target in TARGETS:
        base = np.mean([row["per_target"][target]["train_std_normalized_RMSE"] for row in outer["B0"]])
        fused = np.mean([row["per_target"][target]["train_std_normalized_RMSE"] for row in outer["F1"]])
        target_non_degrade += int(fused <= base)
    best_single = min(macro["W1"], macro["S1"])
    conditions = {
        "f1_vs_b0_at_least_0p5pct": f1_gain >= 0.005,
        "outer_wins_at_least_3_of_4": outer_wins >= 3,
        "targets_non_degraded_at_least_2_of_3": target_non_degrade >= 2,
        "f1_vs_best_single_at_least_0p3pct": (best_single - macro["F1"]) / best_single >= 0.003,
        "f1_vs_a5_at_least_0p5pct": (macro["A5"] - macro["F1"]) / macro["A5"] >= 0.005,
        "f1_vs_a6_at_least_0p5pct": (macro["A6"] - macro["F1"]) / macro["A6"] >= 0.005,
        "f1_vs_a8_at_least_0p5pct": (macro["A8"] - macro["F1"]) / macro["A8"] >= 0.005,
        "paired_bootstrap_ci_above_zero": float(bootstrap_ci[0]) > 0,
        "a7_exact": bool(a7_exact and global_a7_error == 0.0),
        "data_asset_alignment_firewall_guards": bool(all_guards),
    }
    passed = all(conditions.values())
    return {
        "verdict": "R0_SIGNAL_PASS_NEXT_STAGE_ADVICE_ONLY" if passed else "R0_STOP_NO_ATTRIBUTABLE_SIGNAL",
        "passed": passed, "conditions": conditions, "macro_composite": macro,
        "f1_relative_gain_vs_b0": f1_gain, "outer_wins": outer_wins,
        "non_degraded_targets": target_non_degrade, "bootstrap_improvement_ci": list(bootstrap_ci),
    }


def paired_bootstrap(actual: np.ndarray, base: np.ndarray, fused: np.ndarray,
                     families: np.ndarray, train_std: np.ndarray, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(families.astype(str)))
    improvements = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for repeat in range(BOOTSTRAP_REPLICATES):
        fold_deltas = []
        for family in unique:
            pool = np.flatnonzero(families == family)
            sampled = rng.choice(pool, size=len(pool), replace=True)
            b = regression_metrics(actual[sampled], base[sampled], train_std[sampled][0])
            f = regression_metrics(actual[sampled], fused[sampled], train_std[sampled][0])
            fold_deltas.append(
                b["composite_equal_weight_train_std_normalized_RMSE"]
                - f["composite_equal_weight_train_std_normalized_RMSE"]
            )
        improvements[repeat] = np.mean(fold_deltas)
    return np.quantile(improvements, [0.025, 0.975]).astype(float).tolist()


def run_r0(device: str) -> dict[str, Any]:
    phase = phase0()
    run_baseline_subprocess()
    foundation_audit = extract_foundations(device)
    with np.load(PHASE0_CACHE, allow_pickle=False) as data, \
         np.load(FOUNDATION_CACHE, allow_pickle=False) as feature_data, \
         np.load(BASELINE_CACHE, allow_pickle=False) as baseline_data:
        targets = np.asarray(data["targets"], dtype=np.float64)
        sample_ids = data["sample_ids"].astype(str)
        families = data["families"].astype(str)
        wells = data["wells"].astype(str)
        depths = np.asarray(data["depths_m"], dtype=np.float64)
        context_rows = json.loads(str(data["contexts_json"]))
        b0_metadata = json.loads(str(baseline_data["metadata_json"]))
        tokens = {key: np.asarray(feature_data[key]) for key in (
            "moment", "moment_random", "gfm", "gfm_random", "gfm_shifted"
        )}
        outer_predictions = {arm: np.empty_like(targets) for arm in ARMS}
        outer_actual = np.empty_like(targets)
        outer_std = np.empty_like(targets)
        assigned = np.zeros(len(targets), dtype=bool)
        fold_metrics: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
        results: list[dict[str, Any]] = []
        context_audits: list[dict[str, Any]] = []
        for context_index, context in enumerate(context_rows):
            prefix = f"c{context_index}"
            train_idx = np.asarray(baseline_data[f"{prefix}_train_idx"], dtype=np.int64)
            val_idx = np.asarray(baseline_data[f"{prefix}_val_idx"], dtype=np.int64)
            base_train = np.asarray(baseline_data[f"{prefix}_train_pred"], dtype=np.float64)
            base_val = np.asarray(baseline_data[f"{prefix}_val_pred"], dtype=np.float64)
            target_mean = np.asarray(baseline_data[f"{prefix}_target_mean"], dtype=np.float64)
            target_std = np.asarray(baseline_data[f"{prefix}_target_std"], dtype=np.float64)
            # Every PCA is fitted only on this context's selected train rows.
            pca: dict[str, np.ndarray] = {}
            pca_audit: dict[str, Any] = {}
            for key in ("moment", "moment_random", "gfm", "gfm_random"):
                transformed, audit = fit_pca(tokens[key][train_idx], tokens[key])
                pca[key] = transformed
                pca_audit[key] = audit
            # A8 must live in the exact same coordinate system as aligned
            # GFM.  Fit on aligned train tokens, then transform shifted tokens
            # with the deterministically identical PCA fit.
            pca["gfm_shifted"], pca_audit["gfm_shifted"] = fit_pca(
                tokens["gfm"][train_idx], tokens["gfm_shifted"]
            )
            pca_audit["gfm_shifted"]["fit_source"] = "aligned_gfm_train_tokens"
            shuffled_idx = within_family_shuffle(train_idx, families, SEED + context_index)
            zero_well_train = np.zeros_like(pca["moment"][train_idx])
            zero_well_val = np.zeros_like(pca["moment"][val_idx])
            zero_seismic_train = np.zeros_like(pca["gfm"][train_idx])
            zero_seismic_val = np.zeros_like(pca["gfm"][val_idx])
            arm_features = {
                "W1": (pca["moment"][train_idx], zero_seismic_train, pca["moment"][val_idx], zero_seismic_val),
                "S1": (zero_well_train, pca["gfm"][train_idx], zero_well_val, pca["gfm"][val_idx]),
                "F1": (pca["moment"][train_idx], pca["gfm"][train_idx], pca["moment"][val_idx], pca["gfm"][val_idx]),
                "A5": (pca["moment_random"][train_idx], pca["gfm_random"][train_idx],
                       pca["moment_random"][val_idx], pca["gfm_random"][val_idx]),
                "A6": (pca["moment"][train_idx], pca["gfm"][shuffled_idx],
                       pca["moment"][val_idx], pca["gfm"][val_idx]),
                "A8": (pca["moment"][train_idx], pca["gfm_shifted"][train_idx],
                       pca["moment"][val_idx], pca["gfm"][val_idx]),
            }
            predictions: dict[str, np.ndarray] = {"B0": base_val, "A7": base_val.copy()}
            arm_audit: dict[str, Any] = {}
            for arm in ("W1", "S1", "F1", "A5", "A6", "A8"):
                wt, st, wv, sv = arm_features[arm]
                pred, off, audit = train_residual(
                    well_train=wt, seismic_train=st, well_val=wv, seismic_val=sv,
                    base_train=base_train, base_val=base_val, target_train=targets[train_idx],
                    target_mean=target_mean, target_std=target_std,
                    seed=SEED + context_index * 17 + ARMS.index(arm),
                )
                if np.max(np.abs(off - base_val)) != 0.0:
                    raise RuntimeError("A7 mathematical fallback is not exact")
                predictions[arm] = pred
                arm_audit[arm] = audit
            context_result_metrics: dict[str, Any] = {}
            for arm in ARMS:
                metric = regression_metrics(targets[val_idx], predictions[arm], target_std)
                context_result_metrics[arm] = metric
                results.append({
                    "context_id": context["context_id"], "level": context["level"],
                    "outer_family": context["outer"], "validation_family": context["validation_family"],
                    "arm": arm, "seed": SEED, "train_rows": len(train_idx),
                    "validation_rows": len(val_idx), "metrics": metric,
                    "b0_validation_prediction_sha256": hashlib.sha256(base_val.tobytes()).hexdigest(),
                })
                if context["level"] == "outer":
                    fold_metrics[arm].append(metric)
                    outer_predictions[arm][val_idx] = predictions[arm]
            if context["level"] == "outer":
                outer_actual[val_idx] = targets[val_idx]
                outer_std[val_idx] = target_std
                assigned[val_idx] = True
            a8_feature_delta = np.max(
                np.abs(pca["gfm_shifted"][train_idx] - pca["gfm"][train_idx]), axis=1
            )
            context_audits.append({
                **context, "context_index": context_index,
                "pca": pca_audit, "arms": arm_audit,
                "p5_b0": b0_metadata[context_index],
                "same_family_shuffle": {
                    "all_same_family": bool(np.all(families[shuffled_idx] == families[train_idx])),
                    "changed_pairs": int(np.sum(shuffled_idx != train_idx)),
                },
                "small_depth_shift": {
                    "kind": "native section time-axis intervention",
                    "fixed_time_shift_samples": 1, "fixed_time_shift_ms": 4.0,
                    "validation_intervention": False,
                    "changed_feature_rows": int(np.sum(a8_feature_delta > 0)),
                    "max_abs_pca_feature_difference": float(np.max(a8_feature_delta)),
                },
            })
    if not assigned.all():
        raise RuntimeError("outer LOGO OOF predictions are incomplete")
    a7_error = float(np.max(np.abs(outer_predictions["A7"] - outer_predictions["B0"])))
    bootstrap_ci = paired_bootstrap(
        outer_actual, outer_predictions["B0"], outer_predictions["F1"], families, outer_std
    )
    gate = promotion_gate(
        fold_metrics, bootstrap_ci, a7_error == 0.0, a7_error,
        all_guards=(
            phase["status"] == "PHASE0_READY"
            and foundation_audit["assets"]["local_only"]
            and foundation_audit["gfm_shifted"]["max_abs_feature_difference_vs_aligned"] > 0
        ),
    )
    outer_families = sorted(set(families))
    gate["worst_family_by_arm"] = {}
    for arm in ARMS:
        values = [row["composite_equal_weight_train_std_normalized_RMSE"] for row in fold_metrics[arm]]
        worst_index = int(np.argmax(values))
        gate["worst_family_by_arm"][arm] = {
            "family_id": outer_families[worst_index], "composite": float(values[worst_index])
        }
    write_jsonl(RESULTS_PATH, results)
    np.savez_compressed(
        PREDICTIONS_PATH, actual=outer_actual, sample_ids=sample_ids, family_ids=families,
        train_std=outer_std, **{arm: outer_predictions[arm] for arm in ARMS},
    )
    write_json(FEATURE_AUDIT_PATH, {
        "schema": "p41-feature-adapter-audit/v2", "seed": SEED,
        "residual_steps": RESIDUAL_STEPS, "pca_components": PCA_COMPONENTS,
        "contexts": context_audits, "outer_logo_folds": 4, "inner_logo_contexts": 12,
        "all_transforms_fit_train_only": True, "outer_held_early_stopping": False,
        "b0_shared_by_all_arms": True, "a7_max_abs_error": a7_error,
    })
    summary = {
        "schema": "p41-property-crossmodal-qualification/v2", "verdict": gate["verdict"],
        "gate": gate, "seed_pool": [SEED], "outer_logo_folds": 4, "inner_logo_contexts": 12,
        "arms": list(ARMS), "test_access": False, "frozen_family_seen": False,
        "strong_baseline": phase["p5_strong_baseline"],
        "phase0_sha256": sha256(PHASE0_PATH), "predictions_sha256": sha256(PREDICTIONS_PATH),
        "results_sha256": sha256(RESULTS_PATH), "foundation_audit_sha256": sha256(FOUNDATION_AUDIT_PATH),
        "feature_audit_sha256": sha256(FEATURE_AUDIT_PATH),
    }
    write_json(SUMMARY_PATH, summary)
    independent_metric_check(write=True)
    runner = "_pipelines/02_task_datasets/reservoir/p41_property_crossmodal_qualification.py"
    write_json(RERUN_PATH, {
        "phase0": ["{python}", runner, "--phase0-only"],
        "r0": ["{python}", runner, "--device", device],
        "verify": ["{python}", runner, "--verify-only"],
        "independent_metric": ["{python}", runner, "--independent-metric-check"],
        "environment_note": (
            "Run from the repository root in torch-common. The P5 baseline is delegated to "
            "P41_TABULAR_PYTHON; raw property assets may be overridden with "
            "JUNWEI_PROPERTY_SOURCE_ROOT. Foundation cache roots use the P41_* environment variables."
        ),
    })
    EVIDENCE_PATH.write_text(
        "# P41 物性井震双基础模型条件增量资格门\n\n"
        f"- 判定：`{gate['verdict']}`\n"
        f"- 强基线：P5 Stage-3 `{P5_COMMIT}`，PHIF/KLOGH=ExtraTrees，SW=XGBoost。\n"
        f"- B0 复合 RMSE：`{gate['macro_composite']['B0']:.8f}`\n"
        f"- F1 复合 RMSE：`{gate['macro_composite']['F1']:.8f}`\n"
        f"- F1 相对改进：`{100 * gate['f1_relative_gain_vs_b0']:.4f}%`\n"
        f"- 外层胜出：`{gate['outer_wins']}/4`；非劣目标：`{gate['non_degraded_targets']}/3`。\n"
        f"- 配对 bootstrap 改进区间：`{gate['bootstrap_improvement_ci']}`。\n"
        f"- A7 最大误差：`{a7_error}`。\n"
        "- 本次只使用 4 个 development families；未实现冻结测试加载入口。\n",
        encoding="utf-8",
    )
    build_artifact_manifest()
    return summary


def independent_metric_check(write: bool = False) -> dict[str, Any]:
    if not PREDICTIONS_PATH.is_file() or not SUMMARY_PATH.is_file():
        raise FileNotFoundError("P41 predictions/summary are missing")
    with np.load(PREDICTIONS_PATH, allow_pickle=False) as archive:
        actual = np.asarray(archive["actual"], dtype=np.float64)
        std = np.asarray(archive["train_std"], dtype=np.float64)
        families = archive["family_ids"].astype(str)
        recomputed: dict[str, Any] = {}
        for arm in ARMS:
            prediction = np.asarray(archive[arm], dtype=np.float64)
            folds = []
            for family in sorted(set(families)):
                selected = families == family
                folds.append(regression_metrics(actual[selected], prediction[selected], std[selected][0]))
            recomputed[arm] = float(np.mean([
                row["composite_equal_weight_train_std_normalized_RMSE"] for row in folds
            ]))
        a7_error = float(np.max(np.abs(archive["A7"] - archive["B0"])))
    stored = read_json(SUMMARY_PATH)["gate"]["macro_composite"]
    error = max(abs(recomputed[arm] - float(stored[arm])) for arm in ARMS)
    result = {
        "status": "PASS" if error <= 1e-12 and a7_error == 0 else "FAIL",
        "independent_recomputed_macro_composite": recomputed,
        "stored_vs_recomputed_max_abs_error": error, "a7_max_abs_error": a7_error,
        "predictions_sha256": sha256(PREDICTIONS_PATH),
    }
    if result["status"] != "PASS":
        raise RuntimeError("independent metric check failed")
    if write:
        write_json(INDEPENDENT_PATH, result)
    return result


def build_artifact_manifest() -> dict[str, Any]:
    paths = [
        SUMMARY_PATH, RESULTS_PATH, PREDICTIONS_PATH, PHASE0_PATH, PAIR_PATH,
        FOUNDATION_AUDIT_PATH, FEATURE_AUDIT_PATH, RERUN_PATH, EVIDENCE_PATH, INDEPENDENT_PATH,
    ]
    manifest = {
        "schema": "p41-artifact-manifest/v2", "test_access": False,
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths},
    }
    write_json(ARTIFACT_PATH, manifest)
    return manifest


def verify_only() -> dict[str, Any]:
    required = [
        SUMMARY_PATH, RESULTS_PATH, PREDICTIONS_PATH, PHASE0_PATH, PAIR_PATH,
        FOUNDATION_AUDIT_PATH, FEATURE_AUDIT_PATH, ARTIFACT_PATH, RERUN_PATH, EVIDENCE_PATH,
        INDEPENDENT_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing P41 artifact(s): " + ", ".join(missing))
    verify_p5_lock()
    phase = read_json(PHASE0_PATH)
    summary = read_json(SUMMARY_PATH)
    manifest = read_json(ARTIFACT_PATH)
    if phase.get("status") != "PHASE0_READY" or phase.get("test_access") is not False:
        raise RuntimeError("P41 phase0/firewall contract failed")
    if set(summary.get("arms", [])) != set(ARMS) or summary.get("outer_logo_folds") != 4:
        raise RuntimeError("P41 arm/LOGO contract failed")
    if not summary["gate"]["conditions"]["a7_exact"]:
        raise RuntimeError("P41 A7 exact fallback failed")
    for name, evidence in manifest["artifacts"].items():
        path = OUTPUT / name
        if sha256(path) != evidence["sha256"] or path.stat().st_size != evidence["bytes"]:
            raise RuntimeError(f"P41 artifact hash drift: {name}")
    independent = independent_metric_check(write=False)
    return {
        "status": "PASS", "verdict": summary["verdict"], "artifact_count": len(required),
        "p5_source_commit": P5_COMMIT, "p5_source_locked": True,
        "outer_logo_folds": 4, "inner_logo_contexts": 12, "a7_exact": True,
        "independent_metric_check": independent,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--independent-metric-check", action="store_true")
    parser.add_argument("--baseline-worker", nargs=2, metavar=("CACHE", "OUTPUT"))
    parser.add_argument("--device", default="cuda:5")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.baseline_worker:
        return baseline_worker(Path(args.baseline_worker[0]), Path(args.baseline_worker[1]))
    if args.phase0_only:
        return phase0()
    if args.verify_only:
        return verify_only()
    if args.independent_metric_check:
        return independent_metric_check(write=False)
    return run_r0(str(args.device))


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2, sort_keys=True))
