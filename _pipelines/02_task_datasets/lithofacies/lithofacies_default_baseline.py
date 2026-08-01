#!/usr/bin/env python3
"""Adopt and verify the P17 XGBoost configuration as the default baseline.

This runner consumes only the immutable development LOGO4 batch. It retrains
the archived 2/0.2/40 comparator and the canonical adapter default over four
folds and three frozen seeds, then writes portable evidence in a new directory.
No command accepts a dataset directory, HDF5 file, holdout, or test input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _models.lithofacies.p5_adapter_common import (  # noqa: E402
    multimodal_numpy_features,
)
from _models.lithofacies.xgboost_multisoftprob_window import (  # noqa: E402
    DEFAULT_BASELINE_ETA,
    DEFAULT_BASELINE_MAX_DEPTH,
    DEFAULT_BASELINE_ROUNDS,
    XGBoostWindowAdapter,
)
from lithofacies_p5_stage3 import (  # noqa: E402
    FOLD_IDS,
    REPEAT_SEEDS,
    _fold_arrays,
    load_stage3_batch,
)
from p4_contract import classification_metrics_from_logits  # noqa: E402


SCHEMA_VERSION = "lithofacies-default-baseline-adoption/v1"
RESULT_SCHEMA = "lithofacies-default-baseline-cell/v1"
MANIFEST_SCHEMA = "lithofacies-default-baseline-manifest/v1"
EXPECTED_SPLIT_HASH = (
    "a06375429f9e9cf380fb5cdebd7d0cb7b25d7a13d29522b8e2420f4dae1b4555"
)
EXPECTED_P17_LEGACY_MEAN = 0.19493770207563763
EXPECTED_P17_DEFAULT_MEAN = 0.2133487970485067
MATERIALITY_THRESHOLD = 0.005
NUM_CLASSES = 9
VARIANTS = ("legacy_depth2_eta02_rounds40", "default_depth3_eta01_rounds60")
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "default_baseline"
FORBIDDEN_PATH_MARKERS = ("holdout", "frozen", "test.h5")
P17_ORIGINAL_HASHES = {
    "_outputs/agent_chapter/evidence.md": (
        "05664d19f171c0bcb252a435d47315d94e513a844c20bf12b99aac25d7287a18"
    ),
    "_outputs/agent_chapter/summary.json": (
        "182e2f014977baea68ab335678f766d9fd1ec409132b6ad686ea3ad26e85f37f"
    ),
    "_outputs/agent_chapter/results.jsonl": (
        "6a8b7e20431a2557bdb3463dc960cfafdb1cd1771d291179678e19f0b9126175"
    ),
    "_outputs/agent_chapter/artifact_manifest.json": (
        "bc75f1857cc772b18b8578ed8f6d3b63d3b3498787c74a2ff92594442d9f2348"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ensure_development_only_paths(paths: Iterable[Path]) -> None:
    """Reject holdout/test-like inputs before opening anything."""
    for raw_path in paths:
        lowered = [part.lower().replace("-", "_") for part in Path(raw_path).parts]
        if any(marker in part for part in lowered for marker in FORBIDDEN_PATH_MARKERS):
            raise ValueError(f"forbidden holdout/test path: {raw_path}")


def _track_owned_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(TRACK_DIR.resolve())
    except ValueError as exc:
        raise ValueError("output directory must remain under the lithofacies track") from exc
    return resolved


def verify_p17_originals() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in P17_ORIGINAL_HASHES.items():
        path = TRACK_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"P17 original is missing: {path}")
        digest = _sha256(path)
        if digest != expected:
            raise RuntimeError(f"P17 original changed: {path}")
        observed[relative] = digest
    return observed


def _fit(
    *,
    train_well: np.ndarray,
    train_seismic: np.ndarray,
    train_labels: np.ndarray,
    validation_well: np.ndarray,
    validation_seismic: np.ndarray,
    class_counts: np.ndarray,
    seed: int,
    variant: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if variant == VARIANTS[0]:
        model = XGBoostWindowAdapter(
            rounds=40,
            max_depth=2,
            eta=0.2,
            seed=seed,
        )
    elif variant == VARIANTS[1]:
        model = XGBoostWindowAdapter(seed=seed)
    else:
        raise ValueError(f"unknown baseline variant: {variant}")
    model.fit_stage1(
        train_well,
        train_seismic,
        train_labels,
        class_counts=class_counts,
    )
    logits = model.predict_logits(validation_well, validation_seismic)
    config = {
        "max_depth": model.max_depth,
        "eta": model.eta,
        "rounds": model.rounds,
        "seed": model.seed,
        "weight_exponent": 0.5,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "feature_count": int(
            multimodal_numpy_features(train_well, train_seismic).shape[1]
        ),
    }
    return logits, config


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        cells = [row for row in rows if row["variant"] == variant]
        values = np.asarray(
            [row["metrics"]["fixed_schema_macro_f1"] for row in cells],
            dtype=np.float64,
        )
        variants[variant] = {
            "cells": len(cells),
            "mean_fixed_schema_macro_f1": float(values.mean()),
            "std_fixed_schema_macro_f1": float(values.std(ddof=1)),
            "fold_means": {
                str(fold_id): float(
                    np.mean(
                        [
                            row["metrics"]["fixed_schema_macro_f1"]
                            for row in cells
                            if int(row["fold_id"]) == fold_id
                        ]
                    )
                )
                for fold_id in FOLD_IDS
            },
        }
    legacy = variants[VARIANTS[0]]["mean_fixed_schema_macro_f1"]
    default = variants[VARIANTS[1]]["mean_fixed_schema_macro_f1"]
    paired = {
        (int(row["fold_id"]), int(row["repeat_id"])): float(
            row["metrics"]["fixed_schema_macro_f1"]
        )
        for row in rows
        if row["variant"] == VARIANTS[0]
    }
    wins = sum(
        float(row["metrics"]["fixed_schema_macro_f1"])
        > paired[(int(row["fold_id"]), int(row["repeat_id"]))]
        for row in rows
        if row["variant"] == VARIANTS[1]
    )
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for per_class in row["metrics"]["per_class"]
        for value in (
            per_class["precision"],
            per_class["recall"],
            per_class["f1"],
            per_class["iou"],
        )
    )
    tolerance = 1e-12
    matches_p17 = {
        "legacy": abs(legacy - EXPECTED_P17_LEGACY_MEAN) <= tolerance,
        "default": abs(default - EXPECTED_P17_DEFAULT_MEAN) <= tolerance,
    }
    delta = default - legacy
    accepted = (
        len(rows) == 24
        and wins == 12
        and delta >= MATERIALITY_THRESHOLD
        and finite
    )
    return {
        "variants": variants,
        "comparison": {
            "default_minus_legacy": delta,
            "default_wins": wins,
            "paired_cells": 12,
            "materiality_threshold": MATERIALITY_THRESHOLD,
            "all_class_metrics_finite": finite,
            "matches_p17_at_1e_12": matches_p17,
        },
        "decision": {
            "status": "ACCEPT_AS_DEFAULT" if accepted else "DO_NOT_ADOPT",
            "default_variant": VARIANTS[1] if accepted else VARIANTS[0],
            "attribution": "xgboost_hyperparameter_tuning_only",
            "moment_or_large_model_contribution": False,
        },
    }


def _evidence(summary: Mapping[str, Any]) -> str:
    legacy = summary["variants"][VARIANTS[0]]
    default = summary["variants"][VARIANTS[1]]
    comparison = summary["comparison"]
    matches = comparison["matches_p17_at_1e_12"]
    agreement = (
        "The live recomputation matches the P17 recorded means within `1e-12`."
        if all(matches.values())
        else "The live recomputation differs from P17; the exact observed values above govern."
    )
    lines = [
        "# Lithofacies default-baseline adoption evidence",
        "",
        "## Outcome",
        "",
        f"Decision: **{summary['decision']['status']}**.",
        "",
        "| Configuration | LOGO4 x 3 cells | Fixed-schema Macro-F1 | Delta | Wins |",
        "|---|---:|---:|---:|---:|",
        (
            "| archived depth=2, eta=0.2, rounds=40 | "
            f"{legacy['cells']} | {legacy['mean_fixed_schema_macro_f1']:.12f} "
            "| - | - |"
        ),
        (
            "| default depth=3, eta=0.1, rounds=60 | "
            f"{default['cells']} | {default['mean_fixed_schema_macro_f1']:.12f} "
            f"| {comparison['default_minus_legacy']:+.12f} | "
            f"{comparison['default_wins']}/12 |"
        ),
        "",
        agreement,
        (
            "The fixed metric is the arithmetic mean of nine schema-class F1 "
            "values; absent validation classes contribute zero. The split is "
            "the immutable four-development-family LOGO4 contract, repeated "
            "with seeds `1867973658`, `2137841944`, and `3902865753`."
        ),
        "",
        "## Attribution and firewall",
        "",
        (
            "This adoption is solely an XGBoost hyperparameter change. It does "
            "not use MOMENT, pretrained embeddings, a large model, or an LLM "
            "at training/inference time, so no improvement is attributed to "
            "MOMENT or any large model."
        ),
        "",
        (
            "Only the prebuilt development LOGO4 batch was opened. "
            "`frozen_test_accessed=false`, `known_holdout_accessed=false`, and "
            "no `test.h5` path is accepted by the runner. P17 agent-chapter "
            "artifacts were verified against frozen SHA-256 values and were "
            "not rewritten."
        ),
        "",
        "## Fold means",
        "",
        "| Fold | Archived | Default |",
        "|---:|---:|---:|",
    ]
    for fold_id in FOLD_IDS:
        lines.append(
            f"| {fold_id} | {legacy['fold_means'][str(fold_id)]:.12f} | "
            f"{default['fold_means'][str(fold_id)]:.12f} |"
        )
    lines.extend(
        [
            "",
            (
                "The three nominal seeds are deterministic duplicates here "
                "because `subsample=1.0` and `colsample_bytree=1.0`; 12/12 "
                "therefore represents four distinct family outcomes, each "
                "repeated three times."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(development_batch: Path, output_dir: Path) -> dict[str, Any]:
    ensure_development_only_paths((development_batch,))
    output_dir = _track_owned_output(output_dir)
    originals = verify_p17_originals()
    arrays, manifest = load_stage3_batch(development_batch)
    if (
        manifest.get("split_hash") != EXPECTED_SPLIT_HASH
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("test_metrics_used") is not False
    ):
        raise RuntimeError("development batch violates the frozen LOGO4 contract")
    rows: list[dict[str, Any]] = []
    for fold_id in FOLD_IDS:
        fold = _fold_arrays(arrays, fold_id)
        train_well = np.asarray(fold["p_train_well"], dtype=np.float32)
        train_seismic = np.asarray(fold["p_train_seismic"], dtype=np.float32)
        train_labels = np.asarray(fold["p_train_labels"], dtype=np.int64)
        validation_well = np.asarray(fold["p_validation_well"], dtype=np.float32)
        validation_seismic = np.asarray(
            fold["p_validation_seismic"], dtype=np.float32
        )
        validation_labels = np.asarray(
            fold["p_validation_labels"], dtype=np.int64
        )
        counts = np.asarray(fold["class_counts"], dtype=np.int64)
        for repeat_id, seed in enumerate(REPEAT_SEEDS):
            for variant in VARIANTS:
                logits, config = _fit(
                    train_well=train_well,
                    train_seismic=train_seismic,
                    train_labels=train_labels,
                    validation_well=validation_well,
                    validation_seismic=validation_seismic,
                    class_counts=counts,
                    seed=int(seed),
                    variant=variant,
                )
                metrics = classification_metrics_from_logits(
                    validation_labels.tolist(), logits
                )
                rows.append(
                    {
                        "schema_version": RESULT_SCHEMA,
                        "track_id": "lithofacies",
                        "task_id": "gm09_genetic_facies_9class",
                        "variant": variant,
                        "fold_id": fold_id,
                        "repeat_id": repeat_id,
                        "seed": int(seed),
                        "split_hash": EXPECTED_SPLIT_HASH,
                        "training": config,
                        "metrics": metrics,
                        "development_only": True,
                        "frozen_test_accessed": False,
                        "known_holdout_accessed": False,
                    }
                )
    aggregation = summarize_rows(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": "fixed_LOGO4_three_seed_development_only",
        "metric": "fixed_schema_macro_f1",
        "split_hash": EXPECTED_SPLIT_HASH,
        "development_batch_sha256": _sha256(development_batch),
        "folds": list(FOLD_IDS),
        "seeds": [int(seed) for seed in REPEAT_SEEDS],
        "default_config": {
            "max_depth": DEFAULT_BASELINE_MAX_DEPTH,
            "eta": DEFAULT_BASELINE_ETA,
            "rounds": DEFAULT_BASELINE_ROUNDS,
        },
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
        "p17_original_hashes": originals,
        **aggregation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "results.jsonl", rows)
    _write_json(output_dir / "summary.json", summary)
    _write_text(output_dir / "evidence.md", _evidence(summary))
    artifacts = []
    for name in ("results.jsonl", "summary.json", "evidence.md"):
        path = output_dir / name
        artifacts.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    _write_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": MANIFEST_SCHEMA,
            "split_hash": EXPECTED_SPLIT_HASH,
            "artifacts": artifacts,
            "p17_original_hashes": originals,
            "frozen_test_accessed": False,
            "known_holdout_accessed": False,
        },
    )
    verify_artifacts(output_dir)
    return summary


def verify_artifacts(output_dir: Path) -> dict[str, Any]:
    output_dir = _track_owned_output(output_dir)
    originals = verify_p17_originals()
    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("split_hash") != EXPECTED_SPLIT_HASH
        or manifest.get("p17_original_hashes") != originals
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("known_holdout_accessed") is not False
    ):
        raise RuntimeError("default-baseline artifact manifest violates the contract")
    for artifact in manifest["artifacts"]:
        path = output_dir / artifact["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["bytes"])
            or _sha256(path) != artifact["sha256"]
        ):
            raise RuntimeError(f"default-baseline artifact changed: {path}")
    rows = _read_jsonl(output_dir / "results.jsonl")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    counts = Counter(row["variant"] for row in rows)
    if counts != Counter({variant: 12 for variant in VARIANTS}):
        raise RuntimeError(f"expected 12 cells per variant, got {counts}")
    expected_cells = {
        (variant, fold_id, repeat_id)
        for variant in VARIANTS
        for fold_id in FOLD_IDS
        for repeat_id in range(len(REPEAT_SEEDS))
    }
    observed_cells = [
        (str(row["variant"]), int(row["fold_id"]), int(row["repeat_id"]))
        for row in rows
    ]
    if (
        len(set(observed_cells)) != len(observed_cells)
        or set(observed_cells) != expected_cells
    ):
        raise RuntimeError("default-baseline result rows do not cover exact LOGO4 x 3")
    if any(
        row.get("split_hash") != EXPECTED_SPLIT_HASH
        or row.get("development_only") is not True
        or row.get("frozen_test_accessed") is not False
        or row.get("known_holdout_accessed") is not False
        for row in rows
    ):
        raise RuntimeError("default-baseline result row violates the firewall")
    expected_configs = {
        VARIANTS[0]: {"max_depth": 2, "eta": 0.2, "rounds": 40},
        VARIANTS[1]: {
            "max_depth": DEFAULT_BASELINE_MAX_DEPTH,
            "eta": DEFAULT_BASELINE_ETA,
            "rounds": DEFAULT_BASELINE_ROUNDS,
        },
    }
    for row in rows:
        config = row["training"]
        expected = expected_configs[str(row["variant"])]
        observed = {key: config[key] for key in ("max_depth", "eta", "rounds")}
        repeat_id = int(row["repeat_id"])
        if observed != expected or int(config["seed"]) != int(REPEAT_SEEDS[repeat_id]):
            raise RuntimeError("default-baseline result row changed its training config")
    recomputed = summarize_rows(rows)
    if any(summary.get(key) != value for key, value in recomputed.items()):
        raise RuntimeError("default-baseline summary is not reproducible")
    evidence = (output_dir / "evidence.md").read_text(encoding="utf-8")
    for marker in (
        "XGBoost hyperparameter change",
        "no improvement is attributed to MOMENT",
        "frozen_test_accessed=false",
        "12/12",
    ):
        if marker not in evidence:
            raise RuntimeError(f"evidence is missing required marker: {marker}")
    return {
        "status": "verified",
        "rows": len(rows),
        "decision": summary["decision"],
        "comparison": summary["comparison"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--development-batch", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        payload = run(args.development_batch, args.output_dir)
        result = {
            "status": payload["decision"]["status"],
            "comparison": payload["comparison"],
        }
    else:
        result = verify_artifacts(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
