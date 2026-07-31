#!/usr/bin/env python3
"""Build leakage-aware F3 and Penobscot facies segmentation splits.

The two sources deliberately remain separate tasks because their label IDs have
different geological meanings:

* ``facies_f3`` uses F3 labels 0..9.
* ``facies_penobscot`` uses Penobscot labels 0..7.

Both sources use the same split policy.  Inline sections are ordered by inline
number, the lowest 75% are used for training, the next 5% are discarded as a
spatial guard band, and the highest 20% are used for testing.  F3 crossline
sections are intentionally excluded: using orthogonal views would allow the
same voxels to appear in both splits.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _code.dataset_io import save_split  # noqa: E402
from _code.ml_framework.preprocess import (  # noqa: E402
    NormStats,
    denoise_identity,
    denormalize,
    fit_minmax,
    fit_zscore,
    normalize,
)
from pipeline_contract import (  # noqa: E402
    DEFAULT_EXTERNAL_GUARD_FRACTION,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_VALIDATION_GUARD_FRACTION,
    DOMINANT_AMPLITUDE_FRACTION_THRESHOLD,
    NEAR_CONSTANT_PEAK_TO_PEAK_EPSILON,
    PIPELINE_VERSION,
    get_task_schema,
    is_near_constant_patch,
    ordered_spatial_split,
    validate_label_array,
)

DEFAULT_DATA_ROOT = PROJECT_ROOT / "_sandbox" / "f3_penobscot"
F3_TASK = "facies_f3"
PENOBSCOT_TASK = "facies_penobscot"
F3_INLINE_RE = re.compile(r"^masks/inline_(\d+)_mask\.png$")
DEFAULT_SUMMARY_PATH = Path(__file__).resolve().parent / "_outputs" / PIPELINE_VERSION / "data_build_manifest.json"


def portable_path(path: Path, *, external_root: Path | None = None) -> str:
    """Serialize project/external-data paths without host-specific prefixes."""
    resolved = path.resolve()
    if external_root is not None:
        try:
            return str(resolved.relative_to(external_root.resolve()))
        except ValueError:
            pass
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def file_fingerprint(
    path: Path, *, external_root: Path | None = None
) -> dict[str, Any]:
    """Return a reproducible SHA-256 identity with a portable logical path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": portable_path(path, external_root=external_root),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def preprocess_patch(
    patch: np.ndarray, stats: NormStats
) -> tuple[np.ndarray, float]:
    """Apply train-fitted normalization and verify its inverse transform."""
    array = np.asarray(patch, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("seismic patch contains non-finite amplitudes")

    # Do not smooth by default: sharp seismic events may be real geology rather
    # than noise.  The shared identity stage makes that choice explicit.
    denoised = denoise_identity(array)
    normalized = np.asarray(normalize(denoised, stats), dtype=np.float32)
    restored = np.asarray(denormalize(normalized, stats), dtype=np.float32)

    max_abs_error = float(np.max(np.abs(restored - denoised)))
    float32_tolerance = float(
        np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(denoised)))) * 4.0
    )
    if not np.allclose(restored, denoised, rtol=1e-6, atol=float32_tolerance):
        raise ValueError(
            f"{stats.method} normalization round-trip failed: max_abs_error={max_abs_error}"
        )
    return normalized, max_abs_error


def fit_training_normalization(
    samples: list[dict[str, Any]], method: str
) -> NormStats:
    """Fit shared normalization once, exclusively on model-training patches."""
    if not samples:
        raise ValueError("cannot fit normalization on an empty model-training split")
    fit_array = np.stack(
        [np.asarray(sample["seismic_patch"], dtype=np.float32) for sample in samples]
    )
    if method == "zscore":
        stats = fit_zscore(fit_array)
    elif method == "minmax":
        stats = fit_minmax(fit_array)
    else:
        raise ValueError(f"unsupported normalization method: {method}")
    del fit_array
    return stats


def spatial_split(
    line_numbers: Iterable[int], test_fraction: float, guard_fraction: float
) -> tuple[set[int], set[int], set[int]]:
    """Backward-compatible wrapper around the shared local spatial contract."""
    return ordered_spatial_split(line_numbers, test_fraction, guard_fraction)


def patch_origins(
    height: int,
    width: int,
    patch_size: int,
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Choose deterministic, unique random patch origins within one section."""
    if patch_size <= 0 or count <= 0:
        raise ValueError("patch_size and count must be positive")
    if patch_size > height or patch_size > width:
        raise ValueError(
            f"patch_size {patch_size} exceeds section shape {(height, width)}"
        )
    max_row = height - patch_size
    max_col = width - patch_size
    n_positions = (max_row + 1) * (max_col + 1)
    if count > n_positions:
        raise ValueError(f"requested {count} unique patches from only {n_positions} positions")

    rng = np.random.default_rng(seed)
    origins: set[tuple[int, int]] = set()
    while len(origins) < count:
        row = int(rng.integers(0, max_row + 1))
        col = int(rng.integers(0, max_col + 1))
        origins.add((row, col))
    return sorted(origins)


def make_raw_samples(
    seismic: np.ndarray,
    label: np.ndarray,
    *,
    task: str,
    source: str,
    inline_number: int,
    patch_size: int,
    patches_per_inline: int,
    seed: int,
    crossline_start: int,
    sample_interval_ms: float | None,
    near_constant_epsilon: float,
) -> tuple[list[dict[str, Any]], int]:
    """Extract valid raw patches before fitting any normalization statistics."""
    seismic = np.asarray(seismic)
    label = np.asarray(label)
    if seismic.ndim == 3 and seismic.shape[-1] == 1:
        seismic = seismic[..., 0]
    if seismic.ndim != 2 or label.ndim != 2:
        raise ValueError(
            f"expected 2-D seismic/label sections, got {seismic.shape} and {label.shape}"
        )
    if seismic.shape != label.shape:
        raise ValueError(f"seismic/label shape mismatch: {seismic.shape} != {label.shape}")
    schema = get_task_schema(task)
    validate_label_array(label, schema)

    origins = patch_origins(
        seismic.shape[0],
        seismic.shape[1],
        patch_size,
        patches_per_inline,
        seed,
    )
    samples: list[dict[str, Any]] = []
    filtered_near_constant = 0
    for row, col in origins:
        row_stop = row + patch_size
        col_stop = col + patch_size
        time_index = row + patch_size // 2
        raw_patch = np.asarray(
            denoise_identity(seismic[row:row_stop, col:col_stop]), dtype=np.float32
        )
        if is_near_constant_patch(raw_patch, near_constant_epsilon):
            filtered_near_constant += 1
            continue
        label_patch = label[row:row_stop, col:col_stop].astype(np.uint8, copy=True)
        validate_label_array(label_patch, schema)
        samples.append(
            {
                "seismic_patch": raw_patch,
                "well_log_seq": None,
                "position": {
                    "inline": int(inline_number),
                    "crossline": int(crossline_start + col + patch_size // 2),
                    "time_ms": (
                        float(time_index * sample_interval_ms)
                        if sample_interval_ms is not None
                        else None
                    ),
                    "well_name": None,
                },
                "label": label_patch,
                "meta": {
                    "source": source,
                    "label_space": task,
                    "view": "inline",
                    "inline_number": int(inline_number),
                    "patch_origin": [int(row), int(col)],
                    "original_section_shape": list(seismic.shape),
                    "denoise": "identity_no_smoothing_preserves_sharp_geological_features",
                    "split_policy": "ordered_inline_contiguous_with_guard_band",
                    "near_constant_rule": (
                        f"filter_if_peak_to_peak_le_{near_constant_epsilon:g}_or_"
                        "dominant_amplitude_fraction_ge_"
                        f"{DOMINANT_AMPLITUDE_FRACTION_THRESHOLD:g}"
                    ),
                },
            }
        )
    return samples, filtered_near_constant


def apply_training_normalization(
    samples: list[dict[str, Any]],
    *,
    stats: NormStats,
    fit_lines: set[int],
) -> float:
    """Normalize samples in place with one model-train-fitted parameter set."""
    if not samples:
        raise ValueError("cannot normalize an empty sample list")
    max_roundtrip_error = 0.0
    stats_dict = stats.to_dict()
    fit_range = [min(fit_lines), max(fit_lines)]
    for sample in samples:
        normalized, roundtrip_error = preprocess_patch(sample["seismic_patch"], stats)
        sample["seismic_patch"] = normalized
        sample["meta"].update(
            {
                "pipeline_version": PIPELINE_VERSION,
                "normalization": f"model_train_global_{stats.method}",
                "normalization_fit_scope": "model_train_inline_only",
                "normalization_fit_inline_range": fit_range,
                "normalization_stats": stats_dict,
                "normalization_roundtrip_max_abs_error": roundtrip_error,
            }
        )
        max_roundtrip_error = max(max_roundtrip_error, roundtrip_error)
    return max_roundtrip_error


def _describe_split(
    task: str,
    train_lines: set[int],
    guard_lines: set[int],
    test_lines: set[int],
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    def line_range(values: set[int]) -> list[int] | None:
        return [min(values), max(values)] if values else None

    return {
        "task": task,
        "train_lines": len(train_lines),
        "guard_lines": len(guard_lines),
        "test_lines": len(test_lines),
        "train_line_range": line_range(train_lines),
        "guard_line_range": line_range(guard_lines),
        "test_line_range": line_range(test_lines),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
    }


def _label_histogram(samples: list[dict[str, Any]], task: str) -> list[int]:
    schema = get_task_schema(task)
    histogram = np.zeros(schema.num_classes, dtype=np.int64)
    for sample in samples:
        label = np.asarray(sample["label"])
        validate_label_array(label, schema)
        histogram += np.bincount(
            label.reshape(-1), minlength=schema.num_classes
        )[: schema.num_classes]
    return histogram.tolist()


def finalize_and_save(
    *,
    task: str,
    train_lines: set[int],
    guard_lines: set[int],
    test_lines: set[int],
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    filtered_counts: dict[str, int],
    normalization: str,
    validation_fraction: float,
    validation_guard_fraction: float,
) -> dict[str, Any]:
    """Fit on model-train only, transform every split, then save both splits."""
    model_train_lines, validation_guard_lines, validation_lines = ordered_spatial_split(
        train_lines, validation_fraction, validation_guard_fraction
    )
    fit_samples = [
        sample
        for sample in train_samples
        if int(sample["position"]["inline"]) in model_train_lines
    ]
    stats = fit_training_normalization(fit_samples, normalization)
    train_roundtrip = apply_training_normalization(
        train_samples, stats=stats, fit_lines=model_train_lines
    )
    test_roundtrip = apply_training_normalization(
        test_samples, stats=stats, fit_lines=model_train_lines
    )

    train_path = save_split(task, "train", train_samples)
    test_path = save_split(task, "test", test_samples)
    summary = _describe_split(
        task,
        train_lines,
        guard_lines,
        test_lines,
        train_samples,
        test_samples,
    )
    summary.update(
        {
            "pipeline_version": PIPELINE_VERSION,
            "schema": {
                "num_classes": get_task_schema(task).num_classes,
                "valid_label_ids": list(get_task_schema(task).valid_label_ids),
                "ignore_index": get_task_schema(task).ignore_index,
                "source": get_task_schema(task).source,
            },
            "model_train_line_range": [min(model_train_lines), max(model_train_lines)],
            "validation_guard_line_range": [
                min(validation_guard_lines),
                max(validation_guard_lines),
            ],
            "validation_line_range": [min(validation_lines), max(validation_lines)],
            "normalization_fit_samples": len(fit_samples),
            "normalization_stats": stats.to_dict(),
            "normalization_fit_scope": "model_train_inline_only",
            "normalization_roundtrip_max_abs_error": {
                "train": train_roundtrip,
                "test": test_roundtrip,
            },
            "near_constant_filter": {
                "rule": (
                    "filter if raw seismic patch peak-to-peak <= "
                    f"{NEAR_CONSTANT_PEAK_TO_PEAK_EPSILON:g} or one exact amplitude "
                    "occupies >= "
                    f"{DOMINANT_AMPLITUDE_FRACTION_THRESHOLD:g} of pixels"
                ),
                **filtered_counts,
            },
            "train_label_histogram": _label_histogram(train_samples, task),
            "test_label_histogram": _label_histogram(test_samples, task),
            "train_path": portable_path(train_path),
            "test_path": portable_path(test_path),
            "processed_files": [
                file_fingerprint(train_path),
                file_fingerprint(test_path),
            ],
        }
    )
    return summary


def build_f3(
    data_root: Path,
    *,
    patch_size: int,
    patches_per_inline: int,
    test_fraction: float,
    guard_fraction: float,
    validation_fraction: float,
    validation_guard_fraction: float,
    seed: int,
    normalization: str,
    near_constant_epsilon: float,
) -> dict[str, Any]:
    """Build F3 samples from inline TIFFs and their PNG masks."""
    f3_root = data_root / "f3demo"
    inline_zip = f3_root / "inlines.zip"
    mask_tar = f3_root / "masks.tar.gz"
    for path in (inline_zip, mask_tar):
        if not path.is_file():
            raise FileNotFoundError(path)

    with tarfile.open(mask_tar, "r:gz") as masks:
        members: list[tuple[tarfile.TarInfo, int]] = []
        for member in masks.getmembers():
            match = F3_INLINE_RE.fullmatch(member.name)
            if member.isfile() and match:
                members.append((member, int(match.group(1))))
        if not members:
            raise ValueError(f"no usable inline masks found in {mask_tar}")

        line_numbers = [line for _, line in members]
        train_lines, guard_lines, test_lines = spatial_split(
            line_numbers, test_fraction, guard_fraction
        )
        train_samples: list[dict[str, Any]] = []
        test_samples: list[dict[str, Any]] = []
        filtered_counts = {"train": 0, "test": 0}

        with zipfile.ZipFile(inline_zip) as seismic_zip:
            seismic_names = set(seismic_zip.namelist())
            for member, inline_number in members:
                if inline_number in guard_lines:
                    continue
                split = "train" if inline_number in train_lines else "test"
                seismic_name = f"inlines/inline_{inline_number}.tiff"
                if seismic_name not in seismic_names:
                    raise FileNotFoundError(f"{seismic_name} is missing from {inline_zip}")
                mask_file = masks.extractfile(member)
                if mask_file is None:
                    raise OSError(f"could not read {member.name}")
                with Image.open(mask_file) as image:
                    label = np.asarray(image).copy()
                with Image.open(BytesIO(seismic_zip.read(seismic_name))) as image:
                    seismic = np.asarray(image).copy()

                line_seed = int(np.random.SeedSequence([seed, 3, inline_number]).generate_state(1)[0])
                samples, filtered = make_raw_samples(
                    seismic,
                    label,
                    task=F3_TASK,
                    source="f3_demo_netherlands",
                    inline_number=inline_number,
                    patch_size=patch_size,
                    patches_per_inline=patches_per_inline,
                    seed=line_seed,
                    crossline_start=300,
                    sample_interval_ms=None,
                    near_constant_epsilon=near_constant_epsilon,
                )
                filtered_counts[split] += filtered
                (train_samples if split == "train" else test_samples).extend(samples)

    summary = finalize_and_save(
        task=F3_TASK,
        train_lines=train_lines,
        guard_lines=guard_lines,
        test_lines=test_lines,
        train_samples=train_samples,
        test_samples=test_samples,
        filtered_counts=filtered_counts,
        normalization=normalization,
        validation_fraction=validation_fraction,
        validation_guard_fraction=validation_guard_fraction,
    )
    summary["source_files"] = [
        file_fingerprint(inline_zip, external_root=data_root),
        file_fingerprint(mask_tar, external_root=data_root),
    ]
    return summary


def build_penobscot(
    data_root: Path,
    *,
    patch_size: int,
    patches_per_inline: int,
    test_fraction: float,
    guard_fraction: float,
    validation_fraction: float,
    validation_guard_fraction: float,
    seed: int,
    normalization: str,
    near_constant_epsilon: float,
) -> dict[str, Any]:
    """Build Penobscot samples from aligned HDF5 features and labels."""
    dataset_path = data_root / "penobscot" / "dataset.h5"
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    with h5py.File(dataset_path, "r") as dataset:
        if "features" not in dataset or "label" not in dataset:
            raise KeyError(f"{dataset_path} must contain features and label datasets")
        features = dataset["features"]
        labels = dataset["label"]
        if features.shape[:-1] != labels.shape or features.shape[-1] != 1:
            raise ValueError(
                f"unaligned Penobscot arrays: features={features.shape}, label={labels.shape}"
            )
        line_numbers = np.asarray(dataset["line_number"][:], dtype=np.int64)
        if len(line_numbers) != features.shape[0]:
            raise ValueError("line_number length does not match features")

        train_lines, guard_lines, test_lines = spatial_split(
            line_numbers.tolist(), test_fraction, guard_fraction
        )
        train_samples: list[dict[str, Any]] = []
        test_samples: list[dict[str, Any]] = []
        filtered_counts = {"train": 0, "test": 0}
        for index, inline_number_raw in enumerate(line_numbers):
            inline_number = int(inline_number_raw)
            if inline_number in guard_lines:
                continue
            split = "train" if inline_number in train_lines else "test"
            line_seed = int(
                np.random.SeedSequence([seed, 7, inline_number]).generate_state(1)[0]
            )
            samples, filtered = make_raw_samples(
                features[index, ..., 0],
                labels[index, ...],
                task=PENOBSCOT_TASK,
                source="penobscot_canada",
                inline_number=inline_number,
                patch_size=patch_size,
                patches_per_inline=patches_per_inline,
                seed=line_seed,
                crossline_start=1000,
                sample_interval_ms=4.0,
                near_constant_epsilon=near_constant_epsilon,
            )
            filtered_counts[split] += filtered
            (train_samples if split == "train" else test_samples).extend(samples)

    summary = finalize_and_save(
        task=PENOBSCOT_TASK,
        train_lines=train_lines,
        guard_lines=guard_lines,
        test_lines=test_lines,
        train_samples=train_samples,
        test_samples=test_samples,
        filtered_counts=filtered_counts,
        normalization=normalization,
        validation_fraction=validation_fraction,
        validation_guard_fraction=validation_guard_fraction,
    )
    summary["source_files"] = [
        file_fingerprint(dataset_path, external_root=data_root)
    ]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "f3", "penobscot"),
        default="all",
        help="source to build (default: both independent tasks)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-inline", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument(
        "--guard-fraction", type=float, default=DEFAULT_EXTERNAL_GUARD_FRACTION
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument(
        "--validation-guard-fraction",
        type=float,
        default=DEFAULT_VALIDATION_GUARD_FRACTION,
    )
    parser.add_argument("--normalization", choices=("zscore", "minmax"), default="zscore")
    parser.add_argument("--seed", type=int, default=2693)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = {
        "patch_size": args.patch_size,
        "patches_per_inline": args.patches_per_inline,
        "test_fraction": args.test_fraction,
        "guard_fraction": args.guard_fraction,
        "validation_fraction": args.validation_fraction,
        "validation_guard_fraction": args.validation_guard_fraction,
        "normalization": args.normalization,
        "seed": args.seed,
        "near_constant_epsilon": NEAR_CONSTANT_PEAK_TO_PEAK_EPSILON,
    }
    summaries: list[dict[str, Any]] = []
    if args.dataset in ("all", "f3"):
        summaries.append(build_f3(args.data_root, **common))
        gc.collect()
    if args.dataset in ("all", "penobscot"):
        summaries.append(build_penobscot(args.data_root, **common))
        gc.collect()
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "data_root": "provided via --data-root; source file paths are relative to it",
        "configuration": {
            "patch_size": args.patch_size,
            "patches_per_inline": args.patches_per_inline,
            "test_fraction": args.test_fraction,
            "external_guard_fraction": args.guard_fraction,
            "validation_fraction": args.validation_fraction,
            "validation_guard_fraction": args.validation_guard_fraction,
            "normalization": args.normalization,
            "near_constant_peak_to_peak_epsilon": NEAR_CONSTANT_PEAK_TO_PEAK_EPSILON,
            "dominant_amplitude_fraction_threshold": DOMINANT_AMPLITUDE_FRACTION_THRESHOLD,
            "seed": args.seed,
        },
        "tasks": summaries,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
