#!/usr/bin/env python3
"""Build the fault-detection train/test datasets from Layer1 Volve outputs.

The official fault interpretation stores each stick as a 1/2/3 vertex-code
sequence (start/intermediate/end).  Layer1 preserved that field under the
historical name ``stick_no``.  This builder connects consecutive vertices of
each stick with nearest voxel centres; it does not dilate or invent fault
thickness.

Each sample is a 2-D crossline/time patch at one fixed inline.  Train and test
are separated by an explicit guard band, and the three inline sets are asserted
to be disjoint. Fault-centred and annotation-free patches are sampled in equal
numbers, while each label remains the original sparse voxel mask.

Normalization statistics are fitted once from the sampled training patches and
then reused unchanged for test. Audited reruns write reports beneath
``_outputs/runs/<run-name>`` so the historical baseline artifacts stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import segyio


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAYER1_OUTPUTS = PROJECT_ROOT / "_pipelines" / "01_common_preprocess" / "outputs"
FAULT_POINTS_PATH = LAYER1_OUTPUTS / "fault_points.npz"
SEISMIC_INDEX_PATH = LAYER1_OUTPUTS / "seismic_index.npz"
SEGY_DIR = (
    PROJECT_ROOT
    / "_sandbox"
    / "volve_data"
    / "_extracted_seismic"
    / "ST0202"
    / "Stacks"
)
TRACK_DIR = Path(__file__).resolve().parent

# Import the mandatory shared writer without modifying or bypassing it.
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
from dataset_io import save_split  # noqa: E402
from ml_framework.preprocess import (  # noqa: E402
    NormStats,
    denoise_identity,
    denormalize,
    fit_zscore,
    normalize,
)
from audit_utils import (  # noqa: E402
    sha256_file,
    software_versions,
    validated_run_dir,
    verify_historical_artifacts_if_present,
)
from split_utils import ValidationPlan, validation_masks  # noqa: E402


@dataclass(frozen=True)
class PatchShape:
    crosslines: int
    times: int

    def __post_init__(self) -> None:
        if self.crosslines < 3 or self.times < 3:
            raise ValueError("patch dimensions must both be >= 3")
        if self.crosslines % 2 == 0 or self.times % 2 == 0:
            raise ValueError("patch dimensions must be odd so they have one centre voxel")

    @property
    def half_crosslines(self) -> int:
        return self.crosslines // 2

    @property
    def half_times(self) -> int:
        return self.times // 2


@dataclass(frozen=True)
class SplitPlan:
    train: tuple[int, int]
    guard: tuple[int, int]
    test: tuple[int, int]

    def inline_sets(self) -> tuple[set[int], set[int], set[int]]:
        return tuple(set(range(start, end + 1)) for start, end in (self.train, self.guard, self.test))

    def assert_disjoint(self) -> None:
        train, guard, test = self.inline_sets()
        if train & guard or train & test or guard & test:
            raise AssertionError("train/guard/test inline ranges overlap")
        if max(train) + 1 != min(guard) or max(guard) + 1 != min(test):
            raise AssertionError("train/guard/test inline ranges are not contiguous")

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "guard": list(self.guard),
            "test": list(self.test),
        }


def make_split_plan(
    il_min: int, il_max: int, test_fraction: float, guard_inlines: int
) -> SplitPlan:
    if guard_inlines < 1:
        raise ValueError("guard-inlines must be >= 1 so train and test are spatially separated")
    n_inline = il_max - il_min + 1
    test_start = il_min + int(np.floor(n_inline * (1.0 - test_fraction)))
    guard_start = test_start - guard_inlines
    train_end = guard_start - 1
    if train_end < il_min or test_start > il_max:
        raise ValueError("test-fraction and guard-inlines leave an empty train or test range")
    plan = SplitPlan(
        train=(il_min, train_end),
        guard=(guard_start, test_start - 1),
        test=(test_start, il_max),
    )
    plan.assert_disjoint()
    return plan


def load_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not FAULT_POINTS_PATH.exists() or not SEISMIC_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Layer1 outputs are missing; expected fault_points.npz and seismic_index.npz in "
            f"{LAYER1_OUTPUTS}"
        )
    with np.load(FAULT_POINTS_PATH, allow_pickle=True) as z:
        faults = {key: z[key] for key in z.files}
    with np.load(SEISMIC_INDEX_PATH, allow_pickle=False) as z:
        index = {key: z[key] for key in z.files}
    return faults, index


def fault_sticks(vertex_codes: np.ndarray) -> list[np.ndarray]:
    """Return index sequences encoded as 1=start, 2=middle, 3=end.

    Failing loudly matters here: connecting malformed sequences could draw a
    false fault across unrelated interpretations.
    """
    sticks: list[np.ndarray] = []
    current: list[int] = []
    for i, raw_code in enumerate(np.asarray(vertex_codes).tolist()):
        code = int(raw_code)
        if code == 1:
            if current:
                raise ValueError(f"fault stick starting at {current[0]} has no end before index {i}")
            current = [i]
        elif code == 2:
            if not current:
                raise ValueError(f"fault-stick middle vertex at index {i} has no start")
            current.append(i)
        elif code == 3:
            if not current:
                raise ValueError(f"fault-stick end vertex at index {i} has no start")
            current.append(i)
            sticks.append(np.asarray(current, dtype=np.int32))
            current = []
        else:
            raise ValueError(f"unknown fault-stick vertex code {code} at index {i}")
    if current:
        raise ValueError(f"fault stick starting at {current[0]} is unterminated")
    return sticks


def nearest_time_indices(times_ms: np.ndarray, query_ms: np.ndarray) -> np.ndarray:
    """Map times to the nearest explicit seismic sample without assuming 4 ms."""
    samples = np.asarray(times_ms, dtype=np.float64)
    query = np.asarray(query_ms, dtype=np.float64)
    right = np.searchsorted(samples, query, side="left")
    right = np.clip(right, 1, len(samples) - 1)
    left = right - 1
    choose_left = np.abs(query - samples[left]) <= np.abs(samples[right] - query)
    return np.where(choose_left, left, right).astype(np.int32)


def rasterize_fault_voxels(
    faults: dict[str, np.ndarray], index: dict[str, np.ndarray]
) -> np.ndarray:
    """Connect consecutive vertices into unique (inline, crossline, time_idx) voxels."""
    inline = np.asarray(faults["inline"], dtype=np.int32)
    crossline = np.asarray(faults["crossline"], dtype=np.int32)
    time_idx = nearest_time_indices(index["samples_ms"], faults["twt_ms"])
    fault_name = np.asarray(faults["fault_name"]).astype(str)
    sticks = fault_sticks(faults["stick_no"])

    chunks: list[np.ndarray] = []
    for stick in sticks:
        if len(set(fault_name[stick].tolist())) != 1:
            raise ValueError(f"fault name changes within stick beginning at index {int(stick[0])}")
        points = np.column_stack((inline[stick], crossline[stick], time_idx[stick])).astype(np.float64)
        for start, end in zip(points[:-1], points[1:]):
            n_steps = int(np.max(np.abs(end - start))) + 1
            segment = np.rint(np.linspace(start, end, n_steps)).astype(np.int32)
            chunks.append(segment)

    if not chunks:
        raise ValueError("no fault-stick segments were available for rasterization")
    voxels = np.unique(np.concatenate(chunks, axis=0), axis=0)

    valid = (
        (voxels[:, 0] >= int(index["il_min"]))
        & (voxels[:, 0] <= int(index["il_max"]))
        & (voxels[:, 1] >= int(index["xl_min"]))
        & (voxels[:, 1] <= int(index["xl_max"]))
        & (voxels[:, 2] >= 0)
        & (voxels[:, 2] < len(index["samples_ms"]))
    )
    if not np.all(valid):
        raise ValueError(f"rasterization produced {int((~valid).sum())} voxels outside the seismic grid")
    return voxels


def group_voxels_by_inline(voxels: np.ndarray) -> dict[int, np.ndarray]:
    grouped: dict[int, list[tuple[int, int]]] = {}
    for inline, crossline, time_idx in voxels.tolist():
        grouped.setdefault(int(inline), []).append((int(crossline), int(time_idx)))
    return {key: np.asarray(value, dtype=np.int32) for key, value in grouped.items()}


def label_patch(
    voxels_by_inline: dict[int, np.ndarray],
    inline: int,
    centre_crossline: int,
    centre_time_idx: int,
    shape: PatchShape,
) -> np.ndarray:
    label = np.zeros((shape.crosslines, shape.times), dtype=np.uint8)
    points = voxels_by_inline.get(int(inline))
    if points is None:
        return label
    xl0 = centre_crossline - shape.half_crosslines
    t0 = centre_time_idx - shape.half_times
    rows = points[:, 0] - xl0
    cols = points[:, 1] - t0
    keep = (
        (rows >= 0)
        & (rows < shape.crosslines)
        & (cols >= 0)
        & (cols < shape.times)
    )
    label[rows[keep], cols[keep]] = 1
    return label


def find_segy() -> Path:
    candidates = sorted(SEGY_DIR.glob("*.segy"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one extracted SEG-Y in {SEGY_DIR}, found {len(candidates)}")
    return candidates[0]


class SeismicPatchReader:
    def __init__(self, path: Path, index: dict[str, np.ndarray]) -> None:
        self.index = index
        self._file = segyio.open(str(path), "r", ignore_geometry=True)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "SeismicPatchReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read(
        self,
        inline: int,
        centre_crossline: int,
        centre_time_idx: int,
        shape: PatchShape,
    ) -> np.ndarray:
        il_min = int(self.index["il_min"])
        xl_min = int(self.index["xl_min"])
        n_xl = int(self.index["n_xl"])
        xl0 = centre_crossline - shape.half_crosslines
        t0 = centre_time_idx - shape.half_times
        section = np.empty((shape.crosslines, shape.times), dtype=np.float32)
        for row, crossline in enumerate(range(xl0, xl0 + shape.crosslines)):
            trace_index = (inline - il_min) * n_xl + (crossline - xl_min)
            section[row] = np.asarray(
                self._file.trace[int(trace_index)][t0 : t0 + shape.times], dtype=np.float32
            )
        if not np.isfinite(section).all() or float(np.std(section)) <= 1e-8:
            raise ValueError(
                f"invalid seismic patch statistics at inline={inline}, crossline={centre_crossline}, "
                f"time_idx={centre_time_idx}: mean={float(np.mean(section))}, "
                f"std={float(np.std(section))}"
            )

        # Explicitly no denoising: sharp seismic amplitudes may be genuine
        # geological boundaries. The shared identity stage keeps this policy
        # visible and replaceable without silently smoothing the input.
        denoised = denoise_identity(section)
        return denoised[np.newaxis].astype(np.float32)


def choose_positive_centres(
    voxels: np.ndarray, inline_min: int, inline_max: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    candidates = voxels[(voxels[:, 0] >= inline_min) & (voxels[:, 0] <= inline_max)]
    if len(candidates) < n_samples:
        raise ValueError(
            f"only {len(candidates)} positive voxel centres available in inline range "
            f"[{inline_min}, {inline_max}], requested {n_samples}"
        )
    return candidates[rng.choice(len(candidates), size=n_samples, replace=False)]


def choose_negative_centres(
    fault_voxels: np.ndarray,
    voxels_by_inline: dict[int, np.ndarray],
    index: dict[str, np.ndarray],
    inline_min: int,
    inline_max: int,
    n_samples: int,
    shape: PatchShape,
    rng: np.random.Generator,
) -> np.ndarray:
    xl_low = int(index["xl_min"]) + shape.half_crosslines
    xl_high = int(index["xl_max"]) - shape.half_crosslines
    split_faults = fault_voxels[
        (fault_voxels[:, 0] >= inline_min) & (fault_voxels[:, 0] <= inline_max)
    ]
    if not len(split_faults):
        raise ValueError(f"no fault voxels available in inline range [{inline_min}, {inline_max}]")
    # Match the annotated time support. Sampling the whole trace would select
    # late-time zero padding as an easy but geologically meaningless negative.
    t_low = max(shape.half_times, int(split_faults[:, 2].min()))
    t_high = min(
        len(index["samples_ms"]) - shape.half_times - 1,
        int(split_faults[:, 2].max()),
    )
    selected: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    max_attempts = max(10_000, n_samples * 500)
    for _ in range(max_attempts):
        centre = (
            int(rng.integers(inline_min, inline_max + 1)),
            int(rng.integers(xl_low, xl_high + 1)),
            int(rng.integers(t_low, t_high + 1)),
        )
        if centre in seen:
            continue
        if label_patch(voxels_by_inline, *centre, shape).any():
            continue
        selected.append(centre)
        seen.add(centre)
        if len(selected) == n_samples:
            return np.asarray(selected, dtype=np.int32)
    raise RuntimeError(f"found only {len(selected)}/{n_samples} annotation-free negative patches")


def build_samples(
    centres: Iterable[np.ndarray],
    kind: str,
    split: str,
    reader: SeismicPatchReader,
    voxels_by_inline: dict[int, np.ndarray],
    index: dict[str, np.ndarray],
    shape: PatchShape,
    split_plan: SplitPlan,
) -> list[dict]:
    samples: list[dict] = []
    times_ms = np.asarray(index["samples_ms"], dtype=np.float64)
    for inline_raw, crossline_raw, time_idx_raw in centres:
        inline, crossline, time_idx = int(inline_raw), int(crossline_raw), int(time_idx_raw)
        label = label_patch(voxels_by_inline, inline, crossline, time_idx, shape)
        if kind == "fault" and not label.any():
            raise AssertionError("fault-centred patch unexpectedly has no positive voxel")
        if kind == "non_fault" and label.any():
            raise AssertionError("non-fault patch unexpectedly contains annotated fault voxels")
        seismic_patch = reader.read(inline, crossline, time_idx, shape)
        samples.append(
            {
                "seismic_patch": seismic_patch,
                "well_log_seq": None,
                "position": {
                    "inline": inline,
                    "crossline": crossline,
                    "time_ms": float(times_ms[time_idx]),
                    "time_index": time_idx,
                    "well_name": None,
                },
                "label": label,
                "meta": {
                    "source": "Volve_Official_Faults.dat+ST0202",
                    "sample_kind": kind,
                    "rasterization": "linear_between_1_2_3_stick_vertices_no_dilation",
                    "label_dilation_radius": {"inline": 0, "crossline": 0, "time": 0},
                    "label_semantics": "raw_interpreted_fault_stick_skeleton",
                    "patch_axes": ["crossline", "time"],
                    "denoising": "ml_framework.preprocess.denoise_identity",
                    "split_plan": split_plan.to_dict(),
                    "unannotated_patch_assumed_negative": kind == "non_fault",
                    "negative_sampling": "uniform_centres_within_split_annotated_time_support_reject_any_raw_fault_voxel",
                },
            }
        )
    return samples


def apply_training_normalization(
    built: dict[str, list[dict]], val_fraction: float, val_guard_inlines: int
) -> tuple[NormStats, ValidationPlan]:
    """Fit on train-fit only, then transform validation/test unchanged."""
    inlines = np.asarray(
        [int(sample["position"]["inline"]) for sample in built["train"]], dtype=np.int32
    )
    fit_mask, _, validation_mask, validation_plan = validation_masks(
        inlines, val_fraction, val_guard_inlines
    )
    fit_samples = [sample for sample, selected in zip(built["train"], fit_mask) if selected]
    validation_samples = [
        sample for sample, selected in zip(built["train"], validation_mask) if selected
    ]
    if not any(sample["label"].any() for sample in fit_samples) or not any(
        sample["label"].any() for sample in validation_samples
    ):
        raise ValueError("train-fit and validation must both contain positive fault voxels")
    training_fit = np.stack([sample["seismic_patch"] for sample in fit_samples])
    stats = fit_zscore(training_fit)
    for split in ("train", "test"):
        for sample in built[split]:
            physical = sample["seismic_patch"]
            normalized = normalize(physical, stats).astype(np.float32)
            restored = denormalize(normalized, stats)
            error = float(np.max(np.abs(restored - physical)))
            tolerance = max(1e-5, float(np.max(np.abs(physical))) * 2e-6)
            if not np.isfinite(normalized).all() or error > tolerance:
                raise AssertionError(
                    f"train-fitted normalization round-trip failed for {split}: "
                    f"max_abs_error={error}, tolerance={tolerance}"
                )
            sample["seismic_patch"] = normalized
            sample["meta"]["normalization"] = stats.to_dict()
            sample["meta"]["normalization_fit_split"] = "train_fit"
            sample["meta"]["normalization_scope"] = "spatial_train_fit_patch_voxels_only"
            sample["meta"]["normalization_validation_plan"] = validation_plan.to_dict()
            sample["meta"]["normalization_roundtrip_max_abs_error"] = error
    return stats, validation_plan


def rasterization_audit(
    faults: dict[str, np.ndarray],
    index: dict[str, np.ndarray],
    voxels: np.ndarray,
    split_plan: SplitPlan,
) -> dict:
    sticks = fault_sticks(faults["stick_no"])
    time_indices = nearest_time_indices(index["samples_ms"], faults["twt_ms"])
    snapped_times = np.asarray(index["samples_ms"], dtype=np.float64)[time_indices]
    residuals = np.abs(snapped_times - np.asarray(faults["twt_ms"], dtype=np.float64))
    names = np.asarray(faults["fault_name"]).astype(str)
    inlines = np.asarray(faults["inline"], dtype=np.int32)
    train_end = split_plan.train[1]
    test_start = split_plan.test[0]
    spanning_sticks = [
        stick for stick in sticks if int(inlines[stick].min()) <= train_end and int(inlines[stick].max()) >= test_start
    ]
    spanning_names = sorted(
        name
        for name in np.unique(names)
        if int(inlines[names == name].min()) <= train_end
        and int(inlines[names == name].max()) >= test_start
    )
    return {
        "source_points": int(len(inlines)),
        "complete_sticks": int(len(sticks)),
        "fault_names": int(len(np.unique(names))),
        "method": "max-axis 3D linear interpolation, nearest integer voxel, unique union",
        "label_dilation_radius": {"inline": 0, "crossline": 0, "time": 0},
        "scientific_label_contract": "raw interpreted stick skeleton; no surface interpolation between sticks",
        "rasterized_unique_voxels": int(len(voxels)),
        "time_snap_abs_error_ms": {
            "max": float(residuals.max()),
            "mean": float(residuals.mean()),
        },
        "sticks_spanning_train_to_test_across_guard": int(len(spanning_sticks)),
        "fault_names_spanning_train_to_test_across_guard": spanning_names,
        "residual_risk": (
            "Named geological faults can span both spatial blocks; the guard prevents shared/adjacent "
            "sample inlines but does not redefine fault entities."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-crosslines", type=int, default=33)
    parser.add_argument("--patch-times", type=int, default=65)
    parser.add_argument("--train-per-class", type=int, default=128)
    parser.add_argument("--test-per-class", type=int, default=48)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--guard-inlines", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--val-guard-inlines", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2693)
    parser.add_argument("--run-name", default="audited_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical_artifacts = verify_historical_artifacts_if_present()
    run_dir = validated_run_dir(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not 0.05 <= args.test_fraction <= 0.5:
        raise ValueError("test-fraction must be between 0.05 and 0.5")
    if args.train_per_class <= 0 or args.test_per_class <= 0:
        raise ValueError("sample counts must be positive")

    shape = PatchShape(args.patch_crosslines, args.patch_times)
    faults, index = load_inputs()
    voxels = rasterize_fault_voxels(faults, index)
    voxels_by_inline = group_voxels_by_inline(voxels)

    il_min, il_max = int(index["il_min"]), int(index["il_max"])
    split_plan = make_split_plan(il_min, il_max, args.test_fraction, args.guard_inlines)
    split_ranges = {
        "train": split_plan.train,
        "test": split_plan.test,
    }
    rng = np.random.default_rng(args.seed)

    segy_path = find_segy()
    built: dict[str, list[dict]] = {}
    with SeismicPatchReader(segy_path, index) as reader:
        for split, count in (("train", args.train_per_class), ("test", args.test_per_class)):
            split_min, split_max = split_ranges[split]
            positives = choose_positive_centres(voxels, split_min, split_max, count, rng)
            negatives = choose_negative_centres(
                voxels, voxels_by_inline, index, split_min, split_max, count, shape, rng
            )
            samples = build_samples(
                positives,
                "fault",
                split,
                reader,
                voxels_by_inline,
                index,
                shape,
                split_plan,
            )
            samples.extend(
                build_samples(
                    negatives,
                    "non_fault",
                    split,
                    reader,
                    voxels_by_inline,
                    index,
                    shape,
                    split_plan,
                )
            )
            rng.shuffle(samples)
            built[split] = samples

    normalization_stats, normalization_validation_plan = apply_training_normalization(
        built, args.val_fraction, args.val_guard_inlines
    )

    train_inlines = {int(s["position"]["inline"]) for s in built["train"]}
    test_inlines = {int(s["position"]["inline"]) for s in built["test"]}
    full_train, guard_inlines, full_test = split_plan.inline_sets()
    if train_inlines & test_inlines or train_inlines & guard_inlines or test_inlines & guard_inlines:
        raise AssertionError("sampled train/guard/test inline leakage detected")
    if not train_inlines <= full_train or not test_inlines <= full_test:
        raise AssertionError("sample centre fell outside its declared split range")
    train_centres = {
        (int(s["position"]["inline"]), int(s["position"]["crossline"]), int(s["position"]["time_index"]))
        for s in built["train"]
    }
    test_centres = {
        (int(s["position"]["inline"]), int(s["position"]["crossline"]), int(s["position"]["time_index"]))
        for s in built["test"]
    }
    if len(train_centres) != len(built["train"]) or len(test_centres) != len(built["test"]):
        raise AssertionError("duplicate sample centres detected within a split")
    if train_centres & test_centres:
        raise AssertionError("sample centre leakage detected between train and test")
    train_patch_hashes = {
        hashlib.sha256(np.ascontiguousarray(sample["seismic_patch"]).tobytes()).hexdigest()
        for sample in built["train"]
    }
    test_patch_hashes = {
        hashlib.sha256(np.ascontiguousarray(sample["seismic_patch"]).tobytes()).hexdigest()
        for sample in built["test"]
    }
    exact_patch_overlap = train_patch_hashes & test_patch_hashes
    if exact_patch_overlap:
        raise AssertionError(f"exact seismic patch leakage detected: {len(exact_patch_overlap)} hashes")

    # These exact shared-interface calls are the only dataset writes.
    train_path = save_split("fault", "train", built["train"])
    test_path = save_split("fault", "test", built["test"])

    def summarize(split: str) -> dict:
        labels = np.stack([sample["label"] for sample in built[split]])
        kinds = [sample["meta"]["sample_kind"] for sample in built[split]]
        inlines = sorted({int(sample["position"]["inline"]) for sample in built[split]})
        roundtrip_errors = [
            float(sample["meta"]["normalization_roundtrip_max_abs_error"])
            for sample in built[split]
        ]
        return {
            "n_samples": len(labels),
            "fault_centred": kinds.count("fault"),
            "non_fault": kinds.count("non_fault"),
            "positive_voxels": int(labels.sum()),
            "total_voxels": int(labels.size),
            "positive_fraction": float(labels.mean()),
            "inline_min": min(inlines),
            "inline_max": max(inlines),
            "n_unique_inlines": len(inlines),
            "normalization": "train-fit-only ml_framework.fit_zscore reused for validation/test",
            "normalization_roundtrip_max_abs_error": max(roundtrip_errors),
        }

    segy_sha256 = sha256_file(segy_path)
    input_hashes = {
        "fault_points.npz": sha256_file(FAULT_POINTS_PATH),
        "seismic_index.npz": sha256_file(SEISMIC_INDEX_PATH),
        "segy": segy_sha256,
    }
    sample_manifest = {
        split: [
            {
                "inline": int(sample["position"]["inline"]),
                "crossline": int(sample["position"]["crossline"]),
                "time_index": int(sample["position"]["time_index"]),
                "sample_kind": sample["meta"]["sample_kind"],
            }
            for sample in built[split]
        ]
        for split in ("train", "test")
    }
    split_manifest_path = run_dir / "split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "split_plan": split_plan.to_dict(),
                "guard_inline_count": len(guard_inlines),
                "train_guard_test_pairwise_overlap": [],
                "sample_centres": sample_manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "run_name": args.run_name,
        "seed": args.seed,
        "arguments": vars(args),
        "segy": str(segy_path.relative_to(PROJECT_ROOT)),
        "input_sha256": input_hashes,
        "software_versions": software_versions(),
        "source_sha256": {
            "build_dataset.py": sha256_file(Path(__file__)),
            "audit_utils.py": sha256_file(TRACK_DIR / "audit_utils.py"),
            "split_utils.py": sha256_file(TRACK_DIR / "split_utils.py"),
        },
        "patch_shape": [1, shape.crosslines, shape.times],
        "rasterization": rasterization_audit(faults, index, voxels, split_plan),
        "split_plan": split_plan.to_dict(),
        "guard_inline_count": len(guard_inlines),
        "normalization": {
            "fit_split": "train_fit",
            "stats": normalization_stats.to_dict(),
            "validation_plan": normalization_validation_plan.to_dict(),
            "validation_refit": False,
            "test_refit": False,
        },
        "train_path": str(train_path.relative_to(PROJECT_ROOT)),
        "test_path": str(test_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": {
            "train": sha256_file(train_path),
            "test": sha256_file(test_path),
        },
        "split_manifest": str(split_manifest_path.relative_to(TRACK_DIR)),
        "train": summarize("train"),
        "test": summarize("test"),
        "leakage_checks": {
            "train_guard_test_inline_overlap": [],
            "train_test_sample_centre_overlap": [],
            "train_test_exact_patch_sha256_overlap": [],
            "train_test_physical_voxel_overlap": [],
            "guard_inlines_sampled": [],
        },
        "historical_artifacts_verified": historical_artifacts,
    }
    summary_path = run_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verify_historical_artifacts_if_present()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
