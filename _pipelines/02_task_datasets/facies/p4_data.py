"""Read-only facies archive and fixed P4 ``ModelBatch`` adapter.

The processed P3 archives contain patches normalized with an older fixed
train/validation split.  P4 recovers the raw amplitudes with the archived
invertible statistics, then fits a fresh normalizer on each fold-train only.
No smoothing is applied: sharp seismic events may be real geology.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np

from _code.dataset_io import PROCESSED_DIR
from _code.ml_framework.artifacts import hash_payload
from _code.ml_framework.contracts import ModelBatch
from _code.ml_framework.preprocess import (
    NormStats,
    denoise_identity,
    denormalize,
    fit_minmax,
    fit_zscore,
    normalize,
)

from pipeline_contract import get_task_schema, validate_label_array
from p4_tasks import LABEL_VERSIONS, TASK_IDS


@dataclass(frozen=True)
class ArchiveRecord:
    task_id: str
    split: str
    sample_id: str
    storage_key: str
    inline: int
    crossline: int | None
    time_ms: float | None
    patch_shape: tuple[int, int]
    label_support: tuple[int, ...] | None
    source: str


@dataclass(frozen=True)
class FoldPreprocessor:
    task_id: str
    label_version: str
    normalization: NormStats
    class_weights: tuple[float, ...]
    class_histogram: tuple[int, ...]
    fit_sample_count: int
    fit_sample_ids_hash: str
    roundtrip_max_abs_error: float
    denoise: str = "identity"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["normalization"] = self.normalization.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FoldPreprocessor":
        values = dict(payload)
        values["normalization"] = NormStats.from_dict(dict(values["normalization"]))
        values["class_weights"] = tuple(values["class_weights"])
        values["class_histogram"] = tuple(values["class_histogram"])
        return cls(**values)


def inverse_sqrt_class_weights(histogram: np.ndarray) -> np.ndarray:
    """Derive finite weights from fold-train pixels only."""
    counts = np.asarray(histogram, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts <= 0):
        missing = np.flatnonzero(counts <= 0).tolist() if counts.ndim == 1 else []
        raise ValueError(f"fold-train must support every configured class; missing={missing}")
    frequencies = counts / counts.sum()
    weights = 1.0 / np.sqrt(frequencies)
    weights /= weights.mean()
    return np.clip(weights, 0.2, 5.0).astype(np.float32)


class FaciesArchive:
    """Read the unified dataset_io HDF5 schema without mutating its root."""

    def __init__(self, task_id: str, processed_root: Path | None = None) -> None:
        if task_id not in TASK_IDS:
            raise ValueError(f"unknown facies task {task_id!r}")
        self.task_id = task_id
        self.schema = get_task_schema(task_id)
        self.processed_root = Path(PROCESSED_DIR if processed_root is None else processed_root)
        self._index_cache: dict[tuple[str, bool], tuple[ArchiveRecord, ...]] = {}

    def split_path(self, split: str) -> Path:
        if split not in {"train", "test"}:
            raise ValueError("facies archive split must be 'train' or 'test'")
        path = self.processed_root / self.task_id / f"{split}.h5"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {self.task_id}/{split} archive: {path}; provision with --processed-root"
            )
        return path

    def development_index(self) -> tuple[ArchiveRecord, ...]:
        """Index development and read its labels for fold support checks."""
        return self._index_split("train", include_label_support=True)

    def sampled_development_index(self, *, max_candidates: int = 256) -> tuple[ArchiveRecord, ...]:
        """Read evenly spaced real development records for a bounded smoke.

        This is not a replacement for the formal full manifest.  It exists so
        the real-data smoke verifies HDF5/schema/model plumbing without an
        expensive all-group metadata scan and never opens the test archive.
        """
        if max_candidates < 10:
            raise ValueError("max_candidates must be >=10")
        path = self.split_path("train")
        records: list[ArchiveRecord] = []
        with h5py.File(path, "r") as archive:
            keys = sorted(archive.keys())
            candidate_count = min(max_candidates, len(keys))
            window_count = min(4, candidate_count)
            base, remainder = divmod(candidate_count, window_count)
            window_sizes = [base + (1 if index < remainder else 0) for index in range(window_count)]
            positions: list[int] = []
            for index, window_size in enumerate(window_sizes):
                start = int(
                    round(index * (len(keys) - window_size) / max(1, window_count - 1))
                )
                positions.extend(range(start, start + window_size))
            for position_index in sorted(set(positions)):
                key = keys[int(position_index)]
                group = archive[key]
                position = json.loads(group.attrs["position"])
                metadata = json.loads(group.attrs["meta"])
                label = np.asarray(group["label"][()])
                validate_label_array(label, self.schema)
                support = tuple(
                    int(value)
                    for value in np.bincount(
                        label.reshape(-1), minlength=self.schema.num_classes
                    )[: self.schema.num_classes]
                )
                records.append(
                    ArchiveRecord(
                        task_id=self.task_id,
                        split="train",
                        sample_id=f"{self.task_id}:train:{key}",
                        storage_key=key,
                        inline=int(position["inline"]),
                        crossline=(
                            None if position.get("crossline") is None else int(position["crossline"])
                        ),
                        time_ms=(
                            None if position.get("time_ms") is None else float(position["time_ms"])
                        ),
                        patch_shape=tuple(int(value) for value in label.shape),
                        label_support=support,
                        source=str(metadata.get("source", "unknown")),
                    )
                )
        if len(records) < 3:
            raise ValueError("real-data smoke could not sample enough development records")
        return tuple(records)

    def frozen_test_index(self, *, labels_consumed: bool = False) -> tuple[ArchiveRecord, ...]:
        """Index test coordinates only unless the lifecycle already consumed test."""
        return self._index_split("test", include_label_support=labels_consumed)

    def _index_split(self, split: str, *, include_label_support: bool) -> tuple[ArchiveRecord, ...]:
        cache_key = (split, include_label_support)
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]
        path = self.split_path(split)
        records: list[ArchiveRecord] = []
        with h5py.File(path, "r") as archive:
            archived_task = str(archive.attrs.get("task", self.task_id))
            archived_split = str(archive.attrs.get("split", split))
            if archived_task != self.task_id or archived_split != split:
                raise ValueError(
                    f"archive identity mismatch: expected {self.task_id}/{split}, "
                    f"found {archived_task}/{archived_split}"
                )
            for key in sorted(archive.keys()):
                group = archive[key]
                position = json.loads(group.attrs["position"])
                metadata = json.loads(group.attrs["meta"])
                label_dataset = group["label"]
                if len(label_dataset.shape) != 2:
                    raise ValueError(f"{key} label must be 2-D, got {label_dataset.shape}")
                support: tuple[int, ...] | None = None
                if include_label_support:
                    label = np.asarray(label_dataset[()])
                    validate_label_array(label, self.schema)
                    support = tuple(
                        int(value)
                        for value in np.bincount(
                            label.reshape(-1), minlength=self.schema.num_classes
                        )[: self.schema.num_classes]
                    )
                records.append(
                    ArchiveRecord(
                        task_id=self.task_id,
                        split=split,
                        sample_id=f"{self.task_id}:{split}:{key}",
                        storage_key=key,
                        inline=int(position["inline"]),
                        crossline=(
                            None if position.get("crossline") is None else int(position["crossline"])
                        ),
                        time_ms=(
                            None if position.get("time_ms") is None else float(position["time_ms"])
                        ),
                        patch_shape=tuple(int(value) for value in label_dataset.shape),
                        label_support=support,
                        source=str(metadata.get("source", "unknown")),
                    )
                )
        if not records:
            raise ValueError(f"{self.task_id}/{split} archive is empty")
        if len({record.sample_id for record in records}) != len(records):
            raise ValueError(f"{self.task_id}/{split} contains duplicate sample IDs")
        result = tuple(records)
        self._index_cache[cache_key] = result
        return result

    def _read_group(
        self,
        archive: h5py.File,
        record: ArchiveRecord,
        *,
        include_target: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        group = archive[record.storage_key]
        stored = np.asarray(group["seismic_patch"][()], dtype=np.float32)
        metadata = json.loads(group.attrs["meta"])
        stats_payload = metadata.get("normalization_stats")
        raw = stored if stats_payload is None else denormalize(stored, NormStats.from_dict(stats_payload))
        raw = np.asarray(denoise_identity(raw), dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"{record.sample_id} recovered seismic contains NaN/Inf")
        label: np.ndarray | None = None
        if include_target:
            label = np.asarray(group["label"][()], dtype=np.int64)
            validate_label_array(label, self.schema)
        return raw, label, metadata

    def fit_preprocessor(
        self,
        fold_train_records: Sequence[ArchiveRecord],
        *,
        method: str = "zscore",
    ) -> FoldPreprocessor:
        """Fit normalization and class weights using fold-train only."""
        records = tuple(fold_train_records)
        if not records or any(record.task_id != self.task_id or record.split != "train" for record in records):
            raise ValueError("preprocessing fit requires nonempty same-task development records")
        recovered: list[np.ndarray] = []
        histogram = np.zeros(self.schema.num_classes, dtype=np.int64)
        with h5py.File(self.split_path("train"), "r") as archive:
            for record in records:
                raw, label, _ = self._read_group(archive, record, include_target=True)
                assert label is not None
                recovered.append(raw.reshape(-1))
                histogram += np.bincount(
                    label.reshape(-1), minlength=self.schema.num_classes
                )[: self.schema.num_classes]
        fit_values = np.concatenate(recovered).astype(np.float32, copy=False)
        if method == "zscore":
            stats = fit_zscore(fit_values)
        elif method == "minmax":
            stats = fit_minmax(fit_values)
        else:
            raise ValueError(f"unsupported fold normalization {method!r}")
        probe = recovered[0]
        roundtrip = denormalize(normalize(probe, stats), stats)
        error = float(np.max(np.abs(roundtrip - probe)))
        if not np.isfinite(error) or error > 1e-2:
            raise ValueError(f"fold normalization round-trip error is too large: {error}")
        weights = inverse_sqrt_class_weights(histogram)
        return FoldPreprocessor(
            task_id=self.task_id,
            label_version=LABEL_VERSIONS[self.task_id],
            normalization=stats,
            class_weights=tuple(float(value) for value in weights),
            class_histogram=tuple(int(value) for value in histogram),
            fit_sample_count=len(records),
            fit_sample_ids_hash=hash_payload(sorted(record.sample_id for record in records)),
            roundtrip_max_abs_error=error,
        )

    def iter_model_batches(
        self,
        records: Sequence[ArchiveRecord],
        preprocessor: FoldPreprocessor,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        include_targets: bool = True,
    ) -> Iterator[ModelBatch]:
        """Yield the fixed facies P4 envelope from one archive split."""
        selected = tuple(records)
        if not selected:
            raise ValueError("cannot batch zero facies records")
        if batch_size <= 0:
            raise ValueError("batch_size must be >0")
        if preprocessor.task_id != self.task_id:
            raise ValueError("preprocessor task does not match archive task")
        splits = {record.split for record in selected}
        if len(splits) != 1:
            raise ValueError("one batch factory cannot mix train and test archives")
        split = next(iter(splits))
        order = np.arange(len(selected))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        with h5py.File(self.split_path(split), "r") as archive:
            for start in range(0, len(order), batch_size):
                batch_records = [selected[int(index)] for index in order[start : start + batch_size]]
                seismic: list[np.ndarray] = []
                labels: list[np.ndarray] = []
                for record in batch_records:
                    raw, label, _ = self._read_group(
                        archive, record, include_target=include_targets
                    )
                    transformed = normalize(raw, preprocessor.normalization).astype(np.float32)
                    if not np.isfinite(transformed).all():
                        raise ValueError(f"{record.sample_id} normalized seismic contains NaN/Inf")
                    seismic.append(transformed)
                    if label is not None:
                        labels.append(label)
                image_array = np.stack(seismic)[:, None, :, :]
                target_map = np.stack(labels) if include_targets else None
                yield ModelBatch(
                    inputs={"seismic": image_array},
                    targets=(None if target_map is None else {"facies": target_map}),
                    input_masks={},
                    target_masks=(
                        {} if target_map is None else {"facies": np.ones_like(target_map, dtype=bool)}
                    ),
                    sample_ids=[record.sample_id for record in batch_records],
                    groups={"inline": [str(record.inline) for record in batch_records]},
                    coordinates={
                        "inline": np.asarray([record.inline for record in batch_records], dtype=np.int64),
                        "crossline": np.asarray(
                            [-1 if record.crossline is None else record.crossline for record in batch_records],
                            dtype=np.int64,
                        ),
                        "time_ms": np.asarray(
                            [np.nan if record.time_ms is None else record.time_ms for record in batch_records],
                            dtype=np.float64,
                        ),
                    },
                    metadata={
                        "track_id": "facies",
                        "task_id": self.task_id,
                        "label_version": LABEL_VERSIONS[self.task_id],
                        "normalization_fit_sample_ids_hash": preprocessor.fit_sample_ids_hash,
                        "raw_output_contract": "model_returns_logits_BCHW",
                    },
                )


def records_by_id(records: Sequence[ArchiveRecord]) -> dict[str, ArchiveRecord]:
    mapping = {record.sample_id: record for record in records}
    if len(mapping) != len(records):
        raise ValueError("duplicate record IDs")
    return mapping


def label_histogram(records: Sequence[ArchiveRecord], num_classes: int) -> np.ndarray:
    histogram = np.zeros(num_classes, dtype=np.int64)
    for record in records:
        if record.label_support is None:
            raise ValueError(f"label support was not consumed for {record.sample_id}")
        if len(record.label_support) != num_classes:
            raise ValueError("record support width differs from task schema")
        histogram += np.asarray(record.label_support, dtype=np.int64)
    return histogram


def select_records_with_all_classes(
    records: Sequence[ArchiveRecord],
    *,
    num_classes: int,
    max_records: int,
) -> tuple[ArchiveRecord, ...]:
    """Choose a deterministic small subset with all-class support for smoke tests."""
    if max_records <= 0:
        raise ValueError("max_records must be >0")
    remaining = list(records)
    selected: list[ArchiveRecord] = []
    covered = np.zeros(num_classes, dtype=bool)
    while remaining and not covered.all() and len(selected) < max_records:
        best_index = max(
            range(len(remaining)),
            key=lambda index: int(
                np.logical_and(
                    np.asarray(remaining[index].label_support or (), dtype=np.int64) > 0,
                    ~covered,
                ).sum()
            ),
        )
        record = remaining.pop(best_index)
        support = np.asarray(record.label_support or (), dtype=np.int64)
        if support.size != num_classes:
            raise ValueError("smoke selection requires consumed development label support")
        selected.append(record)
        covered |= support > 0
    if not covered.all():
        raise ValueError(f"could not cover classes {np.flatnonzero(~covered).tolist()} in smoke subset")
    for record in remaining:
        if len(selected) >= max_records:
            break
        selected.append(record)
    return tuple(selected)
