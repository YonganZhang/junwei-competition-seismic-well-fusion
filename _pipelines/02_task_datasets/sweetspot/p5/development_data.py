"""Read only an explicitly declared development batch after label approval."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "sweetspot-p5-development-batch/v1"


@dataclass(frozen=True)
class DevelopmentBatch:
    target_id: str
    inputs: Mapping[str, Any]
    target: np.ndarray
    target_mask: np.ndarray
    sample_ids: tuple[str, ...]
    manifest_path: str
    manifest_sha256: str


def _forbidden_test_path(path: Path) -> bool:
    for part in path.parts:
        lowered = part.lower()
        if lowered in {"test", "frozen_test", "test.h5", "test.hdf5", "test.npz"}:
            return True
        if lowered.startswith("frozen_test"):
            return True
    return False


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development_batch(
    manifest_path: Path,
    *,
    target_id: str,
    label_spec_sha256: str,
    limit: int = 64,
) -> DevelopmentBatch:
    """Load a small development-only batch; there is deliberately no test API."""
    path = Path(manifest_path)
    if _forbidden_test_path(path):
        raise PermissionError("development manifest path resolves through a forbidden test component")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported development-batch manifest schema")
    if payload.get("split") != "development" or payload.get("contains_test") is not False:
        raise PermissionError("Stage 1 accepts only contains_test=false development manifests")
    if payload.get("test_accessed") is not False:
        raise PermissionError("development manifest must attest test_accessed=false")
    if payload.get("target_id") != target_id:
        raise ValueError("development manifest target_id does not match the requested independent task")
    if payload.get("label_spec_sha256") != label_spec_sha256:
        raise ValueError("development batch was not built from the approved label_spec hash")
    data_file = (path.parent / str(payload.get("data_file", ""))).resolve()
    if _forbidden_test_path(data_file):
        raise PermissionError("development data path resolves through a forbidden test component")
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    if payload.get("data_sha256") != _sha256(data_file):
        raise ValueError("development data hash does not match its manifest")
    arrays = payload.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("development manifest arrays must be an object")
    input_map = arrays.get("inputs")
    if not isinstance(input_map, Mapping) or not input_map:
        raise ValueError("development manifest must declare at least one input array")

    data_format = payload.get("format")
    inputs: dict[str, Any] = {}
    if data_format == "npz":
        with np.load(data_file, allow_pickle=False) as archive:
            for logical, stored in input_map.items():
                inputs[str(logical)] = np.asarray(archive[str(stored)])[:limit]
            target = np.asarray(archive[str(arrays["target"])])[:limit]
            mask = np.asarray(archive[str(arrays["target_mask"])], dtype=bool)[:limit]
            sample_ids = tuple(str(value) for value in np.asarray(archive[str(arrays["sample_ids"])])[:limit])
    elif data_format == "csv_time_series":
        import pandas as pd

        frame = pd.read_csv(data_file)
        inputs = {str(logical): frame for logical in input_map}
        target_column = str(arrays["target"])
        target = frame[target_column].to_numpy(dtype=float)
        mask = np.isfinite(target)
        sample_column = str(arrays["sample_ids"])
        sample_ids = tuple(frame[sample_column].astype(str).tolist())
    else:
        raise ValueError("development data format must be npz or csv_time_series")
    if not sample_ids or len(sample_ids) != len(target) or len(mask) != len(target):
        raise ValueError("development sample IDs, target and mask must be nonempty and aligned")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("development sample IDs must be unique")
    return DevelopmentBatch(
        target_id=target_id,
        inputs=inputs,
        target=np.asarray(target),
        target_mask=np.asarray(mask, dtype=bool),
        sample_ids=sample_ids,
        manifest_path=path.as_posix(),
        manifest_sha256=_sha256(path),
    )
