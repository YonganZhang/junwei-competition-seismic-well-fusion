from __future__ import annotations

import hashlib
import importlib.metadata as md
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np


HERE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in HERE.parents if (parent / "_pipelines").exists() and (parent / "_data").exists())
RESERVOIR_DIR = PROJECT_ROOT / "_pipelines" / "02_task_datasets" / "reservoir"
OUTPUT_DIR = RESERVOIR_DIR / "_outputs" / "p18_cigbench_property"
EVIDENCE_PATH = OUTPUT_DIR / "evidence.md"
DEV_TRAIN_PATH = PROJECT_ROOT / "_data" / "processed" / "reservoir" / "train.h5"
DEV_GUARD_PATH = RESERVOIR_DIR / "_outputs" / "guard.npz"
BASELINE_METRICS_PATH = RESERVOIR_DIR / "_outputs" / "metrics.json"
BASELINE_RUN_MANIFEST_PATH = RESERVOIR_DIR / "_outputs" / "run_manifest.json"
BASELINE_SPLIT_MANIFEST_PATH = RESERVOIR_DIR / "_outputs" / "split_manifest.json"


@dataclass(frozen=True)
class FileInfo:
    path: Path
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_info(path: Path) -> FileInfo:
    return FileInfo(path=path, sha256=sha256_file(path), size_bytes=path.stat().st_size)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def discover_development_inputs() -> dict[str, Any]:
    with h5py.File(DEV_TRAIN_PATH, "r") as train_h5:
        first_key = next(iter(train_h5.keys()))
        sample = train_h5[first_key]
        seismic_shape = tuple(np.asarray(sample["seismic_patch"]).shape)
        sparse_shape = tuple(np.asarray(sample["well_log_seq"]).shape)
        label_shape = tuple(np.asarray(sample["label"]).shape)
        meta = json.loads(sample.attrs["meta"])
        train_sample_count = len(train_h5)

    guard = np.load(DEV_GUARD_PATH, allow_pickle=False)
    guard_seismic_shape = tuple(guard["seismic_patch"].shape[1:])
    guard_sparse_shape = tuple(guard["well_log_seq"].shape[1:])
    guard_label_shape = tuple(guard["label"].shape[1:])

    return {
        "train": {
            "path": DEV_TRAIN_PATH,
            "sha256": sha256_file(DEV_TRAIN_PATH),
            "sample_count": train_sample_count,
            "sample_key": first_key,
            "seismic_shape": seismic_shape,
            "well_log_seq_shape": sparse_shape,
            "label_shape": label_shape,
            "meta": meta,
        },
        "guard": {
            "path": DEV_GUARD_PATH,
            "sha256": sha256_file(DEV_GUARD_PATH),
            "sample_count": int(guard["seismic_patch"].shape[0]),
            "seismic_shape": guard_seismic_shape,
            "well_log_seq_shape": guard_sparse_shape,
            "label_shape": guard_label_shape,
        },
    }


def inspect_cig_bench_api() -> dict[str, Any]:
    import cig_bench
    from cig_bench.predictor._download import MODELSCOPE_REGISTRY
    from cig_bench.predictor.property import PropertyPredictor

    return {
        "cig_bench_version": md.version("cig_bench"),
        "modelscope_version": md.version("modelscope"),
        "torch_version": md.version("torch"),
        "cig_bench_file": getattr(cig_bench, "__file__", None),
        "property_predictor_signature": str(inspect.signature(PropertyPredictor)),
        "property_registry": {"property": MODELSCOPE_REGISTRY["property"]},
    }


def baseline_snapshot() -> dict[str, Any]:
    metrics = load_json(BASELINE_METRICS_PATH)
    run_manifest = load_json(BASELINE_RUN_MANIFEST_PATH)
    split_manifest = load_json(BASELINE_SPLIT_MANIFEST_PATH)
    return {
        "metrics_path": file_info(BASELINE_METRICS_PATH).__dict__,
        "run_manifest_path": file_info(BASELINE_RUN_MANIFEST_PATH).__dict__,
        "split_manifest_path": file_info(BASELINE_SPLIT_MANIFEST_PATH).__dict__,
        "metrics": metrics,
        "run_manifest": run_manifest,
        "split_manifest": split_manifest,
    }


def try_property_predictor_smoke(
    predictor_factory: Optional[Callable[[], Any]] = None,
    sample_seismic: Optional[np.ndarray] = None,
    sample_sparse: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    try:
        from cig_bench.predictor.property import PropertyPredictor
    except Exception as exc:  # pragma: no cover - local import failure would be blocker.
        return {
            "status": "BLOCKED_DATA_OR_API",
            "reason": "cig_bench import failure",
            "error": repr(exc),
        }

    if predictor_factory is None:
        predictor_factory = lambda: PropertyPredictor(device="cpu", use_autocast=False)

    try:
        predictor = predictor_factory()
    except Exception as exc:
        return {
            "status": "BLOCKED_DATA_OR_API",
            "reason": "property weight download / predictor init failed",
            "error": repr(exc),
        }

    if sample_seismic is None:
        return {
            "status": "BLOCKED_DATA_OR_API",
            "reason": "no sample provided for smoke",
        }
    if sample_sparse is None:
        sample_sparse = np.zeros_like(sample_seismic, dtype=np.float32)

    try:
        prop_vol, used_seis, well_info = predictor.predict(
            sample_seismic.astype(np.float32),
            prop=sample_sparse.astype(np.float32),
            infer_shape=(8, 8, 8),
            resize_back=False,
            normalize_output=False,
        )
        return {
            "status": "ok",
            "prop_shape": tuple(prop_vol.shape),
            "used_shape": tuple(used_seis.shape),
            "well_count": int(len(well_info["positions"])),
        }
    except Exception as exc:
        return {
            "status": "BLOCKED_DATA_OR_API",
            "reason": "predictor forward failed",
            "error": repr(exc),
        }


def build_evidence() -> dict[str, Any]:
    dev = discover_development_inputs()
    api = inspect_cig_bench_api()
    baseline = baseline_snapshot()

    with h5py.File(DEV_TRAIN_PATH, "r") as train_h5:
        first_key = next(iter(train_h5.keys()))
        sample = train_h5[first_key]
        seismic = np.asarray(sample["seismic_patch"], dtype=np.float32)
        sparse = np.asarray(sample["well_log_seq"], dtype=np.float32)

    smoke = try_property_predictor_smoke(sample_seismic=seismic, sample_sparse=sparse)

    candidate_metrics = None
    blocker_reasons = [
        "ModelScope default checkpoint download for douyimin/CIG-Bench / CIG-Bench-Property.pth failed with a reproducible HTTP 404.",
        "The reservoir development tensors are sample-level seismic patches plus well_log_seq feature sequences; they do not provide a legal sparse target-property volume matching CIG-Bench PropertyPredictor's seismic+ sparse property contract without target leakage.",
    ]
    if smoke.get("status") == "ok":
        candidate_metrics = smoke
    else:
        blocker_reasons.insert(0, f"API probe returned {smoke.get('status')}: {smoke.get('reason')}")

    return {
        "verdict": "BLOCKED_DATA_OR_API" if candidate_metrics is None else "READY",
        "api": api,
        "development_inputs": dev,
        "baseline": baseline,
        "smoke": smoke,
        "blocker_reasons": blocker_reasons if candidate_metrics is None else [],
        "candidate_metrics": candidate_metrics,
        "commands": [
            "python3 -m pip install cig_bench",
            "python3 - <<'PY' ... PropertyPredictor(device='cpu', use_autocast=False) ... PY",
        ],
    }


def render_evidence(report: dict[str, Any]) -> str:
    dev = report["development_inputs"]
    baseline = report["baseline"]
    api = report["api"]
    smoke = report["smoke"]

    lines: list[str] = []
    lines.append("# P18 CIG-Bench Property feasibility")
    lines.append("")
    lines.append(f"Verdict: **{report['verdict']}**")
    lines.append("")
    lines.append("## API / package inspection")
    lines.append(f"- cig_bench version: `{api['cig_bench_version']}`")
    lines.append(f"- modelscope version: `{api['modelscope_version']}`")
    lines.append(f"- torch version: `{api['torch_version']}`")
    lines.append(f"- PropertyPredictor signature: `{api['property_predictor_signature']}`")
    lines.append(f"- property registry entry: `{api['property_registry']['property']}`")
    lines.append("")
    lines.append("## Development inputs actually inspected")
    lines.append(f"- train.h5: `{dev['train']['path']}`")
    lines.append(f"  - sha256: `{dev['train']['sha256']}`")
    lines.append(f"  - sample_count: `{dev['train']['sample_count']}`")
    lines.append(f"  - sample_key: `{dev['train']['sample_key']}`")
    lines.append(f"  - seismic_patch shape: `{dev['train']['seismic_shape']}`")
    lines.append(f"  - well_log_seq shape: `{dev['train']['well_log_seq_shape']}`")
    lines.append(f"  - label shape: `{dev['train']['label_shape']}`")
    lines.append(f"  - meta target_names: `{dev['train']['meta'].get('target_names')}`")
    lines.append(f"- guard.npz: `{dev['guard']['path']}`")
    lines.append(f"  - sha256: `{dev['guard']['sha256']}`")
    lines.append(f"  - sample_count: `{dev['guard']['sample_count']}`")
    lines.append(f"  - seismic_patch shape: `{dev['guard']['seismic_shape']}`")
    lines.append(f"  - well_log_seq shape: `{dev['guard']['well_log_seq_shape']}`")
    lines.append(f"  - label shape: `{dev['guard']['label_shape']}`")
    lines.append("")
    lines.append("## Baseline reference already present in repo")
    lines.append(f"- model: `{baseline['run_manifest']['model']}`")
    lines.append(f"- framework: `{baseline['run_manifest']['framework']}`")
    lines.append(f"- split families: `{baseline['split_manifest']['family_partition']}`")
    lines.append(f"- PHIF RMSE/MAE/R2/Pearson: `{baseline['metrics']['per_target']['PHIF']}`")
    lines.append(f"- log1p(KLOGH) RMSE/MAE/R2/Pearson: `{baseline['metrics']['per_target']['log1p(KLOGH)']}`")
    lines.append(f"- SW RMSE/MAE/R2/Pearson: `{baseline['metrics']['per_target']['SW']}`")
    lines.append("")
    lines.append("## Smoke / blocker probe")
    lines.append(f"- smoke status: `{smoke['status']}`")
    if smoke.get("reason"):
        lines.append(f"- smoke reason: `{smoke['reason']}`")
    if smoke.get("error"):
        lines.append(f"- smoke error: `{smoke['error']}`")
    lines.append("")
    lines.append("## Blocker reasons")
    for reason in report["blocker_reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## Commands run")
    for cmd in report["commands"]:
        lines.append(f"- `{cmd}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- No frozen holdout / test.h5 was opened.")
    lines.append("- No candidate metrics were fabricated.")
    lines.append("- Because the API/weight contract is blocked, there is no honest dev-only baseline comparison to report.")
    return "\n".join(lines) + "\n"


def main() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_evidence()
    EVIDENCE_PATH.write_text(render_evidence(report))
    return report


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
