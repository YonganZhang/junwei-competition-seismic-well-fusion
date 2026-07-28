#!/usr/bin/env python3
"""Audit whether the connected SAM-Med3D route can be evaluated honestly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = HERE / "_outputs" / "p9_sammed3d_gate" / "summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    *,
    development_h5: Path,
    p5_gate_manifest: Path,
    routes: Path,
    runtime_smoke: Path,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    gate = json.loads(p5_gate_manifest.read_text(encoding="utf-8"))
    route_payload = json.loads(routes.read_text(encoding="utf-8"))
    route = next(
        item for item in route_payload["routes"] if item["track_id"] == "fault"
    )
    smoke = json.loads(runtime_smoke.read_text(encoding="utf-8"))["tracks"]["fault"]
    kinds: dict[str, int] = {}
    positive_voxels = 0
    zero_voxels = 0
    assumed_negative_samples = 0
    shapes: set[tuple[int, ...]] = set()
    with h5py.File(development_h5, "r") as handle:
        for group in handle.values():
            meta = json.loads(group.attrs["meta"])
            kind = str(meta["sample_kind"])
            kinds[kind] = kinds.get(kind, 0) + 1
            assumed_negative_samples += int(
                bool(meta.get("unannotated_patch_assumed_negative"))
            )
            label = np.asarray(group["label"])
            positive_voxels += int(np.sum(label == 1))
            zero_voxels += int(np.sum(label == 0))
            shapes.add(tuple(int(value) for value in group["seismic_patch"].shape))
    expected_blockers = {
        "AUDITED_VERIFIED_NEGATIVE_COVERAGE_MISSING",
        "COVERAGE_AUDITED_UNKNOWN_MASK_MISSING",
        "DEVELOPMENT_SPLIT_NOT_FEASIBLE",
    }
    actual_blockers = {row["code"] for row in gate["blockers"]}
    if not expected_blockers <= actual_blockers:
        raise RuntimeError("fault P5 data gate no longer matches the P9 assumptions")
    result = {
        "schema_version": "fault-p9-sammed3d-gate/v1",
        "track_id": "fault",
        "foundation_route": {
            "adapter": route["adapter"],
            "model_id": route["model"]["model_id"],
            "real_checkpoint_sha256": smoke["checkpoint_sha256"],
            "source_revision": smoke["source_revision"],
            "synthetic_forward_passed": smoke["finite"],
            "parameter_count": smoke["parameter_count"],
            "input_shape": smoke["input_shape"],
            "output_shape": smoke["output_shape"],
        },
        "development_data_audit": {
            "h5_sha256": _sha256(development_h5),
            "sample_count": sum(kinds.values()),
            "sample_kind_counts": kinds,
            "input_shapes": [list(shape) for shape in sorted(shapes)],
            "patch_axes": ["crossline", "time"],
            "contiguous_3d_volume_count": 0,
            "positive_voxels": positive_voxels,
            "zero_voxels": zero_voxels,
            "unannotated_patch_assumed_negative_samples": assumed_negative_samples,
            "verified_negative_voxels": int(
                gate["coverage"]["negative_provenance"][
                    "verified_negative_labels"
                ]
            ),
            "explicit_unknown_mask_present": False,
        },
        "data_gate": {
            "source_manifest_sha256": _sha256(p5_gate_manifest),
            "effective_development_folds": gate["split"]["effective_n_splits"],
            "blockers": sorted(actual_blockers),
            "frozen_test_accessed": False,
        },
        "decision": {
            "state": "CONNECTED_DATA_GATE_BLOCKED",
            "default_enabled": False,
            "reason": (
                "The real SAM-Med3D checkpoint and forward path work, but the "
                "available development samples are isolated 2-D slices and their "
                "background labels rely on an unannotated-as-negative assumption. "
                "No legal 3-D development score can be produced."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-h5", type=Path, required=True)
    parser.add_argument("--p5-gate-manifest", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--runtime-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
