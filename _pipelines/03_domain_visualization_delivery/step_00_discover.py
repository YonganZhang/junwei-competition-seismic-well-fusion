#!/usr/bin/env python3
"""Discover the active P12 scientific-visualization worktrees and entrypoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STYLE_GUIDE = Path("_meta/_visual_style_guide.yml")
PROFILE_ID = "p12_tracks_1_3_5"
PAUSED_TRACKS = ("facies", "lithofacies", "reconstruction")
TRACKS: dict[str, dict[str, str]] = {
    "fault": {
        "track_number": "1",
        "worktree": "p12-viz-fault",
        "branch": "p12-viz-fault",
        "renderer": "_pipelines/02_task_datasets/fault/p12_visualization.py",
        "test": "_pipelines/02_task_datasets/fault/test_p12_visualization.py",
        "manifest": "_pipelines/02_task_datasets/fault/_outputs/p12_visualization/manifest.json",
    },
    "property": {
        "track_number": "3",
        "worktree": "p12-viz-property",
        "branch": "p12-viz-property",
        "renderer": "_pipelines/02_task_datasets/reservoir/p12_visualization.py",
        "test": "_pipelines/02_task_datasets/reservoir/test_p12_visualization.py",
        "manifest": "_pipelines/02_task_datasets/reservoir/_outputs/p12_visualization/manifest.json",
    },
    "sweetspot": {
        "track_number": "5",
        "worktree": "p12-viz-sweetspot",
        "branch": "p12-viz-sweetspot",
        "renderer": "_pipelines/02_task_datasets/sweetspot/p12_visualization.py",
        "test": "_pipelines/02_task_datasets/sweetspot/tests/test_p12_visualization.py",
        "manifest": "_pipelines/02_task_datasets/sweetspot/_outputs/p12_visualization/manifest.json",
    },
}


class DiscoveryError(RuntimeError):
    """Raised when an advertised P12 visualization entrypoint is unavailable."""


def _git(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise DiscoveryError(f"not a PNG with an IHDR header: {path}")
    return struct.unpack(">II", header[16:24])


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DiscoveryError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise DiscoveryError(f"{field} is not a safe relative path: {value!r}")
    return path


def _resolve_input_path(project_root: Path, worktree: Path, value: Any) -> Path:
    relative = _safe_relative_path(value, "input.path")
    if relative.parts[:2] == (".claude", "worktrees"):
        return project_root / relative
    return worktree / relative


def _validate_contract(
    project_root: Path,
    worktree: Path,
    track: str,
    spec: dict[str, str],
    manifest_path: Path,
    head: str,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = payload.get("p12_contract")
    if not isinstance(contract, dict):
        raise DiscoveryError(f"{track}: manifest is missing p12_contract")
    expected_scalars = {
        "schema_version": "scientific-visualization-contract/v1",
        "profile": PROFILE_ID,
        "track_id": track,
    }
    for field, expected in expected_scalars.items():
        if contract.get(field) != expected:
            raise DiscoveryError(
                f"{track}: p12_contract.{field} must be {expected!r}, "
                f"got {contract.get(field)!r}"
            )

    source_commit = contract.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise DiscoveryError(f"{track}: p12_contract.source_commit is required")
    ancestor = subprocess.run(
        ["git", "-C", str(worktree), "merge-base", "--is-ancestor", source_commit, head],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    if not ancestor:
        raise DiscoveryError(
            f"{track}: source_commit is not an ancestor of current HEAD"
        )

    renderer = contract.get("renderer")
    if not isinstance(renderer, dict):
        raise DiscoveryError(f"{track}: p12_contract.renderer must be an object")
    renderer_rel = _safe_relative_path(renderer.get("path"), "renderer.path")
    if renderer_rel.as_posix() != spec["renderer"]:
        raise DiscoveryError(
            f"{track}: renderer path drift: {renderer_rel} != {spec['renderer']}"
        )
    renderer_path = worktree / renderer_rel
    if not renderer_path.is_file():
        raise DiscoveryError(f"{track}: renderer is missing: {renderer_path}")
    renderer_sha = _sha256_file(renderer_path)
    if renderer.get("sha256") != renderer_sha:
        raise DiscoveryError(f"{track}: renderer hash drift")

    if not contract.get("generated_at") or not contract.get("scientific_caveat"):
        raise DiscoveryError(
            f"{track}: generated_at and scientific_caveat are required"
        )

    inputs = contract.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise DiscoveryError(f"{track}: p12_contract.inputs must be a non-empty list")
    required_input_fields = {
        "path",
        "sha256",
        "shape_or_row_count",
        "scientific_role",
        "split_scope",
    }
    for index, record in enumerate(inputs):
        if not isinstance(record, dict):
            raise DiscoveryError(f"{track}: input {index} must be an object")
        missing = sorted(required_input_fields - record.keys())
        if missing:
            raise DiscoveryError(f"{track}: input {index} missing fields {missing}")
        source_path = _resolve_input_path(project_root, worktree, record["path"])
        if not source_path.is_file():
            raise DiscoveryError(f"{track}: input {index} is missing: {source_path}")
        if _sha256_file(source_path) != record["sha256"]:
            raise DiscoveryError(f"{track}: input {index} hash drift")

    outputs = contract.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise DiscoveryError(f"{track}: p12_contract.outputs must be a non-empty list")
    required_output_fields = {
        "role",
        "path",
        "sha256",
        "width_px",
        "height_px",
        "dpi",
        "vector_companions",
    }
    for index, record in enumerate(outputs):
        if not isinstance(record, dict):
            raise DiscoveryError(f"{track}: output {index} must be an object")
        missing = sorted(required_output_fields - record.keys())
        if missing:
            raise DiscoveryError(f"{track}: output {index} missing fields {missing}")
        output_rel = _safe_relative_path(record["path"], f"output[{index}].path")
        output_path = worktree / output_rel
        if output_path.suffix.lower() != ".png" or not output_path.is_file():
            raise DiscoveryError(f"{track}: output {index} must be an existing PNG")
        if _sha256_file(output_path) != record["sha256"]:
            raise DiscoveryError(f"{track}: output {index} hash drift")
        width, height = _png_dimensions(output_path)
        if (record["width_px"], record["height_px"]) != (width, height):
            raise DiscoveryError(f"{track}: output {index} dimension drift")
        if width < 1200 or height < 700 or record["dpi"] != 300:
            raise DiscoveryError(
                f"{track}: output {index} does not meet 1200x700 at 300 DPI"
            )
        companions = record["vector_companions"]
        if not isinstance(companions, list) or len(companions) != 2:
            raise DiscoveryError(
                f"{track}: output {index} must list PDF and SVG companions"
            )
        companion_paths: list[Path] = []
        for value in companions:
            if value in {"svg", "pdf"}:
                companion_paths.append(output_path.with_suffix(f".{value}"))
            else:
                companion_paths.append(
                    worktree
                    / _safe_relative_path(
                        value, f"output[{index}].vector_companions"
                    )
                )
        if {path.suffix.lower() for path in companion_paths} != {".svg", ".pdf"}:
            raise DiscoveryError(
                f"{track}: output {index} companions must be one PDF and one SVG"
            )
        for companion in companion_paths:
            if not companion.is_file() or companion.stat().st_size == 0:
                raise DiscoveryError(
                    f"{track}: output {index} missing vector companion {companion}"
                )

    review = contract.get("manual_review")
    required_review_fields = {
        "reviewed",
        "reviewed_sha256",
        "reviewer",
        "no_clipping",
        "no_overlap",
        "labels_legible",
        "colors_consistent",
        "scientific_boundary_preserved",
    }
    if not isinstance(review, dict):
        raise DiscoveryError(f"{track}: p12_contract.manual_review must be an object")
    missing_review = sorted(required_review_fields - review.keys())
    if missing_review:
        raise DiscoveryError(
            f"{track}: manual_review missing fields {missing_review}"
        )
    if review["reviewed"] not in {False, True}:
        raise DiscoveryError(f"{track}: manual_review.reviewed must be boolean")

    return {
        "schema_version": contract["schema_version"],
        "profile": contract["profile"],
        "source_commit": source_commit,
        "source_commit_is_ancestor": ancestor,
        "renderer_sha256": renderer_sha,
        "input_count": len(inputs),
        "output_count": len(outputs),
        "manual_review_status": "accepted" if review["reviewed"] else "pending",
    }


def discover(project_root: Path, require_artifacts: bool = False) -> dict[str, Any]:
    project_root = project_root.resolve()
    style_guide = project_root / STYLE_GUIDE
    if not style_guide.is_file():
        raise DiscoveryError(f"shared style guide missing: {style_guide}")

    discovered: list[dict[str, Any]] = []
    failures: list[str] = []
    for track, spec in TRACKS.items():
        worktree = project_root / ".claude" / "worktrees" / spec["worktree"]
        item: dict[str, Any] = {
            "track": track,
            **spec,
            "worktree_path": str(worktree),
            "render_command": f"python3 {spec['renderer']}",
            "test_command": f"python3 {spec['test']}",
        }
        if not worktree.is_dir():
            item["status"] = "missing_worktree"
            failures.append(f"{track}: missing worktree {worktree}")
            discovered.append(item)
            continue

        item["actual_branch"] = _git(worktree, "branch", "--show-current")
        item["head"] = _git(worktree, "rev-parse", "HEAD")
        if item["actual_branch"] != spec["branch"]:
            failures.append(
                f"{track}: expected branch {spec['branch']}, got {item['actual_branch']}"
            )

        file_status: dict[str, bool] = {}
        for field in ("renderer", "test", "manifest"):
            file_status[field] = (worktree / spec[field]).is_file()
            if require_artifacts and not file_status[field]:
                failures.append(f"{track}: missing {field} {spec[field]}")
        item["files"] = file_status
        if file_status["manifest"]:
            try:
                item["contract"] = _validate_contract(
                    project_root,
                    worktree,
                    track,
                    spec,
                    worktree / spec["manifest"],
                    item["head"],
                )
            except (DiscoveryError, json.JSONDecodeError) as exc:
                item["contract_error"] = str(exc)
                failures.append(str(exc))
        item["status"] = (
            "ready"
            if (
                item["actual_branch"] == spec["branch"]
                and all(file_status.values())
                and "contract" in item
            )
            else "in_progress"
        )
        discovered.append(item)

    if failures and require_artifacts:
        raise DiscoveryError("; ".join(failures))
    return {
        "schema_version": "scientific-visualization-discovery/v1",
        "profile": PROFILE_ID,
        "style_guide": str(STYLE_GUIDE),
        "included_tracks": list(TRACKS),
        "paused_tracks": list(PAUSED_TRACKS),
        "require_artifacts": require_artifacts,
        "status": "ready" if not failures and all(x["status"] == "ready" for x in discovered) else "in_progress",
        "tracks": discovered,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail unless every advertised renderer, test and manifest exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = discover(args.project_root, require_artifacts=args.check)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
