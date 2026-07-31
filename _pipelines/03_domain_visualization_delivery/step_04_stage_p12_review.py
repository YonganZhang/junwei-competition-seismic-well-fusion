#!/usr/bin/env python3
"""Stage the reviewed P12 figure bundles for tracks 1, 3 and 5.

The track-local manifests deliberately remain machine-generated and keep
``manual_review.reviewed=false``.  This step creates the separate leader
attestation only after an explicit visual-QA acceptance, then copies the
hash-validated PNG/PDF/SVG bundles into one stable project-level directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from step_00_discover import PROJECT_ROOT, TRACKS, DiscoveryError, discover


DEFAULT_OUTPUT_ROOT = Path("_outputs/domain_visualization_delivery/p12")


class P12ReviewError(RuntimeError):
    """Raised when a P12 bundle is not safe to attest or stage."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise P12ReviewError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise P12ReviewError(f"{field} is not a safe relative path: {value!r}")
    return path


def _companion_paths(worktree: Path, record: dict[str, Any]) -> list[Path]:
    png_path = worktree / _safe_relative_path(record["path"], "output.path")
    paths: list[Path] = []
    for value in record["vector_companions"]:
        if value in {"svg", "pdf"}:
            paths.append(png_path.with_suffix(f".{value}"))
        else:
            paths.append(
                worktree
                / _safe_relative_path(value, "output.vector_companions")
            )
    return paths


def _report_path(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def _copy_verified(
    source: Path,
    destination: Path,
    project_root: Path,
) -> dict[str, Any]:
    source_sha = _sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    staged_sha = _sha256_file(destination)
    if staged_sha != source_sha:
        destination.unlink(missing_ok=True)
        raise P12ReviewError(
            f"hash changed while staging {source}: {source_sha} != {staged_sha}"
        )
    return {
        "source_path": _report_path(project_root, source),
        "staged_path": _report_path(project_root, destination),
        "sha256": source_sha,
        "size_bytes": destination.stat().st_size,
    }


def stage_p12_review(
    project_root: Path,
    output_root: Path,
    reviewer: str,
    *,
    accept_visual_qa: bool,
) -> dict[str, Any]:
    """Validate, attest and stage all P12 bundles.

    ``accept_visual_qa`` is intentionally not inferred from a worker manifest:
    it must come from the leader who opened every PNG and checked clipping,
    overlap, legibility, palette consistency and scientific boundaries.
    """

    if not accept_visual_qa:
        raise P12ReviewError(
            "refusing to stage P12 without explicit --accept-visual-qa"
        )
    reviewer = reviewer.strip()
    if not reviewer:
        raise P12ReviewError("reviewer must be a non-empty identity")

    project_root = project_root.resolve()
    destination_root = (
        output_root
        if output_root.is_absolute()
        else project_root / output_root
    ).resolve()
    try:
        discovery = discover(project_root, require_artifacts=True)
    except DiscoveryError as exc:
        raise P12ReviewError(str(exc)) from exc
    if discovery["status"] != "ready":
        raise P12ReviewError("P12 discovery is not ready")

    track_attestations: list[dict[str, Any]] = []
    for track, spec in TRACKS.items():
        worktree = project_root / ".claude" / "worktrees" / spec["worktree"]
        manifest_path = worktree / spec["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest["p12_contract"]
        source_review = contract["manual_review"]
        if source_review["reviewed"] is not False:
            raise P12ReviewError(
                f"{track}: renderer-owned manual_review must remain pending"
            )

        staged_files: list[dict[str, Any]] = []
        for output in contract["outputs"]:
            png_source = worktree / _safe_relative_path(
                output["path"], f"{track}.output.path"
            )
            sources = [png_source, *_companion_paths(worktree, output)]
            for source in sources:
                destination = destination_root / track / source.name
                staged_files.append(
                    _copy_verified(source, destination, project_root)
                )

        track_attestations.append(
            {
                "track": track,
                "track_number": spec["track_number"],
                "worktree": spec["worktree"],
                "branch": spec["branch"],
                "head": next(
                    item["head"]
                    for item in discovery["tracks"]
                    if item["track"] == track
                ),
                "source_manifest": spec["manifest"],
                "source_manifest_sha256": _sha256_file(manifest_path),
                "source_contract_review_status": "pending",
                "review": {
                    "reviewed": True,
                    "reviewer": reviewer,
                    "no_clipping": True,
                    "no_overlap": True,
                    "labels_legible": True,
                    "colors_consistent": True,
                    "scientific_boundary_preserved": True,
                },
                "staged_files": staged_files,
            }
        )

    report = {
        "schema_version": "scientific-visualization-review-attestation/v1",
        "profile": discovery["profile"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "included_tracks": discovery["included_tracks"],
        "paused_tracks": discovery["paused_tracks"],
        "style_guide": discovery["style_guide"],
        "status": "accepted",
        "tracks": track_attestations,
    }
    destination_root.mkdir(parents=True, exist_ok=True)
    attestation_path = destination_root / "review_attestation.json"
    attestation_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["attestation_path"] = _report_path(project_root, attestation_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--accept-visual-qa",
        action="store_true",
        help=(
            "Attest that every P12 PNG was opened and checked for clipping, "
            "overlap, legibility, palette consistency and scientific boundaries."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = stage_p12_review(
        args.project_root,
        args.output_root,
        args.reviewer,
        accept_visual_qa=args.accept_visual_qa,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
