#!/usr/bin/env python3
"""Fail-closed validation for six-track domain-visualization delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("delivery_manifest.json")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_outputs" / "domain_visualization_delivery" / "v1"
REQUIRED_TRACKS = (
    "fault",
    "facies",
    "property",
    "lithofacies",
    "sweetspot",
    "reconstruction",
)
ALLOWED_CLASSES = {
    "real_model_prediction",
    "real_data_diagnostic",
    "real_domain_evaluation",
}
DISALLOWED_PATH_TOKENS = {
    "status",
    "readiness",
    "protocol",
    "gate",
    "placeholder",
    "not_feasible",
    "summary",
    "p5_r2_visualization",
}
REQUIRED_REVIEW_CHECKS = {
    "domain_content_visible",
    "real_data_or_model_output",
    "not_protocol_or_status",
    "not_placeholder",
    "scientific_boundary_visible_or_captioned",
}


class ValidationError(RuntimeError):
    """Raised when an artifact is unsafe to deliver."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValidationError(f"{path}: not a valid PNG with an IHDR header")
    return struct.unpack(">II", header[16:24])


def run_git(worktree: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def validate_relative_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValidationError(f"{field}: path must be project-relative without '..': {value}")
    return path


def validate_path_policy(image_path: str) -> None:
    lowered_parts = [part.lower() for part in Path(image_path).parts]
    lowered_name = Path(image_path).name.lower()
    for token in DISALLOWED_PATH_TOKENS:
        if any(token in part for part in lowered_parts) or token in lowered_name:
            raise ValidationError(
                f"image path rejected by domain-delivery policy: token={token!r}, path={image_path}"
            )


def validate_track(project_root: Path, item: dict[str, Any]) -> dict[str, Any]:
    track = item.get("track")
    if track not in REQUIRED_TRACKS:
        raise ValidationError(f"unknown track: {track!r}")
    if item.get("artifact_class") not in ALLOWED_CLASSES:
        raise ValidationError(f"{track}: invalid artifact_class={item.get('artifact_class')!r}")

    worktree_name = item.get("worktree", "")
    if not worktree_name or Path(worktree_name).name != worktree_name:
        raise ValidationError(f"{track}: unsafe worktree name={worktree_name!r}")
    worktree = project_root / ".claude" / "worktrees" / worktree_name
    if not worktree.is_dir():
        raise ValidationError(f"{track}: worktree does not exist: {worktree}")

    expected_head = item.get("source_commit")
    if not expected_head:
        raise ValidationError(f"{track}: source_commit is required")
    head = run_git(worktree, "rev-parse", "HEAD").stdout.strip()
    source_is_ancestor = (
        run_git(
            worktree,
            "merge-base",
            "--is-ancestor",
            expected_head,
            head,
            check=False,
        ).returncode
        == 0
    )
    if not source_is_ancestor:
        raise ValidationError(
            f"{track}: source commit is not an ancestor of current HEAD: "
            f"source={expected_head}, actual={head}"
        )

    image_rel = validate_relative_path(item["image_path"], f"{track}.image_path")
    validate_path_policy(item["image_path"])
    image_path = worktree / image_rel
    if not image_path.is_file():
        raise ValidationError(f"{track}: image missing: {image_path}")
    if image_path.suffix.lower() != ".png":
        raise ValidationError(f"{track}: only PNG delivery is allowed: {image_path}")

    width, height = png_dimensions(image_path)
    if width < 500 or height < 300:
        raise ValidationError(f"{track}: image is too small for delivery: {width}x{height}")
    image_sha = sha256_file(image_path)
    if image_sha != item.get("image_sha256"):
        raise ValidationError(
            f"{track}: image hash drift: expected={item.get('image_sha256')}, actual={image_sha}"
        )

    script_rel = validate_relative_path(item["source_script"], f"{track}.source_script")
    script_path = worktree / script_rel
    if not script_path.is_file():
        raise ValidationError(f"{track}: source script missing: {script_path}")
    if run_git(worktree, "ls-files", "--error-unmatch", str(script_rel), check=False).returncode != 0:
        raise ValidationError(f"{track}: source script is not Git tracked: {script_rel}")
    script_blob = run_git(
        worktree,
        "show",
        f"{expected_head}:{script_rel.as_posix()}",
        check=False,
    )
    if script_blob.returncode != 0:
        raise ValidationError(f"{track}: source script is absent at source_commit: {script_rel}")
    current_script_sha = sha256_file(script_path)
    source_script_sha = hashlib.sha256(script_blob.stdout.encode()).hexdigest()
    if current_script_sha != source_script_sha:
        raise ValidationError(
            f"{track}: source script drift after source_commit: {script_rel}"
        )

    evidence_rel = validate_relative_path(item["evidence_path"], f"{track}.evidence_path")
    evidence_path = worktree / evidence_rel
    if not evidence_path.is_file():
        raise ValidationError(f"{track}: evidence missing: {evidence_path}")

    tracked = (
        run_git(worktree, "ls-files", "--error-unmatch", str(image_rel), check=False).returncode
        == 0
    )
    if tracked != bool(item.get("image_tracked")):
        raise ValidationError(
            f"{track}: image_tracked drift: manifest={item.get('image_tracked')}, actual={tracked}"
        )
    expected_policy = "git_tracked" if tracked else "generated_ignored"
    if item.get("artifact_policy") != expected_policy:
        raise ValidationError(
            f"{track}: artifact_policy must be {expected_policy!r}, got {item.get('artifact_policy')!r}"
        )

    review = item.get("review", {})
    if review.get("approved") is not True:
        raise ValidationError(f"{track}: human review is not approved")
    if review.get("reviewed_sha256") != image_sha:
        raise ValidationError(f"{track}: human review hash does not match current image")
    checks = review.get("checks", {})
    missing_checks = sorted(
        check for check in REQUIRED_REVIEW_CHECKS if checks.get(check) is not True
    )
    if missing_checks:
        raise ValidationError(f"{track}: failed human-review checks: {missing_checks}")
    if not item.get("scientific_scope") or not item.get("caption"):
        raise ValidationError(f"{track}: scientific_scope and caption are required")

    delivery_name = Path(item.get("delivery_filename", ""))
    if delivery_name.name != str(delivery_name) or delivery_name.suffix.lower() != ".png":
        raise ValidationError(f"{track}: invalid delivery_filename={delivery_name}")

    return {
        **item,
        "resolved": {
            "worktree_path": str(worktree.relative_to(project_root)),
            "image_path": str(image_path.relative_to(project_root)),
            "source_script": str(script_path.relative_to(project_root)),
            "evidence_path": str(evidence_path.relative_to(project_root)),
            "image_sha256": image_sha,
            "source_script_sha256": current_script_sha,
            "evidence_sha256": sha256_file(evidence_path),
            "png_width": width,
            "png_height": height,
            "image_git_tracked": tracked,
            "worktree_head": head,
            "source_commit_is_ancestor": source_is_ancestor,
        },
        "validation_status": "passed",
    }


def validate_manifest(project_root: Path, manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "domain-visualization-delivery/v1":
        raise ValidationError(f"unsupported schema_version={payload.get('schema_version')!r}")
    if payload.get("delivery_class") != "real_domain_visualization":
        raise ValidationError("delivery_class must be real_domain_visualization")
    items = payload.get("tracks")
    if not isinstance(items, list):
        raise ValidationError("tracks must be a list")
    observed = tuple(item.get("track") for item in items)
    if observed != REQUIRED_TRACKS:
        raise ValidationError(
            f"tracks must be exactly {REQUIRED_TRACKS}, in order; observed={observed}"
        )
    validated = [validate_track(project_root, item) for item in items]
    manifest_sha = sha256_file(manifest_path)
    return {
        "schema_version": "domain-visualization-validation/v1",
        "validation_status": "passed",
        "input_manifest": str(manifest_path.resolve().relative_to(project_root)),
        "input_manifest_sha256": manifest_sha,
        "required_tracks": list(REQUIRED_TRACKS),
        "validated_track_count": len(validated),
        "tracks": validated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest_path = args.manifest.resolve()
    report = validate_manifest(project_root, manifest_path)
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validated_path = output_dir / "validated_manifest.json"
    report_path = output_dir / "validation_report.json"
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    validated_path.write_text(rendered, encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "domain-visualization-validation-report/v1",
                "status": "passed",
                "assertions": {
                    "exact_six_tracks": True,
                    "all_files_exist": True,
                    "all_hashes_match": True,
                    "all_sources_and_evidence_exist": True,
                    "no_protocol_status_or_placeholder_paths": True,
                    "all_human_reviews_match_current_hash": True,
                },
                "validated_manifest_sha256": sha256_file(validated_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PASS: validated {len(report['tracks'])} real domain visualizations")
    print(validated_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
