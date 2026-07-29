#!/usr/bin/env python3
"""Stage only artifacts approved by step 01 and verify copied hashes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from step_01_validate_manifest import PROJECT_ROOT, sha256_file


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_outputs" / "domain_visualization_delivery" / "v1"
DEFAULT_VALIDATED_MANIFEST = DEFAULT_OUTPUT_DIR / "validated_manifest.json"


def stage_delivery(
    project_root: Path,
    validated_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    payload = json.loads(validated_manifest_path.read_text(encoding="utf-8"))
    if payload.get("validation_status") != "passed":
        raise RuntimeError("validated manifest is not passed")
    tracks = payload.get("tracks", [])
    if len(tracks) != 6:
        raise RuntimeError(f"expected six validated tracks, got {len(tracks)}")

    cards_dir = output_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    staged: list[dict[str, Any]] = []
    for item in tracks:
        if item.get("validation_status") != "passed":
            raise RuntimeError(f"{item.get('track')}: validation_status is not passed")
        source = project_root / item["resolved"]["image_path"]
        expected_sha = item["resolved"]["image_sha256"]
        if sha256_file(source) != expected_sha:
            raise RuntimeError(f"{item['track']}: source hash drifted after validation")
        destination = cards_dir / item["delivery_filename"]
        shutil.copy2(source, destination)
        copied_sha = sha256_file(destination)
        if copied_sha != expected_sha:
            raise RuntimeError(f"{item['track']}: copied hash mismatch")
        staged.append(
            {
                "track": item["track"],
                "display_name": item["display_name"],
                "caption": item["caption"],
                "scientific_scope": item["scientific_scope"],
                "artifact_class": item["artifact_class"],
                "source_commit": item["source_commit"],
                "source_image": item["resolved"]["image_path"],
                "staged_image": str(destination.relative_to(project_root)),
                "sha256": copied_sha,
                "bytes": destination.stat().st_size,
                "png_width": item["resolved"]["png_width"],
                "png_height": item["resolved"]["png_height"],
                "stage_assertions": {
                    "source_hash_rechecked": True,
                    "copy_hash_matches": True,
                    "came_from_validated_manifest": True,
                },
            }
        )

    return {
        "schema_version": "domain-visualization-staged/v1",
        "status": "passed",
        "validated_manifest": str(validated_manifest_path.relative_to(project_root)),
        "validated_manifest_sha256": sha256_file(validated_manifest_path),
        "staged_track_count": len(staged),
        "tracks": staged,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--validated-manifest",
        type=Path,
        default=DEFAULT_VALIDATED_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_delivery(
        project_root,
        args.validated_manifest.resolve(),
        output_dir,
    )
    staged_path = output_dir / "staged_manifest.json"
    staged_path.write_text(
        json.dumps(staged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: staged {len(staged['tracks'])} hash-verified domain visualizations")
    print(staged_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
