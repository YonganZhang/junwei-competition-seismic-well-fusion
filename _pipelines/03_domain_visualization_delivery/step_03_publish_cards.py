#!/usr/bin/env python3
"""Publish staged visualizations as permanent card-rendering URLs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from step_01_validate_manifest import PROJECT_ROOT, sha256_file


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_outputs" / "domain_visualization_delivery" / "v1"
DEFAULT_STAGED_MANIFEST = DEFAULT_OUTPUT_DIR / "staged_manifest.json"
DEFAULT_PUBFILE = (
    Path.home() / ".codex" / "skills" / "share-docs" / "scripts" / "pubfile.sh"
)
URL_PATTERN = re.compile(r"https://share\.yongan\.site/[^\s]+")


def publish_cards(
    project_root: Path,
    staged_manifest_path: Path,
    pubfile: Path,
    topic: str,
) -> dict[str, Any]:
    payload = json.loads(staged_manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("staged_track_count") != 6:
        raise RuntimeError("staged manifest is not a passed six-track delivery")
    if not pubfile.is_file():
        raise FileNotFoundError(f"pubfile entrypoint missing: {pubfile}")

    published: list[dict[str, Any]] = []
    for item in payload["tracks"]:
        image_path = project_root / item["staged_image"]
        if sha256_file(image_path) != item["sha256"]:
            raise RuntimeError(f"{item['track']}: staged image hash drift")
        proc = subprocess.run(
            [str(pubfile), str(image_path), f"{topic}-{item['track']}"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        urls = URL_PATTERN.findall(proc.stdout + "\n" + proc.stderr)
        if not urls:
            raise RuntimeError(f"{item['track']}: pubfile returned no permanent URL")
        url = urls[-1].rstrip(".,;)")
        probe = subprocess.run(
            ["curl", "-fsSIL", "--max-time", "20", url],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if probe.returncode != 0 or " 200 " not in probe.stdout:
            raise RuntimeError(
                f"{item['track']}: public URL did not return HTTP 200: {url}"
            )
        published.append(
            {
                **item,
                "url": url,
                "publish_assertions": {
                    "staged_hash_rechecked": True,
                    "permanent_share_url": True,
                    "http_200": True,
                },
            }
        )
    return {
        "schema_version": "domain-visualization-published/v1",
        "status": "passed",
        "staged_manifest": str(staged_manifest_path.relative_to(project_root)),
        "staged_manifest_sha256": sha256_file(staged_manifest_path),
        "published_track_count": len(published),
        "tracks": published,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--staged-manifest",
        type=Path,
        default=DEFAULT_STAGED_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pubfile", type=Path, default=DEFAULT_PUBFILE)
    parser.add_argument("--topic", default="junwei-six-track-domain-viz")
    parser.add_argument("--yes-public", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.yes_public:
        raise SystemExit(
            "Refusing public upload without --yes-public; validation and staging remain local."
        )
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    published = publish_cards(
        project_root,
        args.staged_manifest.resolve(),
        args.pubfile.resolve(),
        args.topic,
    )
    published_path = output_dir / "published_manifest.json"
    published_path.write_text(
        json.dumps(published, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: published {len(published['tracks'])} card-rendering URLs")
    print(published_path)
    for item in published["tracks"]:
        print(f"{item['track']}\t{item['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
