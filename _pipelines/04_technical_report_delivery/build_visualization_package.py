#!/usr/bin/env python3
"""Build the stable, incrementally updateable visualization delivery archive."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "_outputs" / "technical_report_delivery"
OUTPUT_ZIP = OUTPUT_DIR / "junwei_visualizations_latest.zip"
OUTPUT_MANIFEST = OUTPUT_DIR / "package_manifest.json"
OUTPUT_SHA256 = OUTPUT_DIR / "junwei_visualizations_latest.zip.sha256"
STABLE_URL = (
    "https://share.yongan.site/junwei-visualizations/"
    "junwei_visualizations_latest.zip"
)

P12_ROOT = PROJECT_ROOT / "_outputs" / "domain_visualization_delivery" / "p12"
REPORT_FIGURES_ROOT = PROJECT_ROOT / "_paper" / "technical_report" / "figures"
README_PATH = Path(__file__).resolve().with_name("PACKAGE_README.md")
ATTESTATION_PATH = P12_ROOT / "review_attestation.json"
STYLE_GUIDE_PATH = PROJECT_ROOT / "_meta" / "_visual_style_guide.yml"

FIGURE_SUFFIXES = {".png", ".pdf", ".svg"}
FIXED_ZIP_TIME = (2026, 7, 30, 0, 0, 0)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_inputs() -> list[tuple[Path, str, str]]:
    """Return source path, package path and scientific role."""
    records: list[tuple[Path, str, str]] = []

    if not P12_ROOT.is_dir():
        raise FileNotFoundError(f"missing reviewed visualization root: {P12_ROOT}")

    for source in sorted(P12_ROOT.rglob("*")):
        if source.is_file() and source.suffix.lower() in FIGURE_SUFFIXES:
            relative = source.relative_to(P12_ROOT)
            records.append(
                (source, f"visualizations/p12/{relative.as_posix()}", "p12_reviewed_figure")
            )

    if REPORT_FIGURES_ROOT.is_dir():
        for source in sorted(REPORT_FIGURES_ROOT.rglob("*")):
            if source.is_file() and source.suffix.lower() in FIGURE_SUFFIXES:
                relative = source.relative_to(REPORT_FIGURES_ROOT)
                records.append(
                    (
                        source,
                        f"technical_report/figures/{relative.as_posix()}",
                        "technical_report_figure",
                    )
                )

    required_evidence = (
        (ATTESTATION_PATH, "evidence/review_attestation.json", "review_attestation"),
        (STYLE_GUIDE_PATH, "evidence/visual_style_guide.yml", "style_contract"),
        (README_PATH, "README.md", "delivery_readme"),
    )
    for source, archive_path, role in required_evidence:
        if not source.is_file():
            raise FileNotFoundError(f"missing required package evidence: {source}")
        records.append((source, archive_path, role))

    archive_paths = [archive_path for _, archive_path, _ in records]
    if len(archive_paths) != len(set(archive_paths)):
        raise ValueError("duplicate archive paths detected")
    return sorted(records, key=lambda row: row[1])


def zip_write_bytes(bundle: zipfile.ZipFile, archive_path: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(archive_path, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    bundle.writestr(info, payload)


def build_manifest(inputs: list[tuple[Path, str, str]]) -> dict[str, object]:
    files = []
    for source, archive_path, role in inputs:
        files.append(
            {
                "path": archive_path,
                "scientific_role": role,
                "size_bytes": source.stat().st_size,
                "sha256": sha256_path(source),
            }
        )

    return {
        "schema_version": "junwei-visualization-package/v1",
        "package_basename": OUTPUT_ZIP.name,
        "stable_url": STABLE_URL,
        "update_policy": "atomic overwrite of latest pointer; timestamped versions retained",
        "included_tracks_p12": ["fault", "property", "sweetspot"],
        "paused_tracks_p12": ["facies", "lithofacies", "reconstruction"],
        "figure_count": sum(
            1
            for row in files
            if Path(str(row["path"])).suffix.lower() in FIGURE_SUFFIXES
        ),
        "png_count": sum(1 for row in files if str(row["path"]).endswith(".png")),
        "files": files,
    }


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def main() -> None:
    inputs = iter_inputs()
    manifest = build_manifest(inputs)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".zip", dir=OUTPUT_DIR, delete=False
    ) as handle:
        temp_zip = Path(handle.name)

    try:
        with zipfile.ZipFile(
            temp_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as bundle:
            for source, archive_path, _ in inputs:
                zip_write_bytes(bundle, archive_path, source.read_bytes())
            zip_write_bytes(bundle, "PACKAGE_MANIFEST.json", manifest_bytes)

        with zipfile.ZipFile(temp_zip, "r") as bundle:
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise RuntimeError(f"zip integrity failure: {bad_member}")

        os.replace(temp_zip, OUTPUT_ZIP)
    finally:
        if temp_zip.exists():
            temp_zip.unlink()

    zip_sha256 = sha256_path(OUTPUT_ZIP)
    atomic_write(OUTPUT_MANIFEST, manifest_bytes)
    atomic_write(
        OUTPUT_SHA256,
        f"{zip_sha256}  {OUTPUT_ZIP.name}\n".encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT_ZIP.relative_to(PROJECT_ROOT)),
                "sha256": zip_sha256,
                "figure_count": manifest["figure_count"],
                "png_count": manifest["png_count"],
                "stable_url": STABLE_URL,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

