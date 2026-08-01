"""Thin, hash-audited adapter for CIG-Bench's pretrained RGT predictor."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any, Sequence


model_id = "cigbench_rgt"
UPSTREAM_MODEL_ID = "douyimin/CIG-Bench"
UPSTREAM_WEIGHT_FILE = "CIG-Bench-RGT.pth"
UPSTREAM_URL = "https://github.com/douyimin/CIG-bench"
PACKAGE_VERSION = "0.2.0"
PACKAGE_LICENSE = "MIT"
SOURCE_HASHES = {
    "predictor/rgt.py": "d99643bd1efb3017879d24723d81744358619adc349de6847cb0f300e44e10e9",
    "predictor/_download.py": "e682b557315e7ca168c093fe9f0b0eed18256f55b0dbdd304abbb94774d11b0b",
    "networks/hrnet.py": "c19d5e3570c4d2a91f1cc456284688b1f9c42b849e692ca9cd7a8c3eba3abd2f",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["relative_geological_time"],
        "input_modalities": ["seismic_volume_3d"],
        "input_shape": "[T,H,W] (upstream recommended inference grid 400x512x512)",
        "output_shape": "[T,H,W]",
        "foundation_model": UPSTREAM_MODEL_ID,
        "pretraining_domain": "3d_seismic_interpretation",
        "role": "target-free external drift for kriging; never direct porosity prediction",
        "license": PACKAGE_LICENSE,
        "auto_download": True,
        "requires_pretrained_weight": True,
        "supports_missing_mask": False,
    }


def verify_package() -> dict[str, Any]:
    """Verify the installed wheel identity and the exact imported source files."""

    distribution = importlib.metadata.distribution("cig-bench")
    version = distribution.version
    declared_license = str(distribution.metadata.get("License", "")).strip()
    if version != PACKAGE_VERSION:
        raise RuntimeError(
            f"cig-bench version drift: expected {PACKAGE_VERSION}, got {version}"
        )
    if declared_license.upper() != PACKAGE_LICENSE:
        raise RuntimeError(
            f"cig-bench license drift: expected {PACKAGE_LICENSE}, got {declared_license}"
        )
    package_root = Path(distribution.locate_file("cig_bench")).resolve()
    actual = {
        relative: _sha256(package_root / relative) for relative in SOURCE_HASHES
    }
    drift = {
        relative: {"expected": SOURCE_HASHES[relative], "actual": digest}
        for relative, digest in actual.items()
        if digest != SOURCE_HASHES[relative]
    }
    if drift:
        raise RuntimeError(f"cig-bench imported source hash drift: {drift}")
    return {
        "distribution": "cig-bench",
        "version": version,
        "license": declared_license,
        "upstream_url": UPSTREAM_URL,
        "modelscope_model_id": UPSTREAM_MODEL_ID,
        "weight_file": UPSTREAM_WEIGHT_FILE,
        "package_root": str(package_root),
        "source_sha256": actual,
        "installed_metadata_homepage_is_placeholder": (
            distribution.metadata.get("Home-page")
            == "https://github.com/your-org/CIG_Bench"
        ),
    }


def resolve_weight(
    *, cache_dir: Path, revision: str | None = None
) -> tuple[Path, dict[str, Any]]:
    """Use the upstream ModelScope downloader, then byte-lock its result."""

    verify_package()
    from cig_bench.predictor._download import ensure_weight

    path = Path(
        ensure_weight(
            "rgt",
            cache_dir=str(Path(cache_dir).expanduser().resolve()),
            revision=revision,
        )
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CIG-Bench RGT weight missing after download: {path}")
    return path, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "requested_revision": revision,
        "download_provider": "ModelScope",
        "model_id": UPSTREAM_MODEL_ID,
        "file_path": UPSTREAM_WEIGHT_FILE,
    }


def build_predictor(
    *,
    restore_path: Path,
    device: str,
    infer_shape: Sequence[int],
    expected_weight_sha256: str | None = None,
    use_autocast: bool = True,
) -> Any:
    """Build the unmodified upstream predictor from a verified local weight."""

    verify_package()
    restore_path = Path(restore_path).expanduser().resolve()
    if expected_weight_sha256 is not None:
        actual = _sha256(restore_path)
        if actual != expected_weight_sha256:
            raise RuntimeError(
                "CIG-Bench RGT weight hash drift: "
                f"expected {expected_weight_sha256}, got {actual}"
            )
    shape = tuple(int(value) for value in infer_shape)
    if len(shape) != 3 or any(value <= 0 or value % 16 for value in shape):
        raise ValueError("RGT inference dimensions must be positive multiples of 16")
    from cig_bench.predictor.rgt import RGTPredictor

    return RGTPredictor(
        restore_path=str(restore_path),
        device=device,
        infer_shape=shape,
        pad=8,
        use_autocast=use_autocast,
    )
