"""Frozen MOMENT adapter for six-curve, 33-sample reconstruction windows."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any


model_id = "moment_well"
CHANNELS = 6
LENGTH = 33
PATCHES = 4
EMBEDDING_DIM = 768
EXPECTED_WEIGHTS_SHA256 = (
    "1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825"
)
WEIGHT_MODES = ("pretrained", "random_init")


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["measured_depth_six_curve_window"],
        "input_shape": "[B,6,33]",
        "input_mask_shape": "[B,33]",
        "output_shape": "[B,6,4,768]",
        "foundation_model": "AutonLab/MOMENT-1-base",
        "conditioning": "native_33_point_MD_window",
        "supports_missing_mask": True,
        "requires_pretrained_weight": True,
        "supports_same_architecture_random_init": True,
        "auto_download": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_assets(snapshot_path: Path) -> dict[str, Any]:
    snapshot = Path(snapshot_path).expanduser().resolve()
    required = {
        "config": snapshot / "config.json",
        "weights": snapshot / "model.safetensors",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("MOMENT local asset missing: " + ", ".join(missing))
    weights_sha256 = _sha256(required["weights"])
    if weights_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError("MOMENT weights hash drift")
    return {
        "model_id": "AutonLab/MOMENT-1-base",
        "snapshot_revision": snapshot.name,
        "weights_sha256": weights_sha256,
        "config_sha256": _sha256(required["config"]),
        "local_only": True,
        "auto_download": False,
    }


def _add_import_roots(dependency_root: Path, source_root: Path) -> None:
    for root in (dependency_root, source_root):
        resolved = str(Path(root).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _randomize(torch: Any, module: Any, *, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    for name, parameter in module.named_parameters():
        if parameter.ndim >= 2:
            torch.nn.init.xavier_uniform_(parameter)
        elif name.endswith("weight"):
            torch.nn.init.ones_(parameter)
        else:
            torch.nn.init.zeros_(parameter)


def build_encoder(
    *,
    snapshot_path: Path,
    dependency_root: Path,
    source_root: Path,
    device: str,
    weight_mode: str = "pretrained",
    random_seed: int = 2693,
) -> Any:
    """Build a frozen native-length encoder; no task head is exposed."""

    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}")
    audit = verify_local_assets(snapshot_path)
    _add_import_roots(dependency_root, source_root)
    import torch
    from momentfm import MOMENTPipeline

    pipeline = MOMENTPipeline.from_pretrained(
        str(Path(snapshot_path).expanduser().resolve()),
        local_files_only=True,
        model_kwargs={
            "task_name": "classification",
            "n_channels": CHANNELS,
            "num_class": 1,
            "seq_len": LENGTH,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": True,
        },
    )
    pipeline.init()
    if (
        int(pipeline.config.seq_len) != LENGTH
        or int(pipeline.config.patch_len) != 8
        or int(pipeline.config.patch_stride_len) != 8
    ):
        raise RuntimeError("MOMENT native six-curve context contract changed")
    if weight_mode == "random_init":
        _randomize(torch, pipeline, seed=random_seed)
    for parameter in pipeline.parameters():
        parameter.requires_grad = False
    pipeline.eval().to(device)

    class FrozenMomentWellEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pipeline = pipeline
            self.asset_audit = {
                **audit,
                "weight_mode": weight_mode,
                "random_seed": int(random_seed) if weight_mode == "random_init" else None,
                "input_shape": [CHANNELS, LENGTH],
                "token_shape": [CHANNELS, PATCHES, EMBEDDING_DIM],
                "parameter_count": int(sum(p.numel() for p in pipeline.parameters())),
                "trainable_parameter_count": 0,
            }

        def forward(self, values: Any, input_mask: Any) -> Any:
            if values.ndim != 3 or tuple(values.shape[1:]) != (CHANNELS, LENGTH):
                raise ValueError("MOMENT well input must be [B,6,33]")
            if input_mask.ndim != 2 or tuple(input_mask.shape[1:]) != (LENGTH,):
                raise ValueError("MOMENT well input_mask must be [B,33]")
            if not bool(torch.isfinite(values).all()):
                raise ValueError("MOMENT well input contains non-finite values")
            result = self.pipeline.embed(
                x_enc=values,
                input_mask=input_mask.to(dtype=torch.long),
                reduction="none",
            )
            embeddings = getattr(result, "embeddings", None)
            expected = (len(values), CHANNELS, PATCHES, EMBEDDING_DIM)
            if embeddings is None or tuple(embeddings.shape) != expected:
                shape = None if embeddings is None else tuple(embeddings.shape)
                raise RuntimeError(f"unexpected MOMENT well tokens: {shape}")
            if not bool(torch.isfinite(embeddings).all()):
                raise FloatingPointError("MOMENT well tokens are non-finite")
            return embeddings

    return FrozenMomentWellEncoder().to(device)
