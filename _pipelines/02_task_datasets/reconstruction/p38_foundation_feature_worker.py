#!/usr/bin/env python3
"""Local-only frozen MOMENT/GFM feature worker for P38 scratch caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for _root in (str(PROJECT_ROOT), str(HERE)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from _models.reconstruction import geophysical_fm as gfm  # noqa: E402
from _models.reconstruction import moment_well  # noqa: E402


PROJECTION_DIM = 16
PROJECTION_SEED = 2693


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projection(width: int, *, stream: int) -> np.ndarray:
    if int(width) < PROJECTION_DIM:
        raise ValueError("foundation token width is smaller than the locked projection")
    rng = np.random.default_rng(PROJECTION_SEED + int(stream))
    values = rng.normal(size=(int(width), PROJECTION_DIM))
    q, _ = np.linalg.qr(values)
    return q[:, :PROJECTION_DIM].astype(np.float32)


def _project(values: np.ndarray, *, stream: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.einsum("...d,dk->...k", array, _projection(array.shape[-1], stream=stream)).astype(
        np.float32
    )


def _moment(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    with np.load(args.input, allow_pickle=False) as payload:
        values = np.asarray(payload["values"], dtype=np.float32)
        masks = np.asarray(payload["input_mask"], dtype=bool)
    if values.ndim != 3 or tuple(values.shape[1:]) != (6, 33):
        raise ValueError("worker MOMENT values must be [N,6,33]")
    if masks.shape != (len(values), 33):
        raise ValueError("worker MOMENT mask must be [N,33]")
    model = moment_well.build_encoder(
        snapshot_path=args.moment_snapshot,
        dependency_root=args.moment_dependency_root,
        source_root=args.moment_source_root,
        device=args.device,
        weight_mode=args.weight_mode,
        random_seed=args.seed,
    )
    outputs: list[np.ndarray] = []
    forward_batches = 0
    with torch.no_grad():
        for start in range(0, len(values), args.batch_size):
            batch = torch.as_tensor(
                values[start : start + args.batch_size], dtype=torch.float32, device=args.device
            )
            mask = torch.as_tensor(
                masks[start : start + args.batch_size], dtype=torch.long, device=args.device
            )
            tokens = model(batch, mask).detach().cpu().numpy()
            outputs.append(_project(tokens, stream=11))
            forward_batches += 1
    features = np.concatenate(outputs, axis=0)
    audit = {
        "kind": "moment",
        "weight_mode": args.weight_mode,
        "source_token_shape": [6, 4, 768],
        "projected_token_shape": [6, 4, PROJECTION_DIM],
        "fixed_projection_seed": PROJECTION_SEED,
        "forward_batches": forward_batches,
        "asset_audit": model.asset_audit,
        "feature_variance": float(np.var(features)),
        "finite": bool(np.all(np.isfinite(features))),
    }
    return {"features": features, "audit_json": np.asarray(json.dumps(audit, sort_keys=True))}


def _normalize_section(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=(-2, -1), keepdims=True)
    std = np.maximum(np.std(values, axis=(-2, -1), keepdims=True), 1e-6)
    return ((values - mean) / std).astype(np.float32)


def _gfm(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    with np.load(args.input, allow_pickle=False) as payload:
        sections = np.asarray(payload["sections"], dtype=np.float32)
    if sections.ndim != 4 or tuple(sections.shape[1:]) != (3, 400, 160):
        raise ValueError("worker GFM sections must be [S,3,400,160]")
    normalized = _normalize_section(sections)
    model = gfm.build_model(
        SimpleNamespace(track_id="reconstruction"),
        source_root=args.gfm_source_root,
        snapshot_path=args.gfm_snapshot,
        device=args.device,
        freeze_encoder=True,
        encoder_weight_mode=args.weight_mode,
        random_seed=args.seed,
    ).to(args.device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("GFM worker encoder is not frozen")
    output = np.empty((len(sections), 3, 161, PROJECTION_DIM), dtype=np.float32)
    forward_batches = 0
    with torch.no_grad():
        for channel in range(3):
            for start in range(0, len(sections), args.batch_size):
                images = torch.as_tensor(
                    normalized[start : start + args.batch_size, channel, None],
                    dtype=torch.float32,
                    device=args.device,
                )
                tokens = model(images).detach().cpu().numpy()
                output[start : start + len(tokens), channel] = _project(
                    tokens, stream=31 + channel
                )
                forward_batches += 1
    audit = {
        "kind": "gfm",
        "weight_mode": args.weight_mode,
        "source_token_shape": [161, 1200],
        "projected_token_shape": [3, 161, PROJECTION_DIM],
        "trace_tokens_retained": 160,
        "cls_token_retained": True,
        "normalization": "per-section per-channel z-score; label-independent",
        "fixed_projection_seed": PROJECTION_SEED,
        "forward_batches": forward_batches,
        "asset_audit": model.asset_audit,
        "feature_variance": float(np.var(output)),
        "finite": bool(np.all(np.isfinite(output))),
    }
    return {"features": output, "audit_json": np.asarray(json.dumps(audit, sort_keys=True))}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("moment", "gfm"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-mode", choices=("pretrained", "random_init"), required=True)
    parser.add_argument("--seed", type=int, default=2693)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--moment-snapshot", type=Path)
    parser.add_argument("--moment-dependency-root", type=Path)
    parser.add_argument("--moment-source-root", type=Path)
    parser.add_argument("--gfm-snapshot", type=Path)
    parser.add_argument("--gfm-source-root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.kind == "moment":
        required = (args.moment_snapshot, args.moment_dependency_root, args.moment_source_root)
        if any(value is None for value in required):
            raise ValueError("MOMENT worker paths are required")
        payload = _moment(args)
    else:
        required = (args.gfm_snapshot, args.gfm_source_root)
        if any(value is None for value in required):
            raise ValueError("GFM worker paths are required")
        payload = _gfm(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
