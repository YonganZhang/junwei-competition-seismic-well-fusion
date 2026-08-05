#!/usr/bin/env python3
"""Pinned, frozen native-token extractors for the P41 property pilot."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJ3_SITE_PACKAGES = Path(
    os.environ.get(
        "P41_PROJ3_SITE_PACKAGES",
        "/mnt/data/yongan-admin-2/envs/proj3-matrix3d-poc-py310/lib/python3.10/site-packages",
    )
)
LOCAL_SITE_PACKAGES = Path(
    os.environ.get("P41_LOCAL_SITE_PACKAGES", "/mnt/data/yongan-admin-2/.local/lib/python3.10/site-packages")
)


MOMENT_REVISION = "5e44b0ea26376a176360f87831124e018f876d96"
MOMENT_WEIGHTS_SHA256 = "1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825"
GFM_REVISION = "d4a33965730a506cfdb4c85fa2a0a344c53216a2"
GFM_SOURCE_SHA256 = "ce6f97b5be889231107716d71175fe4482a678f8eb4e608c274265ca57612904"
GFM_CONFIG_SHA256 = "3abf4d69533c7aca3075deef079b28098eb9d3ba3ba10ed45e0733aebbb3fefc"
GFM_MODEL_CARD_SHA256 = "3ca7d92f45bac244c7fa9fb8460b703636e853acbbaa35831ae4dc9bdcdbcf5f"
GFM_WEIGHTS_SHA256 = "c905945267bbbc58f0e1848106d182f40b5dc61273959b666a49b384cfcb7446"
GFM_EXPECTED_CONFIG = {
    "img_size": [400, 160], "patch_size": [400, 1], "in_chans": 1,
    "embed_dim": 1200, "depth": 16, "num_heads": 20,
    "decoder_embed_dim": 800, "decoder_depth": 12, "decoder_num_heads": 20,
    "mlp_ratio": 4, "norm_pix_loss": False, "custom_head": False,
    "full_image_loss": True, "classes": 10,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _insert_import_root(root: Path, package: str) -> Path:
    root = Path(root).expanduser().resolve()
    if not (root / package).is_dir():
        raise FileNotFoundError(f"missing local package {package}: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def verify_assets(
    *, moment_snapshot: Path, moment_dependency_root: Path,
    moment_source_root: Path, gfm_snapshot: Path, gfm_source_root: Path,
) -> dict[str, Any]:
    moment_snapshot = Path(moment_snapshot).expanduser().resolve()
    gfm_snapshot = Path(gfm_snapshot).expanduser().resolve()
    moment_dependency_root = Path(moment_dependency_root).expanduser().resolve()
    moment_source_root = Path(moment_source_root).expanduser().resolve()
    gfm_source_root = Path(gfm_source_root).expanduser().resolve()
    required = {
        "moment_weights": moment_snapshot / "model.safetensors",
        "moment_config": moment_snapshot / "config.json",
        "gfm_source": gfm_source_root / "GFM" / "ElasticViTMAE.py",
        "gfm_weights": gfm_snapshot / "model.safetensors",
        "gfm_config": gfm_snapshot / "config.json",
        "gfm_model_card": gfm_snapshot / "README.md",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing P41 local asset(s): " + ", ".join(missing))
    if not moment_dependency_root.is_dir() or not moment_source_root.is_dir():
        raise FileNotFoundError("MOMENT dependency/source root is missing")
    observed = {name: sha256(path) for name, path in required.items()}
    expected = {
        "moment_weights": MOMENT_WEIGHTS_SHA256, "gfm_source": GFM_SOURCE_SHA256,
        "gfm_weights": GFM_WEIGHTS_SHA256, "gfm_config": GFM_CONFIG_SHA256,
        "gfm_model_card": GFM_MODEL_CARD_SHA256,
    }
    drift = {name: {"expected": value, "observed": observed[name]}
             for name, value in expected.items() if observed[name] != value}
    if drift:
        raise RuntimeError(f"P41 local asset hash drift: {drift}")
    if moment_snapshot.name != MOMENT_REVISION or gfm_snapshot.name != GFM_REVISION:
        raise RuntimeError("foundation snapshot revision drift")
    if json.loads(required["gfm_config"].read_text()) != GFM_EXPECTED_CONFIG:
        raise RuntimeError("GFM architecture config drift")
    if "license: apache-2.0" not in required["gfm_model_card"].read_text().lower():
        raise RuntimeError("GFM model card no longer declares Apache-2.0")
    return {
        "local_only": True, "auto_download": False,
        "moment": {"model_id": "AutonLab/MOMENT-1-base", "revision": MOMENT_REVISION,
                   "weights_sha256": observed["moment_weights"],
                   "config_sha256": observed["moment_config"],
                   "source_root": str(moment_source_root),
                   "dependency_root": str(moment_dependency_root)},
        "gfm": {"model_id": "thinkonward/geophysical-foundation-model",
                "revision": GFM_REVISION, "source_sha256": observed["gfm_source"],
                "weights_sha256": observed["gfm_weights"],
                "config_sha256": observed["gfm_config"],
                "model_card_sha256": observed["gfm_model_card"],
                "license": "Apache-2.0", "source_root": str(gfm_source_root)},
    }


def _randomize_backbone(model: Any, *, seed: int) -> None:
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("pipeline.head."):
                continue
            if parameter.ndim >= 2:
                values = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
                values.normal_(mean=0.0, std=0.02, generator=generator)
            elif "layer_norm" in name or "final_layer_norm" in name:
                values = torch.ones(parameter.shape, dtype=parameter.dtype, device="cpu")
            else:
                values = torch.zeros(parameter.shape, dtype=parameter.dtype, device="cpu")
            parameter.copy_(values.to(parameter.device))


def extract_moment_tokens(
    values: np.ndarray, *, snapshot: Path, dependency_root: Path, source_root: Path,
    device: str, random_init: bool, seed: int, valid_length: int = 8,
    batch_size: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return native ``[N,9,4,768]`` tokens for audited 8->33 right padding."""
    # Import the ABI-compatible torch from the active torch-common
    # interpreter before exposing any legacy Python-3.10 dependency trees.
    import torch
    for root in (LOCAL_SITE_PACKAGES, PROJ3_SITE_PACKAGES, dependency_root, source_root):
        root = Path(root).expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.append(str(root))
    import huggingface_hub
    if getattr(huggingface_hub, "__version__", "") >= "1.0":
        huggingface_hub.__version__ = "0.34.0"
    dependency_root = _insert_import_root(dependency_root, "momentfm")
    source_root = _insert_import_root(source_root, "momentfm")
    from momentfm import MOMENTPipeline

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or tuple(array.shape[1:]) != (9, 33):
        raise ValueError(f"P41 MOMENT input must be [N,9,33], got {array.shape}")
    if valid_length != 8 or np.any(array[:, :, valid_length:] != 0):
        raise ValueError("P41 MOMENT adapter requires exact zero right-padding 8->33")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pipeline = MOMENTPipeline.from_pretrained(
        str(Path(snapshot).resolve()), local_files_only=True,
        model_kwargs={"task_name": "classification", "n_channels": 9,
                      "num_class": 3, "seq_len": 33, "freeze_encoder": True,
                      "freeze_embedder": True, "freeze_head": False},
    )
    pipeline.init()
    pipeline = pipeline.to(device).eval()
    if random_init:
        class Wrapper(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__(); self.pipeline = inner
        _randomize_backbone(Wrapper(pipeline), seed=seed)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            batch = torch.as_tensor(array[start:start + batch_size], device=device)
            mask = torch.zeros((len(batch), 33), dtype=torch.long, device=device)
            mask[:, :valid_length] = 1
            embedded = pipeline.embed(x_enc=batch, input_mask=mask, reduction="none")
            tokens = getattr(embedded, "embeddings", None)
            if tokens is None or tuple(tokens.shape[1:]) != (9, 4, 768):
                shape = None if tokens is None else tuple(tokens.shape)
                raise RuntimeError(f"unexpected P41 MOMENT native tokens: {shape}")
            outputs.append(tokens.detach().cpu().numpy().astype(np.float32))
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise FloatingPointError("P41 MOMENT tokens contain non-finite values")
    audit = {
        "weight_mode": "random_init" if random_init else "pretrained",
        "input_shape": list(array.shape), "valid_length": valid_length,
        "adapter": "label-independent per-sample curve normalization; exact zero right-pad 8->33 with input mask",
        "native_token_shape": list(result.shape), "token_width": 768,
        "projection": "none", "feature_variance": float(np.var(result)),
        "parameter_count": int(sum(p.numel() for p in pipeline.parameters())),
        "all_encoder_parameters_frozen": bool(all(not p.requires_grad for p in pipeline.encoder.parameters())),
        "dependency_root": str(dependency_root), "source_root": str(source_root),
    }
    del pipeline
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result, audit


def _gfm_encoder(*, snapshot: Path, source_root: Path, device: str,
                 random_init: bool, seed: int) -> tuple[Any, dict[str, Any]]:
    source_root = _insert_import_root(source_root, "GFM")
    import torch
    from GFM import ElasticViTMAE
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    network = (ElasticViTMAE.ElasticViTMAE(**GFM_EXPECTED_CONFIG) if random_init
               else ElasticViTMAE.ElasticViTMAE.from_pretrained(
                   Path(snapshot).resolve(), local_files_only=True))
    for parameter in network.parameters(): parameter.requires_grad = False
    network = network.to(device).eval()
    return network, {
        "weight_mode": "random_init" if random_init else "pretrained",
        "parameter_count": int(sum(p.numel() for p in network.parameters())),
        "all_encoder_parameters_frozen": all(not p.requires_grad for p in network.parameters()),
        "source_root": str(source_root),
    }


def extract_gfm_sample_tokens(
    sections: np.ndarray, section_ids: np.ndarray, trace_token_ids: np.ndarray,
    *, snapshot: Path, source_root: Path, device: str, random_init: bool,
    seed: int, batch_size: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return native CLS+aligned-complete-trace tokens ``[N,2,1200]``."""
    import torch
    raw = np.asarray(sections, dtype=np.float32)
    if raw.ndim != 3 or tuple(raw.shape[1:]) != (400, 160):
        raise ValueError(f"P41 GFM sections must be [S,400,160], got {raw.shape}")
    mean = raw.mean(axis=(1, 2), keepdims=True)
    scale = np.maximum(raw.std(axis=(1, 2), keepdims=True), 1e-6)
    normalized = ((raw - mean) / scale).astype(np.float32)
    model, audit = _gfm_encoder(snapshot=snapshot, source_root=source_root,
                                device=device, random_init=random_init, seed=seed)
    section_tokens = np.empty((len(raw), 161, 1200), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(raw), batch_size):
            images = torch.as_tensor(normalized[start:start + batch_size, None], device=device)
            indices = torch.arange(160, dtype=torch.float32, device=device)[None].expand(len(images), -1)
            latent, mask, restore = model.forward_encoder(images, indices, 160)
            if tuple(latent.shape[1:]) != (161, 1200) or bool(torch.any(mask != 0)):
                raise RuntimeError("unexpected P41 GFM native token contract")
            expected = torch.arange(160, dtype=restore.dtype, device=device)[None]
            if not bool(torch.equal(restore, expected.expand_as(restore))):
                raise RuntimeError("P41 GFM trace-token order drift")
            section_tokens[start:start + len(images)] = latent.detach().cpu().numpy()
    rows = np.asarray(section_ids, dtype=np.int64)
    traces = np.asarray(trace_token_ids, dtype=np.int64)
    if rows.shape != traces.shape or rows.ndim != 1:
        raise ValueError("P41 GFM token maps must be aligned vectors")
    if rows.min(initial=0) < 0 or rows.max(initial=0) >= len(section_tokens):
        raise ValueError("P41 GFM section map out of bounds")
    if traces.min(initial=0) < 0 or traces.max(initial=0) >= 160:
        raise ValueError("P41 GFM trace-token map out of bounds")
    result = np.stack((section_tokens[rows, 0], section_tokens[rows, 1 + traces]), axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("P41 GFM sample tokens contain non-finite values")
    audit.update({
        "input_section_shape": list(raw.shape), "source_token_shape": [len(raw), 161, 1200],
        "retained_native_token_shape": list(result.shape),
        "retained_token_order": ["CLS", "aligned_complete_trace"], "token_width": 1200,
        "projection": "none", "normalization": "per-section amplitude z-score; label-independent",
        "feature_variance": float(np.var(result)),
    })
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result, audit
