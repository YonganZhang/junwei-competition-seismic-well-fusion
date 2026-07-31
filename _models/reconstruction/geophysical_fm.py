"""ThinkOnward Geophysical Foundation Model feature-extraction adapter.

The upstream model is a 2-D single-channel ViT-MAE whose 400x1 patches encode
complete seismic traces.  This adapter deliberately exposes the frozen encoder
only; P14 owns the auditable 3-D-to-2-D slice construction and the downstream
OOF residual head.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.gaia_dagt.foundation_runtime import consume_config, insert_import_root


model_id = "geophysical_fm"

SNAPSHOT_REVISION = "d4a33965730a506cfdb4c85fa2a0a344c53216a2"
SOURCE_SHA256 = "ce6f97b5be889231107716d71175fe4482a678f8eb4e608c274265ca57612904"
CONFIG_SHA256 = "3abf4d69533c7aca3075deef079b28098eb9d3ba3ba10ed45e0733aebbb3fefc"
MODEL_CARD_SHA256 = "3ca7d92f45bac244c7fa9fb8460b703636e853acbbaa35831ae4dc9bdcdbcf5f"
WEIGHTS_SHA256 = "c905945267bbbc58f0e1848106d182f40b5dc61273959b666a49b384cfcb7446"
EXPECTED_CONFIG = {
    "img_size": [400, 160],
    "patch_size": [400, 1],
    "in_chans": 1,
    "embed_dim": 1200,
    "depth": 16,
    "num_heads": 20,
    "decoder_embed_dim": 800,
    "decoder_depth": 12,
    "decoder_num_heads": 20,
    "mlp_ratio": 4,
    "norm_pix_loss": False,
    "custom_head": False,
    "full_image_loss": True,
    "classes": 10,
}
ENCODER_WEIGHT_MODES = ("pretrained", "random_init")


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["seismic_slice_2d"],
        "input_shape": "[B,1,400,160]",
        "output_shape": "[B,161,1200]",
        "foundation_model": "thinkonward/geophysical-foundation-model",
        "foundation_snapshot": SNAPSHOT_REVISION,
        "pretraining_domain": "synthetic_3d_seismic",
        "pretraining_objective": "trace_masking_vit_mae",
        "license": "Apache-2.0",
        "conditioning": "unmasked_complete_trace_tokens",
        "supports_missing_mask": False,
        "supports_uncertainty": False,
        "requires_pretrained_weight": True,
        "supports_same_architecture_random_init": True,
        "auto_download": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _verify_local_assets_cached(
    source_root_text: str,
    snapshot_path_text: str,
) -> tuple[Path, Path, dict[str, Any]]:
    source_root = Path(source_root_text).expanduser().resolve()
    snapshot_path = Path(snapshot_path_text).expanduser().resolve()
    source_file = source_root / "GFM" / "ElasticViTMAE.py"
    required = {
        "source": source_file,
        "config": snapshot_path / "config.json",
        "model_card": snapshot_path / "README.md",
        "weights": snapshot_path / "model.safetensors",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "GFM local asset(s) missing: " + ", ".join(missing)
        )
    expected_hashes = {
        "source": SOURCE_SHA256,
        "config": CONFIG_SHA256,
        "model_card": MODEL_CARD_SHA256,
        "weights": WEIGHTS_SHA256,
    }
    actual_hashes = {
        name: _sha256(path) for name, path in required.items()
    }
    drift = {
        name: {
            "expected": expected_hashes[name],
            "actual": actual_hashes[name],
        }
        for name in expected_hashes
        if actual_hashes[name] != expected_hashes[name]
    }
    if drift:
        raise RuntimeError(f"GFM local asset hash drift: {drift}")
    config = json.loads(required["config"].read_text(encoding="utf-8"))
    if config != EXPECTED_CONFIG:
        raise RuntimeError("GFM config differs from the locked architecture")
    model_card = required["model_card"].read_text(encoding="utf-8")
    if "license: apache-2.0" not in model_card.lower():
        raise RuntimeError("GFM model card does not declare Apache-2.0")
    if snapshot_path.name != SNAPSHOT_REVISION:
        raise RuntimeError(
            "GFM snapshot revision mismatch: "
            f"expected {SNAPSHOT_REVISION}, got {snapshot_path.name}"
        )
    audit = {
        "model_id": "thinkonward/geophysical-foundation-model",
        "snapshot_revision": SNAPSHOT_REVISION,
        "license": "Apache-2.0",
        "source_sha256": actual_hashes["source"],
        "config_sha256": actual_hashes["config"],
        "model_card_sha256": actual_hashes["model_card"],
        "weights_sha256": actual_hashes["weights"],
        "config": config,
        "local_only": True,
        "auto_download": False,
    }
    return source_root, snapshot_path, audit


def verify_local_assets(
    source_root: Path,
    snapshot_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Hash-lock the vendor source, HF snapshot, config and Apache license."""

    return _verify_local_assets_cached(
        str(Path(source_root).expanduser().resolve()),
        str(Path(snapshot_path).expanduser().resolve()),
    )


def architecture_sha256(network: Any) -> str:
    payload = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in network.named_parameters()
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def encoder_probe_sha256(network: Any) -> str:
    """Fingerprint bounded samples from every encoder tensor.

    The full pretrained artifact is already byte-hashed above.  This bounded
    in-memory probe demonstrates seed-distinct random states without copying
    more than a billion encoder bytes back from the GPU for each ablation.
    """

    digest = hashlib.sha256()
    encoder_names = (
        "patch_embed.",
        "cls_token",
        "pos_embed",
        "blocks.",
        "norm.",
    )
    selected = 0
    for name, tensor in network.state_dict().items():
        if not name.startswith(encoder_names):
            continue
        values = tensor.detach().cpu().reshape(-1)
        count = min(64, values.numel())
        sample = values[:count].contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(sample.numpy().tobytes())
        selected += 1
    if selected == 0:
        raise RuntimeError("GFM encoder fingerprint selected no tensors")
    return digest.hexdigest()


def _make_wrapper(
    torch: Any,
    network: Any,
    *,
    freeze_encoder: bool,
    weight_mode: str,
    asset_audit: dict[str, Any],
) -> Any:
    nn = torch.nn

    class GeophysicalFMEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = network
            self.weight_mode = weight_mode
            self.asset_audit = dict(asset_audit)
            if freeze_encoder:
                for parameter in self.network.parameters():
                    parameter.requires_grad = False

        def forward(self, images: Any) -> Any:
            if images.ndim != 4 or tuple(images.shape[1:]) != (1, 400, 160):
                raise ValueError("GFM input must be [B,1,400,160]")
            if not bool(torch.isfinite(images).all()):
                raise ValueError("GFM input contains non-finite values")
            batch = int(images.shape[0])
            indices = torch.arange(
                160,
                device=images.device,
                dtype=torch.float32,
            ).unsqueeze(0).expand(batch, -1)
            latent, mask, ids_restore = self.network.forward_encoder(
                images,
                indices,
                160,
            )
            if tuple(latent.shape) != (batch, 161, 1200):
                raise RuntimeError(
                    f"unexpected GFM latent shape: {tuple(latent.shape)}"
                )
            if bool(torch.any(mask != 0)):
                raise RuntimeError("GFM feature extraction unexpectedly masked traces")
            expected_restore = torch.arange(
                160,
                device=images.device,
                dtype=ids_restore.dtype,
            ).unsqueeze(0).expand(batch, -1)
            if not bool(torch.equal(ids_restore, expected_restore)):
                raise RuntimeError("GFM identity trace order was not preserved")
            return latent

    return GeophysicalFMEncoder()


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    """Build a local-only pretrained or same-architecture random-init encoder."""

    if task_spec.track_id != "reconstruction":
        raise ValueError("GFM adapter is restricted to reconstruction")
    values = consume_config(
        config,
        required=("source_root", "snapshot_path"),
        optional=(
            "device",
            "freeze_encoder",
            "encoder_weight_mode",
            "random_seed",
        ),
    )
    source_root, snapshot_path, asset_audit = verify_local_assets(
        Path(values["source_root"]),
        Path(values["snapshot_path"]),
    )
    insert_import_root(source_root, "GFM")

    import torch
    from GFM import ElasticViTMAE

    weight_mode = str(values.get("encoder_weight_mode", "pretrained"))
    if weight_mode not in ENCODER_WEIGHT_MODES:
        raise ValueError(
            f"encoder_weight_mode must be one of {ENCODER_WEIGHT_MODES}"
        )
    random_seed = int(values.get("random_seed", 2693))
    if weight_mode == "pretrained":
        network = ElasticViTMAE.ElasticViTMAE.from_pretrained(
            snapshot_path,
            local_files_only=True,
        )
        pretrained_weights_loaded = True
    else:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        network = ElasticViTMAE.ElasticViTMAE(**asset_audit["config"])
        pretrained_weights_loaded = False
    asset_audit = {
        **asset_audit,
        "encoder_weight_mode": weight_mode,
        "pretrained_weights_loaded": pretrained_weights_loaded,
        "random_seed": random_seed if weight_mode == "random_init" else None,
        "architecture_sha256": architecture_sha256(network),
        "encoder_probe_sha256": encoder_probe_sha256(network),
        "parameter_count": int(
            sum(parameter.numel() for parameter in network.parameters())
        ),
    }
    device = str(values.get("device", "cpu"))
    return _make_wrapper(
        torch,
        network,
        freeze_encoder=bool(values.get("freeze_encoder", True)),
        weight_mode=weight_mode,
        asset_audit=asset_audit,
    ).eval().to(device)
