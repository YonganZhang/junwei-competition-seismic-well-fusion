"""GFM native trace-masking reconstruction adapter for P16.

The upstream ThinkOnward model is a ViT-MAE whose ``400 x 1`` patches are
complete seismic traces.  Unlike the frozen-feature P14 adapter, this module
exposes the genuine encoder-decoder path and returns the upstream tutorial's
hybrid interpolation: visible traces remain bitwise input values while masked
traces are replaced by decoder predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.gaia_dagt.foundation_runtime import consume_config, insert_import_root
from _models.reconstruction import geophysical_fm as base_gfm


model_id = "geophysical_fm_denoise"
ENCODER_WEIGHT_MODES = base_gfm.ENCODER_WEIGHT_MODES
TRACE_COUNT = 160
TIME_SAMPLES = 400


@dataclass(frozen=True)
class DenoiseOutput:
    """Auditable result of one native GFM reconstruction forward."""

    reconstruction: Any
    decoder_prediction: Any
    mask: Any
    ids_restore: Any


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["seismic_slice_2d"],
        "input_shape": "[B,1,400,160]",
        "output_shape": "[B,1,400,160]",
        "foundation_model": "thinkonward/geophysical-foundation-model",
        "foundation_snapshot": base_gfm.SNAPSHOT_REVISION,
        "pretraining_domain": "synthetic_3d_seismic",
        "pretraining_objective": "trace_masking_vit_mae",
        "license": "Apache-2.0",
        "reconstruction_semantics": (
            "decoder prediction on masked traces plus original visible traces"
        ),
        "supports_same_architecture_random_init": True,
        "auto_download": False,
    }


def model_probe_sha256(network: Any) -> str:
    """Fingerprint bounded samples from encoder and decoder state tensors."""

    digest = hashlib.sha256()
    selected = 0
    for name, tensor in network.state_dict().items():
        values = tensor.detach().cpu().reshape(-1)
        count = min(64, values.numel())
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(values[:count].contiguous().numpy().tobytes())
        selected += 1
    if selected == 0:
        raise RuntimeError("GFM full-model fingerprint selected no tensors")
    return digest.hexdigest()


def _make_wrapper(
    torch: Any,
    network: Any,
    *,
    freeze_model: bool,
    weight_mode: str,
    asset_audit: dict[str, Any],
) -> Any:
    nn = torch.nn

    class GeophysicalFMDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = network
            self.weight_mode = weight_mode
            self.asset_audit = dict(asset_audit)
            if freeze_model:
                for parameter in self.network.parameters():
                    parameter.requires_grad = False

        def forward(
            self,
            images: Any,
            trace_priorities: Any,
            len_keep: int,
        ) -> DenoiseOutput:
            if images.ndim != 4 or tuple(images.shape[1:]) != (
                1,
                TIME_SAMPLES,
                TRACE_COUNT,
            ):
                raise ValueError("GFM denoiser input must be [B,1,400,160]")
            if trace_priorities.ndim != 2 or tuple(trace_priorities.shape) != (
                int(images.shape[0]),
                TRACE_COUNT,
            ):
                raise ValueError("trace priorities must be [B,160]")
            if not 0 < int(len_keep) <= TRACE_COUNT:
                raise ValueError("len_keep must be in [1,160]")
            if not bool(torch.isfinite(images).all()):
                raise ValueError("GFM denoiser input contains non-finite values")
            if not bool(torch.isfinite(trace_priorities).all()):
                raise ValueError("GFM trace priorities contain non-finite values")

            latent, mask, ids_restore = self.network.forward_encoder(
                images,
                trace_priorities,
                int(len_keep),
            )
            expected_latent = (int(images.shape[0]), int(len_keep) + 1, 1200)
            if tuple(latent.shape) != expected_latent:
                raise RuntimeError(
                    f"unexpected GFM masked latent shape: {tuple(latent.shape)}"
                )
            prediction = self.network.forward_decoder(latent, ids_restore)
            expected_prediction = (
                int(images.shape[0]),
                TRACE_COUNT,
                TIME_SAMPLES,
            )
            if tuple(prediction.shape) != expected_prediction:
                raise RuntimeError(
                    "unexpected GFM decoder prediction shape: "
                    f"{tuple(prediction.shape)}"
                )
            if tuple(mask.shape) != (int(images.shape[0]), TRACE_COUNT):
                raise RuntimeError(f"unexpected GFM mask shape: {tuple(mask.shape)}")
            expected_masked = TRACE_COUNT - int(len_keep)
            if not bool(
                torch.all(mask.sum(dim=1) == float(expected_masked))
            ):
                raise RuntimeError("GFM mask cardinality differs from len_keep")

            # This is the exact interpolation combination shown by the
            # upstream tutorial: the decoder only replaces masked traces.
            visible = self.network.patchify(images)
            combined = (
                prediction * mask.unsqueeze(-1)
                + visible * (1.0 - mask.unsqueeze(-1))
            )
            reconstruction = self.network.unpatchify(combined)
            # Autocast may quantize even the visible branch while running the
            # decoder in BF16.  Re-apply the tutorial's semantic contract in
            # the original FP32 image dtype so visible traces are exact.
            reconstruction = torch.where(
                mask[:, None, None, :].bool(),
                reconstruction.to(dtype=images.dtype),
                images,
            )
            if tuple(reconstruction.shape) != tuple(images.shape):
                raise RuntimeError("GFM unpatchify changed image geometry")
            if not bool(torch.isfinite(reconstruction).all()):
                raise FloatingPointError("GFM reconstruction is non-finite")
            return DenoiseOutput(
                reconstruction=reconstruction,
                decoder_prediction=prediction,
                mask=mask,
                ids_restore=ids_restore,
            )

    return GeophysicalFMDenoiser()


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    """Build a local-only pretrained or same-architecture random MAE."""

    if task_spec.track_id != "reconstruction":
        raise ValueError("GFM denoiser is restricted to reconstruction")
    values = consume_config(
        config,
        required=("source_root", "snapshot_path"),
        optional=(
            "device",
            "freeze_model",
            "encoder_weight_mode",
            "random_seed",
        ),
    )
    source_root, snapshot_path, asset_audit = base_gfm.verify_local_assets(
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
    audit = {
        **asset_audit,
        "encoder_weight_mode": weight_mode,
        "pretrained_weights_loaded": pretrained_weights_loaded,
        "random_seed": random_seed if weight_mode == "random_init" else None,
        "architecture_sha256": base_gfm.architecture_sha256(network),
        "full_model_probe_sha256": model_probe_sha256(network),
        "parameter_count": int(
            sum(parameter.numel() for parameter in network.parameters())
        ),
        "encoder_decoder_forward": True,
        "hybrid_visible_trace_preservation": True,
    }
    device = str(values.get("device", "cpu"))
    return _make_wrapper(
        torch,
        network,
        freeze_model=bool(values.get("freeze_model", True)),
        weight_mode=weight_mode,
        asset_audit=audit,
    ).eval().to(device)
