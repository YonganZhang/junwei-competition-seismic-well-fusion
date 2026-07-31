"""OpenMind ResEnc-L MAE transfer adapter for continuous 3-D property fields."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.gaia_dagt.foundation_runtime import (
    consume_config,
    insert_import_root,
    route_model,
    verify_checkpoint,
    verify_git_source,
)


model_id = "openmind_mae"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["seismic_volume_3d"],
        "input_shape": "[B,3,D,H,W]",
        "output_shape": "[B,1,D,H,W]",
        "foundation_model": "MIC-DKFZ/ResEncL-OpenMind-MAE",
        "conditioning": "masked_volume",
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "requires_pretrained_weight": True,
        "auto_download": False,
    }


def _load_encoder_weights(torch: Any, network: Any, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "network_weights" not in payload:
        raise ValueError("OpenMind checkpoint is missing network_weights")
    pretrained = payload["network_weights"]
    current = network.state_dict()
    prefixes = ("encoder.stem.", "encoder.stages.")
    selected = 0
    for key in current:
        if not key.startswith(prefixes):
            continue
        if key not in pretrained:
            raise ValueError(f"OpenMind checkpoint is missing encoder key: {key}")
        if current[key].shape != pretrained[key].shape:
            raise ValueError(f"OpenMind encoder shape mismatch for {key}")
        current[key] = pretrained[key]
        selected += 1
    if selected == 0:
        raise ValueError("OpenMind checkpoint did not provide any encoder weights")
    network.load_state_dict(current, strict=True)


def _make_network(torch: Any, network: Any, *, freeze_encoder: bool) -> Any:
    nn = torch.nn
    functional = torch.nn.functional

    class OpenMindVolumeRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attribute_projection = nn.Conv3d(3, 1, kernel_size=1)
            self.network = network
            if freeze_encoder:
                for parameter in self.network.encoder.parameters():
                    parameter.requires_grad = False

        def forward(self, inputs: Any) -> Any:
            if inputs.ndim != 5 or inputs.shape[1] != 3:
                raise ValueError("OpenMind reconstruction input must be [B,3,D,H,W]")
            if not bool(torch.isfinite(inputs).all()):
                raise ValueError("OpenMind reconstruction input contains non-finite values")
            original = tuple(int(value) for value in inputs.shape[-3:])
            # ResEnc-L downsamples five times.  A 32-voxel edge collapses to a
            # single deepest voxel and InstanceNorm3d must fail even in eval
            # mode; keep at least two voxels per deepest spatial axis.
            target = tuple(max(64, ((size + 31) // 32) * 32) for size in original)
            pads: list[int] = []
            for size, wanted in reversed(list(zip(original, target))):
                pads.extend((0, wanted - size))
            volume = functional.pad(inputs, tuple(pads)) if any(pads) else inputs
            prediction = self.network(self.attribute_projection(volume))
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            if prediction.ndim != 5 or prediction.shape[1] != 1:
                raise ValueError(
                    f"unexpected OpenMind regression output: {tuple(prediction.shape)}"
                )
            return prediction[..., : original[0], : original[1], : original[2]]

    return OpenMindVolumeRegressor()


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    if task_spec.track_id != "reconstruction":
        raise ValueError("OpenMind MAE adapter is restricted to reconstruction")
    values = consume_config(
        config,
        required=("source_root", "checkpoint_path"),
        optional=("device", "freeze_encoder"),
    )
    model_ref = route_model("reconstruction")
    source_root = verify_git_source(Path(values["source_root"]), model_ref.source_revision)
    checkpoint = verify_checkpoint("reconstruction", Path(values["checkpoint_path"]))
    insert_import_root(source_root / "src", "nnssl")
    import torch
    from nnssl.architectures.architecture_registry import get_res_enc_l

    network = get_res_enc_l(1, 1, deep_supervision=False)
    _load_encoder_weights(torch, network, checkpoint)
    device = str(values.get("device", "cpu"))
    return _make_network(
        torch,
        network,
        freeze_encoder=bool(values.get("freeze_encoder", True)),
    ).to(device)
