"""Partial-fine-tuning adapter for the ThinkOnward seismic ViT-MAE.

P14 deliberately keeps :mod:`geophysical_fm` frozen.  P15 uses this separate
adapter so the committed P14 implementation and evidence hashes remain stable.
The adapter freezes the patch embed and encoder prefix, while exposing the last
one or two genuine transformer blocks plus the encoder LayerNorm for training.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _models.gaia_dagt.foundation_runtime import consume_config, insert_import_root
from _models.reconstruction import geophysical_fm as frozen_adapter


model_id = "geophysical_fm_finetune"
SUPPORTED_TRAINABLE_BLOCK_COUNTS = (1, 2)


def capabilities() -> dict[str, Any]:
    values = dict(frozen_adapter.capabilities())
    values.update(
        {
            "model_id": model_id,
            "encoder_depth": 16,
            "partial_finetuning": True,
            "trainable_tail_block_counts": list(
                SUPPORTED_TRAINABLE_BLOCK_COUNTS
            ),
            "supports_frozen_prefix_cache": True,
            "cached_prefix_precision": "owned_by_evaluation_pipeline",
        }
    )
    return values


def configure_partial_finetune(
    network: Any,
    *,
    trainable_block_count: int,
) -> tuple[int, ...]:
    """Freeze the encoder except its final blocks and terminal LayerNorm."""

    block_count = int(trainable_block_count)
    if block_count not in SUPPORTED_TRAINABLE_BLOCK_COUNTS:
        raise ValueError(
            "trainable_block_count must be one of "
            f"{SUPPORTED_TRAINABLE_BLOCK_COUNTS}"
        )
    blocks = network.blocks
    if len(blocks) != frozen_adapter.EXPECTED_CONFIG["depth"]:
        raise RuntimeError(
            "GFM encoder depth drift: "
            f"{len(blocks)} != {frozen_adapter.EXPECTED_CONFIG['depth']}"
        )
    for parameter in network.parameters():
        parameter.requires_grad = False
    indices = tuple(range(len(blocks) - block_count, len(blocks)))
    for index in indices:
        for parameter in blocks[index].parameters():
            parameter.requires_grad = True
    for parameter in network.norm.parameters():
        parameter.requires_grad = True
    return indices


def _validate_images(torch: Any, images: Any) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 400, 160):
        raise ValueError("GFM input must be [B,1,400,160]")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("GFM input contains non-finite values")


def _make_partial_wrapper(
    torch: Any,
    network: Any,
    *,
    weight_mode: str,
    asset_audit: Mapping[str, Any],
    trainable_block_count: int,
) -> Any:
    nn = torch.nn

    class GeophysicalFMPartialEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = network
            self.weight_mode = str(weight_mode)
            self.trainable_block_indices = configure_partial_finetune(
                self.network,
                trainable_block_count=int(trainable_block_count),
            )
            self.asset_audit = {
                **dict(asset_audit),
                "partial_finetuning": True,
                "trainable_block_indices": list(
                    self.trainable_block_indices
                ),
                "trainable_encoder_parameters": int(
                    sum(
                        parameter.numel()
                        for parameter in self.network.parameters()
                        if parameter.requires_grad
                    )
                ),
                "frozen_encoder_parameters": int(
                    sum(
                        parameter.numel()
                        for parameter in self.network.parameters()
                        if not parameter.requires_grad
                    )
                ),
            }

        @property
        def prefix_block_count(self) -> int:
            return self.trainable_block_indices[0]

        def extract_frozen_prefix(self, images: Any) -> Any:
            """Run the frozen prefix before the first trainable block."""

            _validate_images(torch, images)
            x = self.network.patch_embed(images)
            x = x + self.network.pos_embed[:, 1:, :]
            cls_token = (
                self.network.cls_token + self.network.pos_embed[:, :1, :]
            )
            x = torch.cat(
                (cls_token.expand(x.shape[0], -1, -1), x),
                dim=1,
            )
            for block in self.network.blocks[: self.prefix_block_count]:
                x = block(x)
            expected = (int(images.shape[0]), 161, 1200)
            if tuple(x.shape) != expected:
                raise RuntimeError(
                    f"unexpected GFM prefix shape: {tuple(x.shape)}"
                )
            return x

        def forward_tail(self, prefix_tokens: Any) -> Any:
            if prefix_tokens.ndim != 3 or tuple(
                prefix_tokens.shape[1:]
            ) != (161, 1200):
                raise ValueError("GFM prefix tokens must be [B,161,1200]")
            x = prefix_tokens
            for index in self.trainable_block_indices:
                x = self.network.blocks[index](x)
            return self.network.norm(x)

        def initial_tail(self) -> Any:
            """Return an independent lightweight trainable-tail module."""

            blocks = [
                copy.deepcopy(self.network.blocks[index])
                for index in self.trainable_block_indices
            ]
            norm = copy.deepcopy(self.network.norm)

            class TrainableTail(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.blocks = nn.ModuleList(blocks)
                    self.norm = norm

                def forward(self, prefix_tokens: Any) -> Any:
                    if prefix_tokens.ndim != 3 or tuple(
                        prefix_tokens.shape[1:]
                    ) != (161, 1200):
                        raise ValueError(
                            "GFM prefix tokens must be [B,161,1200]"
                        )
                    x = prefix_tokens
                    for block in self.blocks:
                        x = block(x)
                    return self.norm(x)

            return TrainableTail()

    return GeophysicalFMPartialEncoder()


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    """Build pretrained or matched-random GFM with a trainable encoder tail."""

    if task_spec.track_id != "reconstruction":
        raise ValueError("GFM fine-tuning adapter is restricted to reconstruction")
    values = consume_config(
        config,
        required=("source_root", "snapshot_path"),
        optional=(
            "device",
            "encoder_weight_mode",
            "random_seed",
            "trainable_block_count",
        ),
    )
    source_root, snapshot_path, asset_audit = (
        frozen_adapter.verify_local_assets(
            values["source_root"],
            values["snapshot_path"],
        )
    )
    insert_import_root(source_root, "GFM")

    import torch
    from GFM import ElasticViTMAE

    weight_mode = str(values.get("encoder_weight_mode", "pretrained"))
    if weight_mode not in frozen_adapter.ENCODER_WEIGHT_MODES:
        raise ValueError(
            "encoder_weight_mode must be one of "
            f"{frozen_adapter.ENCODER_WEIGHT_MODES}"
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
        "architecture_sha256": frozen_adapter.architecture_sha256(network),
        "encoder_probe_sha256": frozen_adapter.encoder_probe_sha256(network),
        "parameter_count": int(
            sum(parameter.numel() for parameter in network.parameters())
        ),
    }
    wrapper = _make_partial_wrapper(
        torch,
        network,
        weight_mode=weight_mode,
        asset_audit=audit,
        trainable_block_count=int(values.get("trainable_block_count", 1)),
    )
    return wrapper.to(str(values.get("device", "cpu")))


def tail_state_dict(tail: Any) -> OrderedDict[str, Any]:
    """Clone a trainable tail state to CPU for deterministic OOF restarts."""

    return OrderedDict(
        (
            name,
            tensor.detach().to(device="cpu", dtype=tensor.dtype).clone(),
        )
        for name, tensor in tail.state_dict().items()
    )


def build_tail_module(*, trainable_block_count: int) -> Any:
    """Instantiate the exact lightweight GFM encoder tail architecture."""

    block_count = int(trainable_block_count)
    if block_count not in SUPPORTED_TRAINABLE_BLOCK_COUNTS:
        raise ValueError(
            "trainable_block_count must be one of "
            f"{SUPPORTED_TRAINABLE_BLOCK_COUNTS}"
        )
    import torch
    from timm.models.vision_transformer import Block

    config = frozen_adapter.EXPECTED_CONFIG

    class GFMTrainableTail(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [
                    Block(
                        config["embed_dim"],
                        config["num_heads"],
                        config["mlp_ratio"],
                        qkv_bias=True,
                        norm_layer=torch.nn.LayerNorm,
                    )
                    for _ in range(block_count)
                ]
            )
            self.norm = torch.nn.LayerNorm(config["embed_dim"])

        def forward(self, prefix_tokens: Any) -> Any:
            if prefix_tokens.ndim != 3 or tuple(
                prefix_tokens.shape[1:]
            ) != (161, 1200):
                raise ValueError("GFM prefix tokens must be [B,161,1200]")
            x = prefix_tokens
            for block in self.blocks:
                x = block(x)
            return self.norm(x)

    return GFMTrainableTail()
