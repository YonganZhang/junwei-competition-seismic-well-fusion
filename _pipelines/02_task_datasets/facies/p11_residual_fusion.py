#!/usr/bin/env python3
"""P11 residual-fusion experiment for facies.

The experiment keeps the strong small-model logits as the main route and adds a
bounded SAM2 residual branch:

    final_logits = small_logits + sigmoid(gate) * 0.05 * tanh(residual_logits)

The gate only sees inference-time quantities derived from the input and the
main-branch predictions. Frozen SAM2 feature pyramids are cached once per fold
and encoder condition. No label-, mask-, or validation-derived signal is
passed into the gate or residual inputs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
OUTPUT_ROOT = HERE / "_outputs" / "p11_residual_fusion"
SAM2_SOURCE_ROOT = Path("/mnt/data/yongan-admin-2/.cache/upstream/sam2")
SAM2_CHECKPOINT = Path(
    "/mnt/data/yongan-admin-2/.cache/huggingface/hub/models--facebook--sam2.1-hiera-base-plus/"
    "blobs/a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
)
ROOT_SEED = 2693
FOLDS = (0, 4)
MAX_UPDATES = 40
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 0.0
GATE_REG = 0.05
RESIDUAL_REG = 0.001
MAX_RESIDUAL_CORRECTION = 0.05
MIN_PROMOTION_DELTA = 0.005
SAM2_SOURCE_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_DEPENDENCY_SITE = Path(
    os.environ.get(
        "FACIES_P11_SAM2_SITE_PACKAGES",
        "/mnt/data/yongan-admin-2/envs/atom-sam2-py310/lib/python3.10/site-packages",
    )
)
VARIANTS = (
    "strong_small_baseline",
    "direct_sam2",
    "pretrained_residual",
    "random_sam2_residual",
    "gate_zero",
)
TASK_MANIFEST_HASHES = {
    "facies_f3": "76a501385482bedce4e48dc44dfe5e9854b1b7aa0674fc223b3a6b42760f4614",
    "facies_penobscot": "3440bbbd415158e41c6f0ed14513208fc77075c108716f9f6593ff0847382c81",
}
FORBIDDEN_PATH_PARTS = {
    "test",
    "tests",
    "holdout",
    "known_holdout",
    "frozen_holdout",
}


def _install_local_hydra_compat() -> None:
    try:
        import hydra  # noqa: F401
        import omegaconf  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    hydra_state = {"config_module": None, "initialized": False}

    class AttrDict(dict):
        def __getattr__(self, key: str) -> Any:
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

    def _to_attr(value: Any) -> Any:
        if isinstance(value, dict):
            return AttrDict({key: _to_attr(item) for key, item in value.items()})
        if isinstance(value, list):
            return [_to_attr(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_to_attr(item) for item in value)
        if isinstance(value, str) and re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+",
            value,
        ):
            return float(value)
        return value

    def _set_path(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
        cursor = mapping
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise TypeError(f"override path {dotted_path!r} collides with non-mapping")
        cursor[parts[-1]] = value

    def initialize_config_module(config_module: str, version_base: str | None = None) -> None:
        hydra_state["config_module"] = config_module
        hydra_state["initialized"] = True

    def compose(config_name: str, overrides: Sequence[str] | None = None) -> Any:
        if not hydra_state["initialized"] or not hydra_state["config_module"]:
            raise RuntimeError("hydra compatibility layer has not been initialized")
        import yaml

        package = importlib.import_module(hydra_state["config_module"])
        package_root = Path(package.__file__).resolve().parent
        config_path = package_root / config_name
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload = {} if payload is None else dict(payload)
        for override in overrides or ():
            if not isinstance(override, str):
                raise TypeError(f"override must be a string, got {type(override).__name__}")
            cleaned = override[2:] if override.startswith("++") else override
            if "=" not in cleaned:
                raise ValueError(f"unsupported hydra override {override!r}")
            key, raw = cleaned.split("=", 1)
            import yaml as _yaml

            _set_path(payload, key, _yaml.safe_load(raw))
        return _to_attr(payload)

    def _locate_target(target: str) -> Any:
        module_name, _, object_name = target.rpartition(".")
        if not module_name:
            raise ValueError(f"target must be a dotted path, got {target!r}")
        module = importlib.import_module(module_name)
        return getattr(module, object_name)

    def instantiate(config: Any, _recursive_: bool = True, **kwargs: Any) -> Any:
        if config is None:
            return None
        if isinstance(config, AttrDict):
            config = dict(config)
        if isinstance(config, dict):
            if "_target_" in config:
                values = {
                    key: instantiate(value, _recursive_=_recursive_)
                    if _recursive_ and key != "_target_"
                    else value
                    for key, value in config.items()
                    if key != "_target_"
                }
                values.update(kwargs)
                constructor = _locate_target(str(config["_target_"]))
                return constructor(**values)
            return {
                key: instantiate(value, _recursive_=_recursive_)
                if _recursive_
                else value
                for key, value in config.items()
            }
        if isinstance(config, list):
            return [instantiate(item, _recursive_=_recursive_) for item in config]
        if isinstance(config, tuple):
            return tuple(instantiate(item, _recursive_=_recursive_) for item in config)
        return config

    class GlobalHydra:
        def is_initialized(self) -> bool:
            return bool(hydra_state["initialized"])

    global_hydra = GlobalHydra()
    hydra_module = types.ModuleType("hydra")
    hydra_module.compose = compose
    hydra_module.initialize_config_module = initialize_config_module

    utils_module = types.ModuleType("hydra.utils")
    utils_module.instantiate = instantiate

    core_module = types.ModuleType("hydra.core")
    global_hydra_module = types.ModuleType("hydra.core.global_hydra")
    global_hydra_module.GlobalHydra = type(
        "GlobalHydra",
        (),
        {"instance": staticmethod(lambda: global_hydra)},
    )

    omegaconf_module = types.ModuleType("omegaconf")

    class OmegaConf:
        @staticmethod
        def resolve(config: Any) -> Any:
            return config

    omegaconf_module.OmegaConf = OmegaConf

    sys.modules["hydra"] = hydra_module
    sys.modules["hydra.utils"] = utils_module
    sys.modules["hydra.core"] = core_module
    sys.modules["hydra.core.global_hydra"] = global_hydra_module
    sys.modules["omegaconf"] = omegaconf_module


_install_local_hydra_compat()

for root in (str(PROJECT_ROOT), str(HERE)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.ml_framework.contracts import TaskSpec  # noqa: E402
from _models.facies.sam2_semantic import build_model as build_sam2_direct  # noqa: E402
from _models.facies.smp_deeplabv3plus_r18 import (  # noqa: E402
    build_model as build_penobscot_baseline,
)
from _models.facies.smp_fpn_r18 import build_model as build_f3_baseline  # noqa: E402
from _models.gaia_dagt.foundation_runtime import insert_import_root  # noqa: E402
from _models.gaia_dagt.foundation_runtime import verify_checkpoint  # noqa: E402
from _models.gaia_dagt.foundation_runtime import verify_git_source  # noqa: E402

stage3 = importlib.import_module("_pipelines.02_task_datasets.facies.facies_p5_stage3")
evaluate_probabilities = importlib.import_module(
    "_pipelines.02_task_datasets.facies.p4_metrics"
).evaluate_probabilities


TASK_MODELS = {
    "facies_f3": build_f3_baseline,
    "facies_penobscot": build_penobscot_baseline,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def _is_forbidden_path(path: Path) -> bool:
    lowered_parts = {part.lower() for part in Path(path).parts}
    return bool(lowered_parts & FORBIDDEN_PATH_PARTS) or Path(path).name.lower() == "test.h5"


def validate_development_inputs(
    *,
    f3_manifest: Path,
    penobscot_manifest: Path,
    processed_root: Path,
) -> dict[str, Path]:
    """Resolve only the two locked manifests and development train archives."""
    manifests = {
        "facies_f3": Path(f3_manifest).resolve(),
        "facies_penobscot": Path(penobscot_manifest).resolve(),
    }
    processed_root = Path(processed_root).resolve()
    checked = [*manifests.values(), processed_root]
    checked.extend(processed_root / task_id / "train.h5" for task_id in manifests)
    forbidden = [path for path in checked if _is_forbidden_path(path)]
    if forbidden:
        raise ValueError(
            "frozen holdout path rejected before data access: "
            + ", ".join(str(path) for path in forbidden)
        )
    if manifests["facies_f3"] == manifests["facies_penobscot"]:
        raise ValueError("F3 and Penobscot require separate locked manifests")
    for task_id, manifest in manifests.items():
        if not manifest.is_file():
            raise FileNotFoundError(f"{task_id} split manifest is missing: {manifest}")
        train_h5 = processed_root / task_id / "train.h5"
        if not train_h5.is_file():
            raise FileNotFoundError(f"{task_id} development train archive is missing: {train_h5}")
    return manifests


def _validate_output_root(output_root: Path) -> Path:
    resolved = Path(output_root).resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"P11 output must stay inside the facies track: {resolved}") from exc
    return resolved


def _prepare_sam2_dependency_path() -> Path:
    """Reuse the accepted SAM2 environment's pure-Python support packages."""
    resolved = SAM2_DEPENDENCY_SITE.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            "SAM2 dependency site-packages is missing; set "
            f"FACIES_P11_SAM2_SITE_PACKAGES: {resolved}"
        )
    value = str(resolved)
    if value not in sys.path:
        # Append so torch-common's frozen Torch/SMP stack remains authoritative.
        sys.path.append(value)
    try:
        import iopath  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"SAM2 iopath dependency is unavailable under {resolved}") from exc
    return resolved


def _task_spec(task_id: str) -> TaskSpec:
    return stage3.get_task_spec(task_id)


def _build_small_model(task_id: str, num_classes: int, device: str) -> torch.nn.Module:
    spec = _task_spec(task_id)
    model = TASK_MODELS[task_id](spec, num_classes=num_classes, lane="scratch").to(device)
    return model


def _build_seeded_small_model(
    task_id: str,
    num_classes: int,
    device: str,
    *,
    seed: int,
) -> torch.nn.Module:
    _seed_all(seed)
    return _build_small_model(task_id, num_classes, device)


def _sam2_normalize(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 4 or inputs.shape[1] != 1:
        raise ValueError(f"SAM2 expects [B,1,H,W], got {tuple(inputs.shape)}")
    normalized = torch.clamp(inputs, -5.0, 5.0).add(5.0).div(10.0)
    normalized = F.interpolate(normalized, size=(1024, 1024), mode="bilinear", align_corners=False)
    normalized = normalized.repeat(1, 3, 1, 1)
    mean = normalized.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = normalized.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (normalized - mean) / std


class Sam2FeatureEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        task_id: str,
        num_classes: int,
        device: str,
        randomize: bool = False,
        random_seed: int = ROOT_SEED + 17,
    ) -> None:
        super().__init__()
        _prepare_sam2_dependency_path()
        spec = _task_spec(task_id)
        model = build_sam2_direct(
            spec,
            source_root=SAM2_SOURCE_ROOT,
            checkpoint_path=SAM2_CHECKPOINT,
            num_classes=num_classes,
            device=device,
            freeze_encoder=True,
        )
        self.backbone = model.backbone
        if randomize:
            generator = torch.Generator(device="cpu").manual_seed(random_seed)
            with torch.no_grad():
                for name, parameter in self.backbone.image_encoder.named_parameters():
                    if parameter.ndim >= 2:
                        values = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
                        values.normal_(0.0, 0.02, generator=generator)
                    elif "norm" in name.lower():
                        values = torch.ones(parameter.shape, dtype=parameter.dtype, device="cpu")
                    else:
                        values = torch.zeros(parameter.shape, dtype=parameter.dtype, device="cpu")
                    parameter.copy_(values.to(parameter.device))
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            encoded = self.backbone.image_encoder(_sam2_normalize(inputs))
        maps = list(encoded["backbone_fpn"])[-3:]
        if len(maps) != 3 or any(
            feature.ndim != 4 or feature.shape[1] != 256 for feature in maps
        ):
            raise ValueError(
                "unexpected SAM2 feature pyramid: "
                + repr([tuple(feature.shape) for feature in maps])
            )
        return maps


class DirectSam2Head(torch.nn.Module):
    """The direct frozen-encoder head used by the existing facies SAM2 adapter."""

    def __init__(self, num_classes: int, *, feature_channels: int = 256) -> None:
        super().__init__()
        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(feature_channels, 128, kernel_size=1),
                    torch.nn.GroupNorm(16, 128),
                    torch.nn.GELU(),
                )
                for _ in range(3)
            ]
        )
        self.semantic_head = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.GroupNorm(16, 128),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(
        self,
        sam2_maps: Sequence[torch.Tensor],
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(sam2_maps) != 3:
            raise ValueError(f"direct SAM2 head needs three feature maps, got {len(sam2_maps)}")
        target_size = tuple(int(value) for value in sam2_maps[0].shape[-2:])
        fused = None
        for projection, feature in zip(self.projections, sam2_maps):
            value = projection(feature)
            value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            fused = value if fused is None else fused + value
        logits = self.semantic_head(fused / len(sam2_maps))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class ResidualFusionCore(torch.nn.Module):
    def __init__(self, num_classes: int, *, feature_channels: int = 256) -> None:
        super().__init__()
        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(feature_channels, 64, kernel_size=1),
                    torch.nn.GELU(),
                )
                for _ in range(3)
            ]
        )
        self.residual_head = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, num_classes, kernel_size=1),
        )
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_classes + 4, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 1),
        )
        self.register_buffer(
            "residual_scale",
            torch.tensor(MAX_RESIDUAL_CORRECTION, dtype=torch.float32),
        )
        torch.nn.init.zeros_(self.residual_head[-1].weight)
        torch.nn.init.zeros_(self.residual_head[-1].bias)
        torch.nn.init.zeros_(self.gate_mlp[-1].weight)
        torch.nn.init.constant_(self.gate_mlp[-1].bias, -4.0)

    def forward(
        self,
        small_logits: torch.Tensor,
        sam2_maps: Sequence[torch.Tensor],
        *,
        gate_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_size = tuple(int(v) for v in sam2_maps[0].shape[-2:])
        fused = None
        for projection, feature in zip(self.projections, sam2_maps):
            value = projection(feature)
            value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            fused = value if fused is None else fused + value
        residual_logits = self.residual_head(fused / len(sam2_maps))
        residual_logits = F.interpolate(
            residual_logits, size=small_logits.shape[-2:], mode="bilinear", align_corners=False
        )
        gate_context = torch.cat(
            [
                small_logits.mean(dim=(2, 3)),
                small_logits.softmax(dim=1).amax(dim=1, keepdim=True).mean(dim=(2, 3)),
                *[
                    feature.abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
                    for feature in sam2_maps
                ],
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.gate_mlp(gate_context))
        if gate_override is not None:
            if not 0.0 <= float(gate_override) <= 1.0:
                raise ValueError("gate override must stay inside [0, 1]")
            gate = gate.new_full(gate.shape, float(gate_override))
        bounded_residual = torch.tanh(residual_logits)
        correction = gate[:, :, None, None] * self.residual_scale * bounded_residual
        if float(correction.detach().abs().max().cpu()) > MAX_RESIDUAL_CORRECTION + 1e-6:
            raise RuntimeError("residual correction escaped its fixed bound")
        final_logits = small_logits + correction
        return final_logits, gate.squeeze(1), correction


class ResidualFusionModel(torch.nn.Module):
    def __init__(
        self,
        small_model: torch.nn.Module,
        sam2_encoder: Sam2FeatureEncoder,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.small_model = small_model
        self.sam2_encoder = sam2_encoder
        self.core = ResidualFusionCore(num_classes)
        for parameter in self.small_model.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> "ResidualFusionModel":
        super().train(mode)
        self.small_model.eval()
        self.sam2_encoder.eval()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        gate_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            small_logits = self.small_model(inputs)
            sam2_maps = self.sam2_encoder(inputs)
        return self.core(small_logits, sam2_maps, gate_override=gate_override)


def _make_batches(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    batch_size: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(images))
    for _ in range(MAX_UPDATES):
        choice = rng.choice(indices, size=min(batch_size, len(indices)), replace=False)
        yield (
            torch.as_tensor(images[choice], dtype=torch.float32),
            torch.as_tensor(labels[choice], dtype=torch.long),
        )


def _make_index_batches(sample_count: int, *, seed: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count)
    for _ in range(MAX_UPDATES):
        yield rng.choice(indices, size=min(BATCH_SIZE, sample_count), replace=False)


def _predict_model(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = torch.as_tensor(
                images[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            logits = model(batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _metrics_payload(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
) -> tuple[dict[str, Any], list[list[int]]]:
    probabilities = torch.from_numpy(logits).softmax(dim=1).numpy()
    metrics, confusion = evaluate_probabilities(
        probabilities,
        labels,
        num_classes=num_classes,
        require_all_classes=False,
    )
    return metrics, confusion.tolist()


def _train_ce_model(
    model: torch.nn.Module,
    prepared: Any,
    *,
    device: str,
    seed: int,
    label: str,
) -> dict[str, Any]:
    _seed_all(seed)
    model = model.to(device)
    weights = torch.as_tensor(prepared.class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[float] = []
    model.train()
    for images, labels in _make_batches(prepared.train_images, prepared.train_labels, seed=seed, batch_size=BATCH_SIZE):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits, labels, weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{label} loss became non-finite")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    validation_logits = _predict_model(model, prepared.validation_images, device=device)
    metrics, confusion = _metrics_payload(
        validation_logits,
        prepared.validation_labels,
        num_classes=prepared.num_classes,
    )
    return {
        "model": model,
        "train_loss_last": history[-1],
        "train_loss_mean": float(np.mean(history)),
        "validation_logits": validation_logits,
        "metrics": metrics,
        "confusion": confusion,
    }


def _cache_sam2_features(
    encoder: Sam2FeatureEncoder,
    images: np.ndarray,
    *,
    device: str,
 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    levels: list[list[torch.Tensor]] = [[], [], []]
    encoder = encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = torch.as_tensor(
                images[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            maps = encoder(batch)
            for level, feature in enumerate(maps):
                levels[level].append(feature.detach().to(device="cpu", dtype=torch.float16))
    return tuple(torch.cat(parts, dim=0) for parts in levels)  # type: ignore[return-value]


def _cached_slice(
    features: Sequence[torch.Tensor],
    indices: Sequence[int] | np.ndarray,
    *,
    device: str,
) -> list[torch.Tensor]:
    index = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    return [
        feature.index_select(0, index).to(device=device, dtype=torch.float32)
        for feature in features
    ]


def _predict_direct_head(
    head: DirectSam2Head,
    features: Sequence[torch.Tensor],
    *,
    output_size: tuple[int, int],
    device: str,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(features[0]), BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, len(features[0])))
            logits = head(
                _cached_slice(features, indices, device=device),
                output_size=output_size,
            )
            outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _train_direct_head(
    prepared: Any,
    train_features: Sequence[torch.Tensor],
    validation_features: Sequence[torch.Tensor],
    *,
    device: str,
    seed: int,
) -> dict[str, Any]:
    _seed_all(seed)
    head = DirectSam2Head(prepared.num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    weights = torch.as_tensor(prepared.class_weights, dtype=torch.float32, device=device)
    history: list[float] = []
    output_size = tuple(int(value) for value in prepared.train_labels.shape[-2:])
    head.train()
    for indices in _make_index_batches(len(prepared.train_labels), seed=seed):
        labels = torch.as_tensor(prepared.train_labels[indices], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = head(
            _cached_slice(train_features, indices, device=device),
            output_size=output_size,
        )
        loss = F.cross_entropy(logits, labels, weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("direct SAM2 head loss became non-finite")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    validation_logits = _predict_direct_head(
        head,
        validation_features,
        output_size=tuple(int(value) for value in prepared.validation_labels.shape[-2:]),
        device=device,
    )
    metrics, confusion = _metrics_payload(
        validation_logits,
        prepared.validation_labels,
        num_classes=prepared.num_classes,
    )
    head.cpu()
    return {
        "train_loss_last": history[-1],
        "train_loss_mean": float(np.mean(history)),
        "metrics": metrics,
        "confusion": confusion,
    }


def _predict_residual_core(
    core: ResidualFusionCore,
    small_logits: torch.Tensor,
    features: Sequence[torch.Tensor],
    *,
    device: str,
    gate_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []
    correction_parts: list[np.ndarray] = []
    core.eval()
    with torch.no_grad():
        for start in range(0, len(small_logits), BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, len(small_logits)))
            logits, gate, correction = core(
                small_logits[indices].to(device=device, dtype=torch.float32),
                _cached_slice(features, indices, device=device),
                gate_override=gate_override,
            )
            logits_parts.append(logits.detach().cpu().numpy())
            gate_parts.append(gate.detach().cpu().numpy())
            correction_parts.append(correction.detach().cpu().numpy())
    return (
        np.concatenate(logits_parts, axis=0),
        np.concatenate(gate_parts, axis=0),
        np.concatenate(correction_parts, axis=0),
    )


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _train_residual_core(
    prepared: Any,
    train_small_logits: torch.Tensor,
    validation_small_logits: torch.Tensor,
    train_features: Sequence[torch.Tensor],
    validation_features: Sequence[torch.Tensor],
    *,
    device: str,
    seed: int,
    label: str,
) -> dict[str, Any]:
    _seed_all(seed)
    core = ResidualFusionCore(prepared.num_classes).to(device)
    optimizer = torch.optim.AdamW(core.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    weights = torch.as_tensor(prepared.class_weights, dtype=torch.float32, device=device)
    history: list[float] = []
    train_gates: list[float] = []
    train_corrections: list[float] = []
    core.train()
    for indices in _make_index_batches(len(prepared.train_labels), seed=seed):
        labels = torch.as_tensor(prepared.train_labels[indices], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, gate, correction = core(
            train_small_logits[indices].to(device=device, dtype=torch.float32),
            _cached_slice(train_features, indices, device=device),
        )
        loss = F.cross_entropy(logits, labels, weight=weights)
        loss = loss + GATE_REG * gate.mean() + RESIDUAL_REG * correction.abs().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{label} loss became non-finite")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        train_gates.extend(float(value) for value in gate.detach().cpu().flatten())
        train_corrections.append(float(correction.detach().abs().mean().cpu()))

    validation_logits, gates, corrections = _predict_residual_core(
        core,
        validation_small_logits,
        validation_features,
        device=device,
    )
    metrics, confusion = _metrics_payload(
        validation_logits,
        prepared.validation_labels,
        num_classes=prepared.num_classes,
    )
    gate_zero_logits, gate_zero_gates, gate_zero_corrections = _predict_residual_core(
        core,
        validation_small_logits,
        validation_features,
        device=device,
        gate_override=0.0,
    )
    gate_zero_metrics, gate_zero_confusion = _metrics_payload(
        gate_zero_logits,
        prepared.validation_labels,
        num_classes=prepared.num_classes,
    )
    small_numpy = validation_small_logits.numpy()
    gate_zero_diff = float(np.max(np.abs(gate_zero_logits - small_numpy)))
    if gate_zero_diff > 1e-7:
        raise RuntimeError(f"gate=0 failed to reproduce the small-model logits: {gate_zero_diff}")
    if float(np.max(np.abs(gate_zero_gates))) != 0.0:
        raise RuntimeError("gate=0 produced a nonzero gate")
    if float(np.max(np.abs(gate_zero_corrections))) != 0.0:
        raise RuntimeError("gate=0 produced a nonzero correction")

    gate_stats = _distribution_stats(gates)
    correction_abs = np.abs(corrections)
    residual_stats = {
        "mean_abs": float(np.mean(correction_abs)),
        "max_abs": float(np.max(correction_abs)),
        "fixed_bound": MAX_RESIDUAL_CORRECTION,
        "ratio_to_small_logits": float(
            np.mean(correction_abs) / max(np.mean(np.abs(small_numpy)), 1e-12)
        ),
    }
    core.cpu()
    return {
        "train_loss_last": history[-1],
        "train_loss_mean": float(np.mean(history)),
        "train_gate_mean": float(np.mean(train_gates)),
        "train_correction_mean_abs": float(np.mean(train_corrections)),
        "metrics": metrics,
        "confusion": confusion,
        "gate_stats": gate_stats,
        "residual_stats": residual_stats,
        "gate_zero": {
            "metrics": gate_zero_metrics,
            "confusion": gate_zero_confusion,
            "gate_stats": _distribution_stats(gate_zero_gates),
            "residual_stats": {
                "mean_abs": 0.0,
                "max_abs": 0.0,
                "fixed_bound": MAX_RESIDUAL_CORRECTION,
                "ratio_to_small_logits": 0.0,
            },
            "max_abs_logit_diff": gate_zero_diff,
        },
    }


def _make_sam2_encoder(
    task_id: str,
    num_classes: int,
    device: str,
    *,
    randomize: bool = False,
    random_seed: int = ROOT_SEED + 17,
) -> Sam2FeatureEncoder:
    return Sam2FeatureEncoder(
        task_id=task_id,
        num_classes=num_classes,
        device=device,
        randomize=randomize,
        random_seed=random_seed,
    )


def _fold_seed(fold_id: int) -> int:
    return ROOT_SEED + fold_id


def _run_task(
    task_id: str,
    *,
    manifest_path: Path,
    processed_root: Path,
    device: str,
    run_command: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    budget = stage3.Stage3Budget()
    results: list[dict[str, Any]] = []
    stable_hash: str | None = None
    runner_sha256 = _sha256(Path(__file__))
    git_head = _git_head()
    checkpoint_sha256 = _sha256(SAM2_CHECKPOINT)
    for fold_id in FOLDS:
        fold_started = time.perf_counter()
        prepared = stage3.prepare_fold(
            task_id=task_id,
            fold_id=fold_id,
            manifest_path=manifest_path,
            processed_root=processed_root,
            budget=budget,
        )
        if stable_hash is None:
            stable_hash = prepared.manifest_stable_hash
        elif stable_hash != prepared.manifest_stable_hash:
            raise RuntimeError("fold identity drift detected")
        if prepared.manifest_stable_hash != TASK_MANIFEST_HASHES[task_id]:
            raise RuntimeError(
                f"{task_id} manifest identity changed: {prepared.manifest_stable_hash}"
            )
        seed = _fold_seed(fold_id)

        variant_started = time.perf_counter()
        baseline_model = _build_seeded_small_model(
            task_id,
            prepared.num_classes,
            device,
            seed=seed,
        )
        baseline_result = _train_ce_model(
            baseline_model,
            prepared,
            device=device,
            seed=seed,
            label=f"{task_id}/baseline",
        )
        baseline_seconds = time.perf_counter() - variant_started
        trained_small_model = baseline_result.pop("model")
        train_small_logits = torch.from_numpy(
            _predict_model(trained_small_model, prepared.train_images, device=device)
        )
        validation_small_logits = torch.from_numpy(baseline_result["validation_logits"])
        trained_small_model.cpu()
        del trained_small_model, baseline_model
        gc.collect()
        torch.cuda.empty_cache()

        cache_started = time.perf_counter()
        pretrained_encoder = _make_sam2_encoder(
            task_id,
            prepared.num_classes,
            device,
            randomize=False,
        )
        pretrained_train_features = _cache_sam2_features(
            pretrained_encoder,
            prepared.train_images,
            device=device,
        )
        pretrained_validation_features = _cache_sam2_features(
            pretrained_encoder,
            prepared.validation_images,
            device=device,
        )
        feature_shapes = [list(feature.shape[1:]) for feature in pretrained_train_features]
        pretrained_encoder.cpu()
        del pretrained_encoder
        gc.collect()
        torch.cuda.empty_cache()
        pretrained_cache_seconds = time.perf_counter() - cache_started

        variant_started = time.perf_counter()
        direct_sam2 = _train_direct_head(
            prepared,
            pretrained_train_features,
            pretrained_validation_features,
            device=device,
            seed=seed,
        )
        direct_seconds = time.perf_counter() - variant_started

        variant_started = time.perf_counter()
        pretrained_residual = _train_residual_core(
            prepared,
            train_small_logits,
            validation_small_logits,
            pretrained_train_features,
            pretrained_validation_features,
            device=device,
            seed=seed,
            label=f"{task_id}/pretrained_residual",
        )
        pretrained_residual_seconds = time.perf_counter() - variant_started
        gate_zero = pretrained_residual.pop("gate_zero")
        del pretrained_train_features, pretrained_validation_features
        gc.collect()

        cache_started = time.perf_counter()
        random_encoder = _make_sam2_encoder(
            task_id,
            prepared.num_classes,
            device,
            randomize=True,
            random_seed=seed,
        )
        random_train_features = _cache_sam2_features(
            random_encoder,
            prepared.train_images,
            device=device,
        )
        random_validation_features = _cache_sam2_features(
            random_encoder,
            prepared.validation_images,
            device=device,
        )
        random_encoder.cpu()
        del random_encoder
        gc.collect()
        torch.cuda.empty_cache()
        random_cache_seconds = time.perf_counter() - cache_started

        variant_started = time.perf_counter()
        random_residual = _train_residual_core(
            prepared,
            train_small_logits,
            validation_small_logits,
            random_train_features,
            random_validation_features,
            device=device,
            seed=seed,
            label=f"{task_id}/random_sam2_residual",
        )
        random_residual.pop("gate_zero")
        random_residual_seconds = time.perf_counter() - variant_started
        del random_train_features, random_validation_features
        gc.collect()
        torch.cuda.empty_cache()

        baseline_metric = float(baseline_result["metrics"]["miou"])
        variant_payloads = [
            (
                "strong_small_baseline",
                baseline_result,
                "trained",
                "strong small-model main route trained on the fixed development fold",
                baseline_seconds,
                0.0,
            ),
            (
                "direct_sam2",
                direct_sam2,
                "trained",
                "direct semantic head trained on the same cached pretrained SAM2 features",
                direct_seconds,
                pretrained_cache_seconds,
            ),
            (
                "pretrained_residual",
                pretrained_residual,
                "trained",
                "small-model logits plus bounded pretrained-SAM2 residual correction",
                pretrained_residual_seconds,
                pretrained_cache_seconds,
            ),
            (
                "random_sam2_residual",
                random_residual,
                "trained",
                "same residual architecture with a random SAM2 encoder control",
                random_residual_seconds,
                random_cache_seconds,
            ),
            (
                "gate_zero",
                gate_zero,
                "control",
                "exact gate=0 degeneration of the trained pretrained residual core",
                0.0,
                pretrained_cache_seconds,
            ),
        ]

        for variant, payload, status, notes, duration_seconds, cache_seconds in variant_payloads:
            metrics = payload["metrics"]
            row = {
                "track": "facies",
                "dataset": "F3" if task_id == "facies_f3" else "Penobscot",
                "task_id": task_id,
                "fold_id": fold_id,
                "seed": seed,
                "variant": variant,
                "status": status,
                "command": run_command,
                "metric_name": "miou",
                "metric_value": float(metrics["miou"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "nll": float(metrics["nll"]),
                "brier": float(metrics["brier"]),
                "ece": float(metrics["ece"]),
                "baseline_metric": baseline_metric,
                "delta_abs": float(metrics["miou"] - baseline_metric),
                "gate_mean": float(payload.get("gate_stats", {}).get("mean", 0.0)),
                "gate_std": float(payload.get("gate_stats", {}).get("std", 0.0)),
                "gate_min": float(payload.get("gate_stats", {}).get("min", 0.0)),
                "gate_max": float(payload.get("gate_stats", {}).get("max", 0.0)),
                "gate_p05": float(payload.get("gate_stats", {}).get("p05", 0.0)),
                "gate_p50": float(payload.get("gate_stats", {}).get("p50", 0.0)),
                "gate_p95": float(payload.get("gate_stats", {}).get("p95", 0.0)),
                "residual_mean_abs": float(
                    payload.get("residual_stats", {}).get("mean_abs", 0.0)
                ),
                "residual_max_abs": float(
                    payload.get("residual_stats", {}).get("max_abs", 0.0)
                ),
                "residual_fixed_bound": MAX_RESIDUAL_CORRECTION,
                "residual_ratio_to_small_logits": float(
                    payload.get("residual_stats", {}).get(
                        "ratio_to_small_logits",
                        0.0,
                    )
                ),
                "train_loss_last": payload.get("train_loss_last"),
                "train_loss_mean": payload.get("train_loss_mean"),
                "gate_zero_max_abs_logit_diff": payload.get("max_abs_logit_diff"),
                "duration_seconds": float(duration_seconds),
                "feature_cache_seconds": float(cache_seconds),
                "evidence_path": str(
                    (output_root / "p11_residual_fusion_results.jsonl").relative_to(
                        PROJECT_ROOT
                    )
                ),
                "checkpoint_path": str(SAM2_CHECKPOINT)
                if variant != "strong_small_baseline"
                else "",
                "checkpoint_sha256": checkpoint_sha256
                if variant != "strong_small_baseline"
                else "",
                "git_head_at_run": git_head,
                "runner_sha256": runner_sha256,
                "notes": notes,
                "fold_split_hash": prepared.fold_split_hash,
                "manifest_stable_hash": prepared.manifest_stable_hash,
                "train_samples": len(prepared.train_images),
                "validation_samples": len(prepared.validation_images),
                "cached_feature_dtype": "float16_cpu",
                "cached_feature_shapes": feature_shapes,
                "frozen_test_accessed": False,
            }
            results.append(row)

        del train_small_logits, validation_small_logits
        gc.collect()
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "fold_id": fold_id,
                    "baseline_miou": baseline_metric,
                    "direct_sam2_miou": float(direct_sam2["metrics"]["miou"]),
                    "pretrained_residual_miou": float(
                        pretrained_residual["metrics"]["miou"]
                    ),
                    "random_sam2_residual_miou": float(
                        random_residual["metrics"]["miou"]
                    ),
                    "gate_mean": float(pretrained_residual["gate_stats"]["mean"]),
                    "fold_seconds": time.perf_counter() - fold_started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return results


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


def _plot_primary_metric(summary: Mapping[str, Any], output_root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    tasks = ["F3", "Penobscot"]
    variants = list(VARIANTS)
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    width = 0.15
    plotted_values: list[float] = []
    for task_index, task in enumerate(tasks):
        task_summary = summary["tasks"][task]
        for variant_index, variant in enumerate(variants):
            value = float(task_summary["variant_means"][variant]["miou"])
            plotted_values.append(value)
            x = task_index + (variant_index - 2) * width
            ax.bar(x, value, width=width, color=colors[variant_index], label=variant if task_index == 0 else None)
            ax.text(x, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_ylabel("mIoU")
    ax.set_title("Facies P11 fixed-development mIoU")
    ax.set_ylim(0, max(plotted_values) * 1.25)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    path = output_root / "p11_residual_fusion_miou.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_gate_distribution(summary: Mapping[str, Any], output_root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    tasks = ["F3", "Penobscot"]
    variants = (
        ("pretrained_residual", -0.08, "#7570b3"),
        ("random_sam2_residual", 0.08, "#e7298a"),
    )
    for task_index, task in enumerate(tasks):
        task_summary = summary["tasks"][task]
        for variant, offset, color in variants:
            gates = [
                float(row["gate_mean"])
                for row in task_summary["rows"]
                if row["variant"] == variant
            ]
            ax.scatter(
                [task_index + offset] * len(gates),
                gates,
                s=34,
                alpha=0.85,
                color=color,
                label=variant if task_index == 0 else None,
            )
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_ylabel("mean gate value")
    ax.set_title("Residual gate means on fixed development folds")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    path = output_root / "p11_residual_gate_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _validate_cuda_device(device: str) -> None:
    parsed = torch.device(device)
    if parsed.type != "cuda" or parsed.index not in (None, 0):
        raise ValueError("P11 fixed-development neural ablations require visible cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("P11 requires a real visible CUDA device")


def build(
    *,
    f3_manifest: Path,
    penobscot_manifest: Path,
    processed_root: Path,
    device: str = "cuda:0",
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Path]:
    started = time.perf_counter()
    _validate_cuda_device(device)
    manifests = validate_development_inputs(
        f3_manifest=f3_manifest,
        penobscot_manifest=penobscot_manifest,
        processed_root=processed_root,
    )
    output_root = _validate_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dependency_site = _prepare_sam2_dependency_path()
    source_root = verify_git_source(SAM2_SOURCE_ROOT, SAM2_SOURCE_REVISION)
    checkpoint = verify_checkpoint("facies", SAM2_CHECKPOINT)
    insert_import_root(source_root, "sam2")
    run_command = " ".join(
        [
            sys.executable,
            str(Path(__file__).relative_to(PROJECT_ROOT)),
            "run",
            "--f3-manifest",
            str(manifests["facies_f3"]),
            "--penobscot-manifest",
            str(manifests["facies_penobscot"]),
            "--processed-root",
            str(Path(processed_root).resolve()),
            "--device",
            device,
        ]
    )
    all_rows: list[dict[str, Any]] = []
    task_summaries: dict[str, Any] = {}
    for task_id in ("facies_f3", "facies_penobscot"):
        rows = _run_task(
            task_id,
            manifest_path=manifests[task_id],
            processed_root=Path(processed_root).resolve(),
            device=device,
            run_command=run_command,
            output_root=output_root,
        )
        all_rows.extend(rows)
        task_name = "F3" if task_id == "facies_f3" else "Penobscot"
        variant_means: dict[str, dict[str, float]] = {}
        for variant in VARIANTS:
            subset = [row for row in rows if row["variant"] == variant]
            variant_means[variant] = {
                "miou": float(np.mean([row["metric_value"] for row in subset])),
                "accuracy": float(np.mean([row["accuracy"] for row in subset])),
                "macro_f1": float(np.mean([row["macro_f1"] for row in subset])),
                "nll": float(np.mean([row["nll"] for row in subset])),
                "brier": float(np.mean([row["brier"] for row in subset])),
                "ece": float(np.mean([row["ece"] for row in subset])),
                "gate_mean": float(np.mean([row["gate_mean"] for row in subset])),
                "gate_std_across_folds": float(np.std([row["gate_mean"] for row in subset])),
                "residual_mean_abs": float(
                    np.mean([row["residual_mean_abs"] for row in subset])
                ),
            }
        baseline_miou = variant_means["strong_small_baseline"]["miou"]
        pretrained_miou = variant_means["pretrained_residual"]["miou"]
        random_miou = variant_means["random_sam2_residual"]["miou"]
        baseline_gain = pretrained_miou - baseline_miou
        foundation_specific_gain = pretrained_miou - random_miou
        promoted = (
            baseline_gain >= MIN_PROMOTION_DELTA
            and foundation_specific_gain >= MIN_PROMOTION_DELTA
        )
        task_summaries[task_name] = {
            "task_id": task_id,
            "rows": rows,
            "variant_means": variant_means,
            "best_variant": max(variant_means.items(), key=lambda item: item[1]["miou"])[0],
            "best_miou": max(value["miou"] for value in variant_means.values()),
            "comparison": {
                "pretrained_residual_minus_baseline": baseline_gain,
                "random_sam2_residual_minus_baseline": random_miou - baseline_miou,
                "pretrained_minus_random_sam2_residual": foundation_specific_gain,
                "gate_zero_minus_baseline": (
                    variant_means["gate_zero"]["miou"] - baseline_miou
                ),
            },
            "decision": {
                "state": "PROMOTED" if promoted else "NON_BENEFICIAL",
                "default_enabled": promoted,
                "minimum_promotion_delta": MIN_PROMOTION_DELTA,
                "promotion_rule": (
                    "pretrained_residual_minus_baseline>=0.005_and_"
                    "pretrained_minus_random_sam2_residual>=0.005"
                ),
                "gate_behavior": (
                    "near_zero"
                    if variant_means["pretrained_residual"]["gate_mean"] < 0.05
                    else "open"
                ),
            },
        }

    summary_payload = {
        "schema_version": "facies-p11-residual-fusion/v1",
        "experiment": {
            "main_branch": {
                "facies_f3": "smp_fpn_r18",
                "facies_penobscot": "smp_deeplabv3plus_r18",
            },
            "foundation_model": "facebook/sam2.1-hiera-base-plus",
            "sam2_source_revision": SAM2_SOURCE_REVISION,
            "sam2_checkpoint_sha256": _sha256(checkpoint),
            "sam2_dependency_site": str(dependency_site),
            "feature_policy": "frozen_encoder_features_cached_as_float16_cpu",
            "residual_formula": (
                "small_logits + sigmoid(gate) * 0.05 * tanh(residual_logits)"
            ),
            "residual_max_abs_logit_correction": MAX_RESIDUAL_CORRECTION,
            "gate_inputs": "small_logits_summary_and_cached_sam2_feature_norms_only",
        },
        "evaluation": {
            "evidence_class": "fixed_development_ablation",
            "folds": list(FOLDS),
            "root_seed": ROOT_SEED,
            "budget": asdict(stage3.Stage3Budget()),
            "f3_manifest_stable_hash": TASK_MANIFEST_HASHES["facies_f3"],
            "penobscot_manifest_stable_hash": TASK_MANIFEST_HASHES[
                "facies_penobscot"
            ],
            "frozen_test_accessed": False,
            "holdout_paths_accepted": False,
            "raw_predictions_persisted": False,
        },
        "tasks": {
            task: {
                key: value
                for key, value in task_summary.items()
                if key != "rows"
            }
            for task, task_summary in task_summaries.items()
        },
        "runtime": {
            "device": device,
            "duration_seconds": time.perf_counter() - started,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "command": run_command,
        },
    }
    jsonl_path = _write_jsonl(
        output_root / "p11_residual_fusion_results.jsonl",
        all_rows,
    )
    summary_path = _write_json(
        output_root / "p11_residual_fusion_summary.json",
        summary_payload,
    )
    report_path = _write_report(task_summaries, all_rows, output_root)
    metric_fig = _plot_primary_metric({"tasks": task_summaries}, output_root)
    gate_fig = _plot_gate_distribution({"tasks": task_summaries}, output_root)
    manifest_rows = [
        {"kind": "jsonl", "name": "p11_residual_fusion_results.jsonl", "path": str(jsonl_path.relative_to(PROJECT_ROOT)), "sha256": _sha256(jsonl_path)},
        {"kind": "json", "name": "p11_residual_fusion_summary.json", "path": str(summary_path.relative_to(PROJECT_ROOT)), "sha256": _sha256(summary_path)},
        {"kind": "md", "name": "p11_residual_fusion_report.md", "path": str(report_path.relative_to(PROJECT_ROOT)), "sha256": _sha256(report_path)},
        {"kind": "png", "name": "p11_residual_fusion_miou.png", "path": str(metric_fig.relative_to(PROJECT_ROOT)), "sha256": _sha256(metric_fig)},
        {"kind": "png", "name": "p11_residual_gate_distribution.png", "path": str(gate_fig.relative_to(PROJECT_ROOT)), "sha256": _sha256(gate_fig)},
    ]
    manifest_path_out = _write_csv(
        output_root / "artifact_manifest.csv",
        manifest_rows,
        ["kind", "name", "path", "sha256"],
    )
    return {
        "results": jsonl_path,
        "summary": summary_path,
        "report": report_path,
        "artifact_manifest": manifest_path_out,
        "miou_figure": metric_fig,
        "gate_figure": gate_fig,
    }


def _write_report(task_summaries: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], output_root: Path) -> Path:
    decisions = {
        task: task_summaries[task]["decision"]["state"]
        for task in ("F3", "Penobscot")
    }
    lines = [
        "# Facies P11 residual fusion experiment",
        "",
        "## Summary",
        "",
        "This fixed-development run keeps the strong small-model logits as the main route and tests a cached SAM2 residual branch with a bounded sigmoid gate.",
        "The residual correction is exactly `sigmoid(gate) × 0.05 × tanh(residual_logits)`, so its absolute per-logit contribution cannot exceed 0.05.",
        "The gate sees only inference-time summaries of the main logits and frozen cached SAM2 features; labels are used only by the training loss.",
        "",
        "## Result table",
        "",
        "| Task | Variant | mIoU | Δ vs baseline | Accuracy | Macro F1 | Gate mean | Correction mean abs | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for task in ("F3", "Penobscot"):
        baseline = task_summaries[task]["variant_means"]["strong_small_baseline"]["miou"]
        for variant in VARIANTS:
            metrics = task_summaries[task]["variant_means"][variant]
            note = (
                "exact main-route control"
                if variant == "gate_zero"
                else (
                    decisions[task].lower()
                    if variant == "pretrained_residual"
                    else "trained"
                )
            )
            lines.append(
                f"| {task} | {variant} | {metrics['miou']:.6f} | "
                f"{metrics['miou'] - baseline:+.6f} | {metrics['accuracy']:.6f} | "
                f"{metrics['macro_f1']:.6f} | {metrics['gate_mean']:.6f} | "
                f"{metrics['residual_mean_abs']:.8f} | {note} |"
            )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- F3 decision: **{decisions['F3']}**.",
            f"- Penobscot decision: **{decisions['Penobscot']}**.",
            "- `NON_BENEFICIAL` means the pretrained residual did not beat both the exact same-run strong baseline and the random-SAM2 residual control by the fixed 0.005 mIoU minimum; tie-level numerical changes are not promoted.",
            "",
            "## Residual gate interpretation",
            "",
            "- small logits remain the main route;",
            "- frozen SAM2 features are computed once per fold/encoder condition and cached in CPU float16;",
            "- the SAM2 correction is bounded by tanh, a fixed 0.05 scale, and a sigmoid gate;",
            "- gate=0 reproduces the main-route logits exactly (checked at `1e-7` tolerance);",
            "- random-init SAM2 is kept as the structure control.",
            "",
            "## Evidence boundary",
            "",
            "- No holdout or frozen-test data were opened for this run.",
            "- Only the locked development folds 0 and 4 were evaluated.",
            "- F3 and Penobscot used separate manifests, TaskSpecs, label spaces, models, and output rows.",
            "- Results are bounded development ablations, not fresh-blind or contest-test scores.",
        ]
    )
    report_path = output_root / "p11_residual_fusion_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def verify(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = _validate_output_root(output_root)
    manifest_path = output_root / "artifact_manifest.csv"
    summary_path = output_root / "p11_residual_fusion_summary.json"
    results_path = output_root / "p11_residual_fusion_results.jsonl"
    for path in (manifest_path, summary_path, results_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing P11 evidence artifact: {path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != 5:
        raise ValueError(f"P11 artifact manifest must contain five artifacts, got {len(manifest_rows)}")
    for row in manifest_rows:
        path = PROJECT_ROOT / row["path"]
        try:
            path.resolve().relative_to(HERE)
        except ValueError as exc:
            raise ValueError(f"non-facies artifact escaped into P11 manifest: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"P11 manifest artifact is missing: {path}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"P11 artifact hash mismatch: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "facies-p11-residual-fusion/v1":
        raise ValueError("unsupported P11 summary schema")
    evaluation = summary["evaluation"]
    if tuple(evaluation["folds"]) != FOLDS or evaluation["frozen_test_accessed"]:
        raise ValueError("P11 evidence violates the fixed-development boundary")
    if evaluation["f3_manifest_stable_hash"] != TASK_MANIFEST_HASHES["facies_f3"]:
        raise ValueError("F3 manifest identity drift in P11 summary")
    if (
        evaluation["penobscot_manifest_stable_hash"]
        != TASK_MANIFEST_HASHES["facies_penobscot"]
    ):
        raise ValueError("Penobscot manifest identity drift in P11 summary")

    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 2 * len(FOLDS) * len(VARIANTS):
        raise ValueError(f"P11 results require 20 rows, got {len(rows)}")
    expected_cells = {
        (task_id, fold_id, variant)
        for task_id in ("facies_f3", "facies_penobscot")
        for fold_id in FOLDS
        for variant in VARIANTS
    }
    observed_cells = {
        (row["task_id"], int(row["fold_id"]), row["variant"]) for row in rows
    }
    if observed_cells != expected_cells:
        raise ValueError("P11 results do not contain the exact task/fold/variant grid")
    current_runner_sha = _sha256(Path(__file__))
    for row in rows:
        if row["frozen_test_accessed"]:
            raise ValueError("a P11 result claims frozen-test access")
        if row["runner_sha256"] != current_runner_sha:
            raise ValueError("P11 row is not bound to the current runner")
        if float(row["residual_max_abs"]) > MAX_RESIDUAL_CORRECTION + 1e-6:
            raise ValueError("P11 result exceeded the residual correction bound")
    for task_id in ("facies_f3", "facies_penobscot"):
        for fold_id in FOLDS:
            cell_rows = [
                row
                for row in rows
                if row["task_id"] == task_id and int(row["fold_id"]) == fold_id
            ]
            by_variant = {row["variant"]: row for row in cell_rows}
            baseline = float(by_variant["strong_small_baseline"]["metric_value"])
            gate_zero = by_variant["gate_zero"]
            if abs(float(gate_zero["metric_value"]) - baseline) > 1e-12:
                raise ValueError(f"{task_id} fold {fold_id} gate=0 metric drift")
            if float(gate_zero["gate_zero_max_abs_logit_diff"]) > 1e-7:
                raise ValueError(f"{task_id} fold {fold_id} gate=0 logit drift")

    decisions = {
        task: summary["tasks"][task]["decision"]["state"]
        for task in ("F3", "Penobscot")
    }
    return {
        "schema_version": summary["schema_version"],
        "rows": len(rows),
        "artifacts": len(manifest_rows),
        "folds": list(FOLDS),
        "decisions": decisions,
        "frozen_test_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run both fixed-development ablations")
    run_parser.add_argument("--f3-manifest", type=Path, required=True)
    run_parser.add_argument("--penobscot-manifest", type=Path, required=True)
    run_parser.add_argument("--processed-root", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    verify_parser = commands.add_parser("verify", help="verify portable P11 evidence")
    verify_parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.command == "run":
        outputs = build(
            f3_manifest=args.f3_manifest,
            penobscot_manifest=args.penobscot_manifest,
            processed_root=args.processed_root,
            device=args.device,
            output_root=args.output_root,
        )
        print(
            json.dumps(
                {name: str(path) for name, path in outputs.items()},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps(verify(args.output_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
