"""Thin PyTorch Forecasting TFT adapter with explicit causal-frame input."""
from __future__ import annotations

import io
import hashlib
import resource
import time
from typing import Any, Mapping

import numpy as np

from .p5_common import AdapterSkip, require_single_target

model_id = "temporal_fusion_transformer"


def capabilities() -> dict[str, Any]:
    return {"task_types": ["regression"], "input_modalities": ["causal_time_series_frame"], "supports_missing_mask": True, "supports_uncertainty": True, "stage1_input_key": "time_series_frame", "external_weights": "forbidden"}


class TFTAdapter:
    def __init__(self, task_spec, config):
        self.task_spec = task_spec
        self.config = dict(config)

    def stage1_smoke(self, inputs: Mapping[str, Any], target, target_mask, *, seed):
        del target, target_mask
        if "time_series_frame" not in inputs:
            raise AdapterSkip("input_modality_missing", "TFT requires an approved causal time_series_frame")
        try:
            import torch
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
            from pytorch_forecasting.metrics import MAE
        except ImportError as exc:
            from .p5_common import dependency_skip
            raise dependency_skip("pytorch_forecasting") from exc
        frame = inputs["time_series_frame"].copy()
        required = {"time_idx", "group_id", "target", "known"}
        if not required <= set(frame.columns):
            raise ValueError(f"TFT frame missing columns: {sorted(required - set(frame.columns))}")
        torch.manual_seed(seed)
        dataset = TimeSeriesDataSet(
            frame, time_idx="time_idx", target="target", group_ids=["group_id"],
            max_encoder_length=int(self.config.get("encoder_length", 8)),
            max_prediction_length=int(self.config.get("prediction_length", 2)),
            time_varying_known_reals=["time_idx", "known"],
            time_varying_unknown_reals=["target"],
            add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        )
        batch = next(iter(dataset.to_dataloader(train=True, batch_size=2, num_workers=0)))
        x, y = batch
        model = TemporalFusionTransformer.from_dataset(
            dataset, hidden_size=4, attention_head_size=1, hidden_continuous_size=4,
            lstm_layers=1, dropout=0.0, output_size=1, loss=MAE(), log_interval=-1,
            reduce_on_plateau_patience=2,
        )
        started = time.monotonic(); model.train()
        output = model(x).prediction
        truth = y[0] if isinstance(y, (tuple, list)) else y
        loss = model.loss(output, truth)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite TFT smoke loss")
        loss.backward()
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = TemporalFusionTransformer.from_dataset(
            dataset, hidden_size=4, attention_head_size=1, hidden_continuous_size=4,
            lstm_layers=1, dropout=0.0, output_size=1, loss=MAE(), log_interval=-1,
            reduce_on_plateau_patience=2,
        )
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True)); restored.eval(); model.eval()
        with torch.no_grad():
            before = model(x).prediction; after = restored(x).prediction
        delta = float(torch.max(torch.abs(before - after)).item())
        return {"real_development_batch_samples": int(before.shape[0]), "raw_output_shape": list(before.shape), "finite_output": bool(torch.isfinite(before).all()), "single_step_loss": float(loss.item()), "backward_completed": True, "checkpoint_bytes": buffer.getbuffer().nbytes, "checkpoint_roundtrip_max_abs_delta": delta, "output_sha256": hashlib.sha256(before.detach().cpu().numpy().tobytes()).hexdigest(), "device": "cpu", "peak_vram_bytes": 0, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "download_bytes": 0, "wall_seconds": time.monotonic() - started, "test_accessed": False}


def build_model(task_spec, **config):
    require_single_target(task_spec, capabilities()["task_types"])
    try:
        import pytorch_forecasting  # noqa: F401
    except ImportError as exc:
        from .p5_common import dependency_skip
        raise dependency_skip("pytorch_forecasting") from exc
    return TFTAdapter(task_spec, config)
