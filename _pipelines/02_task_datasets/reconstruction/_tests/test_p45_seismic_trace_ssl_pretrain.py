"""Regression tests for the P45 masked-trace-reconstruction SSL pretraining.

Checks the concrete deliverable asked for: the encoder artifact loads, and
given an arbitrary length-matching waveform it produces a real (non-zero,
non-constant) fixed-length embedding -- not that the SSL task beats the
physics baseline (it isn't meant to, at this stage).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_seismic_trace_ssl_pretrain import (  # noqa: E402
    ENCODER_PATH, SUMMARY_PATH, WINDOW_LEN, EMBED_DIM, TraceEncoder,
)


def _require_artifacts():
    if not ENCODER_PATH.exists() or not SUMMARY_PATH.exists():
        pytest.skip("run p45_seismic_trace_ssl_pretrain.py main() first")


def _load_summary() -> dict:
    _require_artifacts()
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _load_encoder() -> TraceEncoder:
    _require_artifacts()
    ckpt = torch.load(ENCODER_PATH, map_location="cpu")
    cfg = ckpt["config"]
    encoder = TraceEncoder(embed_dim=cfg["embed_dim"], channels=tuple(cfg["channels"]),
                            kernel_size=cfg["kernel_size"])
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()
    return encoder


def test_summary_status_ok_and_enough_real_windows_sampled():
    summary = _load_summary()
    assert summary["status"] == "OK"
    assert summary["sampling_stats"]["n_windows"] >= 2000
    assert summary["dataset"]["n_windows_total"] == summary["sampling_stats"]["n_windows"]


def test_training_loss_actually_dropped_multi_epoch():
    """Not '1 epoch and stop': there must be a real multi-epoch history and
    the val masked-reconstruction MSE must have genuinely decreased."""
    summary = _load_summary()
    history = summary["training"]["history"]
    assert len(history) >= 10
    first_val = history[0]["val_masked_mse"]
    last_val = history[-1]["val_masked_mse"]
    assert last_val < first_val
    assert summary["training"]["converged"] is True


def test_qualitative_reconstruction_is_not_constant_and_tracks_shape():
    summary = _load_summary()
    qc = summary["qualitative_check"]
    assert qc["n_samples"] >= 3
    assert qc["not_constant_output"] is True
    assert qc["recon_masked_values_std"] > 1e-3
    # a totally untrained/broken encoder-decoder would show ~0 or negative
    # correlation between true and reconstructed masked-region values.
    assert qc["masked_region_true_vs_recon_correlation"] > 0.1


def test_model_param_count_is_small():
    summary = _load_summary()
    n_total = summary["model"]["n_params_total"]
    assert 1e4 <= n_total <= 1e6


def test_encoder_loads_from_disk():
    encoder = _load_encoder()
    assert isinstance(encoder, TraceEncoder)


def test_encoder_output_shape_matches_training_window_len():
    encoder = _load_encoder()
    batch = 4
    x = torch.randn(batch, 1, WINDOW_LEN)
    with torch.no_grad():
        z = encoder(x)
    assert z.shape == (batch, EMBED_DIM)


def test_encoder_accepts_a_different_length_than_training_window():
    """Requirement: the encoder must be usable as waveform_window ->
    fixed-length embedding for ANY length-matching input, not just the
    exact WINDOW_LEN used during pretraining (global-avg-pool architecture
    property)."""
    encoder = _load_encoder()
    for length in (128, 384, 512):
        x = torch.randn(3, 1, length)
        with torch.no_grad():
            z = encoder(x)
        assert z.shape == (3, EMBED_DIM)
        assert torch.isfinite(z).all()


def test_encoder_embedding_is_not_zero_or_constant():
    """Guards against a degenerate/collapsed encoder (e.g. dead ReLUs or an
    untrained random init masquerading as a real checkpoint): different real
    input waveforms must produce different, non-zero embeddings."""
    encoder = _load_encoder()
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(8, 1, WINDOW_LEN)).astype(np.float32))
    with torch.no_grad():
        z = encoder(x)
    assert z.shape == (8, EMBED_DIM)
    assert not torch.allclose(z, torch.zeros_like(z))
    # rows should not all collapse to the same vector
    row_std = z.std(dim=0)
    assert float(row_std.mean()) > 1e-4
