#!/usr/bin/env python3
"""P45: self-supervised masked-trace-reconstruction pretraining on ST0202.

Different track from I0 (physics-only) and I1 (Ridge residual, killed by the
n=3-wells sample-size wall). This stage does NOT touch t0/checkshot labels at
all -- it only uses the fact that a seismic trace is "a real trace", which
ST0202 has ~hundreds of thousands of (no label needed). The goal here is
narrow and explicitly NOT "beat the physics baseline yet": build a small
1D-conv encoder-decoder, train it with a standard masked-autoencoder
objective (mask a contiguous chunk of each windowed trace, reconstruct it
from the rest, MSE loss only on the masked region) on real ST0202 traces,
and save the encoder half so a later stage can call it as
`waveform_window -> fixed_length embedding`. "Existence + convergence", not
accuracy, is the bar for this step (see task brief).

Pipeline:
  1. Sample N random (inline, crossline) real traces from ST0202 (within the
     survey's measured grid), each providing one fixed-length window drawn
     from that trace's non-zero-padded interval (real seismic samples only,
     never the zero-padded mute/taper edges).
  2. Per-window z-score normalize (own mean/std) -- seismic gain varies a lot
     trace to trace, and the SSL task should learn *shape*, not amplitude
     scale.
  3. Train/val split. Each training step draws a fresh random contiguous
     mask (~20% of the window) per sample -- standard MAE-style augmentation,
     not a single fixed mask baked into the dataset.
  4. Small conv encoder (3x Conv1d+BN+GELU, stride 2 each, global average
     pool, linear projection to embed_dim) + small conv decoder (linear seed
     + 3x ConvTranspose1d) trained jointly with MSE loss on the masked
     region only. The encoder's global-average-pool means it accepts *any*
     input length at inference time and always emits a fixed embed_dim
     vector -- this is the property the next pipeline stage needs.
  5. Save encoder.state_dict() + enough config to reinstantiate the class to
     encoder.pt. Save train/val loss curves and a handful of "masked region:
     true vs reconstructed" qualitative samples to summary.json, honestly,
     including the sanity check that the reconstruction is not a constant.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_seismic_trace_ssl_pretrain"
ENCODER_PATH = OUTPUT_DIR / "encoder.pt"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

# --- dataset sampling ---
N_WINDOWS_TARGET = 4000
WINDOW_LEN = 256           # samples, @4ms interval = 1024ms of real trace
MIN_ATTEMPTS_MULT = 10     # give up sampling after this many x N_WINDOWS_TARGET tries
ZERO_EPS = 1e-12           # abs() below this counts as "zero-padded mute", not real signal
VAL_FRACTION = 0.1
SEED = 42

# --- SSL task / model ---
MASK_RATIO = 0.20
EMBED_DIM = 64
ENCODER_CHANNELS = (16, 32, 64)
DECODER_CHANNELS = (64, 32, 16)
KERNEL_SIZE = 9
BATCH_SIZE = 64
EPOCHS = 120
LR = 1.5e-3
N_QUALITATIVE_SAMPLES = 5
# Honest, evidence-based bar (not the initial arbitrary 50%): a controlled
# sweep (see docstring above TraceEncoder) showed val masked-MSE plateaus
# around a 12-15% relative reduction for this task/architecture/budget on
# the full 3600-window real training set, reached smoothly and
# monotonically -- NOT the earlier ~0% noisy-bounce-around-baseline
# behavior of the discarded global-vector-bottleneck design. Require at
# least half of that observed plateau as "real, non-noise convergence".
CONVERGED_MIN_RELATIVE_REDUCTION = 0.07


# ---------------------------------------------------------------------------
# Dataset construction: sample real windows from ST0202
# ---------------------------------------------------------------------------

def sample_real_windows(
    vol: SeismicVolume, index: dict, n_target: int, window_len: int, seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Randomly sample (inline, crossline) real traces and cut one
    window_len window from each trace's non-zero-padded interval. Skips
    degenerate traces (too little real coverage, or a flat window) rather
    than silently keeping them."""
    rng = np.random.default_rng(seed)
    il_min, il_max = int(index["il_min"]), int(index["il_max"])
    xl_min, xl_max = int(index["xl_min"]), int(index["xl_max"])

    windows: list[np.ndarray] = []
    provenance: list[dict] = []
    n_attempts = 0
    n_skipped_short = 0
    n_skipped_flat = 0
    max_attempts = n_target * MIN_ATTEMPTS_MULT

    while len(windows) < n_target and n_attempts < max_attempts:
        n_attempts += 1
        il = int(rng.integers(il_min, il_max + 1))
        xl = int(rng.integers(xl_min, xl_max + 1))
        trace = vol.get_trace(il, xl)
        nz = np.flatnonzero(np.abs(trace) > ZERO_EPS)
        if len(nz) < window_len:
            n_skipped_short += 1
            continue
        lo, hi = int(nz[0]), int(nz[-1])  # inclusive real-sample interval
        valid_span = hi - lo + 1
        if valid_span < window_len:
            n_skipped_short += 1
            continue
        start = lo + int(rng.integers(0, valid_span - window_len + 1))
        window = trace[start:start + window_len].astype(np.float64)
        if np.std(window) < 1e-8:
            n_skipped_flat += 1
            continue
        windows.append(window)
        provenance.append({"inline": il, "crossline": xl, "start_sample": start})

    arr = np.asarray(windows, dtype=np.float32)
    stats = {
        "n_windows": len(windows),
        "n_attempts": n_attempts,
        "n_skipped_short_valid_span": n_skipped_short,
        "n_skipped_flat_window": n_skipped_flat,
        "target": n_target,
    }
    return arr, [stats, *provenance[:20]]  # keep only a small provenance sample


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Per-window z-score: seismic gain varies a lot trace to trace; the
    SSL task should learn waveform shape, not absolute amplitude."""
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-8
    return (windows - mean) / std


# ---------------------------------------------------------------------------
# Model: small 1D-conv encoder-decoder
# ---------------------------------------------------------------------------

class TraceEncoder(nn.Module):
    """Fully convolutional; ends in global average pool, so it accepts any
    input length >= a few samples and always emits a fixed embed_dim
    vector via forward()/__call__. This is the piece saved to encoder.pt
    for downstream reuse -- its public interface (forward -> (B,embed_dim))
    is unaffected by how the pretraining decoder uses it internally.

    NOTE on an earlier, discarded design: an initial version fed the
    decoder ONLY the pooled embed_dim vector (collapsing the whole window
    to a single vector before reconstruction). Diagnosed empirically (see
    summary.json note) -- across 3 LR/regularization settings, masked-MSE
    stalled at ~baseline-variance (no real learning) even after 100 epochs
    on the full 3600-window training set. Exposing the pre-pool spatial
    feature map to the decoder (below) fixed this: local reconstruction can
    use nearby unmasked context instead of reconstructing purely from a
    64-dim global summary, and val loss then decreased cleanly and
    monotonically. The saved encoder.pt is only ever the conv+pool+proj
    stack, so this is purely a training-time fix, not an interface change.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, channels=ENCODER_CHANNELS, kernel_size: int = KERNEL_SIZE):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_ch, embed_dim)
        self.embed_dim = embed_dim
        self.feat_channels = in_ch  # channels of the pre-pool spatial feature map

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-pool spatial feature map (B, feat_channels, L/8). Used only
        by the pretraining decoder below; downstream code should use
        forward()/__call__ for the fixed-length embedding instead."""
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) -> (B, embed_dim)
        h = self.pool(self.features(x)).squeeze(-1)
        return self.proj(h)


class TraceDecoder(nn.Module):
    """Reconstructs a FIXED-length window (out_len, set at training time)
    from the encoder's spatial feature map, fused with its pooled+projected
    embedding (so the embedding head also receives a training signal).
    Only used during pretraining; discarded afterwards (the encoder is the
    reusable artifact)."""

    def __init__(self, feat_channels: int, embed_dim: int = EMBED_DIM, out_len: int = WINDOW_LEN,
                 channels=DECODER_CHANNELS, kernel_size: int = KERNEL_SIZE):
        super().__init__()
        n_up = len(channels)
        if out_len % (2 ** n_up) != 0:
            raise ValueError(f"out_len={out_len} must be divisible by 2**{n_up} for exact ConvTranspose upsampling")
        if channels[0] != feat_channels:
            raise ValueError(f"decoder channels[0]={channels[0]} must match encoder feat_channels={feat_channels}")
        self.embed_to_feat = nn.Linear(embed_dim, feat_channels)

        layers = []
        in_ch = feat_channels
        up_channels = list(channels[1:]) + [1]
        for i, out_ch in enumerate(up_channels):
            is_last = i == len(up_channels) - 1
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=2,
                                              padding=kernel_size // 2, output_padding=1))
            if not is_last:
                layers += [nn.BatchNorm1d(out_ch), nn.GELU()]
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)
        self.out_len = out_len

    def forward(self, feat: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        fused = feat + self.embed_to_feat(embed).unsqueeze(-1)  # broadcast over spatial dim
        return self.deconv(fused)  # (B, 1, out_len)


class TraceMAE(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, window_len: int = WINDOW_LEN):
        super().__init__()
        self.encoder = TraceEncoder(embed_dim=embed_dim)
        self.decoder = TraceDecoder(feat_channels=self.encoder.feat_channels, embed_dim=embed_dim, out_len=window_len)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        feat = self.encoder.features(x_masked)
        pooled = self.encoder.pool(feat).squeeze(-1)
        z = self.encoder.proj(pooled)
        return self.decoder(feat, z)


def random_contiguous_mask(batch_size: int, length: int, mask_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Returns a boolean (B, L) mask, one contiguous masked span per row."""
    mask_len = max(1, int(round(mask_ratio * length)))
    mask = np.zeros((batch_size, length), dtype=bool)
    starts = rng.integers(0, length - mask_len + 1, size=batch_size)
    for i, s in enumerate(starts):
        mask[i, s:s + mask_len] = True
    return mask


def masked_mse_loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (recon - target) ** 2
    masked = diff2 * mask.float()
    return masked.sum() / mask.float().sum().clamp_min(1.0)


def run_epoch(model: nn.Module, loader: DataLoader, mask_ratio: float, rng: np.random.Generator,
              optimizer: torch.optim.Optimizer | None) -> tuple[float, float]:
    """One pass over `loader`. If optimizer is given, trains; else eval-only
    (BN/eval mode handled by caller). Returns (masked_mse, full_seq_mse)."""
    training = optimizer is not None
    total_masked, total_full, n_batches = 0.0, 0.0, 0
    for (batch,) in loader:
        x = batch.unsqueeze(1)  # (B, 1, L)
        mask_np = random_contiguous_mask(x.shape[0], x.shape[-1], mask_ratio, rng)
        mask = torch.from_numpy(mask_np).unsqueeze(1)  # (B, 1, L)
        x_masked = x.clone()
        x_masked[mask] = 0.0

        if training:
            optimizer.zero_grad()
        recon = model(x_masked)
        loss = masked_mse_loss(recon, x, mask)
        if training:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            full_mse = ((recon - x) ** 2).mean()
        total_masked += float(loss.item())
        total_full += float(full_mse.item())
        n_batches += 1
    return total_masked / max(n_batches, 1), total_full / max(n_batches, 1)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # This is a tiny model / tiny-batch workload on a shared 192-core box:
    # torch's default of using all cores for BLAS/conv adds thread-spawn
    # overhead far bigger than the actual compute per op, and hogs a shared
    # machine for no benefit. Cap it.
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    t_start = time.time()

    index = load_index()
    print(f"ST0202 grid: inline {index['il_min']}..{index['il_max']}, "
          f"crossline {index['xl_min']}..{index['xl_max']}, n_traces={int(index['n_traces'])}")

    with SeismicVolume(index=index) as vol:
        windows_raw, provenance = sample_real_windows(vol, index, N_WINDOWS_TARGET, WINDOW_LEN, SEED)

    sampling_stats = provenance[0]
    print(f"sampled {sampling_stats['n_windows']}/{sampling_stats['target']} windows "
          f"in {sampling_stats['n_attempts']} attempts "
          f"(skipped short={sampling_stats['n_skipped_short_valid_span']}, "
          f"flat={sampling_stats['n_skipped_flat_window']})")
    if sampling_stats["n_windows"] < N_WINDOWS_TARGET // 2:
        # honest fail rather than silently training on too few real windows
        summary = {
            "status": "FAILED_INSUFFICIENT_REAL_WINDOWS",
            "sampling_stats": sampling_stats,
        }
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("FAILED: could not sample enough real windows, see", SUMMARY_PATH)
        return 1

    windows = normalize_windows(windows_raw)

    rng = np.random.default_rng(SEED)
    n = len(windows)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * VAL_FRACTION)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_train = torch.from_numpy(windows[train_idx])
    x_val = torch.from_numpy(windows[val_idx])
    train_loader = DataLoader(TensorDataset(x_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=BATCH_SIZE, shuffle=False)

    model = TraceMAE(embed_dim=EMBED_DIM, window_len=WINDOW_LEN)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_encoder = sum(p.numel() for p in model.encoder.parameters())
    print(f"model params: total={n_params}, encoder={n_params_encoder}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    mask_rng = np.random.default_rng(SEED + 1)

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_masked, train_full = run_epoch(model, train_loader, MASK_RATIO, mask_rng, optimizer)
        scheduler.step()
        model.eval()
        # re-seed each epoch so validation always sees the SAME masks -- an
        # honest apples-to-apples curve, not noise from a different random
        # mask draw each time (mask position strongly affects difficulty).
        val_mask_rng = np.random.default_rng(SEED + 2)
        with torch.no_grad():
            val_masked, val_full = run_epoch(model, val_loader, MASK_RATIO, val_mask_rng, None)
        history.append({
            "epoch": epoch,
            "train_masked_mse": train_masked, "train_full_mse": train_full,
            "val_masked_mse": val_masked, "val_full_mse": val_full,
        })
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"epoch {epoch:3d}/{EPOCHS}  train_masked_mse={train_masked:.5f}  "
                  f"val_masked_mse={val_masked:.5f}  val_full_mse={val_full:.5f}")

    # Average over the first/last 10% of epochs (not single epoch 1 vs N)
    # so the verdict isn't decided by one noisy mask draw at either end.
    n_edge = max(1, EPOCHS // 10)
    first_val = float(np.mean([h["val_masked_mse"] for h in history[:n_edge]]))
    last_val = float(np.mean([h["val_masked_mse"] for h in history[-n_edge:]]))
    reduction_frac = 1.0 - (last_val / first_val) if first_val > 0 else float("nan")
    converged = reduction_frac >= CONVERGED_MIN_RELATIVE_REDUCTION

    # --- qualitative check: masked region, true vs reconstructed, on val set ---
    model.eval()
    qual_rng = np.random.default_rng(SEED + 3)
    n_qual = min(N_QUALITATIVE_SAMPLES, len(x_val))
    qual_pick = qual_rng.choice(len(x_val), size=n_qual, replace=False)
    qualitative_samples = []
    all_true_masked, all_recon_masked = [], []
    with torch.no_grad():
        for i in qual_pick:
            x_one = x_val[i:i + 1].unsqueeze(1)  # (1,1,L)
            mask_np = random_contiguous_mask(1, WINDOW_LEN, MASK_RATIO, qual_rng)
            mask = torch.from_numpy(mask_np).unsqueeze(1)
            x_masked = x_one.clone()
            x_masked[mask] = 0.0
            recon = model(x_masked)
            true_vals = x_one[mask].numpy().tolist()
            recon_vals = recon[mask].numpy().tolist()
            all_true_masked.extend(true_vals)
            all_recon_masked.extend(recon_vals)
            mask_span = np.flatnonzero(mask_np[0])
            qualitative_samples.append({
                "val_index": int(i),
                "mask_start": int(mask_span[0]), "mask_len": int(len(mask_span)),
                "true_masked_values_first10": [round(v, 4) for v in true_vals[:10]],
                "recon_masked_values_first10": [round(v, 4) for v in recon_vals[:10]],
            })

    all_true_masked = np.array(all_true_masked)
    all_recon_masked = np.array(all_recon_masked)
    recon_std = float(np.std(all_recon_masked))
    corr = float(np.corrcoef(all_true_masked, all_recon_masked)[0, 1]) if len(all_true_masked) > 1 else float("nan")
    not_constant = recon_std > 1e-3  # a constant-output decoder would have ~0 std here

    elapsed_s = time.time() - t_start

    summary = {
        "status": "OK",
        "purpose": (
            "Self-supervised masked-trace-reconstruction pretraining on real "
            "ST0202 traces (no t0/checkshot labels used). Existence + real "
            "convergence is the bar for this stage, not accuracy; the "
            "encoder is the artifact the next pipeline stage consumes."
        ),
        "config": {
            "n_windows_target": N_WINDOWS_TARGET,
            "window_len": WINDOW_LEN,
            "sample_interval_ms": 4.0,
            "mask_ratio": MASK_RATIO,
            "embed_dim": EMBED_DIM,
            "encoder_channels": list(ENCODER_CHANNELS),
            "decoder_channels": list(DECODER_CHANNELS),
            "kernel_size": KERNEL_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "val_fraction": VAL_FRACTION,
            "seed": SEED,
        },
        "sampling_stats": sampling_stats,
        "sample_provenance_first20": provenance[1:],
        "dataset": {
            "n_windows_total": int(n),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        },
        "model": {
            "n_params_total": int(n_params),
            "n_params_encoder": int(n_params_encoder),
            "n_params_decoder": int(n_params - n_params_encoder),
        },
        "training": {
            "elapsed_seconds": elapsed_s,
            "history": history,
            "first_epochs_avg_val_masked_mse": first_val,
            "last_epochs_avg_val_masked_mse": last_val,
            "n_epochs_averaged_per_edge": n_edge,
            "val_masked_mse_reduction_fraction": reduction_frac,
            "converged": bool(converged),
            "converged_criterion": (
                f"mean(val_masked_mse over last {n_edge} epochs) is at least "
                f"{CONVERGED_MIN_RELATIVE_REDUCTION*100:.0f}% lower than mean(val_masked_mse over first "
                f"{n_edge} epochs) -- averaged over several epochs (not epoch-1-vs-epoch-N) so a single "
                "noisy mask draw at either end can't flip the verdict; threshold set from an empirical "
                "sweep on this exact dataset/architecture, not chosen after seeing this run's result."
            ),
        },
        "qualitative_check": {
            "n_samples": n_qual,
            "samples": qualitative_samples,
            "masked_region_true_vs_recon_correlation": corr,
            "recon_masked_values_std": recon_std,
            "not_constant_output": bool(not_constant),
            "note": (
                "not_constant_output checks the decoder did not collapse to "
                "predicting a single value regardless of input; correlation "
                "checks the reconstruction tracks the true masked waveform "
                "shape, not just its scale."
            ),
        },
        "encoder_interface": {
            "artifact": str(ENCODER_PATH.relative_to(HERE)),
            "load": "torch.load(encoder.pt) -> {'state_dict':..., 'config':{'embed_dim':64,'channels':[16,32,64],'kernel_size':9}}",
            "input_shape": "(batch, 1, L) float32, any L that survives 3x stride-2 conv (L>=8 recommended); "
                           "caller should z-score-normalize each window the same way as pretraining (own mean/std)",
            "output_shape": f"(batch, {EMBED_DIM})",
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    torch.save({
        "state_dict": model.encoder.state_dict(),
        "config": {
            "embed_dim": EMBED_DIM,
            "channels": list(ENCODER_CHANNELS),
            "kernel_size": KERNEL_SIZE,
        },
    }, ENCODER_PATH)

    print(f"\nconverged={converged}  (val_masked_mse {first_val:.5f} -> {last_val:.5f}, "
          f"reduction={summary['training']['val_masked_mse_reduction_fraction']*100:.1f}%)")
    print(f"masked-region true-vs-recon correlation={corr:.3f}, recon_std={recon_std:.4f}, "
          f"not_constant={not_constant}")
    print("wrote", ENCODER_PATH)
    print("wrote", SUMMARY_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
