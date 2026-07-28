# Sweetspot P7 — Chronos-2 time-series foundation lane

This lane connects real pretrained Chronos-2 weights only to T3, where the
contract requires forecasting the next 30 days of oil production from causal
production history. A bounded T4 direct-water-forecast experiment is also
recorded, but it is promoted only if its frozen-development average precision
beats the archived CatBoost leader. Static and spatial tracks are explicitly
excluded.

The primary comparison reuses the four frozen P4 development folds and never
opens the already-consumed known holdout. F1 blends the Chronos forecast with a
causal history-mean control using one weight selected from fold-train labels.

Run with the project-specific environment and the shared GPU lock:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/data/yongan-admin-2/.cache/huggingface \
  /mnt/data/yongan-admin-2/envs/volve-chronos2/bin/python \
  -m _pipelines.02_task_datasets.sweetspot.p7.runner \
  --local-files-only \
  --gpu-lock-path /mnt/data/yongan-admin-2/.cache/volve-p5/locks/gpu0.lock
```
