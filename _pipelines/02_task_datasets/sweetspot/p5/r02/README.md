# P5.2 / protocol R2

This package implements the real sweetspot R2 development budget sweep.

Scope:

- T1 reservoir quality, T2 hydrocarbon pay, and T3 productivity only.
- 10 frozen model families.
- Main sweep budgets: 64, 256, 1024.
- One-factor ablations: 256 only.
- T4-T7 remain status-only boundaries and are not merged into the 120-cell sweep.

The runner never opens frozen test artifacts and never creates labels.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /mnt/data/yongan-admin-2/.cache/volve-p5/envs/tabular-cpu/bin/python \
  -m _pipelines.02_task_datasets.sweetspot.p5.r02.runner \
  --device cpu
```

When running InceptionTime cells on GPU, pass the explicit shared lock path required by the protocol:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /mnt/data/yongan-admin-2/.cache/volve-p5/envs/tabular-cpu/bin/python \
  -m _pipelines.02_task_datasets.sweetspot.p5.r02.runner \
  --device cuda \
  --gpu-lock /mnt/data/yongan-admin-2/.cache/volve-p5/locks/gpu0.lock
```

