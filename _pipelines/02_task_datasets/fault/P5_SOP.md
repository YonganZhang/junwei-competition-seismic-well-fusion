# Fault P5 open-model Stage-1 SOP

This SOP is subordinate to `P5_open_model_benchmark_protocol.md` and the
frozen fault P4 mask/test contract. It covers contract smoke only: no HPO, CV,
model ranking, refit, or frozen-test consumption.

## Frozen candidates and sources

The first ten model IDs are exactly:

`monai_segresnet`, `monai_dynunet`, `nnunet_v2_3d_fullres`,
`pytorch3dunet_unet3d`, `faultnet_md`, `faultseg3d_keras`, `monai_vnet`,
`mednext_v1_s_k3`, `uxnet3d`, and `monai_swinunetr`.

Each one is dynamically discovered from `_models/fault/<model_id>.py`. Exact
primary URL, commit/tag, code/weight license, dependency spec, shape constraints,
and weight approval are frozen in `_models/fault/p5_source_locks.json`. The
adapters do not import one another through a central registry and do not copy an
upstream repository.

## Environment and execution

Use the owner-provided shared environment only. Do not install into it from this
track and do not download pretrained weights:

```bash
PYTHON="${VOLVE_P5_TORCH_PYTHON:?set this to the shared torch-common interpreter}"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  _pipelines/02_task_datasets/fault/p5_stage1.py \
  --development-hdf5 <audited-train-only-fault.h5> \
  --device cuda:0
```

The HDF5 must declare `task=fault`, `split=train`, and its SHA-256 must match
the P4 `audited_v2` train hash. The runner records only its logical role, hash,
byte count, and selected sample keys; it does not persist a machine/worktree
path. It has no test loader/path argument.

For a dependency-ready trainable model, Stage-1 first runs a synthetic batch
with explicit positive and verified-negative masks through build, forward,
masked BCE-with-logits, backward, optimizer step, complete checkpoint reload,
and same-seed comparison. It then runs a real development forward. The real
loss/backward/checkpoint gate opens only when that real batch includes both
fault-stick positives and audit-proven `verified_negative_mask` voxels.

The current P4 evidence has zero audited verified negatives. Therefore a model
may carry `synthetic_contract_smoked` evidence and a passed real forward while
its overall Stage-1 status remains structured `skipped` with reason
`NO_AUDITED_VERIFIED_NEGATIVES`. Proxy/non-fault samples and unlabelled zeros
cannot upgrade that result.

Expected static skips are also evidence, not failures:

- `faultnet_md`: required upstream weight has no approved license/hash and no
  weight is downloaded;
- `faultseg3d_keras`: CC BY-NC code/weight use is not approved for this run;
- `nnunet_v2_3d_fullres`: `nnunetv2` remains absent to avoid dependency
  conflicts;
- MedNeXt/UX-Net: no clean local checkout at the exact locked commit.

Unexpected import/build/numerical/checkpoint errors are `failed` and make the
runner exit non-zero. They are never changed into a skip or a passed result.

## Evidence

Portable JSON evidence is written under `_outputs/p5_stage1/`:

- `summary.json`: ten statuses, environment, P4 gate hashes, zero-download and
  frozen-test assertions;
- `models/<model_id>.json`: source lock, dependency/license gate, resource use,
  contract/checkpoint evidence, or a structured skip/failure;
- `manifest.json`: SHA-256 and byte count for every JSON artifact.

Stage-1 checkpoints are temporary round-trip assets. Their hash/size are
recorded, but the checkpoint itself is not committed.

## Tests

From the project root:

```bash
PYTHON="${VOLVE_P5_TORCH_PYTHON:?set this to the shared torch-common interpreter}"
PYTHONNOUSERSITE=1 "$PYTHON" -m unittest discover -v \
  -s _pipelines/02_task_datasets/fault -p 'test_fault_p5.py'
PYTHONNOUSERSITE=1 "$PYTHON" -m unittest discover -v \
  -s _pipelines/02_task_datasets/fault -p 'test_fault_p4.py'
# The portable legacy gate uses the project/default Python because torch-common
# intentionally does not carry the seismic I/O package used by build_dataset.py.
python3 -m unittest discover -v \
  -s _pipelines/02_task_datasets/fault -p 'test_fault_pipeline.py'
```
