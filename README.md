# EA-WAM

This repository contains the isolated EA-WAM extension developed on top of
FastWAM. The upstream FastWAM source and checkpoints remain unchanged.

The current continuation flow variant starts from the frozen FastWAM action
proposal and transports it toward the demonstration action with a conditional
flow model. LDE extracts prediction errors from consecutive visual latents.
TEM maintains error memory across chunk decisions. RAR performs action
refinement with ten flow sampling steps.

## Verified configuration

| Component | Setting |
|---|---|
| FastWAM | Frozen |
| LDE | Hidden 256, 2 layers |
| TEM | Hidden 256 |
| RAR | Hidden 768, 12 layers, 8 heads |
| Added parameters | 120.62M |
| Training data | All four LIBERO suites |
| Training windows | 273,465 |
| Batch size | 64 |
| Training epochs | 10 |
| Selected checkpoint | Epoch 1 |
| Learning rate | 1e-4 |
| Precision | BF16 |
| Flow sampling steps | 10 |
| Replan steps | 8 |
| Arm scale | 0.05 |
| Gripper scale | 0.0 |
| Action noise | 0.005 |

Only the output projection is zero initialized. The FastWAM parameters remain
frozen throughout training.

## Repository layout

```text
configs/
  model/eawam.yaml
  task/eawam_continuation_libero_full_v3.yaml
extensions/
  eawam/
  eawam_continuation/
results/
  libero_comparison.json
  checksums.json
```

The `extensions/eawam` package provides LDE, TEM, the residual RAR baseline,
training utilities and LIBERO evaluation. The
`extensions/eawam_continuation` package implements the continuation flow
variant used for the reported result.

## Installation

Install the official FastWAM repository and its LIBERO environment first.
Copy this repository into the FastWAM root while preserving the directory
structure.

```bash
cp -r extensions /path/to/FastWAM/
cp -r configs /path/to/FastWAM/
cd /path/to/FastWAM
```

Prepare the official FastWAM checkpoint, dataset statistics, LIBERO datasets
and cached text embeddings following the upstream instructions.

## Training

The configuration can build the frozen proposal cache automatically. An
existing sharded cache can be supplied with `proposal_cache_path` to avoid
recomputing FastWAM proposals.

```bash
python -u -m extensions.eawam_continuation.train_libero \
  task=eawam_continuation_libero_full_v3 \
  proposal_cache_path=/path/to/frozen_proposals/manifest.json
```

The run used batch size 64 for ten epochs. The reported model uses
`eawam_epoch_01.pt`.

## Offline scale sweep

```bash
export FASTWAM_ROOT=/path/to/FastWAM
export EAWAM_CONTINUATION_OUTPUT=/path/to/training_run
export EAWAM_PROPOSAL_CACHE=/path/to/frozen_proposals/manifest.json
python -m extensions.eawam_continuation.offline_sweep
```

The selected global action scale is 0.05. Gripper correction is disabled.

## LIBERO evaluation

The following example evaluates all ten tasks from LIBERO Spatial. Replace the
suite name to evaluate Object, Goal and Long.

```bash
python -u extensions/eawam/eval_libero_suite.py \
  --suite-tasks libero_spatial:0,libero_spatial:1,libero_spatial:2,libero_spatial:3,libero_spatial:4,libero_spatial:5,libero_spatial:6,libero_spatial:7,libero_spatial:8,libero_spatial:9 \
  --trials 50 \
  --resume \
  task=eawam_continuation_libero_full_v3 \
  ckpt=/path/to/eawam_epoch_01.pt \
  EVALUATION.output_dir=./evaluate_results/eawam_continuation/libero_spatial \
  EVALUATION.replan_steps=8 \
  EVALUATION.visualize_future_video=false \
  model.module.residual_scale=0.05 \
  +model.module.gripper_residual_scale=0.0 \
  gpu_id=0
```

The official evaluation uses 40 tasks with 50 trials per task, producing 2000
episodes in total.

## Verified LIBERO results

| Method | Spatial | Object | Goal | Long | Overall |
|---|---:|---:|---:|---:|---:|
| FastWAM | 98.0 | 99.2 | 96.8 | 93.0 | 96.75 |
| Previous residual EA-WAM | 97.6 | 98.8 | 96.6 | 93.6 | 96.65 |
| Continuation EA-WAM | 98.0 | 98.2 | 96.6 | 94.2 | 96.75 |

All values are success rates in percent. Continuation EA-WAM improves Long by
1.2 percentage points over FastWAM, while its overall success rate is equal to
FastWAM. The exact counts are available in
`results/libero_comparison.json`.

## Checkpoints

Model weights are not stored in Git because of their size. The verified hashes
are listed in `results/checksums.json`.

The selected extension checkpoint SHA256 is
`b11dfa589660cf02e57ac3076a934422d4b512f862131367744e0778aaae391f`.

The frozen FastWAM checkpoint SHA256 is
`1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`.

## Tests

```bash
python -m extensions.eawam.test_modules
python -m extensions.eawam_continuation.test_smoke
```

The repository is an extension package and requires the upstream FastWAM
runtime for training and full evaluation.
