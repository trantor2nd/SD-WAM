from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from extensions.eawam.train_libero import (
    ShardedProposalDataset,
    _CachedProposalBackbone,
    _move_proposals,
)

from .model import ContinuationErrorAwareFastWAM


ROOT = Path(
    os.environ.get("FASTWAM_ROOT", str(Path.cwd()))
).resolve()
OUTPUT = Path(
    os.environ.get(
        "EAWAM_CONTINUATION_OUTPUT",
        ROOT / "runs/eawam_continuation/libero_full_v3",
    )
)
CACHE = Path(
    os.environ.get(
        "EAWAM_PROPOSAL_CACHE",
        OUTPUT / "frozen_base_proposals_steps10_shards/manifest.json",
    )
)
RESULT = Path(
    os.environ.get(
        "EAWAM_SWEEP_RESULT",
        OUTPUT / "offline_scale_sweep.json",
    )
)
SCALES = [
    float(value)
    for value in os.environ.get(
        "EAWAM_SWEEP_SCALES",
        "0.0,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0",
    ).split(",")
]
GRIPPER_SCALES = [
    float(value)
    for value in os.environ.get(
        "EAWAM_SWEEP_GRIPPER_SCALES",
        "0.0,0.02,0.05,0.1,0.2,0.4,0.6,1.0",
    ).split(",")
]
SAMPLES = int(os.environ.get("EAWAM_SWEEP_SAMPLES", "2048"))
BATCH_SIZE = 64
NUM_STEPS = 10
FIRST_EPOCH = int(os.environ.get("EAWAM_SWEEP_FIRST_EPOCH", "1"))
LAST_EPOCH = int(os.environ.get("EAWAM_SWEEP_LAST_EPOCH", "1000000"))
SAMPLED_SHARDS = int(os.environ.get("EAWAM_SWEEP_SHARDS", "64"))


cfg = OmegaConf.load(OUTPUT / "config.yaml")
module_cfg = dict(OmegaConf.to_container(cfg.model.module, resolve=True))
module_cfg.pop("warmstart_path", None)
device = torch.device("cuda:0")
dtype = torch.bfloat16
backbone = _CachedProposalBackbone(
    action_dim=int(cfg.data.train.processor.action_output_dim),
    latent_dim=int(module_cfg.get("latent_dim", 48)),
    proprio_dim=int(cfg.data.train.processor.proprio_output_dim),
    device=device,
    dtype=dtype,
)
model = ContinuationErrorAwareFastWAM(
    backbone,
    base_checkpoint=str(cfg.model.base_checkpoint),
    **module_cfg,
)

dataset = ShardedProposalDataset(CACHE)
shard_indices = (
    torch.linspace(
        0,
        len(dataset.shards) - 1,
        steps=min(SAMPLED_SHARDS, len(dataset.shards)),
    )
    .round()
    .long()
    .unique()
    .tolist()
)
samples_per_shard = max(SAMPLES // max(len(shard_indices), 1), 1)
indices = []
for shard_index in shard_indices:
    length = dataset.shard_lengths[shard_index]
    local_indices = (
        torch.linspace(
            0,
            length - 1,
            steps=min(samples_per_shard, length),
        )
        .round()
        .long()
        .unique()
        .tolist()
    )
    offset = dataset.shard_offsets[shard_index]
    indices.extend(offset + local_index for local_index in local_indices)
indices = sorted(indices[:SAMPLES])
loader = DataLoader(
    Subset(dataset, indices),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
cached_batches = list(loader)


def checkpoint_epoch(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


checkpoints = sorted(
    [
        path
        for path in OUTPUT.glob("eawam_epoch_*.pt")
        if FIRST_EPOCH <= checkpoint_epoch(path) <= LAST_EPOCH
    ],
    key=checkpoint_epoch,
)
if not checkpoints:
    checkpoints = [OUTPUT / "eawam_best.pt"]

metrics = []
for checkpoint in checkpoints:
    model.load_checkpoint(checkpoint)
    model.eval()
    pairs = [
        (arm_scale, gripper_scale)
        for arm_scale in SCALES
        for gripper_scale in GRIPPER_SCALES
    ]
    sums = {pair: 0.0 for pair in pairs}
    counts = {pair: 0 for pair in pairs}
    with torch.no_grad():
        for sample in cached_batches:
            proposals = _move_proposals(sample, device)
            predicted_video = proposals["predicted_video"]
            observed_video = proposals["observed_video"]
            base_action = proposals["base_action"]
            target_action = proposals["target_action"]
            action_is_pad = proposals["action_is_pad"]
            future_steps = min(
                max(predicted_video.shape[2] - 1, 0),
                max((target_action.shape[1] - 1) // model.refine_horizon, 0),
            )
            memory = None
            for step_index in range(future_steps):
                error_tokens = model.lde(
                    predicted_video[:, :, step_index + 1],
                    observed_video[:, :, step_index + 1],
                )
                memory = model.tem(error_tokens, memory)
                start = (step_index + 1) * model.refine_horizon
                end = min(start + model.refine_horizon, target_action.shape[1])
                if end <= start:
                    continue
                base_segment = base_action[:, start:end]
                target_segment = target_action[:, start:end]
                valid = torch.ones(
                    target_segment.shape[:2],
                    device=device,
                    dtype=torch.bool,
                )
                if action_is_pad is not None:
                    valid = ~action_is_pad[:, start:end]
                original_length = end - start
                base_for_model = base_segment
                if original_length < model.refine_horizon:
                    base_for_model = F.pad(
                        base_segment,
                        (0, 0, 0, model.refine_horizon - original_length),
                    )
                transported = model.rar.sample(
                    base_for_model,
                    error_tokens,
                    memory,
                    num_steps=NUM_STEPS,
                )[:, :original_length]
                raw_correction = transported - base_segment
                valid_float = valid.to(torch.float32)
                token_count = int(valid.sum().item())
                for arm_scale, gripper_scale in pairs:
                    if raw_correction.shape[-1] <= 1:
                        scaled_correction = (
                            raw_correction * float(gripper_scale)
                        )
                    else:
                        scaled_correction = torch.cat(
                            [
                                raw_correction[..., :-1]
                                * float(arm_scale),
                                raw_correction[..., -1:]
                                * float(gripper_scale),
                            ],
                            dim=-1,
                        )
                    correction = model._bounded_correction(
                        scaled_correction
                    )
                    refined = (base_segment + correction).clamp(-1, 1)
                    token_error = (
                        (refined.float() - target_segment.float())
                        .abs()
                        .mean(dim=-1)
                    )
                    pair = (arm_scale, gripper_scale)
                    sums[pair] += float(
                        (token_error * valid_float).sum().item()
                    )
                    counts[pair] += token_count

    base_pair = (0.0, 0.0)
    base_l1 = sums[base_pair] / max(counts[base_pair], 1)
    epoch = (
        checkpoint_epoch(checkpoint)
        if checkpoint.name.startswith("eawam_epoch_")
        else None
    )
    for arm_scale, gripper_scale in pairs:
        pair = (arm_scale, gripper_scale)
        refined_l1 = sums[pair] / max(counts[pair], 1)
        metrics.append(
            {
                "checkpoint": str(checkpoint),
                "epoch": epoch,
                "scale": arm_scale,
                "gripper_scale": gripper_scale,
                "samples": len(indices),
                "valid_tokens": counts[pair],
                "base_action_l1": base_l1,
                "refined_action_l1": refined_l1,
                "relative_l1_improvement": (base_l1 - refined_l1)
                / max(base_l1, 1e-12),
            }
        )

best = min(metrics, key=lambda item: item["refined_action_l1"])
payload = {
    "checkpoints": [str(path) for path in checkpoints],
    "first_epoch": FIRST_EPOCH,
    "last_epoch": LAST_EPOCH,
    "sampled_shards": len(shard_indices),
    "samples": len(indices),
    "num_steps": NUM_STEPS,
    "arm_scales": SCALES,
    "gripper_scales": GRIPPER_SCALES,
    "metrics": metrics,
    "best": best,
}
temporary = RESULT.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
os.replace(temporary, RESULT)
print(json.dumps(payload, indent=2))
