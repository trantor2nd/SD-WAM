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
from extensions.eawam_continuation.model import (
    ContinuationErrorAwareFastWAM,
)


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
CHECKPOINT = Path(
    os.environ.get(
        "EAWAM_CHECKPOINT",
        OUTPUT / "eawam_epoch_01.pt",
    )
)
BOUNDARIES = Path(
    os.environ.get(
        "EAWAM_SUITE_BOUNDARIES",
        OUTPUT / "suite_dataset_boundaries.json",
    )
)
RESULT = Path(
    os.environ.get(
        "EAWAM_SUITE_SWEEP_RESULT",
        OUTPUT / "offline_suite_scale_sweep_epoch01.json",
    )
)
SCALES = [
    float(value)
    for value in os.environ.get(
        "EAWAM_SUITE_SCALES",
        "0.0,0.02,0.05,0.08,0.10,0.15,0.20,0.30",
    ).split(",")
]
SAMPLES_PER_SUITE = int(
    os.environ.get("EAWAM_SUITE_SAMPLES", "1024")
)
BATCH_SIZE = 64
NUM_STEPS = 10


cfg = OmegaConf.load(OUTPUT / "config.yaml")
module_cfg = dict(
    OmegaConf.to_container(cfg.model.module, resolve=True)
)
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
model.load_checkpoint(CHECKPOINT)
model.eval()
dataset = ShardedProposalDataset(CACHE)
boundaries = json.loads(BOUNDARIES.read_text(encoding="utf-8"))[
    "boundaries"
]

suite_results = {}
with torch.no_grad():
    for suite, boundary in boundaries.items():
        start = int(boundary["start"])
        end = int(boundary["end"])
        count = min(SAMPLES_PER_SUITE, end - start)
        indices = (
            torch.linspace(start, end - 1, steps=count)
            .round()
            .long()
            .unique()
            .tolist()
        )
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        sums = {scale: 0.0 for scale in SCALES}
        counts = {scale: 0 for scale in SCALES}
        for sample in loader:
            proposals = _move_proposals(sample, device)
            predicted_video = proposals["predicted_video"]
            observed_video = proposals["observed_video"]
            base_action = proposals["base_action"]
            target_action = proposals["target_action"]
            action_is_pad = proposals["action_is_pad"]
            future_steps = min(
                max(predicted_video.shape[2] - 1, 0),
                max(
                    (target_action.shape[1] - 1)
                    // model.refine_horizon,
                    0,
                ),
            )
            memory = None
            for step_index in range(future_steps):
                error_tokens = model.lde(
                    predicted_video[:, :, step_index + 1],
                    observed_video[:, :, step_index + 1],
                )
                memory = model.tem(error_tokens, memory)
                segment_start = (
                    step_index + 1
                ) * model.refine_horizon
                segment_end = min(
                    segment_start + model.refine_horizon,
                    target_action.shape[1],
                )
                if segment_end <= segment_start:
                    continue
                base_segment = base_action[
                    :, segment_start:segment_end
                ]
                target_segment = target_action[
                    :, segment_start:segment_end
                ]
                valid = torch.ones(
                    target_segment.shape[:2],
                    device=device,
                    dtype=torch.bool,
                )
                if action_is_pad is not None:
                    valid = ~action_is_pad[
                        :, segment_start:segment_end
                    ]
                original_length = segment_end - segment_start
                base_for_model = base_segment
                if original_length < model.refine_horizon:
                    base_for_model = F.pad(
                        base_segment,
                        (
                            0,
                            0,
                            0,
                            model.refine_horizon - original_length,
                        ),
                    )
                transported = model.rar.sample(
                    base_for_model,
                    error_tokens,
                    memory,
                    num_steps=NUM_STEPS,
                )[:, :original_length]
                raw = transported - base_segment
                valid_float = valid.to(torch.float32)
                token_count = int(valid.sum().item())
                for scale in SCALES:
                    if raw.shape[-1] <= 1:
                        correction = torch.zeros_like(raw)
                    else:
                        correction = torch.cat(
                            [
                                raw[..., :-1] * float(scale),
                                torch.zeros_like(raw[..., -1:]),
                            ],
                            dim=-1,
                        )
                    correction = model._bounded_correction(correction)
                    refined = (
                        base_segment + correction
                    ).clamp(-1, 1)
                    token_error = (
                        (
                            refined.float()
                            - target_segment.float()
                        )
                        .abs()
                        .mean(dim=-1)
                    )
                    sums[scale] += float(
                        (token_error * valid_float).sum().item()
                    )
                    counts[scale] += token_count
        base_l1 = sums[0.0] / max(counts[0.0], 1)
        metrics = []
        for scale in SCALES:
            refined_l1 = sums[scale] / max(counts[scale], 1)
            metrics.append(
                {
                    "scale": scale,
                    "samples": len(indices),
                    "valid_tokens": counts[scale],
                    "base_action_l1": base_l1,
                    "refined_action_l1": refined_l1,
                    "relative_l1_improvement": (
                        (base_l1 - refined_l1)
                        / max(base_l1, 1e-12)
                    ),
                }
            )
        suite_results[suite] = {
            "metrics": metrics,
            "best": min(
                metrics,
                key=lambda item: item["refined_action_l1"],
            ),
        }

payload = {
    "checkpoint": str(CHECKPOINT),
    "selection_source": "frozen full demonstration proposal cache",
    "samples_per_suite": SAMPLES_PER_SUITE,
    "num_steps": NUM_STEPS,
    "scales": SCALES,
    "suites": suite_results,
}
temporary = RESULT.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
temporary.replace(RESULT)
print(json.dumps(payload, indent=2))
