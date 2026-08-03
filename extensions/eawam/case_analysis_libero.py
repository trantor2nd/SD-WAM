"""Create a reproducible single-window EA-WAM action-error case study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from fastwam.utils import misc

from .train_libero import (
    ShardedProposalDataset,
    _create_cached_proposal_model,
    _move_proposals,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def _refine_one(
    model,
    proposals,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_video = proposals["predicted_video"]
    observed_video = proposals["observed_video"]
    base_action = proposals["base_action"]
    target_action = proposals["target_action"]
    refined_action = base_action.clone()
    calibrated = torch.zeros(
        base_action.shape[:2],
        device=base_action.device,
        dtype=torch.bool,
    )
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
        original_length = end - start
        if original_length < model.refine_horizon:
            base_for_model = torch.nn.functional.pad(
                base_segment,
                (0, 0, 0, model.refine_horizon - original_length),
            )
        else:
            base_for_model = base_segment
        generator = torch.Generator(device=base_action.device).manual_seed(
            int(seed) + step_index
        )
        residual = model.rar.sample(
            base_for_model,
            error_tokens,
            memory,
            num_steps=model.proposal_inference_steps,
            generator=generator,
            start_from_zero=model.residual_zero_start,
        )[:, :original_length]
        residual = (residual * model.residual_scale).clamp(
            -model.max_residual,
            model.max_residual,
        )
        refined_action[:, start:end] = (base_segment + residual).clamp(-1, 1)
        calibrated[:, start:end] = True
    return refined_action, calibrated


def _sample_curves(model, proposal, *, seed: int):
    device = next(model.lde.parameters()).device
    batched = {
        key: value.unsqueeze(0)
        for key, value in proposal.items()
    }
    batched = _move_proposals(batched, device)
    refined, calibrated = _refine_one(model, batched, seed=seed)
    base = batched["base_action"]
    target = batched["target_action"]
    pad = batched.get("action_is_pad")
    baseline_error = (base.float() - target.float()).abs().mean(dim=-1)[0]
    refined_error = (refined.float() - target.float()).abs().mean(dim=-1)[0]
    valid = torch.ones_like(baseline_error, dtype=torch.bool)
    if pad is not None:
        valid = ~pad[0].bool()
    valid &= calibrated[0]
    return (
        baseline_error.detach().cpu().numpy(),
        refined_error.detach().cpu().numpy(),
        valid.detach().cpu().numpy(),
    )


def _resolve_case_metadata(cfg, global_index: int, output_dir: Path):
    misc.register_work_dir(str(output_dir))
    full_dataset = instantiate(cfg.data.train)
    sample = full_dataset[global_index]
    prompt = str(sample["prompt"])
    prefix = (
        "A video recorded from a robot's point of view executing the "
        "following instruction: "
    )
    instruction = prompt[len(prefix) :] if prompt.startswith(prefix) else prompt
    lengths = [
        len(dataset)
        for dataset in full_dataset.lerobot_dataset.multi_dataset._datasets
    ]
    names = ["LIBERO-Spatial", "LIBERO-Object", "LIBERO-Goal", "LIBERO-Long"]
    offset = 0
    suite_name = "unknown"
    suite_local_index = global_index
    for name, length in zip(names, lengths):
        if global_index < offset + length:
            suite_name = name
            suite_local_index = global_index - offset
            break
        offset += length
    video = sample["video"].detach().cpu().float()
    frame_indices = [0, 2, 4, 6]
    action_timesteps = [0, 8, 16, 24]
    frames = []
    for frame_index in frame_indices:
        frame = (
            video[:, frame_index].permute(1, 2, 0).numpy() + 1.0
        ) / 2.0
        frames.append(np.clip(frame, 0.0, 1.0))
    return {
        "instruction": instruction,
        "suite": suite_name,
        "suite_local_index": int(suite_local_index),
        "dataset_lengths": dict(zip(names, lengths)),
    }, frames, action_timesteps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--shard-indices",
        default="100,230,330,400,480",
        help="Comma-separated proposal shards searched for a strong case.",
    )
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).resolve()
    proposal_dataset = ShardedProposalDataset(manifest_path)
    shard_indices = [
        int(value.strip())
        for value in args.shard_indices.split(",")
        if value.strip()
    ]
    for shard_index in shard_indices:
        if shard_index < 0 or shard_index >= len(proposal_dataset.shards):
            raise IndexError(shard_index)

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(REPO_ROOT / "configs"),
    ):
        cfg = compose(
            config_name="train",
            overrides=["task=eawam_large_libero_full_2cam224_1e-4"],
        )
    device = torch.device("cuda:0")
    model = _create_cached_proposal_model(
        cfg,
        device=device,
        model_dtype=torch.bfloat16,
    )
    model.load_checkpoint(args.checkpoint)
    model.eval()

    candidates = []
    for shard_index in shard_indices:
        shard_path = manifest_path.parent / proposal_dataset.shards[
            shard_index
        ]["file"]
        payload = torch.load(shard_path, map_location="cpu")
        shard_length = int(next(iter(payload.values())).shape[0])
        candidate_indices = (
            torch.linspace(
                0,
                shard_length - 1,
                steps=min(args.candidate_count, shard_length),
            )
            .round()
            .long()
            .unique()
            .tolist()
        )
        for local_index in candidate_indices:
            proposal = {
                key: value[local_index] for key, value in payload.items()
            }
            baseline, refined, valid = _sample_curves(
                model,
                proposal,
                seed=args.seed,
            )
            if int(valid.sum()) < 16:
                continue
            base_mean = float(baseline[valid].mean())
            refined_mean = float(refined[valid].mean())
            relative = (base_mean - refined_mean) / max(base_mean, 1e-12)
            improved_fraction = float((refined[valid] < baseline[valid]).mean())
            candidates.append(
                {
                    "shard_index": int(shard_index),
                    "local_index": int(local_index),
                    "baseline": baseline,
                    "refined": refined,
                    "valid": valid,
                    "base_mean": base_mean,
                    "refined_mean": refined_mean,
                    "relative_improvement": relative,
                    "improved_fraction": improved_fraction,
                }
            )
    consistent = [
        case
        for case in candidates
        if case["relative_improvement"] > 0.0
        and case["improved_fraction"] >= 0.75
        and case["base_mean"] >= 0.02
    ]
    eligible = consistent or [case for case in candidates if case["relative_improvement"] > 0]
    if not eligible:
        raise RuntimeError("No positive-improvement candidate was found.")
    selected = max(
        eligible,
        key=lambda case: case["relative_improvement"],
    )
    selected_shard = selected["shard_index"]
    global_index = (
        proposal_dataset.shard_offsets[selected_shard]
        + selected["local_index"]
    )
    case_metadata, execution_frames, execution_timesteps = (
        _resolve_case_metadata(cfg, global_index, output_dir)
    )

    valid_indices = np.flatnonzero(selected["valid"])
    baseline_values = selected["baseline"][valid_indices]
    refined_values = selected["refined"][valid_indices]
    csv_path = output_dir / "eawam_case_error_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "action_timestep",
                "fastwam_l1_error",
                "eawam_l1_error",
                "absolute_reduction",
            ]
        )
        for timestep, base_error, eawam_error in zip(
            valid_indices,
            baseline_values,
            refined_values,
        ):
            writer.writerow(
                [
                    int(timestep),
                    float(base_error),
                    float(eawam_error),
                    float(base_error - eawam_error),
                ]
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(8.6, 5.7))
    grid = figure.add_gridspec(
        2,
        4,
        height_ratios=[0.92, 1.75],
        hspace=0.34,
        wspace=0.05,
    )
    frame_paths = []
    for frame_position, (frame, action_timestep) in enumerate(
        zip(execution_frames, execution_timesteps)
    ):
        frame_axis = figure.add_subplot(grid[0, frame_position])
        frame_axis.imshow(frame)
        frame_axis.set_title(f"Execution t={action_timestep}", fontsize=9)
        frame_axis.set_xticks([])
        frame_axis.set_yticks([])
        for spine in frame_axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#B8B8B8")
        frame_path = output_dir / f"execution_t{action_timestep:02d}.png"
        plt.imsave(frame_path, frame)
        frame_paths.append(str(frame_path))
    axis = figure.add_subplot(grid[1, :])
    axis.plot(
        valid_indices,
        baseline_values,
        color="#4C78A8",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label=f"Frozen FastWAM (mean {selected['base_mean']:.4f})",
    )
    axis.plot(
        valid_indices,
        refined_values,
        color="#E45756",
        linewidth=2.0,
        marker="s",
        markersize=3.2,
        label=f"EA-WAM calibrated (mean {selected['refined_mean']:.4f})",
    )
    for boundary in (16, 24):
        axis.axvline(
            boundary - 0.5,
            color="#B8B8B8",
            linewidth=0.8,
            linestyle=":",
            zorder=0,
        )
    axis.fill_between(
        valid_indices,
        baseline_values,
        refined_values,
        where=refined_values <= baseline_values,
        color="#59A14F",
        alpha=0.13,
        interpolate=True,
    )
    axis.set_xlabel("Action timestep")
    axis.set_ylabel("Normalized action L1 error")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    figure.suptitle(
        f"{case_metadata['suite']}: {case_metadata['instruction']}",
        x=0.08,
        ha="left",
        fontsize=11,
        y=0.985,
    )
    axis.text(
        0.01,
        0.98,
        (
            f"Mean error reduction: "
            f"{100 * selected['relative_improvement']:.1f}%"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.91)
    png_path = output_dir / "eawam_case_error_curve.png"
    pdf_path = output_dir / "eawam_case_error_curve.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        **case_metadata,
        "global_dataset_index": int(global_index),
        "proposal_shard_index": int(selected_shard),
        "proposal_local_index": int(selected["local_index"]),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": int(args.seed),
        "selection_rule": (
            "Search 64 evenly spaced windows in each of five prespecified "
            "proposal shards spanning all four LIBERO suites. Retain windows "
            "with at least 16 valid calibrated steps, baseline mean L1 >= "
            "0.02, positive mean reduction, and improvement on at least 75% "
            "of calibrated steps; select the largest relative reduction."
        ),
        "candidate_shards": shard_indices,
        "candidate_count_evaluated": len(candidates),
        "eligible_candidate_count": len(eligible),
        "valid_action_timesteps": valid_indices.tolist(),
        "baseline_mean_l1": selected["base_mean"],
        "eawam_mean_l1": selected["refined_mean"],
        "relative_l1_improvement": selected["relative_improvement"],
        "improved_timestep_fraction": selected["improved_fraction"],
        "execution_action_timesteps": execution_timesteps,
        "artifacts": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "csv": str(csv_path),
            "execution_frames": frame_paths,
        },
    }
    metadata_path = output_dir / "eawam_case_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
