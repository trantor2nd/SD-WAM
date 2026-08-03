"""Evaluate exact LIBERO initial states for continuation diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from libero.libero import benchmark
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_DIR = REPO_ROOT / "experiments" / "libero"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LIBERO_DIR) not in sys.path:
    sys.path.insert(0, str(LIBERO_DIR))

from experiments.libero import eval_libero_single as upstream  # noqa: E402
from extensions.eawam.eval_libero_suite import (  # noqa: E402
    _install_cached_text_context,
    _resolve_dataset_stats_path,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import (  # noqa: E402
    FastWAMProcessor,
)
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)


def parse_episode(raw: str) -> tuple[str, int, int]:
    fields = raw.split(":")
    if len(fields) != 3:
        raise ValueError(f"Expected SUITE:TASK:TRIAL, got {raw}")
    return fields[0], int(fields[1]), int(fields[2])


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    args, overrides = parser.parse_known_args()
    episodes = [parse_episode(item) for item in args.episodes.split(",")]
    if not episodes:
        raise ValueError("No episodes were requested")

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(REPO_ROOT / "configs"),
    ):
        cfg = compose(config_name="sim_libero", overrides=overrides)

    output_dir = Path(str(cfg.EVALUATION.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    OmegaConf.save(cfg, output_dir / "suite_config.yaml", resolve=False)

    model_device = upstream._resolve_eval_device(cfg)
    model_dtype = upstream._mixed_precision_to_model_dtype(
        cfg.get("mixed_precision", "bf16")
    )
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    upstream._load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    _install_cached_text_context(model, cfg, model_device)

    stats = load_dataset_stats_from_json(
        str(_resolve_dataset_stats_path(cfg))
    )
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    input_h, input_w = [int(value) for value in cfg.data.train.video_size]
    benchmark_dict = benchmark.get_benchmark_dict()
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for suite, task_id, trial_id in episodes:
        grouped[(suite, task_id)].append(trial_id)

    payload = {
        "checkpoint": str(cfg.ckpt),
        "requested_episodes": [
            {"suite": suite, "task_id": task_id, "trial_id": trial_id}
            for suite, task_id, trial_id in episodes
        ],
        "records": [],
        "successes": 0,
        "total_episodes": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    result_path = output_dir / "selected_trials_results.json"
    for (suite, task_id), trial_ids in grouped.items():
        OmegaConf.update(
            cfg,
            "EVALUATION.task_suite_name",
            suite,
            merge=False,
        )
        OmegaConf.update(cfg, "EVALUATION.task_id", task_id, merge=False)
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        initial_states = list(task_suite.get_task_init_states(task_id))
        env, task_description = upstream.get_libero_env(
            task,
            upstream.LIBERO_ENV_RESOLUTION,
            cfg.get("seed"),
        )
        video_dir = output_dir / suite / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        try:
            for trial_id in sorted(set(trial_ids)):
                if trial_id < 0 or trial_id >= len(initial_states):
                    raise IndexError(
                        f"Trial {trial_id} unavailable for {suite}:{task_id}"
                    )
                if hasattr(model, "reset_feedback"):
                    model.reset_feedback()
                success, frames, _, _ = upstream.run_single_episode(
                    env=env,
                    initial_state=initial_states[trial_id],
                    task_description=task_description,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    episode_idx=trial_id,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=model_device,
                )
                upstream.save_rollout_video(
                    video_dir,
                    frames,
                    f"task{task_id}_trial{trial_id}",
                    success=success,
                    task_description=task_description,
                )
                payload["records"].append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "success": bool(success),
                    }
                )
                payload["successes"] += int(success)
                payload["total_episodes"] += 1
                write_json(result_path, payload)
        finally:
            env.close()
    payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(result_path, payload)
    logging.info(
        "Selected trial diagnostic complete %d/%d",
        payload["successes"],
        payload["total_episodes"],
    )


if __name__ == "__main__":
    main()
