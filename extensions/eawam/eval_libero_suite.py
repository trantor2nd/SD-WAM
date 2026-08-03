"""Evaluate several LIBERO tasks after loading a model only once.

This is an isolated fast-turnaround evaluator. It reuses FastWAM's original
episode and task functions, while replacing the 11 GB online text encoder with
the official precomputed LIBERO text embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from libero.libero import benchmark
from omegaconf import OmegaConf


repo_root = Path(__file__).resolve().parents[2]
libero_dir = repo_root / "experiments" / "libero"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(libero_dir) not in sys.path:
    sys.path.insert(0, str(libero_dir))

from experiments.libero import eval_libero_single as upstream  # noqa: E402
from fastwam.datasets.lerobot.processors.fastwam_processor import (  # noqa: E402
    FastWAMProcessor,
)
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)


def _parse_suite_tasks(raw: str) -> list[tuple[str, int]]:
    tasks = []
    for item in raw.split(","):
        suite, separator, task_id = item.strip().partition(":")
        if not separator or not suite or not task_id:
            raise ValueError(
                "Each --suite-tasks item must be SUITE:TASK_ID, got "
                f"{item!r}"
            )
        tasks.append((suite, int(task_id)))
    if not tasks:
        raise ValueError("--suite-tasks must not be empty")
    return tasks


def _resolve_cache_dir(cfg) -> Path:
    configured = cfg.data.train.get("text_embedding_cache_dir")
    if configured is None:
        raise ValueError("data.train.text_embedding_cache_dir is not configured")
    cache_dir = Path(str(configured)).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
    return cache_dir.resolve()


def _install_cached_text_context(model, cfg, model_device: str) -> None:
    """Inject cached context into the existing infer_action method."""

    cache_dir = _resolve_cache_dir(cfg)
    original_infer_action = model.infer_action
    loaded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def cached_infer_action(*args, **kwargs):
        prompt = kwargs.get("prompt")
        context = kwargs.get("context")
        if context is None:
            if not prompt:
                raise ValueError("A prompt is required to look up cached context")
            if prompt not in loaded:
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                matches = list(
                    cache_dir.glob(f"{hashed}.t5_len*.wan22ti2v5b.pt")
                )
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Expected one cached context for prompt, found "
                        f"{len(matches)} under {cache_dir}"
                    )
                payload = torch.load(matches[0], map_location="cpu")
                prompt_context = payload["context"].clone()
                original_mask = payload["mask"].bool()
                prompt_context[~original_mask] = 0.0
                # Match RobotVideoDataset: padded context is zeroed, while the
                # attention mask is then set to all-valid for Wan2.2.
                context_mask = torch.ones_like(original_mask)
                loaded[prompt] = (
                    prompt_context.unsqueeze(0).to(
                        device=model_device,
                        dtype=model.torch_dtype,
                    ),
                    context_mask.unsqueeze(0).to(device=model_device),
                )
            kwargs["context"], kwargs["context_mask"] = loaded[prompt]
            kwargs["prompt"] = None
        return original_infer_action(*args, **kwargs)

    model.infer_action = cached_infer_action
    logging.info("Using cached LIBERO text contexts from %s", cache_dir)


def _install_episode_reset() -> None:
    original = upstream.run_single_episode
    if getattr(original, "_eawam_resets_feedback", False):
        return

    def resetting_episode(*args, **kwargs):
        model = kwargs.get("model")
        if model is None and len(args) >= 4:
            model = args[3]
        if model is not None and hasattr(model, "reset_feedback"):
            model.reset_feedback()
        return original(*args, **kwargs)

    resetting_episode._eawam_resets_feedback = True
    upstream.run_single_episode = resetting_episode


def _resolve_dataset_stats_path(cfg) -> Path:
    try:
        return upstream._resolve_dataset_stats_path(cfg)
    except FileNotFoundError as original_error:
        candidates = []
        configured = cfg.data.train.get("pretrained_norm_stats")
        if configured:
            candidates.append(Path(str(configured)))
        base_checkpoint = cfg.model.get("base_checkpoint")
        if base_checkpoint:
            base_path = Path(str(base_checkpoint))
            candidates.append(
                base_path.with_name(f"{base_path.stem}_dataset_stats.json")
            )
        checkpoint_path = Path(str(cfg.ckpt))
        candidates.append(
            checkpoint_path.with_name(
                f"{checkpoint_path.stem}_dataset_stats.json"
            )
        )
        for candidate in candidates:
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            candidate = candidate.resolve()
            if candidate.exists():
                return candidate
        raise original_error


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, cls=upstream.NumpyEncoder),
        encoding="utf-8",
    )
    temporary.replace(path)


def _accumulate_task_record(aggregate: dict[str, Any], task_record: dict) -> None:
    aggregate["tasks"].append(task_record)
    aggregate["successes"] += int(task_record["successes"])
    aggregate["total_episodes"] += int(task_record["total_episodes"])
    suite_name = str(task_record["task_suite"])
    suite_record = aggregate["per_suite"].setdefault(
        suite_name,
        {"successes": 0, "total_episodes": 0, "success_rate": None},
    )
    suite_record["successes"] += int(task_record["successes"])
    suite_record["total_episodes"] += int(task_record["total_episodes"])
    suite_record["success_rate"] = (
        suite_record["successes"] / suite_record["total_episodes"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-tasks", required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-task JSON files with the same trial count.",
    )
    args, overrides = parser.parse_known_args()
    suite_tasks = _parse_suite_tasks(args.suite_tasks)
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(repo_root / "configs"),
    ):
        cfg = compose(config_name="sim_libero", overrides=overrides)

    output_dir = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    start_time = time.time()
    OmegaConf.save(cfg, output_dir / "suite_config.yaml", resolve=False)

    model_device = upstream._resolve_eval_device(cfg)
    model_dtype = upstream._mixed_precision_to_model_dtype(
        cfg.get("mixed_precision", "bf16")
    )
    logging.info("Loading model once for %d task(s)", len(suite_tasks))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    upstream._load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    _install_cached_text_context(model, cfg, model_device)
    _install_episode_reset()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    input_h, input_w = [int(value) for value in cfg.data.train.video_size]
    benchmark_dict = benchmark.get_benchmark_dict()

    aggregate = {
        "checkpoint": str(cfg.ckpt),
        "suite_tasks": [f"{suite}:{task_id}" for suite, task_id in suite_tasks],
        "trials_per_task": args.trials,
        "successes": 0,
        "total_episodes": 0,
        "success_rate": None,
        "duration_seconds": None,
        "per_suite": {},
        "tasks": [],
        "errors": [],
    }

    for sequence_index, (suite_name, task_id) in enumerate(suite_tasks, start=1):
        task_start = time.time()
        OmegaConf.update(
            cfg,
            "EVALUATION.task_suite_name",
            suite_name,
            merge=False,
        )
        OmegaConf.update(cfg, "EVALUATION.task_id", task_id, merge=False)
        OmegaConf.update(
            cfg,
            "EVALUATION.num_trials",
            args.trials,
            merge=False,
        )
        suite_output = output_dir / suite_name
        video_dir = suite_output / "videos"
        predicted_video_dir = suite_output / "predicted_videos"
        task_result_path = (
            suite_output / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        if bool(cfg.EVALUATION.get("visualize_future_video", False)):
            predicted_video_dir.mkdir(parents=True, exist_ok=True)

        if args.resume and task_result_path.exists():
            existing = json.loads(task_result_path.read_text(encoding="utf-8"))
            if (
                str(existing.get("task_suite")) == suite_name
                and int(existing.get("task_id", -1)) == task_id
                and int(existing.get("total_episodes", -1)) == args.trials
            ):
                logging.info(
                    "Task %d/%d: %s:%d reused from %s",
                    sequence_index,
                    len(suite_tasks),
                    suite_name,
                    task_id,
                    task_result_path,
                )
                _accumulate_task_record(aggregate, existing)
                aggregate["duration_seconds"] = time.time() - start_time
                aggregate["success_rate"] = (
                    aggregate["successes"] / aggregate["total_episodes"]
                )
                _write_json(output_dir / "aggregate_results.json", aggregate)
                continue

        logging.info(
            "Task %d/%d: %s:%d (%d trial(s))",
            sequence_index,
            len(suite_tasks),
            suite_name,
            task_id,
            args.trials,
        )
        try:
            task_suite = benchmark_dict[suite_name]()
            task = task_suite.get_task(task_id)
            initial_states = list(task_suite.get_task_init_states(task_id))
            while len(initial_states) < args.trials:
                initial_states.extend(
                    initial_states[: args.trials - len(initial_states)]
                )
            task_results = upstream.run_single_task(
                task=task,
                initial_states=initial_states,
                model=model,
                processor=processor,
                cfg=cfg,
                video_dir=video_dir,
                predicted_video_dir=predicted_video_dir,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            task_record = {
                "task_suite": suite_name,
                "task_id": task_id,
                "total_episodes": args.trials,
                "duration_seconds": time.time() - task_start,
                **task_results,
            }
            task_record["success_rate"] = (
                task_record["successes"] / args.trials
            )
            _accumulate_task_record(aggregate, task_record)
            _write_json(task_result_path, task_record)
        except Exception as error:  # keep the remaining paired tasks running
            logging.exception("Task failed: %s:%d", suite_name, task_id)
            aggregate["errors"].append(
                {
                    "task_suite": suite_name,
                    "task_id": task_id,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
        aggregate["duration_seconds"] = time.time() - start_time
        if aggregate["total_episodes"]:
            aggregate["success_rate"] = (
                aggregate["successes"] / aggregate["total_episodes"]
            )
        _write_json(output_dir / "aggregate_results.json", aggregate)

    logging.info(
        "Evaluation complete: %d/%d (%.2f%%), errors=%d, duration=%.1fmin",
        aggregate["successes"],
        aggregate["total_episodes"],
        100.0 * (aggregate["success_rate"] or 0.0),
        len(aggregate["errors"]),
        aggregate["duration_seconds"] / 60.0,
    )


if __name__ == "__main__":
    main()
