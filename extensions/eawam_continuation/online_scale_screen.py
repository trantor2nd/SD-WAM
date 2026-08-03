from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(
    os.environ.get("FASTWAM_ROOT", str(Path.cwd()))
).resolve()
TRAIN_OUTPUT = Path(
    os.environ.get(
        "EAWAM_CONTINUATION_OUTPUT",
        ROOT / "runs/eawam_continuation/libero_full_v3",
    )
)
SWEEP = TRAIN_OUTPUT / "offline_scale_sweep.json"
OUTPUT = TRAIN_OUTPUT / "online_scale_screen"
RESULT = TRAIN_OUTPUT / "online_scale_selection.json"
EARLY_CANDIDATE_OUTPUT = (
    Path(
        os.environ.get(
            "EAWAM_REFERENCE_CANDIDATE_OUTPUT",
            TRAIN_OUTPUT / "reference_candidate_hard100",
        )
    )
)
CONSERVATIVE_CANDIDATE_OUTPUT = (
    Path(
        os.environ.get(
            "EAWAM_CONSERVATIVE_CANDIDATE_OUTPUT",
            TRAIN_OUTPUT / "conservative_candidate_hard100",
        )
    )
)
PAIRED_BASELINE_OUTPUT = (
    Path(
        os.environ.get(
            "EAWAM_PAIRED_BASELINE_OUTPUT",
            TRAIN_OUTPUT / "paired_baseline_hard100",
        )
    )
)
IS_PRIMARY_RUN = bool(
    int(os.environ.get("EAWAM_INCLUDE_REFERENCE_CANDIDATES", "0"))
)
PYTHON = Path(os.environ.get("EAWAM_PYTHON", sys.executable))
DATASET_STATS = Path(
    os.environ.get(
        "EAWAM_DATASET_STATS",
        ROOT
        / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json",
    )
)
TASKS = os.environ.get(
    "EAWAM_SCREEN_TASKS",
    (
        "libero_10:0,libero_10:3,libero_10:4,libero_10:6,"
        "libero_10:9,libero_goal:2,libero_goal:3,libero_goal:9,"
        "libero_spatial:4,libero_object:5"
    ),
).split(",")
TRIALS = int(os.environ.get("EAWAM_SCREEN_TRIALS", "10"))
ALLOWED_GPUS = [
    int(value)
    for value in os.environ.get("EAWAM_SCREEN_GPUS", "0").split(",")
]
MIN_FREE_MIB = int(os.environ.get("EAWAM_MIN_FREE_MIB", "18000"))


def free_memory() -> dict[int, int]:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = {}
    for line in text.splitlines():
        index, free = line.split(",")
        result[int(index.strip())] = int(free.strip())
    return result


def complete_aggregate(output_dir: Path) -> dict | None:
    aggregate_path = output_dir / "aggregate_results.json"
    if not aggregate_path.exists():
        return None
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("errors"):
        return None
    if int(aggregate.get("total_episodes", 0)) != 100:
        return None
    if len(aggregate.get("tasks", [])) != 10:
        return None
    return aggregate


def external_evaluation_active(output_dir: Path) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", str(output_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def command_for(
    checkpoint: Path,
    scale: float,
    gripper_scale: float,
    gpu: int,
    output_dir: Path,
) -> list[str]:
    return [
        str(PYTHON),
        "-u",
        "extensions/eawam/eval_libero_suite.py",
        "--suite-tasks",
        ",".join(TASKS),
        "--trials",
        str(TRIALS),
        "--resume",
        "task=eawam_xl_libero_full_2cam224_1e-4",
        "model._target_=extensions.eawam_continuation.runtime.create_eawam_continuation",
        f"ckpt={checkpoint}",
        f"EVALUATION.output_dir={output_dir}",
        f"EVALUATION.dataset_stats_path={DATASET_STATS}",
        "EVALUATION.replan_steps=8",
        "EVALUATION.visualize_future_video=false",
        "model.load_text_encoder=false",
        "model.module.hidden_dim=256",
        "model.module.lde_layers=2",
        f"model.module.residual_scale={scale}",
        f"+model.module.gripper_residual_scale={gripper_scale}",
        "model.module.action_noise_std=0.005",
        "+model.module.velocity_scale=0.1",
        "+model.module.gripper_velocity_scale=1.0",
        "+model.module.max_gripper_correction=2.0",
        "+model.module.temporal_shift_augmentation=false",
        f"gpu_id={gpu}",
    ]


sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
nonzero = [
    item
    for item in sweep["metrics"]
    if float(item["scale"]) > 0 or float(item["gripper_scale"]) > 0
]
nonzero.sort(key=lambda item: float(item["refined_action_l1"]))
best_per_checkpoint = {}
for item in nonzero:
    best_per_checkpoint.setdefault(item["checkpoint"], item)
candidates = list(best_per_checkpoint.values())[:3]
early_candidate = next(
    (
        item
        for item in sweep["metrics"]
        if IS_PRIMARY_RUN
        if int(item.get("epoch") or -1) == 1
        and abs(float(item["scale"]) - 0.15) < 1e-12
        and abs(float(item["gripper_scale"])) < 1e-12
    ),
    None,
)
if early_candidate is not None and not any(
    item["checkpoint"] == early_candidate["checkpoint"]
    and abs(float(item["scale"]) - 0.15) < 1e-12
    and abs(float(item["gripper_scale"])) < 1e-12
    for item in candidates
):
    candidates.append(early_candidate)
conservative_candidate = next(
    (
        item
        for item in sweep["metrics"]
        if IS_PRIMARY_RUN
        if int(item.get("epoch") or -1) == 1
        and abs(float(item["scale"]) - 0.05) < 1e-12
        and abs(float(item["gripper_scale"])) < 1e-12
    ),
    None,
)
conservative_aggregate = (
    CONSERVATIVE_CANDIDATE_OUTPUT / "aggregate_results.json"
)
if (
    conservative_candidate is not None
    and conservative_aggregate.exists()
    and not any(
        item["checkpoint"] == conservative_candidate["checkpoint"]
        and abs(float(item["scale"]) - 0.05) < 1e-12
        and abs(float(item["gripper_scale"])) < 1e-12
        for item in candidates
    )
):
    conservative_payload = json.loads(
        conservative_aggregate.read_text(encoding="utf-8")
    )
    if (
        not conservative_payload.get("errors")
        and int(conservative_payload.get("total_episodes", 0)) == 100
    ):
        candidates.append(conservative_candidate)

pending = []
running = {}
external_running = []
records = []
OUTPUT.mkdir(parents=True, exist_ok=True)
for candidate in candidates:
    reused_output = None
    if (
        IS_PRIMARY_RUN
        and int(candidate.get("epoch") or -1) == 1
        and abs(float(candidate["gripper_scale"])) < 1e-12
    ):
        if abs(float(candidate["scale"]) - 0.15) < 1e-12:
            reused_output = EARLY_CANDIDATE_OUTPUT
        elif abs(float(candidate["scale"]) - 0.05) < 1e-12:
            reused_output = CONSERVATIVE_CANDIDATE_OUTPUT
    aggregate = (
        None if reused_output is None else complete_aggregate(reused_output)
    )
    if aggregate is not None:
        records.append(
            {
                "scale": float(candidate["scale"]),
                "gripper_scale": 0.0,
                "checkpoint": candidate["checkpoint"],
                "epoch": 1,
                "gpu": 0,
                "successes": int(aggregate["successes"]),
                "total_episodes": 100,
                "success_rate": float(aggregate["success_rate"]),
                "offline_refined_l1": float(
                    candidate["refined_action_l1"]
                ),
                "reused_evaluation": str(reused_output),
            }
        )
        continue
    if (
        reused_output is not None
        and external_evaluation_active(reused_output)
    ):
        external_running.append(
            {
                "candidate": candidate,
                "output_dir": reused_output,
            }
        )
        continue
    pending.append(candidate)

while pending or running or external_running:
    for external in list(external_running):
        candidate = external["candidate"]
        output_dir = external["output_dir"]
        aggregate = complete_aggregate(output_dir)
        if aggregate is not None:
            records.append(
                {
                    "scale": float(candidate["scale"]),
                    "gripper_scale": 0.0,
                    "checkpoint": candidate["checkpoint"],
                    "epoch": 1,
                    "gpu": 0,
                    "successes": int(aggregate["successes"]),
                    "total_episodes": 100,
                    "success_rate": float(aggregate["success_rate"]),
                    "offline_refined_l1": float(
                        candidate["refined_action_l1"]
                    ),
                    "reused_evaluation": str(output_dir),
                }
            )
            external_running.remove(external)
            continue
        if not external_evaluation_active(output_dir):
            raise RuntimeError(
                f"External online screen ended incompletely: {output_dir}"
            )

    for key, record in list(running.items()):
        return_code = record["process"].poll()
        if return_code is None:
            continue
        record["log"].close()
        del running[key]
        aggregate_path = record["output_dir"] / "aggregate_results.json"
        if return_code != 0 or not aggregate_path.exists():
            raise RuntimeError(
                f"Online scale screen failed for scale={record['scale']} "
                f"on gpu={record['gpu']} with exit={return_code}"
            )
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate.get("errors") or int(aggregate["total_episodes"]) != 100:
            raise RuntimeError(
                f"Incomplete online scale screen for scale={record['scale']}"
            )
        records.append(
            {
                "scale": record["scale"],
                "gripper_scale": record["gripper_scale"],
                "checkpoint": str(record["checkpoint"]),
                "epoch": record["epoch"],
                "gpu": record["gpu"],
                "successes": int(aggregate["successes"]),
                "total_episodes": int(aggregate["total_episodes"]),
                "success_rate": float(aggregate["success_rate"]),
                "offline_refined_l1": record["offline_refined_l1"],
            }
        )

    memory = free_memory()
    busy = {record["gpu"] for record in running.values()}
    if external_running:
        busy.add(0)
    available = [
        gpu
        for gpu in sorted(
            ALLOWED_GPUS,
            key=lambda item: memory.get(item, 0),
            reverse=True,
        )
        if memory.get(gpu, 0) >= MIN_FREE_MIB and gpu not in busy
    ]
    while pending and available:
        candidate = pending.pop(0)
        gpu = available.pop(0)
        scale = float(candidate["scale"])
        gripper_scale = float(candidate["gripper_scale"])
        checkpoint = Path(candidate["checkpoint"])
        epoch = candidate.get("epoch")
        scale_key = str(scale).replace(".", "p")
        checkpoint_key = (
            f"epoch_{int(epoch):02d}" if epoch is not None else checkpoint.stem
        )
        gripper_key = str(gripper_scale).replace(".", "p")
        key = (
            f"{checkpoint_key}_arm_{scale_key}_gripper_{gripper_key}"
        )
        output_dir = OUTPUT / key
        output_dir.mkdir(parents=True, exist_ok=True)
        log = (output_dir / "eval.log").open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(ROOT),
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "EAWAM_CONTINUATION_OUTPUT": str(TRAIN_OUTPUT),
                "DIFFSYNTH_MODEL_BASE_PATH": str(ROOT / "checkpoints"),
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        process = subprocess.Popen(
            command_for(
                checkpoint,
                scale,
                gripper_scale,
                gpu,
                output_dir,
            ),
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        running[key] = {
            "process": process,
            "log": log,
            "gpu": gpu,
            "scale": scale,
            "gripper_scale": gripper_scale,
            "checkpoint": checkpoint,
            "epoch": epoch,
            "offline_refined_l1": float(candidate["refined_action_l1"]),
            "output_dir": output_dir,
        }
    time.sleep(30)

records.sort(
    key=lambda item: (
        -item["success_rate"],
        item["offline_refined_l1"],
    )
)
baseline = complete_aggregate(PAIRED_BASELINE_OUTPUT)
while (
    baseline is None
    and external_evaluation_active(PAIRED_BASELINE_OUTPUT)
):
    time.sleep(30)
    baseline = complete_aggregate(PAIRED_BASELINE_OUTPUT)
if baseline is None:
    raise RuntimeError("Paired hard-100 baseline is incomplete")
baseline_successes = int(baseline["successes"])
qualified = [
    record
    for record in records
    if int(record["successes"]) > baseline_successes
]
payload = {
    "tasks": TASKS,
    "trials_per_task": TRIALS,
    "paired_baseline": {
        "output": str(PAIRED_BASELINE_OUTPUT),
        "successes": baseline_successes,
        "total_episodes": 100,
    },
    "candidates": records,
    "qualification_rule": "candidate_successes > paired_baseline_successes",
    "qualified_candidates": qualified,
    "selected": None if not qualified else qualified[0],
}
temporary = RESULT.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
os.replace(temporary, RESULT)
print(json.dumps(payload, indent=2))
if not qualified:
    raise RuntimeError(
        "No continuation candidate strictly beat the paired hard-100 baseline"
    )
