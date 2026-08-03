from __future__ import annotations

import json
import logging
import math
import os
import time
from bisect import bisect_right
from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from fastwam.utils import misc
from fastwam.utils.pytorch_utils import set_global_seed

from .model import ErrorAwareFastWAM


def _dtype_from_name(name: str) -> torch.dtype:
    key = str(name).lower()
    if key == "bf16":
        return torch.bfloat16
    if key == "fp16":
        return torch.float16
    if key == "no":
        return torch.float32
    raise ValueError(f"Unsupported mixed precision: {name}")


def _even_subset(dataset, maximum: int | None):
    if maximum is None or int(maximum) >= len(dataset):
        return dataset
    maximum = int(maximum)
    if maximum <= 0:
        raise ValueError("max_train_samples must be positive or null")
    indices = torch.linspace(0, len(dataset) - 1, steps=maximum).round().long()
    return Subset(dataset, indices.tolist())


def _ensure_writable_hf_cache() -> Path:
    configured = os.environ.get("HF_DATASETS_CACHE")
    if configured:
        configured_path = Path(configured).expanduser()
        parent = configured_path if configured_path.exists() else configured_path.parent
        if parent.exists() and os.access(parent, os.W_OK):
            return configured_path
    fallback = Path.home() / ".cache" / "huggingface" / "datasets"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(fallback)
    return fallback


class ProposalDataset(torch.utils.data.Dataset):
    def __init__(self, payload: dict[str, torch.Tensor]):
        lengths = {value.shape[0] for value in payload.values()}
        if len(lengths) != 1:
            raise ValueError(f"Proposal tensors have inconsistent lengths: {lengths}")
        self.payload = payload
        self.length = lengths.pop()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.payload.items()}


class ShardedProposalDataset(torch.utils.data.Dataset):
    """Disk-backed proposal cache that keeps at most one shard in RAM."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.shards = list(manifest["shards"])
        self.shard_lengths = [int(shard["examples"]) for shard in self.shards]
        self.shard_offsets = [0]
        for length in self.shard_lengths:
            self.shard_offsets.append(self.shard_offsets[-1] + length)
        self.length = self.shard_offsets[-1]
        expected = int(manifest.get("examples_written", self.length))
        if self.length != expected:
            raise ValueError(
                f"Proposal manifest length mismatch: shards={self.length}, "
                f"manifest={expected}"
            )
        self._loaded_shard_index: int | None = None
        self._loaded_payload: dict[str, torch.Tensor] | None = None

    def __len__(self):
        return self.length

    def _load_shard(self, shard_index: int) -> dict[str, torch.Tensor]:
        if self._loaded_shard_index != shard_index:
            shard_path = self.manifest_path.parent / self.shards[shard_index]["file"]
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing proposal shard: {shard_path}")
            self._loaded_payload = torch.load(shard_path, map_location="cpu")
            self._loaded_shard_index = shard_index
        assert self._loaded_payload is not None
        return self._loaded_payload

    def __getitem__(self, index):
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        shard_index = bisect_right(self.shard_offsets, index) - 1
        local_index = index - self.shard_offsets[shard_index]
        payload = self._load_shard(shard_index)
        return {key: value[local_index] for key, value in payload.items()}


class ShardBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Shuffle shards and samples while preserving sequential shard reads."""

    def __init__(
        self,
        dataset: ShardedProposalDataset,
        *,
        batch_size: int,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return sum(
            math.ceil(length / self.batch_size)
            for length in self.dataset.shard_lengths
        )

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_order = torch.randperm(
            len(self.dataset.shards),
            generator=generator,
        ).tolist()
        for shard_index in shard_order:
            length = self.dataset.shard_lengths[shard_index]
            offset = self.dataset.shard_offsets[shard_index]
            order = torch.randperm(length, generator=generator).tolist()
            for start in range(0, length, self.batch_size):
                yield [
                    offset + local_index
                    for local_index in order[start : start + self.batch_size]
                ]


class _CachedProposalBackbone(torch.nn.Module):
    """Shape/device specification used when frozen proposals already exist."""

    def __init__(
        self,
        *,
        action_dim: int,
        latent_dim: int,
        proprio_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.action_expert = SimpleNamespace(action_dim=int(action_dim))
        self.video_expert = SimpleNamespace(in_dim=int(latent_dim))
        self.proprio_dim = int(proprio_dim)
        self.device = torch.device(device)
        self.torch_dtype = dtype

    def load_checkpoint(self, *_args, **_kwargs):
        return {}

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.device = torch.device(device)
        return super().to(device=device, *args, **kwargs)


def _create_cached_proposal_model(
    cfg: DictConfig,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
):
    module_cfg = OmegaConf.to_container(cfg.model.module, resolve=True)
    latent_dim = int(module_cfg.get("latent_dim", 48))
    action_dim = int(cfg.data.train.processor.action_output_dim)
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    backbone = _CachedProposalBackbone(
        action_dim=action_dim,
        latent_dim=latent_dim,
        proprio_dim=proprio_dim,
        device=device,
        dtype=model_dtype,
    )
    return ErrorAwareFastWAM(
        backbone,
        base_checkpoint=str(cfg.model.base_checkpoint),
        **dict(module_cfg),
    )


def _move_proposals(
    proposals: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in proposals.items()
    }


def _build_or_load_proposal_cache(
    *,
    model,
    loader,
    cache_path: Path,
    logger,
    shard_size: int = 0,
) -> ProposalDataset | ShardedProposalDataset:
    if cache_path.name == "manifest.json":
        logger.info(
            "Loaded explicit sharded frozen FastWAM proposal manifest=%s",
            cache_path,
        )
        return ShardedProposalDataset(cache_path)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        logger.info("Loaded frozen FastWAM proposal cache=%s", cache_path)
        return ProposalDataset(payload)

    if shard_size > 0:
        return _build_or_load_sharded_proposal_cache(
            model=model,
            loader=loader,
            cache_path=cache_path,
            logger=logger,
            shard_size=shard_size,
        )

    chunks: dict[str, list[torch.Tensor]] = {}
    started = time.perf_counter()
    for batch_index, sample in enumerate(loader):
        proposals = model.make_training_proposals(sample)
        for key, value in proposals.items():
            if value is None:
                continue
            chunks.setdefault(key, []).append(value.detach().cpu())
        done = min((batch_index + 1) * loader.batch_size, len(loader.dataset))
        if batch_index == 0 or (batch_index + 1) % 10 == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            rate = done / elapsed
            eta = (len(loader.dataset) - done) / max(rate, 1e-9)
            logger.info(
                "proposal_cache=%d/%d rate=%.2f samples/s eta=%.1fmin "
                "peak_mem=%.1fGiB",
                done,
                len(loader.dataset),
                rate,
                eta / 60.0,
                torch.cuda.max_memory_allocated() / (1024**3),
            )

    payload = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    logger.info(
        "Saved frozen FastWAM proposal cache=%s examples=%d size=%.1fMiB",
        cache_path,
        next(iter(payload.values())).shape[0],
        cache_path.stat().st_size / (1024**2),
    )
    return ProposalDataset(payload)


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _build_or_load_sharded_proposal_cache(
    *,
    model,
    loader,
    cache_path: Path,
    logger,
    shard_size: int,
) -> ShardedProposalDataset:
    cache_dir = cache_path.parent / f"{cache_path.stem}_shards"
    manifest_path = cache_dir / "manifest.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_examples = len(loader.dataset)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["total_examples"]) != total_examples:
            raise ValueError(
                "Existing proposal manifest was built for "
                f"{manifest['total_examples']} examples, but this run has "
                f"{total_examples}."
            )
        if int(manifest["shard_size"]) != int(shard_size):
            raise ValueError(
                "Existing proposal manifest shard size does not match: "
                f"{manifest['shard_size']} != {shard_size}"
            )
    else:
        manifest = {
            "format": "eawam-proposals-sharded-v1",
            "total_examples": total_examples,
            "examples_written": 0,
            "shard_size": int(shard_size),
            "complete": False,
            "shards": [],
        }
        _write_manifest(manifest_path, manifest)

    completed = int(manifest["examples_written"])
    if completed > total_examples:
        raise ValueError(
            f"Proposal manifest has too many examples: {completed} > {total_examples}"
        )
    for shard in manifest["shards"]:
        shard_path = cache_dir / shard["file"]
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing completed proposal shard: {shard_path}")
    if bool(manifest.get("complete")):
        if completed != total_examples:
            raise ValueError("Complete proposal manifest has an incomplete length.")
        logger.info(
            "Loaded complete sharded frozen FastWAM proposal cache=%s examples=%d",
            manifest_path,
            completed,
        )
        return ShardedProposalDataset(manifest_path)

    if completed:
        remaining_dataset = Subset(
            loader.dataset,
            range(completed, total_examples),
        )
        loader = DataLoader(
            remaining_dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            worker_init_fn=loader.worker_init_fn,
            generator=loader.generator,
            persistent_workers=loader.num_workers > 0,
            collate_fn=loader.collate_fn,
        )
        logger.info(
            "Resuming proposal cache at %d/%d from %s",
            completed,
            total_examples,
            manifest_path,
        )

    chunks: dict[str, list[torch.Tensor]] = {}
    buffered = 0
    started = time.perf_counter()

    def flush_shard() -> None:
        nonlocal chunks, buffered, completed, manifest
        if buffered == 0:
            return
        payload = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
        shard_index = len(manifest["shards"])
        shard_name = f"part-{shard_index:06d}.pt"
        shard_path = cache_dir / shard_name
        temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, shard_path)
        shard_examples = int(next(iter(payload.values())).shape[0])
        completed += shard_examples
        manifest["examples_written"] = completed
        manifest["shards"].append(
            {
                "file": shard_name,
                "examples": shard_examples,
                "size_bytes": shard_path.stat().st_size,
            }
        )
        manifest["complete"] = completed == total_examples
        _write_manifest(manifest_path, manifest)
        logger.info(
            "saved_proposal_shard=%s examples=%d progress=%d/%d size=%.1fMiB",
            shard_path,
            shard_examples,
            completed,
            total_examples,
            shard_path.stat().st_size / (1024**2),
        )
        chunks = {}
        buffered = 0

    for batch_index, sample in enumerate(loader):
        proposals = model.make_training_proposals(sample)
        batch_examples = None
        for key, value in proposals.items():
            if value is None:
                continue
            cpu_value = value.detach().cpu()
            chunks.setdefault(key, []).append(cpu_value)
            if batch_examples is None:
                batch_examples = int(cpu_value.shape[0])
        if batch_examples is None:
            raise RuntimeError("Frozen FastWAM returned no proposal tensors.")
        buffered += batch_examples
        if buffered >= shard_size:
            flush_shard()
        done = completed + buffered
        if batch_index == 0 or (batch_index + 1) % 10 == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            # Use work completed in this process for a resume-safe ETA.
            processed_now = max(
                batch_index * int(loader.batch_size) + batch_examples,
                1,
            )
            rate = processed_now / elapsed
            eta = (total_examples - done) / max(rate, 1e-9)
            logger.info(
                "proposal_cache=%d/%d rate=%.2f samples/s eta=%.1fmin "
                "peak_mem=%.1fGiB",
                done,
                total_examples,
                rate,
                eta / 60.0,
                torch.cuda.max_memory_allocated() / (1024**3),
            )
    flush_shard()
    if completed != total_examples:
        raise RuntimeError(
            f"Proposal generation ended at {completed}/{total_examples} examples."
        )
    manifest["complete"] = True
    _write_manifest(manifest_path, manifest)
    logger.info(
        "Completed sharded frozen FastWAM proposal cache=%s examples=%d shards=%d",
        manifest_path,
        completed,
        len(manifest["shards"]),
    )
    return ShardedProposalDataset(manifest_path)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="train",
)
def main(cfg: DictConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("EA-WAM training requires CUDA for the frozen FastWAM.")
    output_dir = Path(str(cfg.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("eawam.train")
    hf_cache = _ensure_writable_hf_cache()
    logger.info("Hugging Face dataset cache=%s", hf_cache)
    seed = int(cfg.seed)
    worker_init = set_global_seed(seed, get_worker_init_fn=True)
    device = torch.device("cuda:0")
    model_dtype = _dtype_from_name(cfg.mixed_precision)

    logger.info("Building LIBERO dataset")
    full_dataset = instantiate(cfg.data.train)
    dataset = _even_subset(full_dataset, cfg.get("max_train_samples"))
    generator = torch.Generator().manual_seed(seed)
    proposal_source_loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=int(cfg.num_workers) > 0,
    )

    configured_proposal_cache = cfg.get("proposal_cache_path")
    proposal_cache_override = None
    if configured_proposal_cache:
        proposal_cache_override = Path(
            str(configured_proposal_cache)
        ).expanduser().resolve()
        if not proposal_cache_override.exists():
            raise FileNotFoundError(
                f"Configured proposal cache does not exist: "
                f"{proposal_cache_override}"
            )
        logger.info(
            "Loading EA-WAM extension only; reusing frozen proposal cache=%s",
            proposal_cache_override,
        )
        model = _create_cached_proposal_model(
            cfg,
            device=device,
            model_dtype=model_dtype,
        )
    else:
        logger.info("Loading frozen FastWAM and EA-WAM extension")
        model = instantiate(
            cfg.model,
            model_dtype=model_dtype,
            device=str(device),
        )
    model.train()
    trainable = model.trainable_parameters()
    if any(parameter.requires_grad for parameter in model.base.parameters()):
        raise RuntimeError("Frozen FastWAM unexpectedly has trainable parameters.")
    base_parameter_ids = {id(parameter) for parameter in model.base.parameters()}
    if any(id(parameter) in base_parameter_ids for parameter in trainable):
        raise RuntimeError("Optimizer parameter list overlaps with frozen FastWAM.")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )
    resume = cfg.get("resume_extension")
    start_epoch = 0
    global_step = 0
    if resume:
        payload = model.load_checkpoint(str(resume), optimizer=optimizer)
        start_epoch = int(payload.get("epoch") or 0)
        global_step = int(payload.get("step") or 0)

    proposal_steps = int(model.proposal_inference_steps)
    if proposal_cache_override is not None:
        proposal_cache_path = proposal_cache_override
    else:
        proposal_cache_path = (
            output_dir / f"frozen_base_proposals_steps{proposal_steps}.pt"
        )
    proposal_dataset = _build_or_load_proposal_cache(
        model=model,
        loader=proposal_source_loader,
        cache_path=proposal_cache_path,
        logger=logger,
        shard_size=int(cfg.get("proposal_cache_shard_size", 0)),
    )
    # The 6B backbone is no longer used after caching. Move it off GPU so the
    # actual train phase contains only LDE, TEM, and RAR.
    model.base.to("cpu")
    torch.cuda.empty_cache()
    extension_device = next(model.lde.parameters()).device
    proposal_batch_size = int(cfg.get("proposal_batch_size", 32))
    if isinstance(proposal_dataset, ShardedProposalDataset):
        loader = DataLoader(
            proposal_dataset,
            batch_sampler=ShardBatchSampler(
                proposal_dataset,
                batch_size=proposal_batch_size,
                seed=seed,
            ),
            num_workers=0,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            proposal_dataset,
            batch_size=proposal_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            generator=generator,
        )
    validation_count = min(proposal_batch_size, len(proposal_dataset))
    validation_indices = (
        torch.linspace(
            0,
            len(proposal_dataset) - 1,
            steps=validation_count,
        )
        .round()
        .long()
        .tolist()
    )
    validation_loader = DataLoader(
        Subset(proposal_dataset, validation_indices),
        batch_size=validation_count,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    validation_proposals = _move_proposals(
        next(iter(validation_loader)),
        extension_device,
    )

    accumulation = int(cfg.gradient_accumulation_steps)
    epochs = int(cfg.num_epochs)
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = max(updates_per_epoch * epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_updates,
        eta_min=float(cfg.learning_rate) * 0.01,
    )
    if global_step:
        for _ in range(min(global_step, total_updates)):
            scheduler.step()

    metadata = {
        "dataset_windows_available": len(full_dataset),
        "dataset_windows_used": len(dataset),
        "epochs": epochs,
        "gradient_accumulation_steps": accumulation,
        "trainable_parameters": model.trainable_parameter_count(),
        "base_checkpoint": model.base_checkpoint,
        "base_parameters_trainable": 0,
        "proposal_inference_steps": proposal_steps,
        "proposal_cache_shard_size": int(
            cfg.get("proposal_cache_shard_size", 0)
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Dataset windows=%d/%d, extension parameters=%d, epochs=%d",
        len(dataset),
        len(full_dataset),
        model.trainable_parameter_count(),
        epochs,
    )

    optimizer.zero_grad(set_to_none=True)
    run_start = time.perf_counter()
    micro_step = 0
    best_refined_l1 = float("inf")
    save_every_epochs = max(int(cfg.get("save_every_epochs", 1)), 1)
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_examples = 0
        epoch_start = time.perf_counter()
        for batch_index, sample in enumerate(loader):
            micro_step += 1
            with torch.autocast(
                device_type="cuda",
                dtype=model_dtype,
                enabled=model_dtype != torch.float32,
            ):
                proposals = _move_proposals(sample, extension_device)
                loss, metrics = model.training_loss_from_proposals(proposals)
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            epoch_loss += float(loss.detach().item())
            epoch_examples += int(next(iter(sample.values())).shape[0])

            should_update = micro_step % accumulation == 0 or (
                batch_index + 1 == len(loader)
            )
            if should_update:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    float(cfg.max_grad_norm),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % int(cfg.log_every) == 0:
                    elapsed = max(time.perf_counter() - run_start, 1e-6)
                    updates_per_second = global_step / elapsed
                    remaining = max(total_updates - global_step, 0)
                    eta_seconds = remaining / max(updates_per_second, 1e-9)
                    logger.info(
                        "epoch=%d/%d update=%d/%d loss=%.5f residual_l1=%.5f "
                        "grad=%.4f lr=%.3e eta=%.1fmin peak_mem=%.1fGiB",
                        epoch + 1,
                        epochs,
                        global_step,
                        total_updates,
                        float(loss.detach().item()),
                        metrics["target_residual_l1"],
                        float(grad_norm),
                        optimizer.param_groups[0]["lr"],
                        eta_seconds / 60.0,
                        torch.cuda.max_memory_allocated() / (1024**3),
                    )

        mean_epoch_loss = epoch_loss / max(len(loader), 1)
        logger.info(
            "epoch=%d complete mean_loss=%.6f examples=%d duration=%.1fmin",
            epoch + 1,
            mean_epoch_loss,
            epoch_examples,
            (time.perf_counter() - epoch_start) / 60.0,
        )
        refinement_metrics = model.refinement_metrics_from_proposals(
            validation_proposals,
            num_steps=proposal_steps,
            seed=seed,
        )
        logger.info(
            "epoch=%d offline_action_l1 base=%.6f refined=%.6f "
            "relative_improvement=%.2f%%",
            epoch + 1,
            refinement_metrics["base_action_l1"],
            refinement_metrics["refined_action_l1"],
            100.0 * refinement_metrics["relative_l1_improvement"],
        )
        if refinement_metrics["refined_action_l1"] < best_refined_l1:
            best_refined_l1 = refinement_metrics["refined_action_l1"]
            best_path = output_dir / "eawam_best.pt"
            model.save_checkpoint(
                best_path,
                optimizer=None,
                step=global_step,
                epoch=epoch + 1,
            )
            logger.info("saved_best=%s", best_path)
        if (epoch + 1) % save_every_epochs == 0 or epoch + 1 == epochs:
            checkpoint_path = output_dir / f"eawam_epoch_{epoch + 1:02d}.pt"
            model.save_checkpoint(
                checkpoint_path,
                optimizer=optimizer,
                step=global_step,
                epoch=epoch + 1,
            )
            logger.info("saved=%s", checkpoint_path)

    final_path = output_dir / "eawam_final.pt"
    model.save_checkpoint(
        final_path,
        optimizer=None,
        step=global_step,
        epoch=epochs,
    )
    logger.info("training complete final=%s", final_path)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
