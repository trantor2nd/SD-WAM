from __future__ import annotations

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .model import ErrorAwareFastWAM


def create_eawam(
    base_model,
    base_checkpoint: str,
    module,
    *,
    load_text_encoder: bool | None = None,
    skip_dit_load_from_pretrain: bool | None = None,
    action_dit_pretrained_path: str | None = None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    if isinstance(base_model, DictConfig):
        base_model = OmegaConf.create(
            OmegaConf.to_container(base_model, resolve=True)
        )
        if load_text_encoder is not None:
            base_model.load_text_encoder = bool(load_text_encoder)
        if skip_dit_load_from_pretrain is not None:
            base_model.skip_dit_load_from_pretrain = bool(
                skip_dit_load_from_pretrain
            )
        if action_dit_pretrained_path is not None:
            base_model.action_dit_pretrained_path = action_dit_pretrained_path
        base_model = instantiate(
            base_model,
            model_dtype=model_dtype,
            device=device,
        )
    if isinstance(module, DictConfig):
        module = OmegaConf.to_container(module, resolve=True)
    return ErrorAwareFastWAM(
        base_model,
        base_checkpoint=base_checkpoint,
        **dict(module),
    )
