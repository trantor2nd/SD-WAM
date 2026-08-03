from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .modules import (
    LatentDiscrepancyEncoder,
    ResidualActionRefiner,
    TemporalErrorMemory,
)

logger = logging.getLogger(__name__)


class ErrorAwareFastWAM(nn.Module):
    """EA-WAM as an isolated wrapper around a frozen FastWAM backbone."""

    checkpoint_format = "eawam-extension-v1"

    def __init__(
        self,
        base_model: nn.Module,
        *,
        base_checkpoint: str,
        latent_dim: int = 48,
        hidden_dim: int = 256,
        rar_hidden_dim: Optional[int] = None,
        num_error_tokens: int = 4,
        lde_layers: int = 2,
        rar_layers: int = 4,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        max_visual_tokens: int = 196,
        max_action_horizon: int = 32,
        refine_horizon: int = 8,
        feedback_frame_index: int = 2,
        latent_noise_std: float = 0.02,
        action_noise_std: float = 0.01,
        noise_clip: float = 2.0,
        max_residual: float = 0.25,
        proposal_inference_steps: int = 10,
        residual_zero_start: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.base = base_model
        self.base_checkpoint = str(base_checkpoint)
        self.refine_horizon = int(refine_horizon)
        self.feedback_frame_index = int(feedback_frame_index)
        self.latent_noise_std = float(latent_noise_std)
        self.action_noise_std = float(action_noise_std)
        self.noise_clip = float(noise_clip)
        self.max_residual = float(max_residual)
        self.proposal_inference_steps = int(proposal_inference_steps)
        self.residual_zero_start = bool(residual_zero_start)
        self.residual_scale = float(residual_scale)
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if self.proposal_inference_steps <= 0:
            raise ValueError("proposal_inference_steps must be positive")

        action_dim = int(self.base.action_expert.action_dim)
        rar_hidden_dim = int(
            hidden_dim if rar_hidden_dim is None else rar_hidden_dim
        )
        backbone_latent_dim = int(self.base.video_expert.in_dim)
        if int(latent_dim) != backbone_latent_dim:
            raise ValueError(
                "EA-WAM latent_dim must match the frozen FastWAM video latent "
                f"channels: configured={latent_dim}, backbone={backbone_latent_dim}"
            )
        self.lde = LatentDiscrepancyEncoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_error_tokens=num_error_tokens,
            num_layers=lde_layers,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            max_visual_tokens=max_visual_tokens,
        )
        self.tem = TemporalErrorMemory(hidden_dim=hidden_dim)
        self.rar = ResidualActionRefiner(
            action_dim=action_dim,
            hidden_dim=rar_hidden_dim,
            condition_dim=hidden_dim,
            max_action_horizon=max_action_horizon,
            num_layers=rar_layers,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
        )

        self.base.load_checkpoint(self.base_checkpoint)
        self._freeze_base()
        self.lde.to(device=self.device, dtype=self.torch_dtype)
        self.tem.to(device=self.device, dtype=self.torch_dtype)
        self.rar.to(device=self.device, dtype=self.torch_dtype)
        self.reset_feedback()

    @property
    def device(self):
        return self.base.device

    @property
    def torch_dtype(self):
        return self.base.torch_dtype

    @property
    def action_expert(self):
        return self.base.action_expert

    @property
    def proprio_dim(self):
        return self.base.proprio_dim

    def _freeze_base(self) -> None:
        self.base.eval()
        self.base.requires_grad_(False)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def trainable_parameters(self):
        return list(self.lde.parameters()) + list(self.tem.parameters()) + list(
            self.rar.parameters()
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def reset_feedback(self) -> None:
        self._cached_predicted_latent: Optional[torch.Tensor] = None
        self._memory: Optional[torch.Tensor] = None

    def _bounded_noise(self, tensor: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return torch.zeros_like(tensor)
        bound = self.noise_clip * std
        return torch.randn_like(tensor).mul_(std).clamp_(-bound, bound)

    @torch.no_grad()
    def _sample_base_proposals(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Sample frozen FastWAM proposals with its actual Euler schedule.

        This mirrors the deployed joint video/action sampler but keeps latent
        video output on device instead of decoding it to RGB.
        """
        inputs = self.base.build_inputs(sample)
        input_latents = inputs["input_latents"]
        target_action = inputs["action"]
        batch_size = input_latents.shape[0]
        predicted_video = torch.randn_like(input_latents)
        predicted_action = torch.randn_like(target_action)
        if inputs["first_frame_latents"] is not None:
            predicted_video[:, :, :1] = inputs["first_frame_latents"]

        video_timesteps, video_deltas = (
            self.base.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=self.proposal_inference_steps,
                device=self.device,
                dtype=predicted_video.dtype,
            )
        )
        action_timesteps, action_deltas = (
            self.base.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=self.proposal_inference_steps,
                device=self.device,
                dtype=predicted_action.dtype,
            )
        )
        for video_time, video_delta, action_time, action_delta in zip(
            video_timesteps,
            video_deltas,
            action_timesteps,
            action_deltas,
        ):
            video_velocity, action_velocity = self.base._predict_joint_noise(
                latents_video=predicted_video,
                latents_action=predicted_action,
                timestep_video=video_time.expand(batch_size).to(
                    dtype=predicted_video.dtype
                ),
                timestep_action=action_time.expand(batch_size).to(
                    dtype=predicted_action.dtype
                ),
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                fuse_vae_embedding_in_latents=inputs[
                    "fuse_vae_embedding_in_latents"
                ],
                gt_action=None,
            )
            predicted_video = self.base.infer_video_scheduler.step(
                video_velocity,
                video_delta,
                predicted_video,
            )
            predicted_action = self.base.infer_action_scheduler.step(
                action_velocity,
                action_delta,
                predicted_action,
            )
            if inputs["first_frame_latents"] is not None:
                predicted_video[:, :, :1] = inputs["first_frame_latents"]

        return {
            "predicted_video": predicted_video,
            "observed_video": input_latents,
            "base_action": predicted_action.clamp(-1, 1),
            "target_action": target_action,
            "action_is_pad": inputs["action_is_pad"],
        }

    @torch.no_grad()
    def make_training_proposals(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self._sample_base_proposals(sample)

    def training_loss_from_proposals(
        self,
        proposals: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        predicted_video = proposals["predicted_video"]
        observed_video = proposals["observed_video"]
        base_action = proposals["base_action"]
        target_action = proposals["target_action"]
        action_is_pad = proposals["action_is_pad"]

        future_steps = min(
            max(predicted_video.shape[2] - 1, 0),
            max((target_action.shape[1] - 1) // self.refine_horizon, 0),
        )
        if future_steps <= 0:
            raise ValueError(
                "EA-WAM training needs at least one future latent and one future "
                "action segment."
            )

        batch_size = target_action.shape[0]
        memory = None
        losses = []
        residual_l1_values = []
        temporal_shift = int(
            torch.randint(
                0,
                max(self.refine_horizon, 1),
                (1,),
                device=target_action.device,
            ).item()
        )

        for step_index in range(future_steps):
            predicted_latent = predicted_video[:, :, step_index + 1]
            observed_latent = observed_video[:, :, step_index + 1]
            predicted_latent = predicted_latent + self._bounded_noise(
                predicted_latent, self.latent_noise_std
            )
            observed_latent = observed_latent + self._bounded_noise(
                observed_latent, self.latent_noise_std
            )
            error_tokens = self.lde(predicted_latent, observed_latent)
            memory = self.tem(error_tokens, memory)

            start = max(
                (step_index + 1) * self.refine_horizon - temporal_shift,
                0,
            )
            end = min(start + self.refine_horizon, target_action.shape[1])
            if end <= start:
                continue
            base_segment = base_action[:, start:end]
            target_segment = target_action[:, start:end]
            valid = torch.ones(
                batch_size,
                end - start,
                device=target_action.device,
                dtype=torch.bool,
            )
            if action_is_pad is not None:
                valid = ~action_is_pad[:, start:end]

            if end - start < self.refine_horizon:
                pad_length = self.refine_horizon - (end - start)
                base_segment = F.pad(base_segment, (0, 0, 0, pad_length))
                target_segment = F.pad(target_segment, (0, 0, 0, pad_length))
                valid = F.pad(valid, (0, pad_length), value=False)

            base_segment = (
                base_segment + self._bounded_noise(
                    base_segment, self.action_noise_std
                )
            ).clamp(-1, 1)
            target_residual = target_segment - base_segment
            flow_time = torch.rand(
                batch_size,
                device=target_action.device,
                dtype=target_action.dtype,
            )
            epsilon = torch.randn_like(target_residual)
            noisy_residual = (
                flow_time[:, None, None] * target_residual
                + (1.0 - flow_time[:, None, None]) * epsilon
            )
            target_velocity = target_residual - epsilon
            predicted_velocity = self.rar(
                noisy_residual,
                flow_time,
                base_segment,
                error_tokens,
                memory,
            )
            token_loss = F.mse_loss(
                predicted_velocity.float(),
                target_velocity.float(),
                reduction="none",
            ).mean(dim=-1)
            valid_float = valid.to(token_loss.dtype)
            loss = (token_loss * valid_float).sum() / valid_float.sum().clamp(min=1)
            losses.append(loss)
            residual_l1_values.append(
                (
                    target_residual.float().abs().mean(dim=-1) * valid_float
                ).sum()
                / valid_float.sum().clamp(min=1)
            )

        if not losses:
            raise RuntimeError("No valid residual segment was produced for this batch.")
        total_loss = torch.stack(losses).mean()
        residual_l1 = torch.stack(residual_l1_values).mean()
        return total_loss, {
            "loss_rar": float(total_loss.detach().item()),
            "target_residual_l1": float(residual_l1.detach().item()),
            "temporal_steps": float(len(losses)),
        }

    def training_loss(
        self,
        sample: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self.training_loss_from_proposals(
            self.make_training_proposals(sample)
        )

    @torch.no_grad()
    def refinement_metrics_from_proposals(
        self,
        proposals: dict[str, Any],
        *,
        num_steps: int = 10,
        seed: int = 42,
    ) -> dict[str, float]:
        """Measure normalized action L1 before/after residual refinement."""

        predicted_video = proposals["predicted_video"]
        observed_video = proposals["observed_video"]
        base_action = proposals["base_action"]
        target_action = proposals["target_action"]
        action_is_pad = proposals["action_is_pad"]
        future_steps = min(
            max(predicted_video.shape[2] - 1, 0),
            max((target_action.shape[1] - 1) // self.refine_horizon, 0),
        )
        memory = None
        base_errors = []
        refined_errors = []
        for step_index in range(future_steps):
            error_tokens = self.lde(
                predicted_video[:, :, step_index + 1],
                observed_video[:, :, step_index + 1],
            )
            memory = self.tem(error_tokens, memory)
            start = (step_index + 1) * self.refine_horizon
            end = min(start + self.refine_horizon, target_action.shape[1])
            if end <= start:
                continue
            base_segment = base_action[:, start:end]
            target_segment = target_action[:, start:end]
            valid = torch.ones(
                target_segment.shape[:2],
                device=target_segment.device,
                dtype=torch.bool,
            )
            if action_is_pad is not None:
                valid = ~action_is_pad[:, start:end]
            original_length = end - start
            if original_length < self.refine_horizon:
                pad_length = self.refine_horizon - original_length
                base_for_model = F.pad(base_segment, (0, 0, 0, pad_length))
            else:
                base_for_model = base_segment
            generator = torch.Generator(device=base_action.device).manual_seed(
                int(seed) + step_index
            )
            residual = self.rar.sample(
                base_for_model,
                error_tokens,
                memory,
                num_steps=num_steps,
                generator=generator,
                start_from_zero=self.residual_zero_start,
            )[:, :original_length]
            residual = (residual * self.residual_scale).clamp(
                -self.max_residual,
                self.max_residual,
            )
            refined_segment = (base_segment + residual).clamp(-1, 1)
            valid_float = valid.to(torch.float32)
            base_token_error = (
                (base_segment.float() - target_segment.float())
                .abs()
                .mean(dim=-1)
            )
            refined_token_error = (
                (refined_segment.float() - target_segment.float())
                .abs()
                .mean(dim=-1)
            )
            denominator = valid_float.sum().clamp(min=1)
            base_errors.append(
                (base_token_error * valid_float).sum() / denominator
            )
            refined_errors.append(
                (refined_token_error * valid_float).sum() / denominator
            )
        if not base_errors:
            raise RuntimeError("No valid segment for refinement metrics")
        base_l1 = torch.stack(base_errors).mean()
        refined_l1 = torch.stack(refined_errors).mean()
        relative = (base_l1 - refined_l1) / base_l1.clamp(min=1e-8)
        return {
            "base_action_l1": float(base_l1.item()),
            "refined_action_l1": float(refined_l1.item()),
            "relative_l1_improvement": float(relative.item()),
        }

    @staticmethod
    def _pil_to_tensor(frame: Image.Image | np.ndarray) -> torch.Tensor:
        if isinstance(frame, Image.Image):
            array = np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()
        else:
            array = np.asarray(frame, dtype=np.uint8)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().mul_(2.0 / 255.0).sub_(1.0)

    @torch.no_grad()
    def _encode_observation(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(device=self.device, dtype=self.torch_dtype)
        latent = self.base._encode_input_image_latents_tensor(image)
        return latent[:, :, 0]

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        num_video_frames: int = 9,
    ) -> dict[str, Any]:
        self.eval()
        current_latent = self._encode_observation(input_image)
        prediction = self.base.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=None,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
        base_action = prediction["action"].to(
            device=self.device, dtype=self.torch_dtype
        )
        refined_action = base_action.clone()

        if self._cached_predicted_latent is not None:
            error_tokens = self.lde(
                self._cached_predicted_latent,
                current_latent,
            )
            self._memory = self.tem(error_tokens, self._memory)
            horizon = min(self.refine_horizon, base_action.shape[0])
            base_segment = base_action[:horizon].unsqueeze(0)
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(
                    int(seed) + 100003
                )
            residual = self.rar.sample(
                base_segment,
                error_tokens,
                self._memory,
                num_steps=num_inference_steps,
                generator=generator,
                start_from_zero=self.residual_zero_start,
            )[0]
            residual = (residual * self.residual_scale).clamp(
                -self.max_residual,
                self.max_residual,
            )
            refined_action[:horizon] = (base_action[:horizon] + residual).clamp(
                -1, 1
            )

        predicted_frames = prediction["video"]
        if not predicted_frames:
            raise RuntimeError("Frozen FastWAM returned no predicted video frames.")
        frame_index = min(
            max(self.feedback_frame_index, 0),
            len(predicted_frames) - 1,
        )
        predicted_frame = self._pil_to_tensor(predicted_frames[frame_index])
        self._cached_predicted_latent = self._encode_observation(predicted_frame)
        return {
            "action": refined_action.detach().cpu().float(),
            "base_action": base_action.detach().cpu().float(),
        }

    def extension_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "lde": self.lde.state_dict(),
            "tem": self.tem.state_dict(),
            "rar": self.rar.state_dict(),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer=None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        payload = {
            "format": self.checkpoint_format,
            "eawam_state": self.extension_state_dict(),
            "base_checkpoint": self.base_checkpoint,
            "step": step,
            "epoch": epoch,
            "trainable_parameters": self.trainable_parameter_count(),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, str(path))

    def load_checkpoint(self, path: str | Path, optimizer=None):
        payload = torch.load(str(path), map_location="cpu")
        if isinstance(payload, dict) and "eawam_state" in payload:
            state = payload["eawam_state"]
            self.lde.load_state_dict(state["lde"], strict=True)
            self.tem.load_state_dict(state["tem"], strict=True)
            self.rar.load_state_dict(state["rar"], strict=True)
            if optimizer is not None and "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            logger.info("Loaded EA-WAM extension checkpoint: %s", path)
            return payload
        logger.warning(
            "Checkpoint %s is not an EA-WAM extension; loading it into the frozen base.",
            path,
        )
        self.base.load_checkpoint(str(path), optimizer=None)
        self._freeze_base()
        return payload
