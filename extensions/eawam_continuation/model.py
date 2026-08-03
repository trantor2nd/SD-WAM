from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
from torch.nn import functional as F

from extensions.eawam.model import ErrorAwareFastWAM

from .modules import ContinuationActionRefiner


logger = logging.getLogger(__name__)


class ContinuationErrorAwareFastWAM(ErrorAwareFastWAM):
    """EA-WAM variant that transports the FastWAM action directly to the target."""

    checkpoint_format = "eawam-continuation-v1"

    def __init__(
        self,
        base_model,
        *,
        base_checkpoint: str,
        rar_hidden_dim: Optional[int] = None,
        velocity_scale: float = 0.1,
        gripper_velocity_scale: float = 1.0,
        max_gripper_correction: float = 2.0,
        gripper_residual_scale: Optional[float] = None,
        temporal_shift_augmentation: bool = False,
        warmstart_path: Optional[str] = None,
        freeze_error_encoder: bool = False,
        **kwargs,
    ):
        hidden_dim = int(kwargs.get("hidden_dim", 256))
        rar_layers = int(kwargs.get("rar_layers", 4))
        num_heads = int(kwargs.get("num_heads", 8))
        ffn_multiplier = int(kwargs.get("ffn_multiplier", 4))
        max_action_horizon = int(kwargs.get("max_action_horizon", 32))
        super().__init__(
            base_model,
            base_checkpoint=base_checkpoint,
            rar_hidden_dim=rar_hidden_dim,
            **kwargs,
        )
        action_dim = int(self.base.action_expert.action_dim)
        actual_rar_hidden = int(
            hidden_dim if rar_hidden_dim is None else rar_hidden_dim
        )
        self.velocity_scale = float(velocity_scale)
        self.gripper_velocity_scale = float(gripper_velocity_scale)
        self.max_gripper_correction = float(max_gripper_correction)
        self.gripper_residual_scale = (
            float(self.residual_scale)
            if gripper_residual_scale is None
            else float(gripper_residual_scale)
        )
        self.temporal_shift_augmentation = bool(temporal_shift_augmentation)
        self.freeze_error_encoder = bool(freeze_error_encoder)
        self.rar = ContinuationActionRefiner(
            action_dim=action_dim,
            hidden_dim=actual_rar_hidden,
            condition_dim=hidden_dim,
            max_action_horizon=max_action_horizon,
            num_layers=rar_layers,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            velocity_scale=self.velocity_scale,
            gripper_velocity_scale=self.gripper_velocity_scale,
        ).to(device=self.device, dtype=self.torch_dtype)

        if warmstart_path:
            payload = torch.load(str(warmstart_path), map_location="cpu")
            state = payload.get("eawam_state", payload)
            self.lde.load_state_dict(state["lde"], strict=True)
            self.tem.load_state_dict(state["tem"], strict=True)
            logger.info(
                "Warm-started LDE and TEM from residual EA-WAM checkpoint: %s",
                warmstart_path,
            )
        if self.freeze_error_encoder:
            self.lde.requires_grad_(False)
            self.tem.requires_grad_(False)
            self.lde.eval()
            self.tem.eval()
            logger.info("Frozen warm-started LDE and TEM; training RAR only")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_error_encoder:
            self.lde.eval()
            self.tem.eval()
        return self

    def _bounded_correction(self, correction: torch.Tensor) -> torch.Tensor:
        if correction.shape[-1] <= 1:
            return correction.clamp(-self.max_gripper_correction, self.max_gripper_correction)
        arm = correction[..., :-1].clamp(-self.max_residual, self.max_residual)
        gripper = correction[..., -1:].clamp(
            -self.max_gripper_correction,
            self.max_gripper_correction,
        )
        return torch.cat([arm, gripper], dim=-1)

    def _refine(
        self,
        base_segment: torch.Tensor,
        error_tokens: torch.Tensor,
        memory: torch.Tensor,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        transported = self.rar.sample(
            base_segment,
            error_tokens,
            memory,
            num_steps=num_steps,
        )
        raw_correction = transported - base_segment
        if raw_correction.shape[-1] <= 1:
            scaled_correction = raw_correction * self.gripper_residual_scale
        else:
            scaled_correction = torch.cat(
                [
                    raw_correction[..., :-1] * self.residual_scale,
                    raw_correction[..., -1:] * self.gripper_residual_scale,
                ],
                dim=-1,
            )
        correction = self._bounded_correction(scaled_correction)
        return (base_segment + correction).clamp(-1, 1)

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
            raise ValueError("Continuation training needs at least one future segment")

        batch_size = target_action.shape[0]
        memory = None
        losses = []
        correction_l1_values = []
        temporal_shift = 0
        if self.temporal_shift_augmentation:
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
                pad = self.refine_horizon - (end - start)
                base_segment = F.pad(base_segment, (0, 0, 0, pad))
                target_segment = F.pad(target_segment, (0, 0, 0, pad))
                valid = F.pad(valid, (0, pad), value=False)

            base_segment = (
                base_segment
                + self._bounded_noise(base_segment, self.action_noise_std)
            ).clamp(-1, 1)
            target_correction = self._bounded_correction(
                target_segment - base_segment
            )
            reachable_target = (base_segment + target_correction).clamp(-1, 1)
            flow_time = torch.rand(
                batch_size,
                device=target_action.device,
                dtype=target_action.dtype,
            )
            action_state = (
                (1.0 - flow_time[:, None, None]) * base_segment
                + flow_time[:, None, None] * reachable_target
            )
            target_velocity = target_correction / self.rar.velocity_scales(
                target_correction
            )
            predicted_velocity = self.rar(
                action_state,
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
            denominator = valid_float.sum().clamp(min=1)
            losses.append((token_loss * valid_float).sum() / denominator)
            correction_l1_values.append(
                (
                    target_correction.float().abs().mean(dim=-1)
                    * valid_float
                ).sum()
                / denominator
            )

        if not losses:
            raise RuntimeError("No valid continuation segment was produced")
        loss = torch.stack(losses).mean()
        correction_l1 = torch.stack(correction_l1_values).mean()
        return loss, {
            "loss_rar": float(loss.detach().item()),
            "target_residual_l1": float(correction_l1.detach().item()),
            "temporal_steps": float(len(losses)),
        }

    @torch.no_grad()
    def refinement_metrics_from_proposals(
        self,
        proposals: dict[str, Any],
        *,
        num_steps: int = 10,
        seed: int = 42,
    ) -> dict[str, float]:
        del seed
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
            base_for_model = base_segment
            if original_length < self.refine_horizon:
                base_for_model = F.pad(
                    base_segment,
                    (0, 0, 0, self.refine_horizon - original_length),
                )
            refined = self._refine(
                base_for_model,
                error_tokens,
                memory,
                num_steps=num_steps,
            )[:, :original_length]
            valid_float = valid.to(torch.float32)
            denominator = valid_float.sum().clamp(min=1)
            base_errors.append(
                (
                    (base_segment.float() - target_segment.float())
                    .abs()
                    .mean(dim=-1)
                    * valid_float
                ).sum()
                / denominator
            )
            refined_errors.append(
                (
                    (refined.float() - target_segment.float())
                    .abs()
                    .mean(dim=-1)
                    * valid_float
                ).sum()
                / denominator
            )
        base_l1 = torch.stack(base_errors).mean()
        refined_l1 = torch.stack(refined_errors).mean()
        relative = (base_l1 - refined_l1) / base_l1.clamp(min=1e-8)
        return {
            "base_action_l1": float(base_l1.item()),
            "refined_action_l1": float(refined_l1.item()),
            "relative_l1_improvement": float(relative.item()),
        }

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
            device=self.device,
            dtype=self.torch_dtype,
        )
        refined_action = base_action.clone()
        if self._cached_predicted_latent is not None:
            error_tokens = self.lde(
                self._cached_predicted_latent,
                current_latent,
            )
            self._memory = self.tem(error_tokens, self._memory)
            horizon = min(self.refine_horizon, base_action.shape[0])
            refined_action[:horizon] = self._refine(
                base_action[:horizon].unsqueeze(0),
                error_tokens,
                self._memory,
                num_steps=num_inference_steps,
            )[0]

        predicted_frames = prediction["video"]
        if not predicted_frames:
            raise RuntimeError("Frozen FastWAM returned no predicted video frames")
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
