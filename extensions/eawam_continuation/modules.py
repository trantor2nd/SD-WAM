from __future__ import annotations

import torch
from torch import nn

from extensions.eawam.modules import ResidualDiTBlock, sinusoidal_embedding


class ContinuationActionRefiner(nn.Module):
    """Conditional flow from a FastWAM action proposal to a refined action."""

    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 256,
        max_action_horizon: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        condition_dim: int | None = None,
        velocity_scale: float = 0.1,
        gripper_velocity_scale: float = 1.0,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if velocity_scale <= 0:
            raise ValueError("velocity_scale must be positive")
        if gripper_velocity_scale <= 0:
            raise ValueError("gripper_velocity_scale must be positive")
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = int(
            self.hidden_dim if condition_dim is None else condition_dim
        )
        self.max_action_horizon = int(max_action_horizon)
        self.velocity_scale = float(velocity_scale)
        self.gripper_velocity_scale = float(gripper_velocity_scale)

        self.base_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.action_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.type_embedding = nn.Parameter(torch.zeros(2, self.hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_action_horizon, self.hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.error_projection = (
            nn.Identity()
            if self.condition_dim == self.hidden_dim
            else nn.Linear(self.condition_dim, self.hidden_dim)
        )
        self.memory_projection = nn.Linear(self.condition_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualDiTBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.action_dim)
        self.reset_parameters()

    def velocity_scales(self, reference: torch.Tensor) -> torch.Tensor:
        scales = torch.full_like(reference, self.velocity_scale)
        if self.action_dim > 1:
            scales[..., -1] = self.gripper_velocity_scale
        return scales

    def reset_parameters(self) -> None:
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        action_state: torch.Tensor,
        flow_time: torch.Tensor,
        base_action: torch.Tensor,
        error_tokens: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if action_state.shape != base_action.shape:
            raise ValueError(
                "action_state and base_action must match, got "
                f"{tuple(action_state.shape)} and {tuple(base_action.shape)}"
            )
        horizon = action_state.shape[1]
        if horizon > self.max_action_horizon:
            raise ValueError(
                f"action horizon {horizon} exceeds max {self.max_action_horizon}"
            )
        positions = self.position_embedding[:, :horizon]
        base_tokens = (
            self.base_projection(base_action) + positions + self.type_embedding[0]
        )
        action_tokens = (
            self.action_projection(action_state)
            + positions
            + self.type_embedding[1]
        )
        tokens = torch.cat([base_tokens, action_tokens], dim=1)
        time_embedding = self.time_mlp(
            sinusoidal_embedding(flow_time, self.hidden_dim).to(tokens.dtype)
        )
        condition = torch.cat(
            [
                self.error_projection(error_tokens),
                self.memory_projection(memory).unsqueeze(1),
            ],
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens, condition, time_embedding)
        return self.output(self.output_norm(tokens[:, horizon:]))

    @torch.no_grad()
    def sample(
        self,
        base_action: torch.Tensor,
        error_tokens: torch.Tensor,
        memory: torch.Tensor,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        action_state = base_action.clone()
        delta = 1.0 / float(num_steps)
        for step in range(num_steps):
            flow_time = torch.full(
                (base_action.shape[0],),
                step / float(num_steps),
                device=base_action.device,
                dtype=base_action.dtype,
            )
            normalized_velocity = self(
                action_state,
                flow_time,
                base_action,
                error_tokens,
                memory,
            )
            action_state = (
                action_state
                + delta
                * self.velocity_scales(normalized_velocity)
                * normalized_velocity
            )
        return action_state
