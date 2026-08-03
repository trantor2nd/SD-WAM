from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous sinusoidal embedding for flow time in [0, 1]."""
    if timestep.ndim == 0:
        timestep = timestep[None]
    half = dim // 2
    if half == 0:
        return timestep[:, None]
    scale = -math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(
        torch.arange(half, device=timestep.device, dtype=torch.float32) * scale
    )
    angles = timestep.float()[:, None] * frequencies[None] * 1000.0
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class LatentDiscrepancyEncoder(nn.Module):
    """Jointly encode predicted and observed visual latents into error tokens."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_error_tokens: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        max_visual_tokens: int = 196,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_error_tokens = int(num_error_tokens)
        self.max_visual_tokens = int(max_visual_tokens)

        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.source_embedding = nn.Parameter(torch.zeros(2, self.hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_visual_tokens, self.hidden_dim)
        )
        self.error_tokens = nn.Parameter(
            torch.zeros(1, self.num_error_tokens, self.hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.error_tokens, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.source_embedding, std=0.02)

    def _tokens(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            latent = latent.flatten(2).transpose(1, 2)
        elif latent.ndim != 3:
            raise ValueError(
                f"latent must be [B,C,H,W] or [B,N,C], got {tuple(latent.shape)}"
            )
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent feature dim must be {self.latent_dim}, got {latent.shape[-1]}"
            )
        if latent.shape[1] > self.max_visual_tokens:
            latent = F.adaptive_avg_pool1d(
                latent.transpose(1, 2), self.max_visual_tokens
            ).transpose(1, 2)
        return self.latent_projection(latent)

    def forward(
        self,
        predicted_latent: torch.Tensor,
        observed_latent: torch.Tensor,
    ) -> torch.Tensor:
        predicted = self._tokens(predicted_latent)
        observed = self._tokens(observed_latent)
        if predicted.shape != observed.shape:
            raise ValueError(
                "predicted and observed latent tokens must match, got "
                f"{tuple(predicted.shape)} and {tuple(observed.shape)}"
            )
        batch_size, num_tokens, _ = predicted.shape
        positions = self.position_embedding[:, :num_tokens]
        predicted = predicted + positions + self.source_embedding[0]
        observed = observed + positions + self.source_embedding[1]
        queries = self.error_tokens.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([queries, predicted, observed], dim=1))
        return self.output_norm(encoded[:, : self.num_error_tokens])


class TemporalErrorMemory(nn.Module):
    """GRU memory over pooled error tokens."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def forward(
        self,
        error_tokens: torch.Tensor,
        previous_memory: torch.Tensor | None,
    ) -> torch.Tensor:
        pooled_error = error_tokens.mean(dim=1)
        if previous_memory is None:
            previous_memory = self.initial_state(
                pooled_error.shape[0],
                device=pooled_error.device,
                dtype=pooled_error.dtype,
            )
        return self.gru(pooled_error, previous_memory)


class ResidualDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int,
        dropout: float,
    ):
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        modulated = self.norm_self(tokens + time_embedding[:, None])
        attended, _ = self.self_attention(
            modulated, modulated, modulated, need_weights=False
        )
        tokens = tokens + attended
        query = self.norm_cross(tokens)
        attended, _ = self.cross_attention(
            query, condition, condition, need_weights=False
        )
        tokens = tokens + attended
        return tokens + self.ffn(self.norm_ffn(tokens))


class ResidualActionRefiner(nn.Module):
    """A compact DiT that predicts the residual flow velocity field."""

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
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = int(
            self.hidden_dim if condition_dim is None else condition_dim
        )
        self.max_action_horizon = int(max_action_horizon)
        self.base_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.residual_projection = nn.Linear(self.action_dim, self.hidden_dim)
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
        self.memory_projection = nn.Linear(
            self.condition_dim,
            self.hidden_dim,
        )
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

    def reset_parameters(self) -> None:
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        flow_time: torch.Tensor,
        base_action: torch.Tensor,
        error_tokens: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_residual.shape != base_action.shape:
            raise ValueError(
                "noisy_residual and base_action must match, got "
                f"{tuple(noisy_residual.shape)} and {tuple(base_action.shape)}"
            )
        horizon = noisy_residual.shape[1]
        if horizon > self.max_action_horizon:
            raise ValueError(
                f"action horizon {horizon} exceeds max {self.max_action_horizon}"
            )
        positions = self.position_embedding[:, :horizon]
        base_tokens = (
            self.base_projection(base_action) + positions + self.type_embedding[0]
        )
        residual_tokens = (
            self.residual_projection(noisy_residual)
            + positions
            + self.type_embedding[1]
        )
        tokens = torch.cat([base_tokens, residual_tokens], dim=1)
        time_embedding = self.time_mlp(
            sinusoidal_embedding(flow_time, self.hidden_dim).to(tokens.dtype)
        )
        error_condition = self.error_projection(error_tokens)
        memory_token = self.memory_projection(memory).unsqueeze(1)
        condition = torch.cat([error_condition, memory_token], dim=1)
        for block in self.blocks:
            tokens = block(tokens, condition, time_embedding)
        residual_tokens = tokens[:, horizon:]
        return self.output(self.output_norm(residual_tokens))

    @torch.no_grad()
    def sample(
        self,
        base_action: torch.Tensor,
        error_tokens: torch.Tensor,
        memory: torch.Tensor,
        *,
        num_steps: int,
        generator: torch.Generator | None = None,
        start_from_zero: bool = False,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if start_from_zero:
            residual = torch.zeros_like(base_action)
        else:
            residual = torch.randn(
                base_action.shape,
                device=base_action.device,
                dtype=base_action.dtype,
                generator=generator,
            )
        delta = 1.0 / float(num_steps)
        for step in range(num_steps):
            flow_time = torch.full(
                (base_action.shape[0],),
                step / float(num_steps),
                device=base_action.device,
                dtype=base_action.dtype,
            )
            velocity = self(
                residual,
                flow_time,
                base_action,
                error_tokens,
                memory,
            )
            residual = residual + delta * velocity
        return residual
