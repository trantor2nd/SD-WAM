import torch

from extensions.eawam.modules import (
    LatentDiscrepancyEncoder,
    ResidualActionRefiner,
    TemporalErrorMemory,
)


def main():
    lde = LatentDiscrepancyEncoder(
        latent_dim=16,
        hidden_dim=64,
        num_error_tokens=4,
        num_layers=1,
        num_heads=4,
        max_visual_tokens=32,
    )
    tem = TemporalErrorMemory(hidden_dim=64)
    rar = ResidualActionRefiner(
        action_dim=7,
        hidden_dim=64,
        max_action_horizon=8,
        num_layers=2,
        num_heads=4,
    )
    predicted = torch.randn(2, 16, 8, 8)
    observed = torch.randn_like(predicted)
    error = lde(predicted, observed)
    memory = tem(error, None)
    base_action = torch.randn(2, 8, 7)
    noisy_residual = torch.randn_like(base_action)
    flow_time = torch.rand(2)
    velocity = rar(
        noisy_residual,
        flow_time,
        base_action,
        error,
        memory,
    )
    loss = velocity.square().mean()
    loss.backward()
    assert error.shape == (2, 4, 64)
    assert memory.shape == (2, 64)
    assert velocity.shape == (2, 8, 7)
    assert any(parameter.grad is not None for parameter in lde.parameters())
    assert any(parameter.grad is not None for parameter in tem.parameters())
    assert any(parameter.grad is not None for parameter in rar.parameters())
    print(
        "EA-WAM module smoke test passed:",
        tuple(error.shape),
        tuple(memory.shape),
        tuple(velocity.shape),
    )


if __name__ == "__main__":
    main()
