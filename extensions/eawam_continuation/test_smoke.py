import torch

from extensions.eawam.train_libero import _CachedProposalBackbone
from extensions.eawam_continuation.model import ContinuationErrorAwareFastWAM


device = torch.device("cuda:0")
dtype = torch.bfloat16
backbone = _CachedProposalBackbone(
    action_dim=7,
    latent_dim=48,
    proprio_dim=7,
    device=device,
    dtype=dtype,
)
model = ContinuationErrorAwareFastWAM(
    backbone,
    base_checkpoint="unused.pt",
    latent_dim=48,
    hidden_dim=256,
    rar_hidden_dim=768,
    num_error_tokens=4,
    lde_layers=2,
    rar_layers=12,
    num_heads=8,
    ffn_multiplier=4,
    max_visual_tokens=196,
    max_action_horizon=32,
    refine_horizon=8,
    latent_noise_std=0.02,
    action_noise_std=0.005,
    max_residual=0.25,
    residual_scale=1.0,
    velocity_scale=0.1,
    gripper_velocity_scale=1.0,
    max_gripper_correction=2.0,
    temporal_shift_augmentation=False,
).train()
batch = 2
proposals = {
    "predicted_video": torch.randn(
        batch, 48, 5, 14, 14, device=device, dtype=dtype
    ),
    "observed_video": torch.randn(
        batch, 48, 5, 14, 14, device=device, dtype=dtype
    ),
    "base_action": torch.rand(batch, 40, 7, device=device, dtype=dtype) * 2 - 1,
    "target_action": torch.rand(
        batch, 40, 7, device=device, dtype=dtype
    ) * 2 - 1,
    "action_is_pad": torch.zeros(batch, 40, device=device, dtype=torch.bool),
}
loss, training_metrics = model.training_loss_from_proposals(proposals)
loss.backward()
model.eval()
metrics = model.refinement_metrics_from_proposals(
    proposals,
    num_steps=2,
)
print(
    {
        "parameters": model.trainable_parameter_count(),
        "loss": float(loss.detach()),
        "training_metrics": training_metrics,
        "refinement_metrics": metrics,
        "output_weight_nonzero_grad": bool(
            model.rar.output.weight.grad.float().abs().sum() > 0
        ),
    }
)
