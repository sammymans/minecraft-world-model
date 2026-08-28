from __future__ import annotations

from pathlib import Path

import torch

from mcwm.model import SpatialLatentDynamics, SpatialLatentVideoFlow
from mcwm.spatial_flow import FlowStepAdapter, flow_matching_loss, load_spatial_flow_checkpoint


def _model(*, horizon: int = 2) -> SpatialLatentVideoFlow:
    return SpatialLatentVideoFlow(
        latent_channels=2,
        action_dim=3,
        horizon=horizon,
        hidden_channels=8,
        condition_dim=16,
    )


def test_flow_preserves_clip_shape_and_uses_conditioning() -> None:
    torch.manual_seed(2)
    model = _model()
    noisy = torch.randn(3, 2, 2, 8, 8)
    context = torch.randn(3, 2, 2, 8, 8)
    base = torch.randn_like(noisy)
    actions = torch.randn(3, 2, 3)
    flow_time = torch.rand(3)

    conditioned = model(noisy, context, base, actions, flow_time)
    unconditioned = model(noisy, context, base, actions, flow_time, torch.zeros(3))

    assert conditioned.shape == noisy.shape
    assert conditioned.isfinite().all()
    assert not torch.equal(conditioned, unconditioned)


def test_flow_sampler_integrates_a_perfect_constant_velocity() -> None:
    class PerfectFlow(SpatialLatentVideoFlow):
        velocity: torch.Tensor

        def forward(
            self, noisy_future, context, base_future, actions, flow_time,
            condition_mask=None
        ):
            return self.velocity.expand_as(noisy_future)

    model = PerfectFlow(
        latent_channels=1, action_dim=1, horizon=2, hidden_channels=8, condition_dim=8
    )
    initial = torch.randn(1, 2, 1, 4, 4)
    target_residual = torch.randn_like(initial)
    target = context_current = torch.randn(1, 1, 1, 4, 4)
    target = target + target_residual
    model.velocity = target_residual - initial
    context = torch.cat((torch.randn_like(context_current), context_current), dim=1)
    actions = torch.randn(1, 2, 1)
    base = context_current.expand_as(initial)

    sampled = model.sample(
        context, base, actions, steps=4, guidance_scale=1, initial_noise=initial
    )

    assert torch.allclose(sampled, target, atol=1e-6)


def test_flow_loss_can_overfit_one_fixed_clip() -> None:
    torch.manual_seed(3)
    model = _model()
    batch = {
        "latents": torch.randn(1, 4, 2, 8, 8),
        "actions": torch.randn(1, 2, 3),
    }
    noise = torch.randn(1, 2, 2, 8, 8)
    base = torch.randn_like(noise)
    flow_time = torch.tensor([0.4])
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    initial = flow_matching_loss(
        model, batch, base, action_dropout=0, noise=noise, flow_time=flow_time
    ).item()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = flow_matching_loss(
            model, batch, base, action_dropout=0, noise=noise, flow_time=flow_time
        )
        loss.backward()
        optimizer.step()
    final = flow_matching_loss(
        model, batch, base, action_dropout=0, noise=noise, flow_time=flow_time
    ).item()

    assert final < initial * 0.5


def test_flow_checkpoint_and_live_adapter(tmp_path: Path) -> None:
    model = _model()
    checkpoint = tmp_path / "flow.pt"
    torch.save(
        {
            "model_type": "spatial_latent_video_flow",
            "architecture": "v1_refinement_rectified_flow_v3",
            "latent_channels": 2,
            "action_dim": 3,
            "horizon": 2,
            "hidden_channels": 8,
            "condition_dim": 16,
            "sampling_steps": 2,
            "guidance_scale": 1.5,
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    loaded, metadata = load_spatial_flow_checkpoint(checkpoint, torch.device("cpu"))
    base_dynamics = SpatialLatentDynamics(
        latent_channels=2, action_dim=3, hidden_channels=8, blocks=1
    )
    adapter = FlowStepAdapter(loaded, base_dynamics, steps=2, guidance_scale=1.5)
    previous = torch.randn(1, 2, 8, 8)
    current = torch.randn(1, 2, 8, 8)
    action = torch.randn(1, 3)

    predicted = adapter(previous, current, action)
    adapter.reset_sampling()
    repeated = adapter(previous, current, action)
    adapter(previous, repeated, action)

    assert metadata["horizon"] == 2
    assert predicted.shape == current.shape
    assert torch.equal(predicted, repeated)
    assert adapter.cache_index == 2
