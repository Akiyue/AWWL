"""End-to-end one-step training smoke test on synthetic CIFAR-like data.

This exercises the full training-step machinery (UNet forward, scheduler
add-noise, AWWL loss, optimizer.step) without touching disk or HuggingFace.
"""

from __future__ import annotations

import torch
from diffusers import DDPMScheduler

from awwl.losses import get_loss_function
from awwl.models import build_ddpm_unet


def test_one_training_step_runs():
    torch.manual_seed(0)
    # block_out_channels must be divisible by GroupNorm's num_groups (32 in diffusers).
    unet = build_ddpm_unet(
        image_size=16,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    scheduler = DDPMScheduler(num_train_timesteps=50)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-3)
    loss_fn = get_loss_function(
        "adaptive_wavelet",
        noise_scheduler=scheduler,
        alpha=0.8, power=2.0, wavelet_type="db1", levels=1, weighting="boosted",
    )

    images = torch.randn(2, 3, 16, 16)
    noise = torch.randn_like(images)
    timesteps = torch.randint(0, 50, (2,), dtype=torch.long)
    noisy = scheduler.add_noise(images, noise, timesteps)

    pred = unet(noisy, timesteps).sample
    loss = loss_fn(pred, noise, timesteps=timesteps)
    assert torch.isfinite(loss)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
