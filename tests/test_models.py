"""Smoke tests for model factories."""

from __future__ import annotations

import torch

from awwl.models import build_ddpm_unet


def test_ddpm_unet_forward_shape():
    """A tiny DDPM UNet should round-trip a synthetic batch on CPU quickly."""
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
    x = torch.randn(2, 3, 16, 16)
    t = torch.zeros(2, dtype=torch.long)
    out = unet(x, t).sample
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
