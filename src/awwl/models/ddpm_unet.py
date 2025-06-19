"""Unconditional pixel-space DDPM UNet builder (used by AWWL-Diff)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from diffusers import UNet2DModel

logger = logging.getLogger(__name__)


def build_ddpm_unet(
    *,
    image_size: int = 32,
    in_channels: int = 3,
    out_channels: int = 3,
    layers_per_block: int = 2,
    block_out_channels: tuple[int, ...] = (128, 256, 256, 256),
    down_block_types: tuple[str, ...] = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    ),
    up_block_types: tuple[str, ...] = (
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
) -> UNet2DModel:
    """Build a fresh ``UNet2DModel`` matching the AWWL-Diff CIFAR-10 recipe.

    The defaults reproduce the architecture used in the original ``train.py``
    and ``train_cifar10.py`` ablations.
    """
    return UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )


def load_or_build_ddpm_unet(
    *,
    weights_path: str | Path | None,
    builder_kwargs: dict[str, Any],
) -> UNet2DModel:
    """Either resume from a saved checkpoint or build a fresh UNet.

    A saved checkpoint may either be a bare UNet folder or a
    ``DDPMPipeline``-style folder with the UNet in a ``unet/`` subdir; both
    layouts are tried.
    """
    if weights_path is None:
        return build_ddpm_unet(**builder_kwargs)
    p = Path(weights_path)
    logger.info("resuming DDPM UNet from %s", p)
    try:
        return UNet2DModel.from_pretrained(p, subfolder="unet")
    except OSError:
        return UNet2DModel.from_pretrained(p)
