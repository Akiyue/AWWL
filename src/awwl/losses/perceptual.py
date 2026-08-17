"""VGG-16 perceptual loss used as a baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class PerceptualLoss(nn.Module):
    """Multi-scale L1 distance between intermediate VGG-16 activations.

    Inputs are resized to 224×224 and converted to 3-channel RGB if needed,
    matching the VGG receptive field. The four feature maps that are
    typically used for perceptual losses are extracted and an L1 distance
    accumulated across them.

    .. warning::

       Two things make this a poor fit for epsilon-prediction diffusion, and
       both are properties of the setting rather than of this implementation.

       *Cost.* At ``resize=True`` a 32x32 CIFAR batch is upsampled 49-fold and
       pushed through most of VGG-16 twice per step. That is a few hundred
       times the cost of the 32x32 UNet being trained, so a run using it is
       bounded by the loss rather than by the model.

       *Meaning.* The trainers call this on ``(predicted_noise, noise)``. The
       epsilon target is white Gaussian noise and carries no perceptual
       content, so VGG activations of it measure nothing the name suggests.
       Read any number it produces as a control, not as a perceptual result.

    Args:
        resize: When ``True`` (default) inputs are resized to 224×224 before
            running through VGG. Off when callers know their inputs already
            match.
    """

    def __init__(self, *, resize: bool = True) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList(
            [vgg[:4].eval(), vgg[4:9].eval(), vgg[9:16].eval(), vgg[16:23].eval()]
        )
        for p in self.parameters():
            p.requires_grad = False
        self.resize = resize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = x.device
        self.to(device)

        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        channels = x.shape[1]
        if channels not in (1, 3):
            # `repeat(1, 3, 1, 1)` turned 4-channel latents into 12 channels and
            # VGG rejected them, which is the right outcome reached the wrong
            # way. VGG features are defined on RGB; a Stable Diffusion latent
            # channel is not a colour, so there is no correct mapping to make
            # here and inventing one would put an uninterpretable number in a
            # table.
            raise ValueError(
                f"perceptual loss needs 1 or 3 channels, got {channels}. VGG features "
                "are defined on RGB images; latent channels have no perceptual reading. "
                "Use this loss in pixel space, or use a different objective."
            )
        if channels == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode="bilinear", align_corners=False)

        loss = x.new_zeros(())
        for block in self.blocks:
            x, y = block(x), block(y)
            loss = loss + F.l1_loss(x, y)
        return loss
