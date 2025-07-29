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
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] != 3:
            y = y.repeat(1, 3, 1, 1)
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode="bilinear", align_corners=False)

        loss = x.new_zeros(())
        for block in self.blocks:
            x, y = block(x), block(y)
            loss = loss + F.l1_loss(x, y)
        return loss
