"""Loss zoo: a single :func:`get_loss_function` factory + the AWWL module."""

from __future__ import annotations

from awwl.losses.adaptive_wavelet import AdaptiveWaveletLoss
from awwl.losses.factory import get_loss_function, known_losses
from awwl.losses.perceptual import PerceptualLoss

__all__ = [
    "AdaptiveWaveletLoss",
    "PerceptualLoss",
    "get_loss_function",
    "known_losses",
]
