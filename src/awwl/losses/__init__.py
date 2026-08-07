"""Loss zoo: a single :func:`get_loss_function` factory + the wavelet losses.

``adaptive_wavelet`` is the published objective and is frozen. The
``wavelet_*`` family shares one implementation
(:class:`GeneralizedWaveletLoss`) and varies a single axis each: per-subband
schedules, spatial adaptivity, learned weights, or a learned basis.
"""

from __future__ import annotations

from awwl.losses.adaptive_wavelet import AdaptiveWaveletLoss
from awwl.losses.factory import (
    get_loss_function,
    known_losses,
    loss_module,
    trainable_loss_parameters,
)
from awwl.losses.generalized_wavelet import GeneralizedWaveletLoss
from awwl.losses.gradnorm import GradNormBalancer
from awwl.losses.lifting import LiftingWavelet
from awwl.losses.perceptual import PerceptualLoss
from awwl.losses.weighting import (
    BandWeighting,
    RationalWeighting,
    SpatialWeighting,
    UncertaintyWeighting,
    build_weighting,
)

__all__ = [
    "AdaptiveWaveletLoss",
    "BandWeighting",
    "GeneralizedWaveletLoss",
    "GradNormBalancer",
    "LiftingWavelet",
    "PerceptualLoss",
    "RationalWeighting",
    "SpatialWeighting",
    "UncertaintyWeighting",
    "build_weighting",
    "get_loss_function",
    "known_losses",
    "loss_module",
    "trainable_loss_parameters",
]
