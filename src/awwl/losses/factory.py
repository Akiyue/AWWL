"""Build a loss callable from a config dict.

The returned callable always has the signature::

    loss_fn(model_pred, target, *, timesteps=None) -> torch.Tensor

so trainers don't have to special-case which losses need timesteps. Losses
that don't use ``timesteps`` simply ignore it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from awwl.core.exceptions import UnknownLossError
from awwl.losses import analytic
from awwl.losses.adaptive_wavelet import AdaptiveWaveletLoss
from awwl.losses.perceptual import PerceptualLoss

LossFn = Callable[..., torch.Tensor]

_KNOWN: tuple[str, ...] = (
    "mse",
    "l1",
    "huber",
    "charbonnier",
    "kl_x0",
    "vlb",
    "snr_weighted",
    "perceptual",
    "adaptive_wavelet",
)


def get_loss_function(
    name: str,
    *,
    noise_scheduler: Any | None = None,
    **kwargs: Any,
) -> LossFn:
    """Return a unified loss callable for ``name``.

    Args:
        name: One of: ``mse``, ``l1``, ``huber``, ``charbonnier``, ``kl_x0``,
            ``vlb``, ``snr_weighted``, ``perceptual``, ``adaptive_wavelet``.
        noise_scheduler: Required for ``vlb``, ``snr_weighted``, and
            ``adaptive_wavelet``. Must expose ``alphas_cumprod``.
        **kwargs: Per-loss hyperparameters. The wavelet loss reads ``levels``,
            ``wavelet_type``, ``alpha``, ``power``, ``weighting``.

    Raises:
        UnknownLossError: ``name`` is not a known loss type.
        ValueError: A loss that needs ``noise_scheduler`` was not given one.
    """
    if name not in _KNOWN:
        raise UnknownLossError(f"unknown loss {name!r}; known: {', '.join(_KNOWN)}")

    if name == "adaptive_wavelet":
        if noise_scheduler is None:
            raise ValueError("adaptive_wavelet requires noise_scheduler")
        wavelet = AdaptiveWaveletLoss(
            levels=int(kwargs.get("levels", 1)),
            wavelet_type=str(kwargs.get("wavelet_type", "db1")),
            alpha=float(kwargs.get("alpha", 0.8)),
            power=float(kwargs.get("power", 2.0)),
            weighting=str(kwargs.get("weighting", "normalized")),  # type: ignore[arg-type]
        )

        def _adaptive(model_pred: torch.Tensor, target: torch.Tensor, *, timesteps: torch.Tensor) -> torch.Tensor:
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(model_pred.device)
            sigmas = torch.sqrt(1.0 - alphas_cumprod[timesteps])
            return wavelet(model_pred, target, sigmas)

        return _adaptive

    if name == "perceptual":
        net = PerceptualLoss().eval()

        def _perceptual(model_pred: torch.Tensor, target: torch.Tensor, *, timesteps=None) -> torch.Tensor:
            del timesteps
            return net(model_pred, target)

        return _perceptual

    if name == "vlb":
        if noise_scheduler is None:
            raise ValueError("vlb requires noise_scheduler")

        def _vlb(model_pred: torch.Tensor, target: torch.Tensor, *, timesteps: torch.Tensor) -> torch.Tensor:
            return analytic.vlb_loss(
                model_pred, target,
                alphas_cumprod=noise_scheduler.alphas_cumprod, timesteps=timesteps,
            )

        return _vlb

    if name == "snr_weighted":
        if noise_scheduler is None:
            raise ValueError("snr_weighted requires noise_scheduler")

        def _snr(model_pred: torch.Tensor, target: torch.Tensor, *, timesteps: torch.Tensor) -> torch.Tensor:
            return analytic.snr_weighted_loss(
                model_pred, target,
                alphas_cumprod=noise_scheduler.alphas_cumprod, timesteps=timesteps,
            )

        return _snr

    # Stateless point-wise losses.
    pointwise: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "mse": analytic.mse,
        "l1": analytic.l1,
        "huber": analytic.huber,
        "charbonnier": analytic.charbonnier_loss,
        "kl_x0": analytic.kl_loss_x0,
    }
    fn = pointwise[name]

    def _stateless(model_pred: torch.Tensor, target: torch.Tensor, *, timesteps=None) -> torch.Tensor:
        del timesteps
        return fn(model_pred, target)

    return _stateless


def known_losses() -> tuple[str, ...]:
    """Return the tuple of registered loss names (for help text)."""
    return _KNOWN
