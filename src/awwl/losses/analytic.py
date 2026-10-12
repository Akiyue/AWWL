"""Closed-form regression losses used as baselines.

Each helper returns a scalar :class:`torch.Tensor`. The diffusion-aware ones
(``vlb_loss``, ``snr_weighted_loss``) need a ``DDPMScheduler`` so they can
look up ``alphas_cumprod`` for the relevant timesteps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Charbonnier (smooth-L1-style) loss: ``mean(sqrt((p-t)^2 + eps^2))``.

    A robust alternative to L2 that down-weights large residuals less harshly
    than Huber while staying differentiable everywhere.
    """
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def kl_loss_x0(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric Gaussian KL between unit-variance distributions centred at
    ``pred`` and ``target``.

    Reduces to ``0.5 * mean((pred - target)^2)`` plus a constant when both
    log-variances are zero, but is exposed separately to mirror the original
    AWWL ablation grid.
    """
    mean1 = pred
    mean2 = target
    logvar1 = torch.zeros_like(pred)
    logvar2 = torch.zeros_like(target)
    return 0.5 * (
        logvar2
        - logvar1
        + (torch.exp(logvar1) + (mean1 - mean2).pow(2)) / torch.exp(logvar2)
        - 1
    ).mean()


def vlb_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Simplified variational lower bound weighting: ``(1 - α_t) / α_t * MSE``.

    Args:
        pred: ``(N, C, H, W)`` model output.
        target: Same shape as ``pred``.
        alphas_cumprod: 1-D tensor from the diffusion scheduler.
        timesteps: ``(N,)`` integer timesteps for indexing into
            ``alphas_cumprod``.
    """
    alpha_t = alphas_cumprod.to(pred.device)[timesteps].view(-1, 1, 1, 1)
    weight = (1 - alpha_t) / (alpha_t + 1e-8)
    return (weight * (pred - target).pow(2)).mean()


def snr_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Raw signal-to-noise-ratio weighted MSE: ``(α_t / (1 - α_t)) * MSE``.

    This is the schedule the AWWL/AWWL-Diff baselines used, kept so the
    published numbers can be reproduced. It is **not** Min-SNR-γ
    :func:`min_snr_loss` and must not be reported as such: the weight is
    unbounded as ``t -> 0`` (SNR exceeds 1e3 at the low-noise end of a linear
    schedule), so the gradient is dominated by near-clean timesteps and
    from-scratch training diverges. That divergence is a property of the
    weighting, not a bug — Min-SNR exists precisely to clamp it.
    """
    alphas_cumprod = alphas_cumprod.to(pred.device)
    snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
    snr = snr.view(-1, 1, 1, 1)
    return (snr * (pred - target).pow(2)).mean()


def min_snr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    gamma: float = 5.0,
) -> torch.Tensor:
    """Min-SNR-γ weighted MSE (Hang et al., 2023), ε-prediction form.

    The weight is ``min(SNR_t, γ) / SNR_t``: unity wherever the SNR is below
    the clamp (the high-noise end) and decaying as ``γ / SNR_t`` above it, so
    low-noise timesteps stop dominating the gradient. For ε-prediction the
    division by ``SNR_t`` is part of the definition; the x0- and v-prediction
    forms differ and neither trainer here uses them.

    Contrast :func:`snr_weighted_loss`, which applies the *unclamped* SNR and
    is a different objective entirely.

    Args:
        pred: ``(N, C, H, W)`` predicted noise.
        target: The noise that was added, same shape as ``pred``.
        alphas_cumprod: 1-D tensor from the diffusion scheduler.
        timesteps: ``(N,)`` integer timesteps indexing ``alphas_cumprod``.
        gamma: The clamp. 5.0 is the value Hang et al. recommend and use.
    """
    from awwl.losses.weighting import min_snr_weight

    alpha_t = alphas_cumprod.to(pred.device)[timesteps]
    sigma = (1.0 - alpha_t).clamp_min(0.0).sqrt()
    weight = min_snr_weight(sigma, gamma).view(-1, 1, 1, 1)
    return (weight * (pred - target).pow(2)).mean()


# Thin wrappers exposed via the factory so every loss has the same calling
# signature (model_pred, target, **kwargs) -> tensor.
def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def huber(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)
