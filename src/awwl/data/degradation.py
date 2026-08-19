"""On-the-fly degradation for image-restoration training and evaluation.

The restoration reframing studies a network that maps a degraded image back
to its clean counterpart. Degradation is applied in the training loop from a
per-sample severity parameter (sigma), so a single config drives every loss
arm identically and the *same* sigma that produced a sample can be handed to
the loss, the model's conditioning, and the evaluation.

Why Gaussian noise in particular: sigma is the natural severity variable. It
has a precise meaning (the standard deviation of the additive noise, relative
to the ``[-1, 1]`` image range), it is sampled per image, and it maps one-to-one
onto the noise magnitude that ``AdaptiveWaveletLoss`` weights its bands by.
AWWL's claim is that high noise calls for structure (LL) and low noise for
detail (HF); denoising is the setting where that trade-off is the objective
itself rather than something inferred through FID.
"""

from __future__ import annotations

import torch


def sample_sigmas(
    n: int,
    *,
    sigma_min: float = 0.05,
    sigma_max: float = 0.5,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``n`` noise levels uniformly from ``[sigma_min, sigma_max]``.

    The range is deliberately wide: at ``sigma_min`` the target is nearly
    clean (the loss should focus on texture), at ``sigma_max`` it is heavily
    corrupted (the loss should focus on structure). Training on the whole
    range is what makes the adaptive weighting testable at all — a fixed sigma
    would collapse the schedule to a single point.
    """
    if sigma_min < 0:
        raise ValueError(f"sigma_min must be >= 0, got {sigma_min}")
    if sigma_max <= sigma_min:
        raise ValueError(f"sigma_max must be > sigma_min, got {sigma_min}..{sigma_max}")
    return torch.empty(n, device=device, dtype=torch.float32).uniform_(
        sigma_min, sigma_max, generator=generator
    )


def add_noise(images: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Add Gaussian noise of per-sample standard deviation ``sigmas``.

    Args:
        images: ``(N, C, H, W)`` in ``[-1, 1]``.
        sigmas: ``(N,)`` noise levels, normalised to the image range. A sigma
            of ``0.5`` means a noise standard deviation of a quarter of the
            full ``[-1, 1]`` dynamic range.
    """
    if images.ndim != 4:
        raise ValueError(f"images must be NCHW, got shape {tuple(images.shape)}")
    noise = torch.randn_like(images) * sigmas.view(-1, 1, 1, 1)
    return (images + noise).clamp(-1.0, 1.0)


def sigma_to_timesteps(
    sigmas: torch.Tensor,
    *,
    num_timesteps: int,
    sigma_min: float = 0.05,
    sigma_max: float = 0.5,
) -> torch.Tensor:
    """Map a degradation level onto a ``UNet2DModel`` timestep index.

    The UNet's time embedding gives the network a way to be told how badly a
    sample is corrupted, which a plain denoiser would otherwise have to
    reverse-engineer from the input alone. The mapping is linear in the
    sigma fraction so the conditioning matches the degradation distribution.
    """
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
    fraction = (sigmas - sigma_min) / (sigma_max - sigma_min)
    return (fraction.clamp(0.0, 1.0) * (num_timesteps - 1)).round().long()
