"""Adaptive Wavelet-Weighted Loss (AWWL).

Implements the loss from Thao *et al.*, "Adaptive Weight Wavelet Loss: A
Dynamic Frequency-Aware Loss Function for Diffusion Model Training" (CITA
2026), eqs. (4)–(7). Decomposes prediction and target with a 2-D discrete
wavelet transform and weights the low-frequency (LL) and high-frequency
(LH/HL/HH) bands by a function of the diffusion-noise level σ. The intuition:
at large σ the denoising target is dominated by global structure (LL), at
small σ it is dominated by fine detail (high-frequency bands). The weighting
smoothly follows that schedule per-sample.

Two weighting schemes are exposed:

* ``normalized`` *(default; matches paper eqs. 4-5)*::

    w_LL  = α · σ^p / (σ^p + (1-σ)^p)
    w_det = (1-α) · (1-σ)^p / (σ^p + (1-σ)^p)

  The denominator is the same for both; the two weights sum to a constant.
  Every reported number in the paper (Tables 1-3) was produced under this
  formula.

* ``boosted`` *(alternative)*: ``w_ll = α · (0.5 + σ^p)``, symmetric for
  ``w_det``. Weights never collapse to zero, so neither band is ignored at
  any timestep. Useful for ablations but not the published configuration.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from torch import nn

logger = logging.getLogger(__name__)

WeightingScheme = Literal["normalized", "boosted"]


class AdaptiveWaveletLoss(nn.Module):
    """Per-sample wavelet-band-weighted regression loss.

    Args:
        levels: Number of dyadic decomposition levels ``J``. Each extra level
            halves the spatial resolution of the LL band.
        wavelet_type: Any name accepted by :class:`pytorch_wavelets.DWTForward`
            (e.g. ``"db1"``, ``"db4"``, ``"bior1.3"``).
        alpha: Weight on the LL band. ``1 - alpha`` is split across the
            high-frequency bands of every level.
        power: Exponent ``p`` controlling how sharply the weighting follows σ.
        weighting: Either ``"normalized"`` (default; the paper's eqs. 4-5)
            or ``"boosted"`` (an alternative explored in ablations). See
            module docstring.
        loss_func: Pointwise loss applied per band. Must accept
            ``reduction="none"`` and return a tensor matching its inputs.
            Defaults to :func:`torch.nn.functional.mse_loss`.
    """

    def __init__(
        self,
        *,
        levels: int = 1,
        wavelet_type: str = "db1",
        alpha: float = 0.8,
        power: float = 2.0,
        weighting: WeightingScheme = "normalized",
        loss_func=F.mse_loss,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if power < 0:
            raise ValueError(f"power must be >= 0, got {power}")
        if weighting not in ("normalized", "boosted"):
            raise ValueError(f"weighting must be 'normalized' or 'boosted', got {weighting!r}")

        self.levels = levels
        self.wavelet_type = wavelet_type
        self.alpha = alpha
        self.power = power
        self.weighting = weighting
        self.loss_func = loss_func
        self.dwt = DWTForward(J=levels, wave=wavelet_type, mode="zero")

    def _adaptive_weights(self, sigmas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(w_LL, w_details)`` shaped like ``sigmas`` (broadcastable to NCHW)."""
        if self.weighting == "normalized":
            # Paper eqs. (4)-(5).
            w_ll_raw = sigmas.pow(self.power)
            w_high_raw = (1.0 - sigmas).pow(self.power)
            denom = w_ll_raw + w_high_raw + 1e-8
            w_ll = w_ll_raw / denom
            w_high = w_high_raw / denom
        else:  # boosted
            w_ll = 0.5 + sigmas.pow(self.power)
            w_high = 0.5 + (1.0 - sigmas).pow(self.power)
        return self.alpha * w_ll, (1.0 - self.alpha) * w_high

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the wavelet-weighted loss.

        Args:
            model_pred: ``(N, C, H, W)`` model output (e.g. predicted noise).
            target: Ground-truth tensor with the same shape as ``model_pred``.
            sigmas: ``(N,)`` per-sample noise level, expected in ``[0, 1]``
                (typically ``sqrt(1 - alphas_cumprod[t])``).
        """
        # The DWT op is fp16-unfriendly under autocast; force fp32 here.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred = model_pred.to(torch.float32)
            tgt = target.to(torch.float32)
            sig = sigmas.to(torch.float32).view(-1, 1, 1, 1)

            self.dwt = self.dwt.to(pred.device)
            w_ll, w_high = self._adaptive_weights(sig)

            pred_ll, pred_h = self.dwt(pred)
            target_ll, target_h = self.dwt(tgt)

            loss_ll = (w_ll * self.loss_func(pred_ll, target_ll, reduction="none")).mean()
            total = loss_ll

            for level_idx in range(self.levels):
                p_lh, p_hl, p_hh = torch.unbind(pred_h[level_idx], dim=2)
                t_lh, t_hl, t_hh = torch.unbind(target_h[level_idx], dim=2)
                p_details = torch.cat([p_lh, p_hl, p_hh], dim=1)
                t_details = torch.cat([t_lh, t_hl, t_hh], dim=1)
                detail_loss = (
                    w_high * self.loss_func(p_details, t_details, reduction="none")
                ).mean()
                total = total + detail_loss

        return total
