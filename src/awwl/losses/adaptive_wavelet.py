"""Adaptive Wavelet-Weighted Loss (AWWL).

Implements the loss from Thao *et al.*, "Adaptive Weight Wavelet Loss: A
Dynamic Frequency-Aware Loss Function for Diffusion Model Training" (CITA
2026), eqs. (4)-(7). Decomposes prediction and target with a 2-D discrete
wavelet transform and weights the low-frequency (LL) and high-frequency
(LH/HL/HH) bands by a function of the diffusion-noise level σ.

Because the DWT is *linear*, ``DWT(ε̂) - DWT(ε) = DWT(ε̂ - ε)``: the bands
being weighted are bands of the **prediction residual**, not of any image.
For an orthonormal basis (``db1``, ``db4``, …) Parseval then gives

    pixel_MSE  =  ¼ · ( mean(LL²) + mean(LH²) + mean(HL²) + mean(HH²) )

so an *unweighted* wavelet loss is exactly the pixel MSE, and AWWL is a
reallocation of the same error budget across orthogonal sub-bands. See
``scripts/verify_loss_math.py``, which checks this identity numerically.


Weighting schemes
-----------------

* ``normalized`` *(default; matches paper eqs. 4-5)*::

    w_LL  = α · σ^p / (σ^p + (1-σ)^p)
    w_det = (1-α) · (1-σ)^p / (σ^p + (1-σ)^p)

  Every reported number in the paper (Tables 1-3) was produced under this
  formula with the default reductions below.

* ``boosted`` *(alternative)*: ``w_ll = α · (0.5 + σ^p)``, symmetric for
  ``w_det``. Weights never collapse to zero, so neither band is ignored at
  any timestep. Useful for ablations but not the published configuration.


Total-weight confound (``normalize_weights``)
---------------------------------------------

The paper's eqs. (4)-(5) share a denominator but do **not** sum to a
constant. Their effective total is

    w_LL + k·w_det  →  α       as σ → 1   (high noise)
                    →  1 - α   as σ → 0   (low noise)

with ``k`` the detail multiplicity (below). So ``α`` does two things at
once: it sets the LL/detail balance *and* it rescales the overall gradient
magnitude across the noise schedule — at ``α=0.8`` early timesteps receive
4× the gradient of late ones, at ``α=0.2`` the reverse. That second effect
is a global timestep reweighting of exactly the kind Min-SNR / P2 apply,
and it is a confound for any claim that ``α`` selects a *frequency* trade-off.

Passing ``normalize_weights=True`` divides both weights by ``w_LL + k·w_det``
so the total is constant at every σ, isolating the frequency balance from the
gradient-magnitude schedule. This is off by default: the published runs did
not use it.

``normalize_scale`` sets that constant, and it matters more than it looks.
The published total averages **0.5** over the schedule, so normalising to the
default 1.0 also doubles the mean gradient magnitude — which is an effective
learning-rate change riding along with the ablation, and makes any difference
uninterpretable. Set ``normalize_scale=0.5`` to hold the average fixed so that
only the *shape* of the weighting changes. That is the comparison worth
reporting; ``1.0`` conflates two things at once.


Reductions
----------

``detail_reduction`` selects how the three detail bands of a level combine:

* ``"mean"`` *(default — what produced the paper's numbers)*: the three
  bands are averaged.
* ``"sum"``: the three bands are summed, which is what eq. (7) literally
  writes (``L_details = ‖·‖² + ‖·‖² + ‖·‖²``).

The two differ by a factor of 3 on the detail term, i.e. they correspond to
different effective ``α``. ``"mean"`` is kept as the default so existing
configs reproduce; use ``"sum"`` when you want the published equation.

``level_reduction`` does the same across decomposition levels when
``levels > 1``: ``"sum"`` (default) makes the detail term grow with the
number of levels, ``"mean"`` keeps its magnitude level-independent.
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
Reduction = Literal["mean", "sum"]


class AdaptiveWaveletLoss(nn.Module):
    """Per-sample wavelet-band-weighted regression loss.

    Args:
        levels: Number of dyadic decomposition levels ``J``. Each extra level
            halves the spatial resolution of the LL band.
        wavelet_type: Any name accepted by :class:`pytorch_wavelets.DWTForward`
            (e.g. ``"db1"``, ``"db4"``, ``"bior1.3"``). Orthonormal families
            (``db*``, ``sym*``, ``coif*``) satisfy the Parseval identity above;
            biorthogonal ones (``bior*``) do not.
        alpha: Weight on the LL band. ``1 - alpha`` goes to the detail bands.
        power: Exponent ``p`` controlling how sharply the weighting follows σ.
        weighting: Either ``"normalized"`` (default; the paper's eqs. 4-5)
            or ``"boosted"``. See module docstring.
        normalize_weights: Rescale so ``w_LL + k·w_det ≡ 1`` at every σ,
            removing the total-gradient-magnitude confound. Default ``False``
            (the published behaviour).
        detail_reduction: ``"mean"`` (default) or ``"sum"`` over the three
            detail bands of a level. ``"sum"`` matches paper eq. (7).
        level_reduction: ``"sum"`` (default) or ``"mean"`` over levels.
        dwt_mode: Padding mode handed to :class:`DWTForward`. The published
            runs used ``"zero"``.
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
        normalize_weights: bool = False,
        normalize_scale: float = 1.0,
        detail_reduction: Reduction = "mean",
        level_reduction: Reduction = "sum",
        dwt_mode: str = "zero",
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
        if detail_reduction not in ("mean", "sum"):
            raise ValueError(f"detail_reduction must be 'mean' or 'sum', got {detail_reduction!r}")
        if level_reduction not in ("mean", "sum"):
            raise ValueError(f"level_reduction must be 'mean' or 'sum', got {level_reduction!r}")

        self.levels = levels
        self.wavelet_type = wavelet_type
        self.alpha = alpha
        self.power = power
        self.weighting = weighting
        self.normalize_weights = normalize_weights
        self.normalize_scale = float(normalize_scale)
        self.detail_reduction = detail_reduction
        self.level_reduction = level_reduction
        self.dwt_mode = dwt_mode
        self.loss_func = loss_func
        self.dwt = DWTForward(J=levels, wave=wavelet_type, mode=dwt_mode)

    @property
    def detail_multiplicity(self) -> float:
        """``k``: how many band-means the detail term effectively contributes.

        Used to normalise the total weight. With the defaults
        (``detail_reduction="mean"``, ``level_reduction="sum"``, one level)
        this is 1, so ``w_LL + w_det`` is the quantity being normalised —
        the sum written in the paper.
        """
        per_level = 3.0 if self.detail_reduction == "sum" else 1.0
        n_levels = float(self.levels) if self.level_reduction == "sum" else 1.0
        return per_level * n_levels

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

        w_ll = self.alpha * w_ll
        w_high = (1.0 - self.alpha) * w_high

        if self.normalize_weights:
            total = w_ll + self.detail_multiplicity * w_high + 1e-8
            w_ll = self.normalize_scale * w_ll / total
            w_high = self.normalize_scale * w_high / total
        return w_ll, w_high

    @torch.no_grad()
    def weights_at(self, sigmas: torch.Tensor) -> dict[str, torch.Tensor]:
        """Diagnostic: the weights this module would apply at each σ.

        Returns a dict with ``w_ll``, ``w_det`` and ``total``
        (``w_ll + k·w_det``, the effective gradient scale). Used by
        ``scripts/verify_loss_math.py`` and the weighting plots; performs no
        DWT and touches no image data.
        """
        sig = sigmas.to(torch.float32).reshape(-1)
        w_ll, w_high = self._adaptive_weights(sig)
        return {
            "sigma": sig,
            "w_ll": w_ll,
            "w_det": w_high,
            "total": w_ll + self.detail_multiplicity * w_high,
        }

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

            total = (w_ll * self.loss_func(pred_ll, target_ll, reduction="none")).mean()

            detail_total = pred.new_zeros(())
            for level_idx in range(self.levels):
                p_lh, p_hl, p_hh = torch.unbind(pred_h[level_idx], dim=2)
                t_lh, t_hl, t_hh = torch.unbind(target_h[level_idx], dim=2)
                p_details = torch.cat([p_lh, p_hl, p_hh], dim=1)
                t_details = torch.cat([t_lh, t_hl, t_hh], dim=1)
                per_band = (w_high * self.loss_func(p_details, t_details, reduction="none")).mean()
                # `.mean()` over the concatenation already averages the three
                # bands; eq. (7) sums them, hence the ×3.
                if self.detail_reduction == "sum":
                    per_band = per_band * 3.0
                detail_total = detail_total + per_band

            if self.level_reduction == "mean":
                detail_total = detail_total / self.levels

            total = total + detail_total

        return total
