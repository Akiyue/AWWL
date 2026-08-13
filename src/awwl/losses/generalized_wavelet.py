"""The generalised wavelet loss: one body, pluggable weighting and transform.

:class:`~awwl.losses.adaptive_wavelet.AdaptiveWaveletLoss` stays frozen as the
published objective. This is its successor, which keeps the same structure —
decompose the residual, weight the sub-bands, sum — while making the three
fixed choices in the original variable:

===================  ==========================================  ==========
What was fixed       Now supplied by                             Direction
===================  ==========================================  ==========
one shared detail    :class:`~awwl.losses.weighting.RationalWeighting`  A1
weight, one level    with per-level and per-direction schedules
weights depend       :class:`~awwl.losses.weighting.SpatialWeighting`   A2
only on the timestep
weights are          :class:`~awwl.losses.weighting.UncertaintyWeighting`  A3
hand-designed        or :class:`~awwl.losses.gradnorm.GradNormBalancer`
the basis is fixed   :class:`~awwl.losses.lifting.LiftingWavelet`       A4
===================  ==========================================  ==========

Because these are independent axes, they compose: a learned, per-subband,
spatially-adaptive weighting over a learned basis is
``weighting_strategy="uncertainty", spatial=True, transform="lifting"``. That is the
"spatio-temporal-spectral adaptive" objective the extension notes describe,
and the pieces can also be turned on one at a time so each one's contribution
is separately attributable.

Any of these can also make things worse — a learned weighting has more ways to
fail than a two-parameter curve, and a learned basis gives up orthonormality.
Each is therefore built to degrade to the published objective: zero
``strength`` recovers the fixed spatial behaviour, ``learnable=False`` freezes
the basis at Haar, and the rational weighting with no overrides is eqs.
(4)-(5) exactly.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from awwl.losses.gradnorm import GradNormBalancer
from awwl.losses.lifting import LiftingWavelet
from awwl.losses.weighting import (
    LL,
    BandWeighting,
    build_weighting,
    min_snr_weight,
    split_key,
)

logger = logging.getLogger(__name__)

TransformName = Literal["dwt", "lifting"]


class GeneralizedWaveletLoss(nn.Module):
    """Sub-band-weighted regression loss with a pluggable weighting strategy.

    Args:
        levels: Decomposition depth ``J``.
        transform: ``"dwt"`` for a fixed basis via ``pytorch_wavelets``, or
            ``"lifting"`` for the learnable basis of :mod:`awwl.losses.lifting`.
        wavelet_type: Basis name, ``"dwt"`` only.
        dwt_mode: Padding mode, ``"dwt"`` only.
        weighting_strategy: Either a ready :class:`BandWeighting` or a name
            understood by :func:`~awwl.losses.weighting.build_weighting`.
            Deliberately *not* called ``weighting``: the published config uses
            ``loss.weighting`` for the scheme name of the old objective, and a
            merged YAML would otherwise hand this the string ``"normalized"``.
        weighting_kwargs: Forwarded when ``weighting_strategy`` is a name.
        spatial: Wrap the weighting in :class:`SpatialWeighting` (A2).
        normalize_weights: Hold the summed weight at 1 for every σ, so the
            strategy redistributes gradient across bands without also
            rescaling it across timesteps. Strongly recommended for any new
            strategy — this is the confound that makes the published ``α``
            ambiguous.
        gradnorm: Balance the sub-bands with GradNorm instead of the weighting
            strategy's own output (A3). Requires the trainer to call
            :meth:`gradnorm_loss` each step.
        snr_gamma: Compose with Min-SNR-γ timestep weighting (B2). The two act
            on different axes — Min-SNR across timesteps, the wavelet
            weighting across spatial frequencies — so this tests whether the
            frequency schedule contributes anything beyond the implicit
            timestep reweighting it already performs. Applied after
            normalisation, which would otherwise cancel it. ``None`` disables.
        lifting_kernel_size / learnable_basis: Options for ``"lifting"``.
        loss_func: Pointwise loss per band, ``reduction="none"``.
    """

    def __init__(
        self,
        *,
        levels: int = 1,
        transform: TransformName = "dwt",
        wavelet_type: str = "db1",
        dwt_mode: str = "zero",
        weighting_strategy: BandWeighting | str = "rational",
        weighting_kwargs: dict | None = None,
        spatial: bool = False,
        normalize_weights: bool = True,
        normalize_scale: float = 1.0,
        gradnorm: bool = False,
        gradnorm_asymmetry: float = 1.5,
        snr_gamma: float | None = None,
        lifting_kernel_size: int = 3,
        learnable_basis: bool = True,
        loss_func=F.mse_loss,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")

        self.levels = levels
        self.transform_name = transform
        self.snr_gamma = float(snr_gamma) if snr_gamma is not None else None
        self.loss_func = loss_func

        if transform == "dwt":
            from pytorch_wavelets import DWTForward

            self.transform = DWTForward(J=levels, wave=wavelet_type, mode=dwt_mode)
        elif transform == "lifting":
            if levels != 1:
                raise ValueError("the lifting transform currently supports levels=1 only")
            self.transform = LiftingWavelet(
                kernel_size=lifting_kernel_size, learnable=learnable_basis
            )
        else:
            raise ValueError(f"transform must be 'dwt' or 'lifting', got {transform!r}")

        if isinstance(weighting_strategy, BandWeighting):
            self.weighting = weighting_strategy
        else:
            kwargs = dict(weighting_kwargs or {})
            name = (
                "spatial" if (spatial and weighting_strategy == "rational") else weighting_strategy
            )
            self.weighting = build_weighting(
                name,
                levels=levels,
                normalize=normalize_weights,
                normalize_scale=normalize_scale,
                **kwargs,
            )
            if spatial and weighting_strategy != "rational":
                from awwl.losses.weighting import SpatialWeighting

                self.weighting = SpatialWeighting(
                    self.weighting,
                    strength=float(kwargs.get("strength", 1.0)),
                    normalize=normalize_weights,
                )

        self.keys = self.weighting.keys
        self.gradnorm = (
            GradNormBalancer(n_tasks=len(self.keys), asymmetry=gradnorm_asymmetry)
            if gradnorm
            else None
        )
        self._last_band_losses: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------ decompose

    def decompose(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split ``(N,C,H,W)`` into a dict of sub-band coefficients."""
        ll, highs = self.transform(x)
        bands = {LL: ll}
        for level_idx, high in enumerate(highs, start=1):
            lh, hl, hh = torch.unbind(high, dim=2)
            bands[f"lh_{level_idx}"] = lh
            bands[f"hl_{level_idx}"] = hl
            bands[f"hh_{level_idx}"] = hh
        return bands

    # -------------------------------------------------------------- forward

    def band_losses(
        self, model_pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Unweighted mean squared error per sub-band."""
        pred_bands = self.decompose(model_pred)
        target_bands = self.decompose(target)
        return {
            key: self.loss_func(pred_bands[key], target_bands[key], reduction="none")
            for key in self.keys
        }

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the weighted loss.

        Args:
            model_pred: ``(N, C, H, W)`` predicted noise.
            target: Ground truth, same shape.
            sigmas: ``(N,)`` noise level in ``[0, 1]``.
        """
        # The transforms are fp16-unfriendly under autocast; force fp32.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred = model_pred.to(torch.float32)
            tgt = target.to(torch.float32)
            sig = sigmas.to(torch.float32).view(-1, 1, 1, 1)

            self.transform = self.transform.to(pred.device)
            pred_bands = self.decompose(pred)
            target_bands = self.decompose(tgt)

            per_element = {
                key: self.loss_func(pred_bands[key], target_bands[key], reduction="none")
                for key in self.keys
            }

            if self.gradnorm is not None:
                # GradNorm owns the weights; the strategy is bypassed.
                scalars = [per_element[key].mean() for key in self.keys]
                self._last_band_losses = dict(zip(self.keys, scalars, strict=True))
                return self.gradnorm.combine(scalars)

            # The activity map keys off the *target* coefficients, never the
            # prediction, so the model cannot lower its own weight by shrinking
            # its outputs.
            weights = self.weighting(sig, references=target_bands)
            if self.snr_gamma is not None:
                # After normalisation: a global per-timestep factor would be
                # divided straight back out if applied before it.
                snr = min_snr_weight(sig, self.snr_gamma)
                weights = {k: v * snr for k, v in weights.items()}
            self._last_band_losses = {k: v.mean() for k, v in per_element.items()}

            total = pred.new_zeros(())
            for key in self.keys:
                total = total + (weights[key] * per_element[key]).mean()
            return total + self.weighting.penalty().to(total.device)

    # ------------------------------------------------------------- gradnorm

    def gradnorm_loss(self, shared_parameters) -> torch.Tensor | None:
        """GradNorm's weight objective for the most recent forward pass.

        Returns ``None`` when GradNorm is off. Call before the main
        ``backward`` — it builds a second-order graph through the shared
        parameters.
        """
        if self.gradnorm is None or self._last_band_losses is None:
            return None
        losses = [self._last_band_losses[key] for key in self.keys]
        return self.gradnorm.gradnorm_loss(losses, shared_parameters)

    def after_optimizer_step(self) -> None:
        """Renormalise GradNorm's weights. No-op otherwise."""
        if self.gradnorm is not None:
            self.gradnorm.renormalise()

    # ----------------------------------------------------------- diagnostics

    @torch.no_grad()
    def weight_profile(self, sigmas: torch.Tensor) -> dict[str, list[float]]:
        """Per-band weights across σ — for plotting the learned curriculum.

        With a learned strategy this is the headline figure: it shows whether
        the network rediscovered coarse-to-fine on its own or chose something
        else entirely.
        """
        if self.gradnorm is not None:
            weights = self.gradnorm.weights.detach()
            return {key: [float(weights[i])] for i, key in enumerate(self.keys)}
        profile = self.weighting.describe(sigmas)
        return {key: [float(v) for v in profile[key]] for key in self.keys}

    def band_summary(self) -> dict[str, dict[str, int | str]]:
        """Which direction and level each key refers to."""
        return {
            key: dict(zip(("direction", "level"), split_key(key), strict=True))
            for key in self.keys
        }
