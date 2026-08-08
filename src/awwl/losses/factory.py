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
from awwl.losses.generalized_wavelet import GeneralizedWaveletLoss
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
    # Extension losses. All share one implementation
    # (:class:`GeneralizedWaveletLoss`) and differ only in the weighting
    # strategy and transform they select, so their results are directly
    # comparable and any difference is attributable to that one change.
    "wavelet_subband",   # A1: per-level / per-direction schedules
    "wavelet_spatial",   # A2: weights vary over image position
    "wavelet_learned",   # A3: uncertainty weighting, optionally sigma-conditioned
    "wavelet_gradnorm",  # A3: GradNorm balancing across sub-bands
    "wavelet_lifting",   # A4: learnable wavelet basis
    "wavelet_minsnr",    # B2: AWWL composed with Min-SNR timestep weighting
)

# Defaults per extension loss. Each turns on exactly one axis so a sweep can
# attribute effects; combining them is done by overriding in the config.
_EXTENSION_DEFAULTS: dict[str, dict] = {
    "wavelet_subband": {"weighting_strategy": "rational"},
    "wavelet_spatial": {"weighting_strategy": "rational", "spatial": True},
    "wavelet_learned": {"weighting_strategy": "uncertainty"},
    "wavelet_gradnorm": {"weighting_strategy": "rational", "gradnorm": True},
    "wavelet_lifting": {"weighting_strategy": "rational", "transform": "lifting"},
    "wavelet_minsnr": {"weighting_strategy": "rational", "snr_gamma": 5.0},
}

# Keys consumed by GeneralizedWaveletLoss itself; everything else in the loss
# config is forwarded to the weighting strategy.
_LOSS_BODY_KEYS = frozenset(
    {
        "levels",
        "transform",
        "wavelet_type",
        "dwt_mode",
        "weighting_strategy",
        "spatial",
        "normalize_weights",
        "gradnorm",
        "gradnorm_asymmetry",
        "snr_gamma",
        "lifting_kernel_size",
        "learnable_basis",
    }
)


def get_loss_function(
    name: str,
    *,
    noise_scheduler: Any | None = None,
    **kwargs: Any,
) -> LossFn:
    """Return a unified loss callable for ``name``.

    Args:
        name: Any entry of :func:`known_losses` — the nine published
            objectives plus the ``wavelet_*`` extension family.
        noise_scheduler: Required for ``vlb``, ``snr_weighted``,
            ``adaptive_wavelet`` and every ``wavelet_*`` loss. Must expose
            ``alphas_cumprod``.
        **kwargs: Per-loss hyperparameters, normally the whole ``loss:``
            config block. ``adaptive_wavelet`` reads ``levels``,
            ``wavelet_type``, ``alpha``, ``power``, ``weighting``,
            ``normalize_weights``, ``detail_reduction``, ``level_reduction``
            and ``dwt_mode``. The ``wavelet_*`` losses split their kwargs
            between the loss body and its weighting strategy, and **ignore**
            keys the chosen strategy has no use for, so a config written for
            the published loss can select a learned one by changing
            ``loss.name`` alone.

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
            normalize_weights=bool(kwargs.get("normalize_weights", False)),
            detail_reduction=str(kwargs.get("detail_reduction", "mean")),  # type: ignore[arg-type]
            level_reduction=str(kwargs.get("level_reduction", "sum")),  # type: ignore[arg-type]
            dwt_mode=str(kwargs.get("dwt_mode", "zero")),
        )

        def _adaptive(model_pred: torch.Tensor, target: torch.Tensor, *, timesteps: torch.Tensor) -> torch.Tensor:
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(model_pred.device)
            sigmas = torch.sqrt(1.0 - alphas_cumprod[timesteps])
            return wavelet(model_pred, target, sigmas)

        _adaptive.module = wavelet
        return _adaptive

    if name in _EXTENSION_DEFAULTS:
        if noise_scheduler is None:
            raise ValueError(f"{name} requires noise_scheduler")

        settings = {**_EXTENSION_DEFAULTS[name], **kwargs}
        # `weighting` names the *published* scheme (eqs. 4-5 = "normalized").
        # RationalWeighting implements exactly that, so the key is redundant
        # here; consume it rather than letting a merged config leak it into
        # the strategy selector.
        legacy_scheme = settings.pop("weighting", None)
        if legacy_scheme == "boosted":
            raise ValueError(
                f"{name} does not implement the 'boosted' scheme; it is an ablation "
                "of the published loss. Use loss.name=adaptive_wavelet for that."
            )
        body = {k: v for k, v in settings.items() if k in _LOSS_BODY_KEYS}
        strategy_kwargs = {k: v for k, v in settings.items() if k not in _LOSS_BODY_KEYS}
        module = GeneralizedWaveletLoss(weighting_kwargs=strategy_kwargs, **body)

        def _generalized(model_pred: torch.Tensor, target: torch.Tensor, *, timesteps: torch.Tensor) -> torch.Tensor:
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(model_pred.device)
            sigmas = torch.sqrt(1.0 - alphas_cumprod[timesteps])
            return module(model_pred, target, sigmas)

        _generalized.module = module
        return _generalized

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


def loss_module(loss_fn: LossFn) -> torch.nn.Module | None:
    """The :class:`torch.nn.Module` behind a loss callable, if it has one.

    The factory returns closures so trainers need not care which losses are
    stateful; this recovers the module for the ones that are.
    """
    return getattr(loss_fn, "module", None)


def trainable_loss_parameters(loss_fn: LossFn) -> list[torch.nn.Parameter]:
    """Parameters of a loss that are learned alongside the network.

    Non-empty for the learned-weighting and learnable-basis objectives. These
    **must** be added to the optimiser — a loss whose weights never receive an
    update silently degenerates into a fixed-weight objective at its
    initialisation, which would look like a working experiment while testing
    nothing. The trainer puts them in their own parameter group so they can
    take a different learning rate, and includes them in the checkpoint so a
    resumed run does not reset the learned schedule.
    """
    module = loss_module(loss_fn)
    if module is None:
        return []
    return [p for p in module.parameters() if p.requires_grad]
