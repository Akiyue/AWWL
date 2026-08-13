"""Band-weighting strategies for the wavelet loss.

The published AWWL weights exactly two things — the LL band and "the details"
lumped together — with one scalar ``α`` and one scalar ``p``, both hand-tuned
per dataset. This module generalises that into pluggable strategies so the
loss body stays fixed while the weighting varies:

* :class:`RationalWeighting` — eqs. (4)-(5), optionally with a **separate
  ``α`` and ``p`` per decomposition level and per direction** (A1). Fig. 1 of
  the paper illustrates a 3-level decomposition while the method uses one
  level and one shared detail weight; this closes that gap and makes the
  question "does the diagonal band HH need a steeper schedule than LH/HL?"
  directly answerable.
* :class:`SpatialWeighting` — a modulation layer that makes the weight depend
  on **where** in the image the error is, not only on the timestep (A2).
* :class:`UncertaintyWeighting` — **learns** the weights instead of hand-
  designing them (A3), either as static per-band uncertainties (Kendall et
  al.) or as a small network conditioned on σ, which learns the coarse-to-fine
  curriculum itself and removes ``α`` and ``p`` entirely.

Every strategy returns a dict keyed by band name — ``"ll"`` plus
``"lh_1"``, ``"hl_1"``, ``"hh_1"``, ``"lh_2"`` … — so the loss body never has
to know which strategy it is talking to.

**Normalisation.** All strategies accept ``normalize``, which rescales so the
weights sum to 1 at every σ. This is not cosmetic: without it the total weight
varies across the noise schedule, which means the weighting silently rescales
gradient magnitude per timestep as well as redistributing it across bands.
That confound is what makes the published ``α`` uninterpretable (see
``scripts/verify_loss_math.py``); any new strategy that reintroduces it would
be just as hard to draw conclusions from, so normalisation is available to
every one of them.
"""

from __future__ import annotations

import inspect
import logging
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

DIRECTIONS = ("lh", "hl", "hh")
LL = "ll"


def min_snr_weight(sigmas: torch.Tensor, gamma: float) -> torch.Tensor:
    """Min-SNR-γ timestep weight, expressed in terms of σ.

    With ``σ = sqrt(1 - ᾱ_t)`` the signal-to-noise ratio is
    ``SNR = ᾱ_t / (1 - ᾱ_t) = (1 - σ²) / σ²``, and Min-SNR scales each
    timestep's loss by ``min(SNR, γ) / SNR``. That leaves high-noise steps
    untouched and progressively damps the low-noise ones, whose SNR would
    otherwise let them dominate the gradient.

    This exists so AWWL and Min-SNR can be **combined**. The two act on
    different axes — Min-SNR redistributes across *timesteps*, the wavelet
    weighting across *spatial frequencies* — so composing them is a real
    experiment rather than a redundancy, and it is the concrete test of
    whether the paper's schedule adds anything beyond an implicit timestep
    reweighting. Applied *after* any normalisation, since normalising would
    otherwise cancel a global per-timestep factor.
    """
    sigma_sq = sigmas.clamp(1e-4, 1 - 1e-4).pow(2)
    snr = (1.0 - sigma_sq) / sigma_sq
    return snr.clamp(max=gamma) / snr


def band_keys(levels: int) -> list[str]:
    """Names of every sub-band produced by an ``levels``-deep decomposition."""
    keys = [LL]
    for level in range(1, levels + 1):
        keys += [f"{d}_{level}" for d in DIRECTIONS]
    return keys


def split_key(key: str) -> tuple[str, int]:
    """``"hh_2"`` -> ``("hh", 2)``; ``"ll"`` -> ``("ll", 0)``."""
    if key == LL:
        return LL, 0
    direction, _, level = key.partition("_")
    return direction, int(level)


class BandWeighting(nn.Module):
    """Base class: map a noise level to one weight per sub-band.

    Subclasses implement :meth:`raw_weights`. The base class handles
    normalisation and the optional extra loss term some strategies need.
    """

    def __init__(
        self, *, levels: int = 1, normalize: bool = False, normalize_scale: float = 1.0
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.levels = levels
        self.normalize = normalize
        # The published weighting averages 0.5 over the schedule. Normalising
        # to 1.0 therefore doubles the mean gradient magnitude, smuggling an
        # effective learning-rate change into what is meant to be a pure
        # shape ablation. Pass 0.5 to hold the average fixed.
        self.normalize_scale = float(normalize_scale)
        self.keys = band_keys(levels)

    def raw_weights(
        self,
        sigmas: torch.Tensor,
        *,
        references: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        sigmas: torch.Tensor,
        *,
        references: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``{band_key: weight}``, each broadcastable to ``(N,C,H,W)``."""
        weights = self.raw_weights(sigmas, references=references)
        if not self.normalize:
            return weights
        # Sum over bands, averaging away any spatial extent first so that a
        # spatially-varying strategy is normalised in the mean rather than
        # flattened back to a constant.
        total = sum(w.mean(dim=tuple(range(1, w.dim())), keepdim=True) for w in weights.values())
        total = total + 1e-8
        return {k: self.normalize_scale * w / total for k, w in weights.items()}

    def penalty(self) -> torch.Tensor:
        """Extra term added to the total loss (zero for hand-designed schemes)."""
        return torch.zeros(())

    def describe(self, sigmas: torch.Tensor) -> dict[str, torch.Tensor]:
        """Diagnostic snapshot of the weights at the given σ values."""
        with torch.no_grad():
            weights = self.forward(sigmas.reshape(-1, 1, 1, 1))
        return {k: v.reshape(-1) for k, v in weights.items()}


def _per_band_value(
    default: float,
    band_overrides: dict[str, float] | None,
    level_overrides: list[float] | dict[int, float] | None,
    key: str,
) -> float:
    """Resolve a scalar for one band: level override beats direction beats default."""
    direction, level = split_key(key)
    value = default
    if band_overrides and direction in band_overrides:
        value = float(band_overrides[direction])
    if level_overrides and level > 0:
        if isinstance(level_overrides, dict):
            if level in level_overrides:
                value = float(level_overrides[level])
        elif 0 <= level - 1 < len(level_overrides):
            value = float(level_overrides[level - 1])
    return value


class RationalWeighting(BandWeighting):
    """The paper's rational schedule, generalised to per-level / per-direction.

    With no overrides this reproduces eqs. (4)-(5) exactly: the LL band gets
    ``α·σ^p / (σ^p + (1-σ)^p)`` and every detail band shares
    ``(1-α)·(1-σ)^p / (σ^p + (1-σ)^p)``.

    Supplying ``direction_powers`` or ``level_powers`` gives each sub-band its
    own schedule — the A1 generalisation. ``direction_powers={"hh": 2.0}``
    asks the diagonal band to switch on later and more abruptly than LH/HL,
    which is the natural hypothesis given that diagonal detail is the hardest
    component to learn.

    ``share_detail_budget`` controls whether ``1-α`` is split across the detail
    bands (so adding levels does not inflate the detail term) or given to each
    in full (the published behaviour at one level).
    """

    def __init__(
        self,
        *,
        levels: int = 1,
        alpha: float = 0.2,
        power: float = 1.0,
        direction_alphas: dict[str, float] | None = None,
        direction_powers: dict[str, float] | None = None,
        level_alphas: list[float] | dict[int, float] | None = None,
        level_powers: list[float] | dict[int, float] | None = None,
        share_detail_budget: bool = False,
        normalize: bool = False,
        normalize_scale: float = 1.0,
    ) -> None:
        super().__init__(levels=levels, normalize=normalize, normalize_scale=normalize_scale)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.power = power
        self.share_detail_budget = share_detail_budget
        self._alphas = {
            k: _per_band_value(alpha, direction_alphas, level_alphas, k) for k in self.keys
        }
        self._powers = {
            k: _per_band_value(power, direction_powers, level_powers, k) for k in self.keys
        }
        self._n_detail = max(1, len(self.keys) - 1)

    def raw_weights(self, sigmas, *, references=None):
        del references
        weights: dict[str, torch.Tensor] = {}
        detail_share = self._n_detail if self.share_detail_budget else 1.0

        for key in self.keys:
            p = self._powers[key]
            a = self._alphas[key]
            low = sigmas.pow(p)
            high = (1.0 - sigmas).pow(p)
            denom = low + high + 1e-8
            if key == LL:
                weights[key] = a * low / denom
            else:
                weights[key] = (1.0 - a) * high / denom / detail_share
        return weights

    def extra_repr(self) -> str:
        return f"levels={self.levels}, alpha={self.alpha}, power={self.power}"


class SpatialWeighting(BandWeighting):
    """Make the weight depend on *where* the error is, not only on when (A2).

    Wraps another strategy and multiplies its detail weights by a per-pixel map
    derived from local coefficient activity: regions with strong edges or
    texture (large neighbouring coefficient magnitudes) are prioritised, flat
    regions de-emphasised. The map is computed from the **target**
    coefficients, never the prediction, so the weighting cannot be gamed by the
    model shrinking its own outputs.

    Two design choices keep the comparison interpretable:

    * The modulation strength scales with ``1-σ``, so it is inactive during the
      high-noise phase (where there is no meaningful spatial structure in the
      residual to key on) and strongest during the detail phase.
    * The map is renormalised to **mean 1** within each sample and band, so
      spatial reweighting redistributes gradient across positions without
      changing the total. Skipping this would reintroduce, spatially, exactly
      the total-weight confound that makes the published ``α`` ambiguous.

    Args:
        base: The strategy supplying the temporal/band weights.
        strength: Exponent on the normalised activity map. 0 disables the
            effect and recovers ``base``; larger values sharpen the preference.
        window: Side length of the box filter used to measure local activity.
        apply_to_ll: Whether the LL band is modulated too. Off by default —
            LL carries global structure, where a texture prior is not obviously
            wanted.
    """

    def __init__(
        self,
        base: BandWeighting,
        *,
        strength: float = 1.0,
        window: int = 3,
        apply_to_ll: bool = False,
        normalize: bool = False,
        normalize_scale: float = 1.0,
    ) -> None:
        super().__init__(
            levels=base.levels, normalize=normalize, normalize_scale=normalize_scale
        )
        if strength < 0:
            raise ValueError(f"strength must be >= 0, got {strength}")
        if window < 1 or window % 2 == 0:
            raise ValueError(f"window must be a positive odd integer, got {window}")
        self.base = base
        self.strength = strength
        self.window = window
        self.apply_to_ll = apply_to_ll

    def _activity_map(self, reference: torch.Tensor) -> torch.Tensor:
        """Local mean absolute coefficient, normalised to mean 1 per sample."""
        magnitude = reference.abs()
        pad = self.window // 2
        smoothed = F.avg_pool2d(magnitude, kernel_size=self.window, stride=1, padding=pad)
        mean = smoothed.mean(dim=(1, 2, 3), keepdim=True) + 1e-8
        return smoothed / mean

    def raw_weights(self, sigmas, *, references=None):
        weights = self.base.raw_weights(sigmas, references=references)
        if references is None or self.strength == 0.0:
            return weights

        # Inactive at high noise, full strength as sigma -> 0.
        exponent = self.strength * (1.0 - sigmas)

        out: dict[str, torch.Tensor] = {}
        for key, weight in weights.items():
            reference = references.get(key)
            if reference is None or (key == LL and not self.apply_to_ll):
                out[key] = weight
                continue
            activity = self._activity_map(reference)
            factor = activity.clamp_min(1e-6).pow(exponent)
            # Renormalise to mean 1 so only the spatial *distribution* changes.
            factor = factor / (factor.mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
            out[key] = weight * factor
        return out

    def penalty(self) -> torch.Tensor:
        return self.base.penalty()


class UncertaintyWeighting(BandWeighting):
    """Learn the band weights instead of hand-designing them (A3).

    Two modes, both replacing ``α`` and ``p`` with parameters fitted during
    training — which is the point: the published method needs ``α`` retuned per
    dataset (0.8 for DreamBooth, 0.2 for CIFAR-10), so it is not the
    plug-and-play objective the paper claims.

    * ``conditioned=False`` — one learnable log-variance per band, i.e. the
      homoscedastic uncertainty weighting of Kendall et al. Each band
      contributes ``L_i / (2σ_i²) + log σ_i``; the log term is what stops the
      trivial solution of sending every weight to zero. Bands the network
      cannot fit are automatically down-weighted.

    * ``conditioned=True`` — a small MLP maps the diffusion noise level to a
      log-variance per band, so the model **learns the curriculum itself**
      rather than being told to follow ``σ^p``. This is the version that makes
      the coarse-to-fine schedule an outcome instead of an assumption, and it
      is what lets the weighting adapt across datasets without a grid search.

    The learnable parameters live on this module, so they must reach the
    optimiser — :func:`awwl.losses.factory.trainable_loss_parameters` exposes
    them and the trainer adds them as their own parameter group.
    """

    def __init__(
        self,
        *,
        levels: int = 1,
        conditioned: bool = True,
        hidden: int = 32,
        init_log_var: float = 0.0,
        normalize: bool = False,
        normalize_scale: float = 1.0,
    ) -> None:
        super().__init__(levels=levels, normalize=normalize, normalize_scale=normalize_scale)
        self.conditioned = conditioned
        n_bands = len(self.keys)
        if conditioned:
            self.net = nn.Sequential(
                nn.Linear(1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, n_bands),
            )
            # Start near the unweighted objective so early training is stable.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, init_log_var)
        else:
            self.log_vars = nn.Parameter(torch.full((n_bands,), float(init_log_var)))
        self._last_log_vars: torch.Tensor | None = None

    def _log_vars_for(self, sigmas: torch.Tensor) -> torch.Tensor:
        """``(N, n_bands)`` log-variances."""
        flat = sigmas.reshape(sigmas.shape[0], -1)[:, :1]
        if self.conditioned:
            return self.net(flat)
        return self.log_vars.unsqueeze(0).expand(flat.shape[0], -1)

    def raw_weights(self, sigmas, *, references=None):
        del references
        log_vars = self._log_vars_for(sigmas)
        self._last_log_vars = log_vars
        precision = torch.exp(-log_vars)  # 1 / sigma_i^2
        return {
            key: 0.5 * precision[:, idx].reshape(-1, 1, 1, 1)
            for idx, key in enumerate(self.keys)
        }

    def penalty(self) -> torch.Tensor:
        """Kendall's ``+ log σ_i``, without which every weight collapses to zero."""
        if self._last_log_vars is None:
            return torch.zeros(())
        return 0.5 * self._last_log_vars.mean()

    def extra_repr(self) -> str:
        return f"levels={self.levels}, conditioned={self.conditioned}"


WeightingName = Literal["rational", "spatial", "uncertainty"]


def _accepted(cls: type, kwargs: dict) -> dict:
    """Keep only the kwargs ``cls.__init__`` declares, logging what was dropped.

    Loss configs are merged YAML, so ``configs/finetune.yaml`` hands every
    strategy the ``alpha`` and ``power`` it defines for the published
    objective — meaningless to a learned weighting, which has no such knobs.
    Filtering here means switching ``loss.name`` alone is enough to change
    strategy, instead of also requiring the now-irrelevant keys to be deleted
    from the config.
    """
    allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
    kept = {k: v for k, v in kwargs.items() if k in allowed}
    ignored = sorted(set(kwargs) - set(kept))
    if ignored:
        logger.info("%s ignores config key(s): %s", cls.__name__, ", ".join(ignored))
    return kept


def build_weighting(
    name: WeightingName | str,
    *,
    levels: int = 1,
    normalize: bool = False,
    normalize_scale: float = 1.0,
    **kwargs,
) -> BandWeighting:
    """Construct a weighting strategy by name.

    ``spatial`` wraps a ``rational`` base, so it accepts the rational
    arguments too (``alpha``, ``power``, ``direction_powers``, …) alongside its
    own ``strength`` / ``window``. Keys a strategy does not understand are
    dropped with a log line rather than raising — see :func:`_accepted`.
    """
    if name == "rational":
        return RationalWeighting(
            levels=levels,
            normalize=normalize,
            normalize_scale=normalize_scale,
            **_accepted(RationalWeighting, kwargs),
        )
    if name == "uncertainty":
        return UncertaintyWeighting(
            levels=levels,
            normalize=normalize,
            normalize_scale=normalize_scale,
            **_accepted(UncertaintyWeighting, kwargs),
        )
    if name == "spatial":
        spatial_kwargs = _accepted(SpatialWeighting, kwargs)
        spatial_kwargs.pop("base", None)
        base = RationalWeighting(levels=levels, **_accepted(RationalWeighting, kwargs))
        return SpatialWeighting(
            base, normalize=normalize, normalize_scale=normalize_scale, **spatial_kwargs
        )
    raise ValueError(f"unknown weighting {name!r}; known: rational, spatial, uncertainty")
