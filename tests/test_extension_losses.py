"""Tests for the loss extensions A1-A4.

Each axis is checked for the property that makes it a valid experiment: that
it reduces to the published objective when switched off, that its learnable
parts actually learn, and that it does not silently reintroduce the
total-weight confound the published weighting suffers from.
"""

from __future__ import annotations

import pytest
import torch

from awwl.losses import (
    GeneralizedWaveletLoss,
    GradNormBalancer,
    LiftingWavelet,
    RationalWeighting,
    SpatialWeighting,
    UncertaintyWeighting,
    get_loss_function,
    trainable_loss_parameters,
)
from awwl.losses.weighting import band_keys, build_weighting, split_key


class _StubScheduler:
    def __init__(self, num_train_timesteps: int = 1000) -> None:
        self.alphas_cumprod = torch.linspace(0.999, 0.001, num_train_timesteps)


SIGMAS = torch.tensor([0.05, 0.5, 0.95]).view(-1, 1, 1, 1)


# ------------------------------------------------------------------ plumbing


def test_band_keys_and_split():
    assert band_keys(1) == ["ll", "lh_1", "hl_1", "hh_1"]
    assert band_keys(2)[-3:] == ["lh_2", "hl_2", "hh_2"]
    assert split_key("hh_2") == ("hh", 2)
    assert split_key("ll") == ("ll", 0)


# ---------------------------------------------------------------- A1 subband


def test_rational_reproduces_paper_equations():
    """With no overrides the generalised strategy must be eqs. (4)-(5)."""
    alpha, p = 0.2, 1.0
    w = RationalWeighting(alpha=alpha, power=p)(SIGMAS)
    s = SIGMAS
    expected_ll = alpha * s.pow(p) / (s.pow(p) + (1 - s).pow(p))
    expected_det = (1 - alpha) * (1 - s).pow(p) / (s.pow(p) + (1 - s).pow(p))
    assert torch.allclose(w["ll"], expected_ll, atol=1e-6)
    for band in ("lh_1", "hl_1", "hh_1"):
        assert torch.allclose(w[band], expected_det, atol=1e-6)


def test_per_direction_powers_give_hh_its_own_schedule():
    """A1's research question: can the diagonal band switch on later?"""
    w = RationalWeighting(alpha=0.2, power=1.0, direction_powers={"hh": 3.0})(SIGMAS)
    assert torch.allclose(w["lh_1"], w["hl_1"]), "untouched directions should still match"
    assert not torch.allclose(w["hh_1"], w["lh_1"]), "hh did not get its own schedule"
    # A steeper power holds the diagonal band back while noise is still high.
    high_noise = 2
    assert w["hh_1"][high_noise] < w["lh_1"][high_noise]


def test_per_level_powers_differ_across_levels():
    w = RationalWeighting(levels=2, alpha=0.2, power=1.0, level_powers=[1.0, 3.0])(SIGMAS)
    assert not torch.allclose(w["lh_1"], w["lh_2"])


def test_share_detail_budget_keeps_detail_mass_constant():
    """Adding levels must not silently inflate the detail term."""
    one = RationalWeighting(levels=1, share_detail_budget=True)(SIGMAS)
    three = RationalWeighting(levels=3, share_detail_budget=True)(SIGMAS)
    detail_one = sum(v for k, v in one.items() if k != "ll")
    detail_three = sum(v for k, v in three.items() if k != "ll")
    assert torch.allclose(detail_one, detail_three, atol=1e-6)


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_normalize_holds_total_at_one(levels):
    """No strategy may reintroduce the published total-weight confound."""
    w = RationalWeighting(levels=levels, alpha=0.8, power=2.0, normalize=True)(SIGMAS)
    total = sum(w.values())
    assert torch.allclose(total, torch.ones_like(total), atol=1e-4)


# ---------------------------------------------------------------- A2 spatial


def _references(shape=(3, 2, 8, 8)) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    refs = {k: torch.randn(*shape) for k in band_keys(1)}
    # A strong edge in one corner of the detail bands.
    for k in ("lh_1", "hl_1", "hh_1"):
        refs[k][:, :, :4, :4] *= 8.0
    return refs


def test_spatial_weighting_prefers_active_regions():
    base = RationalWeighting(alpha=0.2, power=1.0)
    weighting = SpatialWeighting(base, strength=1.0)
    refs = _references()
    w = weighting(SIGMAS, references=refs)["hh_1"]

    low_noise = 0  # sigma = 0.05, where the modulation is strongest
    busy = w[low_noise, :, :4, :4].mean()
    flat = w[low_noise, :, 4:, 4:].mean()
    assert busy > flat, "textured region was not prioritised during the detail phase"


def test_spatial_modulation_is_inactive_at_high_noise():
    """The residual has no meaningful spatial structure early on."""
    weighting = SpatialWeighting(RationalWeighting(), strength=1.0)
    w = weighting(SIGMAS, references=_references())["hh_1"]
    spread_low_noise = w[0].std()
    spread_high_noise = w[2].std()
    assert spread_high_noise < spread_low_noise


def test_spatial_preserves_the_mean_weight():
    """Spatial reweighting must redistribute, not rescale."""
    base = RationalWeighting(alpha=0.2, power=1.0)
    plain = base(SIGMAS)["hh_1"]
    spatial = SpatialWeighting(base, strength=2.0)(SIGMAS, references=_references())["hh_1"]
    assert torch.allclose(spatial.mean(dim=(1, 2, 3)), plain.reshape(-1), rtol=1e-4)


def test_zero_strength_recovers_the_base_strategy():
    base = RationalWeighting(alpha=0.2, power=1.0)
    w = SpatialWeighting(base, strength=0.0)(SIGMAS, references=_references())
    for key, value in base(SIGMAS).items():
        assert torch.allclose(w[key], value)


# ---------------------------------------------------------------- A3 learned


def test_uncertainty_weighting_has_learnable_parameters():
    for conditioned in (False, True):
        weighting = UncertaintyWeighting(conditioned=conditioned)
        assert list(weighting.parameters()), "nothing to learn"


def test_uncertainty_gradients_reach_the_weights():
    """A loss whose weights never update is a fixed-weight loss in disguise."""
    weighting = UncertaintyWeighting(conditioned=False)
    w = weighting(SIGMAS)
    (sum(v.sum() for v in w.values()) + weighting.penalty()).backward()
    assert weighting.log_vars.grad is not None
    assert torch.isfinite(weighting.log_vars.grad).all()


def test_conditioned_weighting_varies_with_sigma():
    """The point of the conditioned variant: it can learn a curriculum."""
    weighting = UncertaintyWeighting(conditioned=True)
    with torch.no_grad():  # break the zero-init so the net is not constant
        for param in weighting.net[-1].parameters():
            param.add_(torch.randn_like(param))
    w = weighting(SIGMAS)["hh_1"].reshape(-1)
    assert w.std() > 0, "weights are constant in sigma; no curriculum can be learned"


def test_penalty_blocks_the_collapse_to_zero_weights():
    """Kendall's log term: driving every weight to zero must cost something."""
    weighting = UncertaintyWeighting(conditioned=False)
    weighting(SIGMAS)
    small = weighting.penalty()
    with torch.no_grad():
        weighting.log_vars.fill_(10.0)  # huge variance == near-zero weights
    weighting(SIGMAS)
    assert weighting.penalty() > small


def test_gradnorm_weights_stay_normalised():
    balancer = GradNormBalancer(n_tasks=4)
    with torch.no_grad():
        balancer.weights.copy_(torch.tensor([5.0, 0.1, 2.0, 0.4]))
    balancer.renormalise()
    assert float(balancer.weights.detach().sum()) == pytest.approx(4.0, abs=1e-5)
    assert (balancer.weights > 0).all()


def test_gradnorm_produces_a_weight_gradient():
    """The whole mechanism: band gradient norms must drive the weights."""
    torch.manual_seed(0)
    shared = torch.nn.Conv2d(3, 3, 1)
    pred = shared(torch.randn(2, 3, 8, 8))
    target = torch.randn(2, 3, 8, 8)

    balancer = GradNormBalancer(n_tasks=3)
    losses = [(pred - target).pow(2).mean() * scale for scale in (1.0, 2.0, 4.0)]
    aux = balancer.gradnorm_loss(losses, list(shared.parameters()))
    aux.backward(inputs=[balancer.weights])

    assert balancer.weights.grad is not None
    assert torch.isfinite(balancer.weights.grad).all()
    assert balancer.weights.grad.abs().sum() > 0


def test_gradnorm_initial_losses_survive_a_reload():
    """Without this the inverse training rate is measured against a new baseline."""
    balancer = GradNormBalancer(n_tasks=2)
    balancer.combine([torch.tensor(3.0), torch.tensor(7.0)])
    restored = GradNormBalancer(n_tasks=2)
    restored.load_state_dict(balancer.state_dict())
    assert torch.allclose(restored.initial_losses, torch.tensor([3.0, 7.0]))
    assert float(restored.initialised.item()) == 1.0


# ---------------------------------------------------------------- A4 lifting


def test_lifting_is_perfectly_invertible_even_with_random_filters():
    """The property that makes a learnable basis safe to train.

    Checked in float64: arbitrary filters amplify the intermediate
    coefficients, so float32 rounding alone accounts for ~1e-4 and would hide
    an actual algebraic error behind a loose tolerance.
    """
    torch.manual_seed(0)
    wavelet = LiftingWavelet(kernel_size=3).double()
    with torch.no_grad():
        wavelet.predict.add_(torch.randn(3, dtype=torch.float64))
        wavelet.update.add_(torch.randn(3, dtype=torch.float64))
        wavelet.scale.add_(0.3)

    x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    ll, highs = wavelet(x)
    assert torch.allclose(wavelet.inverse(ll, highs), x, atol=1e-12)


def test_lifting_round_trip_is_stable_in_float32():
    """The dtype training actually runs in."""
    wavelet = LiftingWavelet(kernel_size=3)
    x = torch.randn(2, 3, 16, 16)
    ll, highs = wavelet(x)
    assert torch.allclose(wavelet.inverse(ll, highs), x, atol=1e-5)


def test_lifting_initialises_to_haar():
    """Training must start from the published basis, not a different one."""
    from pytorch_wavelets import DWTForward

    x = torch.randn(2, 3, 16, 16)
    ll_lift, highs_lift = LiftingWavelet()(x)
    ll_dwt, highs_dwt = DWTForward(J=1, wave="db1", mode="zero")(x)

    assert torch.allclose(ll_lift, ll_dwt, atol=1e-5)
    # Detail bands match up to the sign convention of the highpass filter.
    assert torch.allclose(highs_lift[0].abs(), highs_dwt[0].abs(), atol=1e-5)


def test_lifting_reports_its_orthogonality_defect():
    """Learned filters lose Parseval; the departure must be measurable."""
    wavelet = LiftingWavelet()
    assert wavelet.orthogonality_defect() < 1e-5, "Haar init should preserve energy"

    with torch.no_grad():
        wavelet.predict.add_(torch.tensor([0.4, 0.0, -0.3]))
    assert wavelet.orthogonality_defect() > 1e-3, "drift went unreported"


def test_lifting_filters_receive_gradients():
    wavelet = LiftingWavelet()
    ll, highs = wavelet(torch.randn(2, 3, 8, 8))
    (ll.pow(2).mean() + highs[0].pow(2).mean()).backward()
    assert wavelet.predict.grad is not None and wavelet.predict.grad.abs().sum() > 0
    assert wavelet.update.grad is not None


def test_frozen_basis_is_an_exact_control():
    wavelet = LiftingWavelet(learnable=False)
    assert not any(p.requires_grad for p in wavelet.parameters())


# ------------------------------------------------------- the composed losses


EXTENSION_LOSSES = [
    "wavelet_subband",
    "wavelet_spatial",
    "wavelet_learned",
    "wavelet_gradnorm",
    "wavelet_lifting",
]


@pytest.mark.parametrize("name", EXTENSION_LOSSES)
def test_extension_losses_train_and_backprop(name):
    pred = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randn(2, 3, 16, 16)
    loss_fn = get_loss_function(name, noise_scheduler=_StubScheduler())
    out = loss_fn(pred, target, timesteps=torch.randint(0, 1000, (2,)))
    assert out.ndim == 0 and torch.isfinite(out)
    out.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize(
    ("name", "expects_parameters"),
    [
        ("wavelet_subband", False),
        ("wavelet_spatial", False),
        ("wavelet_learned", True),
        ("wavelet_gradnorm", True),
        ("wavelet_lifting", True),
    ],
)
def test_trainable_parameters_are_exposed_to_the_optimizer(name, expects_parameters):
    """If these are not surfaced, the trainer never optimises them."""
    loss_fn = get_loss_function(name, noise_scheduler=_StubScheduler())
    assert bool(trainable_loss_parameters(loss_fn)) is expects_parameters


def test_generalized_loss_defaults_to_normalised_weights():
    """New strategies should not inherit the published confound by default."""
    loss = GeneralizedWaveletLoss()
    total = sum(loss.weighting(SIGMAS).values())
    assert torch.allclose(total, torch.ones_like(total), atol=1e-4)


def test_weight_profile_exposes_the_learned_curriculum():
    loss = GeneralizedWaveletLoss(weighting_strategy="uncertainty")
    profile = loss.weight_profile(torch.linspace(0.05, 0.95, 5))
    assert set(profile) == set(band_keys(1))
    assert len(profile["hh_1"]) == 5


def test_composition_of_all_axes_runs():
    """Learned weights, spatial adaptivity and a learned basis together."""
    loss = GeneralizedWaveletLoss(
        weighting_strategy="uncertainty", spatial=True, transform="lifting", normalize_weights=True
    )
    pred = torch.randn(2, 3, 16, 16, requires_grad=True)
    out = loss(pred, torch.randn(2, 3, 16, 16), torch.rand(2))
    out.backward()
    assert torch.isfinite(out)
    assert any(p.grad is not None for p in loss.parameters())


def test_lifting_loss_rejects_multilevel():
    with pytest.raises(ValueError, match="levels=1"):
        GeneralizedWaveletLoss(levels=2, transform="lifting")


def test_build_weighting_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown weighting"):
        build_weighting("telepathy")


@pytest.mark.parametrize("name", EXTENSION_LOSSES)
def test_extension_losses_tolerate_the_published_config_keys(name):
    """Switching ``loss.name`` must be enough to switch strategy.

    ``configs/finetune.yaml`` always carries ``alpha``, ``power``,
    ``wavelet_type``, ``detail_reduction`` and friends, which the pipeline
    merges into every run. A strategy with no such knobs must ignore them
    rather than raise — otherwise a sweep dies on its first learned-weighting
    job, hours in.
    """
    published_keys = {
        "alpha": 0.2,
        "power": 1.0,
        "wavelet_type": "db1",
        "levels": 1,
        "weighting": "normalized",
        "detail_reduction": "mean",
        "level_reduction": "sum",
        "normalize_weights": True,
    }
    loss_fn = get_loss_function(name, noise_scheduler=_StubScheduler(), **published_keys)
    out = loss_fn(
        torch.randn(2, 3, 16, 16, requires_grad=True),
        torch.randn(2, 3, 16, 16),
        timesteps=torch.randint(0, 1000, (2,)),
    )
    assert torch.isfinite(out)
