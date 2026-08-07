"""Smoke tests for the loss zoo: every loss is finite and has gradients."""

from __future__ import annotations

import pytest
import torch

from awwl.core.exceptions import UnknownLossError
from awwl.losses import AdaptiveWaveletLoss, get_loss_function, known_losses


class _StubScheduler:
    """Minimal duck-typed substitute for ``DDPMScheduler``."""

    def __init__(self, num_train_timesteps: int = 1000) -> None:
        self.alphas_cumprod = torch.linspace(0.999, 0.001, num_train_timesteps)


@pytest.mark.parametrize("name", [n for n in known_losses() if n != "perceptual"])
def test_loss_runs_and_backprops(name, synthetic_latents):
    """Every non-perceptual loss should produce a finite scalar with gradients."""
    pred = synthetic_latents.clone().requires_grad_(True)
    target = synthetic_latents + 0.01
    timesteps = torch.randint(0, 1000, (pred.shape[0],))
    scheduler = _StubScheduler()
    loss_fn = get_loss_function(name, noise_scheduler=scheduler, alpha=0.8, power=2.0)
    out = loss_fn(pred, target, timesteps=timesteps)
    assert out.ndim == 0
    assert torch.isfinite(out)
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("weighting", ["boosted", "normalized"])
def test_adaptive_wavelet_both_schemes(weighting):
    """Both weighting schemes should produce finite, positive losses."""
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randn(2, 4, 8, 8)
    sigmas = torch.tensor([0.1, 0.9])
    loss = AdaptiveWaveletLoss(weighting=weighting)
    out = loss(pred, target, sigmas)
    assert out > 0
    assert torch.isfinite(out)
    out.backward()
    assert torch.isfinite(pred.grad).all()


def test_unknown_loss_raises():
    with pytest.raises(UnknownLossError):
        get_loss_function("does_not_exist")


def test_adaptive_wavelet_rejects_bad_args():
    with pytest.raises(ValueError):
        AdaptiveWaveletLoss(levels=0)
    with pytest.raises(ValueError):
        AdaptiveWaveletLoss(alpha=1.5)
    with pytest.raises(ValueError):
        AdaptiveWaveletLoss(weighting="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AdaptiveWaveletLoss(detail_reduction="median")  # type: ignore[arg-type]


def test_parseval_identity_holds_for_orthonormal_basis():
    """An unweighted wavelet loss *is* the pixel MSE — the method's real footing.

    With ``db1`` the four sub-band mean-squared errors average back to the
    pixel MSE exactly, which is what makes AWWL a reallocation of a fixed
    error budget across orthogonal bands rather than a new error signal.
    Mirrors ``scripts/verify_loss_math.py``.
    """
    from pytorch_wavelets import DWTForward

    torch.manual_seed(0)
    pred = torch.randn(4, 3, 32, 32)
    target = torch.randn(4, 3, 32, 32)

    ll, highs = DWTForward(J=1, wave="db1", mode="zero")(pred - target)
    lh, hl, hh = torch.unbind(highs[0], dim=2)
    quarter_sum = sum(b.pow(2).mean() for b in (ll, lh, hl, hh)) / 4.0

    assert torch.allclose(quarter_sum, torch.nn.functional.mse_loss(pred, target), rtol=1e-5)


def test_eq7_sum_is_three_times_the_coded_mean_on_the_detail_term():
    """The paper sums the three detail bands; the code averages them.

    The gap is exactly the detail term's weight times 2 (3x minus the 1x the
    implementation applies), so a published alpha does not mean what eq. (7)
    says. Locks in the discrepancy so it cannot be "fixed" unnoticed.
    """
    torch.manual_seed(0)
    pred = torch.randn(4, 3, 32, 32)
    target = torch.randn(4, 3, 32, 32)
    sigmas = torch.rand(4)

    alpha = 0.2
    kwargs = {"alpha": alpha, "power": 1.0, "wavelet_type": "db1"}
    coded = AdaptiveWaveletLoss(detail_reduction="mean", **kwargs)(pred, target, sigmas)
    written = AdaptiveWaveletLoss(detail_reduction="sum", **kwargs)(pred, target, sigmas)

    # L = w_ll·LL + k·w_det·detail_mean with k = 1 ("mean") or 3 ("sum"), so the
    # gap is 2·w_det·detail_mean. alpha=0 zeroes the LL term and scales w_det to
    # 1, isolating detail_mean·w_det_shape.
    detail_unit = AdaptiveWaveletLoss(alpha=0.0, power=1.0, wavelet_type="db1")(
        pred, target, sigmas
    )
    assert torch.allclose(written - coded, 2.0 * (1.0 - alpha) * detail_unit, rtol=1e-4)


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_published_weighting_total_varies_from_alpha_to_one_minus_alpha(alpha):
    """Eqs. (4)-(5) do not sum to a constant — alpha also rescales gradients.

    This is the confound behind the "Alpha Paradox": alpha is simultaneously a
    frequency balance and a Min-SNR-style timestep reweighting.
    """
    loss = AdaptiveWaveletLoss(alpha=alpha, power=1.0)
    totals = loss.weights_at(torch.tensor([0.001, 0.5, 0.999]))["total"]
    assert totals[0] == pytest.approx(1.0 - alpha, abs=2e-3)
    assert totals[1] == pytest.approx(0.5, abs=1e-3)
    assert totals[2] == pytest.approx(alpha, abs=2e-3)


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("detail_reduction", ["mean", "sum"])
def test_normalize_weights_holds_the_total_at_one(alpha, detail_reduction):
    """The ablation that isolates frequency balance from gradient magnitude."""
    loss = AdaptiveWaveletLoss(
        alpha=alpha, power=1.0, normalize_weights=True, detail_reduction=detail_reduction
    )
    totals = loss.weights_at(torch.linspace(0.01, 0.99, 9))["total"]
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-4)


def test_normalize_weights_preserves_the_band_ratio():
    """Normalisation must rescale both weights, not re-balance them."""
    sigmas = torch.linspace(0.1, 0.9, 5)
    plain = AdaptiveWaveletLoss(alpha=0.8, power=2.0).weights_at(sigmas)
    scaled = AdaptiveWaveletLoss(alpha=0.8, power=2.0, normalize_weights=True).weights_at(sigmas)
    assert torch.allclose(plain["w_ll"] / plain["w_det"], scaled["w_ll"] / scaled["w_det"], rtol=1e-4)


def test_static_variant_ignores_sigma():
    """p=0 is the ablative baseline: weights must not move with the timestep."""
    loss = AdaptiveWaveletLoss(alpha=0.2, power=0.0)
    weights = loss.weights_at(torch.tensor([0.05, 0.5, 0.95]))
    assert torch.allclose(weights["w_ll"], weights["w_ll"][0].expand(3), atol=1e-6)


def test_level_reduction_controls_magnitude_growth():
    """Multi-level detail terms must be comparable to the single-level one."""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)
    sigmas = torch.rand(2)

    summed = AdaptiveWaveletLoss(levels=3, level_reduction="sum")(pred, target, sigmas)
    averaged = AdaptiveWaveletLoss(levels=3, level_reduction="mean")(pred, target, sigmas)
    assert summed > averaged


def test_factory_forwards_the_new_options():
    scheduler = _StubScheduler()
    loss_fn = get_loss_function(
        "adaptive_wavelet",
        noise_scheduler=scheduler,
        alpha=0.2,
        power=1.0,
        normalize_weights=True,
        detail_reduction="sum",
    )
    pred = torch.randn(2, 3, 32, 32, requires_grad=True)
    out = loss_fn(pred, torch.randn(2, 3, 32, 32), timesteps=torch.randint(0, 1000, (2,)))
    assert torch.isfinite(out)
