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
