"""Focal Frequency Loss: the properties the paper's argument rests on.

This objective is in the repository as a *second* member of the
frequency-aware family, so what these tests pin is not "the code runs" but
the three claims the manuscript makes about it: that Parseval holds (so it
reallocates a fixed budget, like the audited objective), that its weights
cannot be gamed by the model, and that it carries the timestep confound of
`\\S confound` through a different mechanism than the audited objective does.

The last one is a measurement, not a design property, so it is tested rather
than asserted in prose.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from awwl.losses import FocalFrequencyLoss, get_loss_function


@pytest.fixture
def residual() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)


# ------------------------------------------------------------------- Parseval


def test_unweighted_is_exactly_the_pixel_mse(residual):
    """The claim that makes this a member of the same family.

    With ``norm="ortho"`` the DFT is unitary, so a frequency loss with every
    weight at 1 *is* the pixel MSE. Any gain the weighting produces is
    therefore a reallocation of one fixed error budget and never new
    information -- the analytic point of the manuscript's family section,
    which was argued there for an orthonormal wavelet basis and holds here
    for the same reason.
    """
    pred, target = residual
    loss = FocalFrequencyLoss()(pred, target, weights=torch.ones_like(pred))
    assert torch.allclose(loss, F.mse_loss(pred, target), rtol=1e-6, atol=1e-8)


def test_alpha_zero_reaches_the_same_identity_through_the_weight_matrix(residual):
    """``alpha=0`` is the unweighted control, and it must be exact.

    This is the only control available for this objective: unlike the audited
    one there is no parameter that separates the weighting's shape from its
    total, so there is nothing to hold fixed. If this identity drifts, the
    control is not a control.
    """
    pred, target = residual
    loss = FocalFrequencyLoss(alpha=0.0)(pred, target)
    assert torch.allclose(loss, F.mse_loss(pred, target), rtol=1e-6, atol=1e-8)


# -------------------------------------------------------------- weight matrix


def test_the_weight_carries_no_gradient(residual):
    """The model must not be able to lower its loss by shrinking the weight.

    Left in the graph, the weight is minimised by making the error's spectrum
    uniform rather than small, which is the degenerate solution.
    """
    pred, target = residual
    w = FocalFrequencyLoss().weight_matrix(pred.requires_grad_(True), target)
    assert not w.requires_grad
    assert w.grad_fn is None


def test_weights_are_bounded_to_the_unit_interval(residual):
    pred, target = residual
    w = FocalFrequencyLoss().weight_matrix(pred, target)
    assert float(w.min()) >= 0.0
    assert float(w.max()) <= 1.0 + 1e-6
    # Normalisation is per sample, so each one must reach its own maximum.
    peaks = w.flatten(1).max(dim=1).values
    assert torch.allclose(peaks, torch.ones_like(peaks), atol=1e-5)


def test_larger_alpha_concentrates_the_objective(residual):
    """Sharper weights put more of the budget on fewer frequencies."""
    pred, target = residual
    spreads = [
        float(FocalFrequencyLoss(alpha=a).weight_matrix(pred, target).mean())
        for a in (0.5, 1.0, 2.0, 4.0)
    ]
    assert spreads == sorted(spreads, reverse=True), spreads


# ------------------------------------------------------- the timestep confound


def test_mean_weight_is_invariant_to_the_residual_scale():
    """No scale-driven confound: the weighting is exactly scale-free.

    Per-sample max normalisation divides the scale straight out, so unlike
    the audited objective's total this cannot drift with the magnitude of the
    residual -- which is the part of the confound the manuscript's protocol
    holds fixed by hand.
    """
    torch.manual_seed(0)
    d = torch.randn(8, 3, 32, 32)
    ffl = FocalFrequencyLoss()
    means = [
        float(ffl.weight_matrix(c * d, torch.zeros_like(d)).mean())
        for c in (1e-2, 1e-1, 1.0, 1e1, 1e2)
    ]
    assert max(means) - min(means) < 1e-6, means


def test_mean_weight_does_depend_on_the_residual_spectrum():
    """But a shape-driven confound is present, and it is large.

    The residual's spectral shape is what changes across the noise schedule
    --- the manuscript's own premise -- so keying the weight on the error
    imposes a timestep reweighting as a side effect, with no parameter that
    switches it off.

    The threshold here is a floor with margin, not the paper's number: the
    size of the effect depends on how the comparison residual is filtered, so
    the figure to quote is the one ``scripts/verify_loss_math.py`` prints.
    """
    torch.manual_seed(0)
    white = torch.randn(8, 3, 32, 32)
    freq = torch.arange(32).view(-1, 1) ** 2 + torch.arange(32).view(1, -1) ** 2
    low_pass = torch.fft.ifft2(torch.fft.fft2(white) * (freq < 100)).real
    low_pass = low_pass / low_pass.std() * white.std()

    ffl = FocalFrequencyLoss()
    mean_white = float(ffl.weight_matrix(white, torch.zeros_like(white)).mean())
    mean_lp = float(ffl.weight_matrix(low_pass, torch.zeros_like(low_pass)).mean())
    assert mean_white > 4 * mean_lp, (mean_white, mean_lp)


# ------------------------------------------------------------------- plumbing


def test_the_factory_does_not_leak_the_wavelet_alpha():
    """A config written for the audited objective must not retune this one.

    Both losses have a parameter called alpha and they mean different things:
    the wavelet one is a frequency balance in ``(0, 1)``, this one a weight
    sharpness. Reading ``loss.alpha`` here would silently run the sweep's
    ``alpha=0.2`` as a sharpness of 0.2, which is a different objective from
    the published Focal Frequency Loss and would be reported as it.
    """
    fn = get_loss_function("focal_frequency", alpha=0.2, power=1.0)
    assert fn.module.alpha == 1.0
    fn_explicit = get_loss_function("focal_frequency", ffl_alpha=2.0)
    assert fn_explicit.module.alpha == 2.0


def test_it_runs_through_the_factory_and_backprops(residual):
    pred, target = residual
    pred = pred.requires_grad_(True)
    fn = get_loss_function("focal_frequency")
    out = fn(pred, target, timesteps=torch.randint(0, 1000, (pred.shape[0],)))
    assert out.ndim == 0 and torch.isfinite(out)
    out.backward()
    assert torch.isfinite(pred.grad).all()


def test_shape_mismatch_is_refused(residual):
    pred, _ = residual
    with pytest.raises(ValueError, match="shape mismatch"):
        FocalFrequencyLoss()(pred, torch.randn(4, 3, 16, 16))
