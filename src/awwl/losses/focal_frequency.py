"""Focal Frequency Loss (Jiang *et al.*, ICCV 2021).

A second, independently published member of the frequency-aware family, added
so the paper's claims about the *family* rest on more than one instance. It
shares the family's premise --- weight the loss across the frequency axis ---
and differs from the audited objective on both of the axes that matter:

===================  ===========================  =========================
                     Audited objective            Focal Frequency Loss
===================  ===========================  =========================
basis                wavelet (localised)          Fourier (global)
weight source        the noise level :math:`σ_t`  the current error itself
weight timing        fixed by a schedule          recomputed every step
===================  ===========================  =========================

The weight is the error's own magnitude, normalised to :math:`[0, 1]` and
**detached**, so the objective is a per-step reweighting of a squared error
rather than a new quantity: frequencies the model currently gets worst are
pushed hardest. Jiang *et al.* call this a spectral analogue of focal loss.

Where it sits in this paper's analysis
--------------------------------------

The trainer calls every objective on ``(ε̂, ε)``, so as with the wavelet
losses the transform lands on the **prediction residual**, not on an image
(see :mod:`awwl.losses.adaptive_wavelet`). The DFT is unitary up to a
constant, so Parseval applies here exactly as it does to an orthonormal
wavelet basis: an *unweighted* frequency loss is the pixel MSE, and this
objective too reallocates one fixed error budget. That is the claim
``\\S family`` makes about the family, and this module is the second member
it is checked against.

The confound of ``\\S confound`` recurs here, by a different route, and the
difference is measurable --- ``scripts/verify_loss_math.py`` checks 4 and 5
report it, and those numbers are the ones to quote, since the size of the
shape effect depends on how the comparison residuals are filtered.

Because the weight matrix is normalised to a maximum of 1 per sample, the
mean weight --- and therefore this objective's total gradient magnitude
relative to the pixel MSE --- is **exactly** invariant to the residual's
scale. It is *not* invariant to the residual's spectral shape: at identical
variance, a white residual and a low-pass one give mean weights differing by
several fold.

That matters because the residual's spectral shape is exactly what varies
across the noise schedule --- it is this paper's premise that it does. So
this member also imposes a timestep reweighting, not by design as the
audited objective does through :math:`α`, but as an emergent consequence of
keying the weight on the error. The confound is therefore a property of
frequency-weighted objectives more broadly than of one algebraic form, which
is a stronger version of ``\\S confound``'s claim than that section makes ---
and it is why this member is worth running rather than merely citing.

What differs is that here the effect cannot be switched off by holding a
budget fixed: there is no parameter separating the shape from the total. The
control is the unweighted limit, ``alpha=0``, which Parseval makes exactly
the pixel MSE.

Reference
---------
L. Jiang, B. Dai, W. Wu, C. C. Loy. "Focal Frequency Loss for Image
Reconstruction and Synthesis." ICCV 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """Squared spectral error weighted by that error's own magnitude.

    Args:
        alpha: Sharpness of the weight matrix. The weight is the spectral
            error magnitude raised to this power, so ``alpha=0`` degenerates
            to an unweighted frequency loss (i.e. the pixel MSE up to the
            transform's constant) and larger values concentrate the objective
            on fewer, worse frequencies. Jiang *et al.* use 1.0.
        log_matrix: Compress the weight matrix through ``log(1 + w)`` before
            normalising. Their remedy for spectra whose dynamic range is wide
            enough that a handful of frequencies take the whole budget.
        batch_matrix: Normalise the weight by the maximum over the *batch*
            rather than per sample. Couples samples in a batch, which is
            their default off and ours too.
        ave_spectrum: Average the spectra over the batch before comparing.
            Off by default: it changes what is being matched from each
            sample's spectrum to the batch's mean spectrum.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        log_matrix: bool = False,
        batch_matrix: bool = False,
        ave_spectrum: bool = False,
    ) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = float(alpha)
        self.log_matrix = bool(log_matrix)
        self.batch_matrix = bool(batch_matrix)
        self.ave_spectrum = bool(ave_spectrum)

    @staticmethod
    def _spectrum(x: torch.Tensor) -> torch.Tensor:
        """``(N, C, H, W)`` real input to its orthonormal 2-D DFT.

        ``norm="ortho"`` is what makes Parseval hold with the same constant
        in both domains, so an unweighted version of this loss is comparable
        with the pixel MSE rather than off by ``H*W``.
        """
        return torch.fft.fft2(x.float(), norm="ortho")

    def weight_matrix(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The per-frequency weight, detached.

        Detaching is not an optimisation: with the weight left in the graph
        the model could lower its loss by making the weight small, which is
        the degenerate solution rather than the intended emphasis.
        """
        with torch.no_grad():
            diff = self._spectrum(pred) - self._spectrum(target)
            w = diff.abs() ** self.alpha
            if self.log_matrix:
                w = torch.log(w + 1.0)
            if self.batch_matrix:
                w = w / w.max().clamp_min(1e-12)
            else:
                # Per sample: the maximum over channel and both frequency
                # axes, keeping the sample dimension.
                peak = w.flatten(1).max(dim=1).values.clamp_min(1e-12)
                w = w / peak.view(-1, 1, 1, 1)
            # NaNs can only arise from a degenerate spectrum; clamping keeps
            # one bad sample from poisoning the batch's gradient.
            return torch.nan_to_num(w, nan=0.0).clamp(0.0, 1.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weighted mean squared spectral error.

        Args:
            pred: ``(N, C, H, W)`` predicted noise.
            target: The noise that was added, same shape.
            weights: Override the computed weight matrix. Only for tests and
                for the unweighted control; training never passes it.
        """
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

        f_pred, f_target = self._spectrum(pred), self._spectrum(target)
        if self.ave_spectrum:
            f_pred = f_pred.mean(dim=0, keepdim=True)
            f_target = f_target.mean(dim=0, keepdim=True)

        # (real - real)^2 + (imag - imag)^2, i.e. the squared modulus of the
        # complex difference, written so autograd stays on real tensors.
        delta = f_pred - f_target
        sq = delta.real.pow(2) + delta.imag.pow(2)

        w = self.weight_matrix(pred, target) if weights is None else weights
        if self.ave_spectrum and weights is None:
            w = w.mean(dim=0, keepdim=True)
        return (w * sq).mean()

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, log_matrix={self.log_matrix}, "
            f"batch_matrix={self.batch_matrix}, ave_spectrum={self.ave_spectrum}"
        )
