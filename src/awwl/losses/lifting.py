"""Learnable wavelet basis via the lifting scheme (A4).

Fixing the basis to Haar or ``db4`` is an assumption, not a result — the
paper's own ablation shows the two differ, which means the choice matters and
was made by search rather than derived. The lifting scheme lets the filters be
*learned* jointly with the denoiser instead.

The scheme factorises a wavelet transform into three trivially invertible
steps applied along one axis:

    split      e = x[0::2],  o = x[1::2]
    predict    d = o - P(e)
    update     s = e + U(d)

``P`` and ``U`` are arbitrary — here, small learnable filters. **Perfect
reconstruction holds for any values they take**, because the inverse is just
the steps run backwards (``e = s - U(d)``, ``o = d + P(e)``). That is what
makes this safe to train: no constraint, penalty or re-orthogonalisation is
needed to keep the transform invertible, and a degenerate filter costs
representational quality but can never make the decomposition lossy.

Initialised to reproduce the Haar transform (``P = 1``, ``U = ½``), so
training starts from the published configuration and any improvement is
attributable to the learning rather than to a different starting basis.

Caveat worth stating in a paper: learned filters are generally **not
orthonormal**, so the Parseval identity that makes an unweighted wavelet loss
equal to the pixel MSE no longer holds exactly. :meth:`LiftingWavelet.
orthogonality_defect` measures how far it has drifted, so the departure can be
reported rather than assumed away.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


def _filter_along_last(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply a 1-D filter along the last axis with reflection padding."""
    k = weight.shape[0]
    pad = k // 2
    if pad == 0:
        return weight[0] * x
    length = x.shape[-1]
    # Reflection padding of a single axis is only defined for 2-D/3-D inputs,
    # so collapse the leading axes into one batch dimension first.
    flat = x.reshape(-1, 1, length)
    padded = F.pad(flat, (pad, pad), mode="reflect").reshape(*x.shape[:-1], length + 2 * pad)
    out = torch.zeros_like(x)
    for j in range(k):
        out = out + weight[j] * padded[..., j : j + length]
    return out


class LiftingWavelet(nn.Module):
    """A one-level 2-D wavelet transform with learnable lifting filters.

    Args:
        kernel_size: Taps in the predict/update filters. Must be odd; 3 gives
            a neighbourhood comparable to ``db2`` while staying cheap.
        channels: Unused placeholder for API symmetry with ``DWTForward``;
            filters are shared across channels, as a wavelet basis should be.
        learnable: Set ``False`` to freeze at the Haar initialisation, which
            gives an exact ablation control for "did learning the basis help?"
    """

    def __init__(
        self,
        *,
        kernel_size: int = 3,
        channels: int | None = None,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        del channels
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.kernel_size = kernel_size
        centre = kernel_size // 2

        predict = torch.zeros(kernel_size)
        update = torch.zeros(kernel_size)
        predict[centre] = 1.0   # d = o - e
        update[centre] = 0.5    # s = e + d/2   -> the Haar pair
        self.predict = nn.Parameter(predict, requires_grad=learnable)
        self.update = nn.Parameter(update, requires_grad=learnable)
        # Haar's orthonormal scaling; learnable so the basis can renormalise.
        self.scale = nn.Parameter(torch.tensor(2.0**0.5), requires_grad=learnable)

    def _lift_1d(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split-predict-update along the last axis. Returns ``(approx, detail)``."""
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"lifting needs an even length along the last axis, got {x.shape[-1]}")
        even = x[..., 0::2]
        odd = x[..., 1::2]
        detail = odd - _filter_along_last(even, self.predict)
        approx = even + _filter_along_last(detail, self.update)
        return approx * self.scale, detail / self.scale

    def _inverse_1d(self, approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        approx = approx / self.scale
        detail = detail * self.scale
        even = approx - _filter_along_last(detail, self.update)
        odd = detail + _filter_along_last(even, self.predict)
        out = torch.stack([even, odd], dim=-1)
        return out.reshape(*even.shape[:-1], -1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Decompose ``(N,C,H,W)``.

        Returns ``(ll, [highs])`` with ``highs[0]`` shaped ``(N,C,3,H/2,W/2)``
        holding ``(lh, hl, hh)`` — the same layout
        :class:`pytorch_wavelets.DWTForward` uses, so the two are drop-in
        interchangeable inside the loss.
        """
        # Columns first (filter along W), then rows (filter along H).
        low_w, high_w = self._lift_1d(x)
        low_w = low_w.transpose(-1, -2)
        high_w = high_w.transpose(-1, -2)

        ll, lh = self._lift_1d(low_w)
        hl, hh = self._lift_1d(high_w)

        ll = ll.transpose(-1, -2)
        lh = lh.transpose(-1, -2)
        hl = hl.transpose(-1, -2)
        hh = hh.transpose(-1, -2)
        return ll, [torch.stack([lh, hl, hh], dim=2)]

    def inverse(self, ll: torch.Tensor, highs: list[torch.Tensor]) -> torch.Tensor:
        """Exact inverse of :meth:`forward` — used to verify invertibility."""
        lh, hl, hh = torch.unbind(highs[0], dim=2)
        low_w = self._inverse_1d(ll.transpose(-1, -2), lh.transpose(-1, -2))
        high_w = self._inverse_1d(hl.transpose(-1, -2), hh.transpose(-1, -2))
        return self._inverse_1d(low_w.transpose(-1, -2), high_w.transpose(-1, -2))

    @torch.no_grad()
    def orthogonality_defect(self, *, size: int = 32, samples: int = 4) -> float:
        """How far the learned basis has drifted from energy preservation.

        Returns ``|¼·Σ band means − pixel MSE| / pixel MSE`` on white noise:
        zero for an orthonormal basis, growing as the filters specialise.
        Report this alongside any result from a learned basis — it is the
        quantity that decides whether the Parseval framing still applies.
        """
        x = torch.randn(samples, 3, size, size, device=self.predict.device)
        ll, highs = self.forward(x)
        lh, hl, hh = torch.unbind(highs[0], dim=2)
        quarter = sum(b.pow(2).mean() for b in (ll, lh, hl, hh)) / 4.0
        reference = x.pow(2).mean()
        return float((quarter - reference).abs() / reference)
