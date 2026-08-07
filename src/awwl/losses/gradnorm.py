"""GradNorm balancing for the wavelet sub-band losses (A3).

The paper cites GradNorm [9] and uncertainty weighting [10] in Related Work as
"explored in other fields", then hand-designs a two-parameter schedule instead.
This closes that gap by actually implementing GradNorm over the sub-bands, so
the comparison the citation implies can be run.

GradNorm treats each sub-band as a task and tunes per-task weights so that the
*gradient magnitudes* the tasks impose on the shared trunk stay balanced,
while letting tasks that are training slowly pull more weight:

    G_i   = ‖∇_W (w_i · L_i)‖₂            gradient norm on the shared layer
    L̃_i   = L_i / L_i(0)                   inverse training rate
    r_i   = L̃_i / mean(L̃)                  relative inverse training rate
    L_grad = Σ | G_i − mean(G) · r_i^α |

``L_grad`` is minimised with respect to the weights only — never the network —
and the weights are renormalised to sum to the task count each step so the
overall loss scale cannot drift.

Why this is worth running even though it is more invasive than uncertainty
weighting: it targets the mechanism the paper's motivation actually appeals
to. AWWL exists because "different frequencies recover at different rates";
GradNorm measures those rates directly from gradients rather than assuming
they follow ``σ^p``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)


class GradNormBalancer(nn.Module):
    """Per-task loss weights tuned by gradient-norm balancing.

    Args:
        n_tasks: Number of sub-band losses being balanced.
        asymmetry: GradNorm's ``α``. 0 balances gradient norms exactly; larger
            values push harder toward equalising *training rates* instead.
            1.5 is the value used in the original paper for uneven tasks.
        eps: Floor for the initial-loss normaliser.

    The weights are parameters, so they must reach the optimiser. Use a
    separate parameter group — see :func:`awwl.losses.factory.
    trainable_loss_parameters`.
    """

    def __init__(self, *, n_tasks: int, asymmetry: float = 1.5, eps: float = 1e-8) -> None:
        super().__init__()
        if n_tasks < 1:
            raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
        self.n_tasks = n_tasks
        self.asymmetry = float(asymmetry)
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(n_tasks))
        # Registered as a buffer so the L_i(0) reference survives a resume;
        # without it, a restarted run would rescale every task's inverse
        # training rate against a different baseline and drift from the
        # trajectory it was resuming.
        self.register_buffer("initial_losses", torch.zeros(n_tasks))
        self.register_buffer("initialised", torch.zeros(1))

    def _record_initial(self, losses: torch.Tensor) -> None:
        if float(self.initialised.item()) == 0.0:
            self.initial_losses.copy_(losses.detach().clamp_min(self.eps))
            self.initialised.fill_(1.0)

    def combine(self, losses: Sequence[torch.Tensor]) -> torch.Tensor:
        """Weighted sum of the per-task losses, for the network's backward pass."""
        stacked = torch.stack([loss for loss in losses])
        self._record_initial(stacked)
        return (self.weights * stacked).sum()

    def gradnorm_loss(
        self,
        losses: Sequence[torch.Tensor],
        shared_parameters: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """The objective whose gradient updates the weights.

        Must be called *before* the network's ``backward``, because it needs a
        second-order graph through the shared parameters
        (``create_graph=True``).

        Returns a scalar to be backpropagated into ``self.weights`` alone.
        """
        stacked = torch.stack([loss for loss in losses])
        self._record_initial(stacked)

        shared = [p for p in shared_parameters if p.requires_grad]
        if not shared:
            raise ValueError("GradNorm needs at least one shared parameter that requires grad")

        norms = []
        for idx in range(self.n_tasks):
            grads = torch.autograd.grad(
                self.weights[idx] * stacked[idx],
                shared,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            flat = torch.cat(
                [g.reshape(-1) for g in grads if g is not None]
                or [torch.zeros(1, device=stacked.device)]
            )
            norms.append(flat.norm(p=2))
        grad_norms = torch.stack(norms)

        with torch.no_grad():
            relative = stacked.detach() / self.initial_losses.clamp_min(self.eps)
            rate = relative / relative.mean().clamp_min(self.eps)
            target = grad_norms.detach().mean() * rate.pow(self.asymmetry)

        return (grad_norms - target).abs().sum()

    @torch.no_grad()
    def renormalise(self) -> None:
        """Keep the weights positive and summing to ``n_tasks``.

        Called after the optimiser step. Without it the weights drift in
        overall scale, which is indistinguishable from a learning-rate change
        and would confound any comparison against a fixed-weight baseline.
        """
        self.weights.clamp_(min=self.eps)
        self.weights.mul_(self.n_tasks / self.weights.sum().clamp_min(self.eps))

    def extra_repr(self) -> str:
        return f"n_tasks={self.n_tasks}, asymmetry={self.asymmetry}"
