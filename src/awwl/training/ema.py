"""Exponential moving average of model weights.

Standard practice for DDPM training and the reason published CIFAR-10 FIDs
sit near 3 rather than near 17: samples are drawn from the EMA shadow copy,
not the raw optimiser iterate. The original AWWL-Diff recipe omitted it, so
this is opt-in via ``train.use_ema`` and off by default — turning it on
changes every number in the CIFAR-10 table and must be reported as its own
configuration, not silently mixed with pre-EMA runs.

Thin wrapper over :class:`diffusers.training_utils.EMAModel` so the trainer
does not have to branch on "EMA enabled?" everywhere and so the checkpoint
format stays stable if the diffusers API shifts.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class EmaHelper:
    """Maintain an EMA shadow copy of ``model``'s parameters.

    Args:
        model: The module whose parameters are tracked.
        decay: Maximum EMA decay (the usual DDPM value is ``0.9999``).
        use_warmup: Ramp the decay up early in training so the shadow copy is
            not pinned to the random initialisation.
        inv_gamma / power: Warm-up schedule shape, forwarded to diffusers.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.9999,
        use_warmup: bool = True,
        inv_gamma: float = 1.0,
        power: float = 0.75,
    ) -> None:
        from diffusers.training_utils import EMAModel

        self._ema = EMAModel(
            model.parameters(),
            decay=decay,
            use_ema_warmup=use_warmup,
            inv_gamma=inv_gamma,
            power=power,
            model_cls=type(model),
            model_config=getattr(model, "config", None),
        )
        self.decay = decay
        logger.info("EMA enabled (decay=%s, warmup=%s)", decay, use_warmup)

    def to(self, device: torch.device | str) -> EmaHelper:
        self._ema.to(device)
        return self

    def step(self, model: nn.Module) -> None:
        """Update the shadow copy from ``model``'s current parameters."""
        self._ema.step(model.parameters())

    @contextlib.contextmanager
    def as_active(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap the EMA weights into ``model``.

        Used around checkpoint export and sampling so the artefacts on disk
        are the EMA weights while training continues from the raw ones.
        """
        self._ema.store(model.parameters())
        self._ema.copy_to(model.parameters())
        try:
            yield
        finally:
            self._ema.restore(model.parameters())

    def state_dict(self) -> dict[str, Any]:
        return self._ema.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._ema.load_state_dict(state)


def build_ema(model: nn.Module, train_cfg: dict[str, Any]) -> EmaHelper | None:
    """Return an :class:`EmaHelper` when ``train.use_ema`` is set, else ``None``."""
    if not bool(train_cfg.get("use_ema", False)):
        return None
    return EmaHelper(
        model,
        decay=float(train_cfg.get("ema_decay", 0.9999)),
        use_warmup=bool(train_cfg.get("ema_warmup", True)),
    )
