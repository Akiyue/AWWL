"""Keep optimiser state on the same device as the parameters it belongs to.

``Optimizer.load_state_dict`` casts each parameter's state onto that
parameter's device *at the moment of loading*. Anything that moves a parameter
afterwards moves the parameter alone, and the mismatch does not surface until
the first optimiser step, which on a resumed run is a long way from the code
that caused it.

The failure this exists for: a learned objective's parameters were loaded
while the loss module was still on the CPU, then moved to the GPU after
``accelerator.prepare``. Training resumed at epoch 20 and died one step later
inside ``_multi_tensor_adam`` --- cuda gradients, CPU moments.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def align_optimizer_state(optimizer: Any) -> int:
    """Move every state tensor onto its parameter's device.

    Args:
        optimizer: A ``torch.optim.Optimizer``, or an Accelerate wrapper that
            proxies ``param_groups`` and ``state``.

    Returns:
        How many tensors were moved. Zero on a healthy run, which is the
        normal case; a non-zero count is logged because it means something
        upstream reordered a device move and the next such bug should be
        easier to find than this one was.
    """
    state = getattr(optimizer, "state", None)
    if not state:
        return 0

    moved = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            entry = state.get(param)
            if not entry:
                continue
            for key, value in entry.items():
                # `step` is a scalar tensor that torch deliberately keeps on
                # the CPU for the fused paths; moving it would be the bug.
                if key == "step" or not isinstance(value, torch.Tensor):
                    continue
                if value.device != param.device:
                    entry[key] = value.to(param.device)
                    moved += 1

    if moved:
        logger.warning(
            "moved %d optimiser state tensor(s) onto their parameters' device; "
            "a device move happened after the optimiser state was loaded",
            moved,
        )
    return moved
