"""Deterministic-seeding helpers shared by every entry point."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Integer seed broadcast to every RNG.
        deterministic: When ``True`` also sets cuDNN to deterministic mode and
            asks PyTorch to error on non-deterministic ops. Slower, but the
            only way to get bit-identical runs across machines for the same
            seed. Off by default because mixed-precision training depends on
            non-deterministic kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover - depends on torch build
            logger.warning("could not enable deterministic algorithms: %s", exc)

    logger.info("seed=%d deterministic=%s", seed, deterministic)
