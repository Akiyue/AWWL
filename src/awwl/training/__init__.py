"""Cross-method training helpers: accelerator, loss-history JSON."""

from __future__ import annotations

from awwl.training.accelerator import build_accelerator, compute_dtype_for
from awwl.training.loss_history import LossHistoryLogger

__all__ = [
    "LossHistoryLogger",
    "build_accelerator",
    "compute_dtype_for",
]
