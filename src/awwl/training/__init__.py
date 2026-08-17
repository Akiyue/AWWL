"""Cross-method training helpers: accelerator, checkpointing, EMA, loss log."""

from __future__ import annotations

from awwl.training.accelerator import build_accelerator, compute_dtype_for
from awwl.training.checkpointing import CheckpointManager, ResumeState
from awwl.training.ema import EmaHelper, build_ema
from awwl.training.loss_history import LossHistoryLogger
from awwl.training.optimizer_state import align_optimizer_state

__all__ = [
    "CheckpointManager",
    "EmaHelper",
    "LossHistoryLogger",
    "ResumeState",
    "align_optimizer_state",
    "build_accelerator",
    "build_ema",
    "compute_dtype_for",
]
