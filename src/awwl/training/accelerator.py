"""Wrap :class:`accelerate.Accelerator` setup behind a small helper."""

from __future__ import annotations

from typing import Literal

import torch
from accelerate import Accelerator

MixedPrecision = Literal["no", "fp16", "bf16"]


def build_accelerator(
    *,
    mixed_precision: MixedPrecision = "fp16",
    gradient_accumulation_steps: int = 1,
) -> Accelerator:
    """Construct an :class:`Accelerator` with the project's defaults."""
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )


def compute_dtype_for(mixed_precision: MixedPrecision) -> torch.dtype:
    """Return the autocast compute dtype implied by ``mixed_precision``."""
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32
