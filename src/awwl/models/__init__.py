"""Model factories: SD components, DDPM UNet, LoRA injection.

The SD-component loader pulls in :mod:`transformers`, which isn't needed for
the Finetune (DDPM) path. Callers that don't need it shouldn't pay the
import cost or fail when transformers is absent — so :func:`load_sd_components`
and :class:`SDComponents` are exposed via lazy ``__getattr__``.
"""

from __future__ import annotations

from typing import Any

from awwl.models.ddpm_unet import build_ddpm_unet, load_or_build_ddpm_unet
from awwl.models.lora import add_lora_to_unet

__all__ = [
    "SDComponents",
    "add_lora_to_unet",
    "build_ddpm_unet",
    "cast_module_dtype",
    "load_or_build_ddpm_unet",
    "load_sd_components",
]

_LAZY = {"SDComponents", "cast_module_dtype", "load_sd_components"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from awwl.models import sd_components

        return getattr(sd_components, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
