"""Inject LoRA attention processors into a Stable-Diffusion UNet.

Diffusers has shuffled this API around; this module hides the cross-version
brittleness behind a stable :func:`add_lora_to_unet` entry point.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _import_lora_processor():
    """Resolve whichever LoRA attention-processor class is available."""
    try:
        from diffusers.models.attention_processor import (
            LoRAAttnProcessor2_0 as Cls,
        )
        return Cls
    except Exception:  # pragma: no cover - depends on installed diffusers
        try:
            from diffusers.models.attention_processor import LoRAAttnProcessor as Cls
            return Cls
        except Exception:
            return None


def _infer_hidden_size(proc: Any, unet_cfg: Any) -> int:
    """Best-effort inference of ``hidden_size`` for an attention processor."""
    if proc is None:
        return 768
    if hasattr(proc, "hidden_size") and isinstance(proc.hidden_size, int):
        return proc.hidden_size
    to_q = getattr(proc, "to_q", None)
    if to_q is not None:
        in_features = getattr(to_q, "in_features", None)
        if isinstance(in_features, int):
            return in_features
        if hasattr(to_q, "weight"):
            try:
                return to_q.weight.shape[1]
            except Exception:
                pass
    for attr in ("cross_attention_dim", "hidden_size", "sample_size"):
        v = getattr(unet_cfg, attr, None)
        if isinstance(v, int):
            return v
    block_out = getattr(unet_cfg, "block_out_channels", None)
    if block_out:
        return block_out[0]
    return 768


def _construct_lora(cls, *, rank: int, hidden_size: int | None, cross_attention_dim: int | None):
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    kwargs: dict[str, Any] = {}
    if "hidden_size" in params and hidden_size is not None:
        kwargs["hidden_size"] = hidden_size
    if "cross_attention_dim" in params and cross_attention_dim is not None:
        kwargs["cross_attention_dim"] = cross_attention_dim
    if "rank" in params:
        kwargs["rank"] = rank
    elif "lora_rank" in params:
        kwargs["lora_rank"] = rank
    else:
        raise RuntimeError(
            f"{cls.__name__} accepts neither 'rank' nor 'lora_rank'; signature: {sig}"
        )
    return cls(**kwargs)


def add_lora_to_unet(unet, *, rank: int = 4):
    """Replace every attention processor in ``unet`` with a LoRA variant.

    Args:
        unet: A ``UNet2DConditionModel`` exposing ``attn_processors``.
        rank: LoRA rank shared across every attention block.

    Returns:
        The same ``unet`` instance, mutated in place.

    Raises:
        RuntimeError: No suitable LoRA attention-processor class can be
            imported from the installed diffusers.
    """
    cls = _import_lora_processor()
    if cls is None:
        raise RuntimeError(
            "no LoRAAttnProcessor available; install a diffusers version that ships one"
        )
    if not hasattr(unet, "attn_processors"):
        raise RuntimeError("UNet has no attribute 'attn_processors'")

    cfg = getattr(unet, "config", None)
    existing = dict(unet.attn_processors) if unet.attn_processors else {}
    logger.info("found %d existing attn processors; replacing with LoRA(rank=%d)", len(existing), rank)

    new_processors: dict[str, Any] = {}
    for name, proc in existing.items():
        try:
            hidden = _infer_hidden_size(proc, cfg)
            cross_dim = getattr(proc, "cross_attention_dim", None)
            if cross_dim is None and cfg is not None:
                cross_dim = getattr(cfg, "cross_attention_dim", None)
            new_processors[name] = _construct_lora(
                cls, rank=rank, hidden_size=hidden, cross_attention_dim=cross_dim
            )
        except Exception as exc:
            logger.warning("could not create LoRA processor for %s: %s", name, exc)

    if not new_processors:
        logger.warning("no LoRA processors created; returning UNet unchanged")
        return unet

    if hasattr(unet, "set_attn_processor"):
        unet.set_attn_processor(new_processors)
    elif hasattr(unet, "set_attention_processor"):
        unet.set_attention_processor(new_processors)
    else:  # pragma: no cover - very old diffusers
        for k, v in new_processors.items():
            unet.attn_processors[k] = v
    return unet
