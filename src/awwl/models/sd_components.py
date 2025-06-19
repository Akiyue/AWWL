"""Loaders for the Stable-Diffusion sub-models DreamBooth fine-tunes.

We keep the loader separate from the trainer so test code can stub it with
tiny networks and the same trainer code path runs end-to-end on CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SDComponents:
    """A bundle of frozen + trainable components from Stable Diffusion."""

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler


def load_sd_components(
    pretrained_model_name_or_path: str,
    *,
    unet_override_path: str | Path | None = None,
) -> SDComponents:
    """Download (or load from cache/disk) every SD-1.5 component AWWL needs.

    Args:
        pretrained_model_name_or_path: HF hub id or local directory containing
            the standard SD-1.5 layout.
        unet_override_path: Optional path to a previously-trained UNet folder
            (``save_pretrained`` output). When set, only the UNet is replaced;
            VAE, tokenizer, text encoder and scheduler still come from the
            base model.

    The text encoder and VAE are returned with ``requires_grad_(False)`` already
    applied — the trainer only updates the UNet.
    """
    logger.info("loading SD components from %s", pretrained_model_name_or_path)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    if unet_override_path is not None:
        logger.info("overriding UNet with %s", unet_override_path)
        unet = UNet2DConditionModel.from_pretrained(str(unet_override_path))
    else:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        )
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    return SDComponents(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        noise_scheduler=noise_scheduler,
    )


def cast_module_dtype(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """In-place cast every parameter, gradient and buffer of ``module`` to ``dtype``.

    Used to align the DWT operator (which has integer-typed buffers) with the
    autocast compute dtype. Most callers should prefer plain ``module.to(dtype)``
    where possible; this helper exists for the wavelet edge case.
    """
    for p in module.parameters(recurse=True):
        if p.data is not None:
            p.data = p.data.to(dtype)
        if p.grad is not None:
            p.grad = p.grad.to(dtype)
    for b in module.buffers():
        if b is not None and getattr(b, "data", None) is not None:
            b.data = b.data.to(dtype)
