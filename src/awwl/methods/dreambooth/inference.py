"""DreamBooth inference: load a fine-tuned UNet, sample images.

Replaces ``AWWL/inference.py`` (SD pipeline path) and ``AWWL/inference_lora.py``
(manual DDPM loop). Per Q4 we keep the SD pipeline as canonical for
DreamBooth — it gives the same quality with less code and benefits from
diffusers' scheduler improvements over time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def build_pipeline(
    *,
    base_model: str,
    unet_dir: str | Path,
    device: str,
    torch_dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """Construct an SD pipeline whose UNet is the fine-tuned weights at ``unet_dir``."""
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=torch_dtype)
    return pipe.to(device)


def generate_images(
    *,
    pipeline: StableDiffusionPipeline,
    prompt: str,
    seeds: list[int],
    output_dir: str | Path,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
) -> list[Path]:
    """Generate one image per seed under ``output_dir`` and return their paths."""
    out = ensure_dir(output_dir)
    device = pipeline.device
    paths: list[Path] = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
        ).images[0]
        path = out / f"seed_{seed}.png"
        image.save(path)
        paths.append(path)
    logger.info("wrote %d images to %s", len(paths), out)
    return paths
