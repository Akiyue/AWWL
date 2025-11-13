"""Sample images from a trained DDPM checkpoint.

Used both for qualitative grids and as the generation step before FID/IS
evaluation (replaces the generation loop inside ``AWWL-Diff/eval_fid.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from diffusers import DDPMPipeline
from tqdm.auto import tqdm

from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def generate_samples(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    num_samples: int = 10000,
    batch_size: int = 128,
    num_inference_steps: int = 1000,
    device: str = "cuda",
) -> Path:
    """Sample ``num_samples`` images from a DDPM checkpoint to ``output_dir``.

    Skips generation if the directory already contains at least ``num_samples``
    PNGs — re-running the eval pipeline is then cheap.

    Returns:
        ``output_dir`` as a :class:`pathlib.Path`.
    """
    out = ensure_dir(output_dir)
    existing = sum(1 for _ in out.glob("*.png"))
    if existing >= num_samples:
        logger.info("output already has %d images; skipping generation", existing)
        return out

    pipeline = DDPMPipeline.from_pretrained(str(checkpoint_path)).to(device)
    pipeline.set_progress_bar_config(disable=True)

    count = existing
    pbar = tqdm(total=num_samples - existing, desc="sampling")
    with torch.no_grad():
        while count < num_samples:
            current = min(batch_size, num_samples - count)
            images = pipeline(batch_size=current, num_inference_steps=num_inference_steps).images
            for img in images:
                img.save(out / f"{count:05d}.png")
                count += 1
                pbar.update(1)
    pbar.close()
    return out
