"""Sample images from a trained DDPM checkpoint.

Used both for qualitative grids and as the generation step before FID/IS
evaluation (replaces the generation loop inside ``AWWL-Diff/eval_fid.py``).

Sampling, not training, is what makes a multi-seed study expensive: 50 000
images through 1 000 ancestral DDPM steps is a couple of GPU-hours *per
configuration*. Passing ``sampler="ddim"`` with 100 steps cuts that by an
order of magnitude, which is what makes a 5-seed × 8-loss table affordable.
FID is sensitive to the sampler, so a table must use one setting throughout —
never mix DDPM-1000 rows with DDIM-100 rows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import torch
from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline
from tqdm.auto import tqdm

from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)

Sampler = Literal["ddpm", "ddim"]


def build_sampling_pipeline(
    checkpoint_path: str | Path,
    *,
    sampler: Sampler = "ddpm",
    device: str = "cuda",
):
    """Load a checkpoint as a DDPM or DDIM sampling pipeline.

    The DDIM path reuses the trained UNet and re-derives a
    :class:`DDIMScheduler` from the saved DDPM schedule, so the two samplers
    differ only in the reverse process.
    """
    pipeline = DDPMPipeline.from_pretrained(str(checkpoint_path))
    if sampler == "ddim":
        pipeline = DDIMPipeline(
            unet=pipeline.unet,
            scheduler=DDIMScheduler.from_config(pipeline.scheduler.config),
        )
    elif sampler != "ddpm":
        raise ValueError(f"sampler must be 'ddpm' or 'ddim', got {sampler!r}")
    return pipeline.to(device)


def generate_samples(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    num_samples: int = 10000,
    batch_size: int = 128,
    num_inference_steps: int | None = None,
    sampler: Sampler = "ddpm",
    seed: int | None = None,
    device: str = "cuda",
) -> Path:
    """Sample ``num_samples`` images from a checkpoint into ``output_dir``.

    Generation resumes: images already on disk are counted and only the
    remainder is produced, so a job killed halfway through 50 000 samples
    picks up where it left off instead of restarting.

    Args:
        num_inference_steps: Defaults to 1000 for ``ddpm`` and 100 for
            ``ddim`` — the usual operating points for each.
        seed: Seeds the sampling noise. Distinct from the *training* seed;
            fix it so that re-running an evaluation is reproducible.

    Returns:
        ``output_dir`` as a :class:`pathlib.Path`.
    """
    out = ensure_dir(output_dir)
    existing = sum(1 for _ in out.glob("*.png"))
    if existing >= num_samples:
        logger.info("output already has %d images; skipping generation", existing)
        return out

    if num_inference_steps is None:
        num_inference_steps = 1000 if sampler == "ddpm" else 100

    pipeline = build_sampling_pipeline(checkpoint_path, sampler=sampler, device=device)
    pipeline.set_progress_bar_config(disable=True)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        # Offset by the resume point so a restart does not re-draw the same
        # noise it already turned into the images sitting on disk.
        generator.manual_seed(int(seed) + existing)

    logger.info(
        "sampling %d images (%s, %d steps) from %s",
        num_samples - existing,
        sampler,
        num_inference_steps,
        checkpoint_path,
    )

    count = existing
    pbar = tqdm(total=num_samples - existing, desc="sampling")
    with torch.no_grad():
        while count < num_samples:
            current = min(batch_size, num_samples - count)
            images = pipeline(
                batch_size=current,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images
            for img in images:
                img.save(out / f"{count:05d}.png")
                count += 1
                pbar.update(1)
    pbar.close()
    return out
