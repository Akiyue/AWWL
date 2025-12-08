"""Per-timestep low/high-frequency error analysis for DDPM models.

Replaces ``AWWL-Diff/timestep.py``. Given two trained DDPM checkpoints (e.g.
MSE baseline vs AWWL), reports the noise-prediction error in the wavelet LL
band and high-frequency bands at each timestep, averaged over a held-out
split. Used to argue *where* AWWL helps in the diffusion process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMPipeline
from pytorch_wavelets import DWTForward
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TimestepProfile:
    """Output of :func:`evaluate_timestep_errors`."""

    timesteps: np.ndarray
    low_freq_error: np.ndarray
    high_freq_error: np.ndarray


def _load_pipeline(model_path: str | Path, device: str) -> DDPMPipeline:
    """Be tolerant of "pipeline folder" vs "checkpoint folder" layouts."""
    try:
        return DDPMPipeline.from_pretrained(str(model_path)).to(device)
    except Exception:
        return DDPMPipeline.from_pretrained(str(Path(model_path).parent)).to(device)


def evaluate_timestep_errors(
    *,
    model_path: str | Path,
    dataloader: DataLoader,
    device: str,
    timestep_stride: int = 10,
    wavelet: str = "db1",
) -> TimestepProfile:
    """Average wavelet-band noise prediction errors across timesteps.

    Args:
        model_path: A DDPM checkpoint (pipeline or unet folder).
        dataloader: Yields ``{"images": tensor}`` batches in ``[-1, 1]``.
        device: Compute device.
        timestep_stride: Distance between timesteps probed (10 yields 100
            samples for a 1000-step scheduler).
        wavelet: DWT type. The original analysis used Haar (``db1``).
    """
    pipeline = _load_pipeline(model_path, device)
    model = pipeline.unet
    scheduler = pipeline.scheduler
    dwt = DWTForward(J=1, wave=wavelet, mode="zero").to(device)

    timesteps = list(range(0, scheduler.config.num_train_timesteps, timestep_stride))
    low_acc = np.zeros(len(timesteps))
    high_acc = np.zeros(len(timesteps))
    total = 0

    for batch in tqdm(dataloader, desc=f"timestep {Path(model_path).name}"):
        images = batch["images"].to(device)
        bsz = images.shape[0]
        total += bsz
        noise = torch.randn_like(images)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((bsz,), t, device=device, dtype=torch.long)
            with torch.no_grad():
                noisy = scheduler.add_noise(images, noise, t_tensor)
                noise_pred = model(noisy, t_tensor).sample
            diff = noise_pred - noise
            ll, high = dwt(diff)
            low_acc[i] += (ll**2).mean().item() * bsz
            high_acc[i] += (high[0] ** 2).mean().item() * bsz

    return TimestepProfile(
        timesteps=np.array(timesteps),
        low_freq_error=low_acc / max(total, 1),
        high_freq_error=high_acc / max(total, 1),
    )
