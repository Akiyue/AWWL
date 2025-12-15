"""Radial FFT power-spectrum profiling.

Replaces the analysis half of ``AWWL-Diff/spectrum_plot.py`` (the plotting
half lives under :mod:`awwl.plotting.spectrum`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

_VALID_EXTS = (".png", ".jpg", ".jpeg")


def radial_profile(folder: str | Path, *, max_images: int = 10000) -> np.ndarray | None:
    """Average radially-binned FFT log-magnitude across images in ``folder``.

    Returns:
        A 1-D NumPy array of length ``min(image_height, image_width)//2 + 1``,
        or ``None`` if the folder is empty.
    """
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.suffix.lower() in _VALID_EXTS][:max_images]
    if not files:
        return None

    profiles: list[np.ndarray] = []
    for f in tqdm(files, desc=f"fft {folder.name}", leave=False):
        try:
            data = np.array(Image.open(f).convert("L"))
        except Exception as exc:
            logger.warning("could not read %s: %s", f, exc)
            continue
        fft = np.fft.fftshift(np.fft.fft2(data))
        psd = 20 * np.log(np.abs(fft) + 1e-8)

        h, w = data.shape
        ys, xs = np.indices((h, w))
        center = np.array([h // 2, w // 2])
        r = np.sqrt((xs - center[1]) ** 2 + (ys - center[0]) ** 2).astype(int)
        tbin = np.bincount(r.ravel(), psd.ravel())
        nr = np.bincount(r.ravel())
        profiles.append(tbin / (nr + 1e-8))

    if not profiles:
        return None
    min_len = min(len(p) for p in profiles)
    profiles = [p[:min_len] for p in profiles]
    return np.mean(profiles, axis=0)
