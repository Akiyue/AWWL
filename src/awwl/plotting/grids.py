"""Random-sample image-grid figures.

Replaces ``AWWL-Diff/create_grid.py``. Builds an 8×8 grid (configurable) of
randomly-sampled PNGs from a folder.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

logger = logging.getLogger(__name__)


def make_image_grid(
    *,
    folder: str | Path,
    output_path: str | Path,
    grid_rows: int = 8,
    grid_cols: int = 8,
    padding: int = 2,
    pad_value: float = 1.0,
    seed: int | None = None,
) -> Path:
    """Sample ``grid_rows × grid_cols`` images from ``folder`` and save a grid.

    Args:
        folder: Source folder containing ``*.png`` files.
        output_path: PNG file to write.
        grid_rows: Number of rows in the grid.
        grid_cols: Number of columns in the grid.
        padding: Pixel padding between cells.
        pad_value: Padding colour in ``[0, 1]`` (1 = white).
        seed: Random seed for reproducible sampling. ``None`` = use process state.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"folder not found: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")

    n = grid_rows * grid_cols
    if seed is not None:
        random.seed(seed)
    chosen = files if len(files) < n else random.sample(files, n)

    to_tensor = transforms.ToTensor()
    batch = torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in chosen])
    grid = make_grid(batch, nrow=grid_cols, padding=padding, pad_value=pad_value)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out)
    logger.info("wrote grid %dx%d (%d images) to %s", grid_rows, grid_cols, len(chosen), out)
    return out
