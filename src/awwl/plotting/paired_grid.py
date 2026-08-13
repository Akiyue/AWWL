"""Matched side-by-side sample grids: same noise, different loss.

The sweep seeds every configuration's sampler identically, so image ``i`` in
one run's folder started from the same initial noise as image ``i`` in
another's. Sampling from the same latent makes the comparison a controlled
one: any visible difference is attributable to the training objective rather
than to which noise each model happened to draw. Random grids from each folder
— the usual qualitative figure, and the one the paper uses — cannot support
that reading.

This exists to settle a specific question. The replication found that AWWL
measurably closes the spectral gap to real images while FID, IS and KID do not
improve. Either the spectral shift is visible, in which case the metrics are
missing something a reader can see, or it is not, in which case the shift is
real but sub-perceptual. Both are publishable; they are different papers, and
only looking decides which.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw

from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)

_LABEL_HEIGHT = 22


def _images(folder: Path, indices: list[int]) -> list[Image.Image]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")
    return [Image.open(files[i]).convert("RGB") for i in indices if i < len(files)]


def paired_sample_grid(
    folders: dict[str, str | Path],
    *,
    output_path: str | Path,
    count: int = 12,
    start: int = 0,
    scale: int = 3,
    padding: int = 2,
) -> Path:
    """One row per configuration, one column per shared initial noise.

    Args:
        folders: ``{label: sample_folder}``. Row order follows insertion order.
        count: Columns, i.e. how many matched samples to show.
        start: First image index, so a different slice can be inspected
            without regenerating anything.
        scale: Integer upscaling. CIFAR-10 at 32px is unreadable at native
            size in a paper figure; nearest-neighbour keeps the pixels honest
            rather than smoothing the very detail under discussion.
    """
    resolved = {k: Path(v) for k, v in folders.items()}
    missing = [k for k, v in resolved.items() if not v.is_dir()]
    if missing:
        raise FileNotFoundError(f"no sample folder for: {', '.join(missing)}")

    indices = list(range(start, start + count))
    rows = {label: _images(folder, indices) for label, folder in resolved.items()}
    rows = {k: v for k, v in rows.items() if v}
    if not rows:
        raise ValueError("no images found in any folder")

    tile = next(iter(rows.values()))[0].size[0] * scale
    label_w = 150
    width = label_w + len(indices) * (tile + padding) + padding
    height = _LABEL_HEIGHT + len(rows) * (tile + padding) + padding

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"same initial noise per column (samples {start}-{start + count - 1})", fill="black")

    for row_idx, (label, images) in enumerate(rows.items()):
        y = _LABEL_HEIGHT + padding + row_idx * (tile + padding)
        draw.text((8, y + tile // 2 - 6), label, fill="black")
        for col_idx, image in enumerate(images):
            x = label_w + padding + col_idx * (tile + padding)
            canvas.paste(image.resize((tile, tile), Image.NEAREST), (x, y))

    out = Path(output_path)
    ensure_dir(out.parent)
    canvas.save(out)
    logger.info("wrote %s (%d rows x %d columns)", out, len(rows), len(indices))
    return out
