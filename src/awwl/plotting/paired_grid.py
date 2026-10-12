"""Matched side-by-side sample grids: same noise, different loss.

The sweep seeds every configuration's sampler identically, so image ``i`` in
one run's folder started from the same initial noise as image ``i`` in
another's. Sampling from the same latent makes the comparison a controlled
one: any visible difference is attributable to the training objective rather
than to which noise each model happened to draw.

Drawn with matplotlib under the shared paper style, so the row labels use the
paper's names and type face at print size. Each 32-pixel sample is enlarged
by an integer factor of device pixels (nearest neighbour), so every image
pixel prints as the same square and no detail is smoothed away. An optional
last row shows the absolute difference between two arms, amplified, which is
where a sub-perceptual change becomes visible at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _images(folder: Path, indices: list[int]) -> list[np.ndarray]:
    from PIL import Image

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")
    return [np.asarray(Image.open(files[i]).convert("RGB")) for i in indices if i < len(files)]


def plot_paired_samples(
    folders: dict[str, str | Path],
    *,
    out_path: str | Path,
    count: int = 8,
    start: int = 0,
    difference: tuple[str, str] | None = ("awwl", "mse"),
    gain: float = 8.0,
    fraction: float = 1.0,
    gap_px: int = 6,
) -> Path:
    """One row per configuration, one column per shared initial noise.

    Args:
        folders: ``{group: sample_folder}``; rows follow insertion order and
            are labelled with the paper's display names.
        count: Columns, i.e. how many matched samples to show.
        start: First image index.
        difference: ``(a, b)``: add a row of ``|a - b|`` averaged over RGB and
            multiplied by ``gain`` (dark = large difference), or ``None``.
        fraction: Printed width as a fraction of the text width.
    """
    import matplotlib.pyplot as plt

    from awwl.plotting._style import DPI, TEXT_WIDTH_IN, apply_style, save_figure
    from awwl.plotting.paper_figures import INK, MUTED, display_name

    resolved = {k: Path(v) for k, v in folders.items()}
    missing = [k for k, v in resolved.items() if not v.is_dir()]
    if missing:
        raise FileNotFoundError(f"no sample folder for: {', '.join(missing)}")

    indices = list(range(start, start + count))
    rows = {k: _images(v, indices) for k, v in resolved.items()}
    rows = {k: v for k, v in rows.items() if v}
    if not rows:
        raise ValueError("no images found in any folder")
    ncol = min(len(v) for v in rows.values())

    grid: list[tuple[str, list[np.ndarray], bool]] = [
        (display_name(k), v[:ncol], False) for k, v in rows.items()
    ]
    if difference is not None and all(k in rows for k in difference):
        a, b = difference
        diffs = [
            np.abs(x.astype(float) - y.astype(float)).mean(axis=2) * gain
            for x, y in zip(rows[a][:ncol], rows[b][:ncol], strict=True)
        ]
        label = (f"|{display_name(a)} − {display_name(b)}|"
                 f"\n×{gain:g}, RGB mean")
        grid.append((label, diffs, True))

    apply_style()
    width_px = int(round(TEXT_WIDTH_IN * fraction * DPI))
    size = rows[next(iter(rows))][0].shape[0]
    right_px = 2
    # The label column is as wide as its longest label needs, measured at the
    # size it is set in, and never narrower than 14% of the line.
    probe = plt.figure(figsize=(width_px / DPI, 1), dpi=DPI)
    renderer = probe.canvas.get_renderer()
    widest = max(probe.text(0, 0, label, fontsize=8 if not is_diff else 7.5, linespacing=1.2)
                 .get_window_extent(renderer).width for label, _images, is_diff in grid)
    plt.close(probe)
    label_px = max(int(0.14 * width_px), int(widest) + 10)
    # Largest integer enlargement that fits the columns beside the labels.
    scale = max(1, (width_px - label_px - right_px - (ncol - 1) * gap_px) // (ncol * size))
    tile = scale * size
    label_px = width_px - right_px - ncol * tile - (ncol - 1) * gap_px
    nrow = len(grid)
    top_px, bottom_px = 4, 4
    height_px = top_px + bottom_px + nrow * tile + (nrow - 1) * gap_px

    fig = plt.figure(figsize=(width_px / DPI, height_px / DPI), dpi=DPI)
    for r, (label, images, is_diff) in enumerate(grid):
        y0 = height_px - top_px - (r + 1) * tile - r * gap_px
        for c, image in enumerate(images):
            x0 = label_px + c * (tile + gap_px)
            ax = fig.add_axes((x0 / width_px, y0 / height_px, tile / width_px, tile / height_px))
            if is_diff:
                ax.imshow(np.clip(image, 0, 255), cmap="gray_r", vmin=0, vmax=255,
                          interpolation="nearest")
            else:
                ax.imshow(image, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(is_diff)
                spine.set_color(MUTED)
                spine.set_linewidth(0.5)
        fig.text((label_px - 8) / width_px, (y0 + tile / 2) / height_px, label,
                 ha="right", va="center", fontsize=8 if not is_diff else 7.5,
                 color=INK if not is_diff else MUTED, linespacing=1.2)
    out = save_figure(fig, out_path, layout=False)
    logger.info("wrote %s (%d rows x %d columns, %dx enlargement)", out, nrow, ncol, scale)
    return out


def paired_sample_grid(
    folders: dict[str, str | Path],
    *,
    output_path: str | Path,
    count: int = 8,
    start: int = 0,
    scale: int | None = None,
    padding: int | None = None,
    difference: tuple[str, str] | None = ("awwl", "mse"),
    gain: float = 8.0,
) -> Path:
    """Backward-compatible entry point; ``scale``/``padding`` are now derived
    from the printed width and ignored."""
    del scale, padding
    return plot_paired_samples(folders, out_path=output_path, count=count, start=start,
                               difference=difference, gain=gain)
