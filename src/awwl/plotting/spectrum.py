"""Spectral-deviation plot: each arm's radial spectrum relative to real images.

Pure visualisation; the statistic is computed by
:func:`awwl.evaluation.price_fid.radial_profile` (luminance, ``20 log10 |F|``
averaged within integer-radius bins about the ``fftshift`` centre, then over
images), or by the older :func:`awwl.evaluation.spectrum.radial_profile`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from awwl.plotting._style import apply_style, figure_size, save_figure

logger = logging.getLogger(__name__)

#: Point size of the end-labels. The de-collision gap is derived from it, so
#: changing the size here keeps the spacing correct.
LABEL_PT = 8.0
#: Minimum centre-to-centre separation between two labels, as a multiple of
#: LABEL_PT. A factor of 1 leaves the ascenders of one name touching the
#: descenders of the one below; this is ordinary typographic leading.
LABEL_LEADING = 1.45


def _axial_nyquist(n_bins: int) -> int:
    """Nyquist radius of the shorter image axis, from the number of radial bins.

    An ``N x N`` image has integer radii up to ``floor(sqrt(2) N / 2)``, so
    ``n_bins - 1`` is about ``sqrt(2)`` times the axial Nyquist ``N / 2``.
    """
    return int(round((n_bins - 1) / np.sqrt(2.0)))


def plot_spectrum_deviation(
    *,
    real: np.ndarray,
    baselines: dict[str, np.ndarray],
    highlight: tuple[str, np.ndarray],
    output_path: str | Path,
    bands: int = 3,
    nyquist: int | None = None,
    fraction: float = 0.93,
) -> Path:
    """Deviation of each arm's radial spectrum from the real one, in dB.

    Profiles may be 1-D (one mean profile) or 2-D ``(seeds, bins)``; for 2-D
    input the curve is the mean over seeds and the shading is ±1 s.d. across
    seeds. The x-axis is the integer radius in cycles per image, the unit the
    statistic is binned in. The band thirds used by the band table are
    marked, and so is the axial Nyquist radius: bins beyond it exist only
    along the diagonals of the spectrum.

    A configuration keeps the colour, dash and name it has in every other
    figure (:data:`awwl.plotting.paper_figures.SERIES`), and each series is
    labelled at its right-hand end instead of in a legend box.

    Args:
        real: Radial profile of real images, dB (the zero line here).
        baselines: Map of ledger group name -> radial profile(s).
        highlight: ``(group, profile)`` drawn heavier.
        output_path: PNG to write.
        bands: How many equal index bands to mark (3: low, mid, high).
        nyquist: Axial Nyquist radius; derived from the bin count if omitted.
    """
    from awwl.plotting.paper_figures import GRID, INK, MUTED, SHADE, display_name, style_of

    apply_style()
    fig, ax = plt.subplots(figsize=figure_size(fraction, 1.7))
    real = np.asarray(real, dtype=float)
    n = len(real)
    radius = np.arange(n)

    # Band thirds, as :func:`band_deviations` splits them, and axial Nyquist.
    edges = np.linspace(0, n, bands + 1).astype(int)
    names = ["low", "mid", "high"] if bands == 3 else [f"band {i + 1}" for i in range(bands)]
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if i % 2 == 1:
            ax.axvspan(a - 0.5, b - 0.5, color=SHADE, linewidth=0, zorder=0)
        ax.text((a + b - 1) / 2, 1.0, names[i], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ny = _axial_nyquist(n) if nyquist is None else nyquist
    ax.axvline(ny, color=MUTED, linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=1)

    ax.axhline(0, color=INK, linewidth=0.8, zorder=3)

    ends: list[tuple[float, str, str]] = []
    series = list(baselines.items()) + [highlight]
    lows = [0.0]
    for group, profile in series:
        prof = np.atleast_2d(np.asarray(profile, dtype=float))
        m = min(n, prof.shape[1])
        delta = prof[:, :m] - real[None, :m]
        mean = delta.mean(axis=0)
        is_highlight = group == highlight[0]
        colour, dash, _marker = style_of(group)
        if delta.shape[0] > 1:
            sd = delta.std(axis=0, ddof=1)
            ax.fill_between(radius[:m], mean - sd, mean + sd, color=colour, alpha=0.18,
                            linewidth=0, zorder=2)
            lows.append(float((mean - sd).min()))
        ax.plot(radius[:m], mean, color=colour, linestyle=dash,
                linewidth=1.5 if is_highlight else 1.2, zorder=4 if is_highlight else 3)
        lows.append(float(mean.min()))
        ends.append((float(mean[-1]), display_name(group), colour))

    ax.set_xlabel("radial frequency $r$ (cycles per image)")
    ax.set_ylabel("deviation from real (dB)")
    ax.set_xlim(0, n - 1)
    ax.set_xticks(list(range(0, n, 2)))
    lo = min(lows)
    ax.set_ylim(lo - 0.06 * abs(lo), 0.06 * abs(lo))
    ax.annotate(f"axial Nyquist ($r={ny}$)", xy=(ny, lo), xytext=(-3, 2),
                textcoords="offset points", ha="right", va="bottom", fontsize=7,
                color=MUTED)
    # Under the zero line at the right, where the curves are far below it and
    # the band names above the frame are clear of it.
    ax.annotate("real images", xy=(n - 1, 0.0), xytext=(0, -3), textcoords="offset points",
                ha="right", va="top", fontsize=7.5, color=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    # Room at the right for the longest end label, measured rather than
    # guessed, so a longer name cannot run off the canvas.
    renderer = fig.canvas.get_renderer()
    widest = 0.0
    for _value, name, _colour in ends:
        probe = fig.text(0, 0, name, fontsize=LABEL_PT)
        widest = max(widest, probe.get_window_extent(renderer).width)
        probe.remove()
    fig_px = fig.get_size_inches()[0] * fig.dpi
    fig.subplots_adjust(right=1.0 - (widest + 0.04 * fig_px) / fig_px, top=0.93)

    # Direct labels at the right end, nudged apart so two arms that finish
    # close together stay legible. The minimum separation is a text height,
    # measured in display units and converted to data units. The sweep is a
    # single upward pass, which terminates by construction.
    fig.canvas.draw()
    lo_y, hi_y = ax.get_ylim()
    axes_px = ax.get_window_extent().height
    gap = LABEL_PT * LABEL_LEADING * (fig.dpi / 72.0) * (hi_y - lo_y) / axes_px
    x_end = n - 1
    dx = 0.012 * (n - 1)

    ends.sort(key=lambda e: e[0])
    placed: list[float] = []
    for value, name, colour in ends:
        y = value if not placed else max(value, placed[-1] + gap)
        placed.append(y)
        # The dot stays on the curve; only the text moves. When the two part
        # company, a hairline leader says which name belongs to which line.
        if abs(y - value) > 1e-9:
            ax.plot([x_end, x_end + dx], [value, y], color=MUTED, linewidth=0.6,
                    zorder=4, clip_on=False)
        ax.annotate(name, xy=(x_end + 1.4 * dx, y), xycoords="data", ha="left",
                    va="center", fontsize=LABEL_PT, color=INK, annotation_clip=False)
        ax.plot([x_end], [value], marker="o", markersize=3.5, color=colour,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5, clip_on=False)

    # Pushing labels up can carry the topmost one past the frame.
    if placed and placed[-1] > hi_y - gap / 2:
        ax.set_ylim(lo_y, placed[-1] + gap / 2)

    return save_figure(fig, output_path, layout=False)
