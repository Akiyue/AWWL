"""Spectral-deviation plot: real vs MSE vs AWWL radial profiles.

Replaces the plotting half of ``AWWL-Diff/spectrum_plot.py``. Pure
visualisation — the FFT computation lives in :mod:`awwl.evaluation.spectrum`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from awwl.plotting._style import apply_style

logger = logging.getLogger(__name__)


def plot_spectrum_deviation(
    *,
    real: np.ndarray,
    baselines: dict[str, np.ndarray],
    highlight: tuple[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Plot deviation from real spectrum for one or more methods.

    Profiles must be in decibels (20*log10), matching
    :func:`awwl.evaluation.spectrum.radial_profile`. Labelling the axis
    explicitly is not cosmetic: an earlier revision produced profiles in
    20*ln, and a figure whose units disagree with the table beside it is the
    kind of inconsistency a referee finds before the authors do.

    Args:
        real: Radial profile of real images, dB (zero-line in the plot).
        baselines: Map of label → radial profile to plot as dashed lines.
        highlight: ``(label, profile)`` rendered as a bold solid red line —
            typically the AWWL run.
        output_path: PNG to write.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    n = len(real)
    freqs = np.linspace(0, 1, n)

    ax.axhline(0, color="black", linestyle="-", linewidth=1.5, label="Real (reference)", alpha=0.8)
    for label, profile in baselines.items():
        m = min(n, len(profile))
        ax.plot(freqs[:m], profile[:m] - real[:m], linestyle="--", linewidth=2.0, label=f"{label} deviation")
    label, profile = highlight
    m = min(n, len(profile))
    ax.plot(freqs[:m], profile[:m] - real[:m], color="#D62728", linewidth=2.5, label=f"{label} deviation")

    ax.set_title("Spectral deviation in dB (closer to 0 is better)")
    ax.set_xlabel("Normalized frequency (low → high)")
    ax.set_ylabel("Magnitude difference, dB (model - real)")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fancybox=True)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
