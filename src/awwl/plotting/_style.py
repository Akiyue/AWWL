"""Shared matplotlib style for publication-quality figures.

Apply once at the top of any plotting entry point with :func:`apply_style`.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_style() -> None:
    """Set serif fonts and academic-default sizes."""
    plt.rc("font", family="serif")
    mpl.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
        }
    )
