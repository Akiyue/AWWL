"""Publication-quality figure helpers.

Every function here writes a file and never opens a window, so a
non-interactive backend is selected before any submodule imports ``pyplot``.
Without this, importing the package on a headless training server picks an
interactive backend and fails on a missing display or a broken Tk install —
turning "save a figure" into a crash in the middle of a sweep. Set
``MPLBACKEND`` to override.
"""

from __future__ import annotations

import os as _os

import matplotlib as _mpl

if not _os.environ.get("MPLBACKEND"):
    _mpl.use("Agg", force=True)

from awwl.plotting.ablation import AlphaSeries, PowerSeries, plot_ablation
from awwl.plotting.bar_scatter import plot_dreambooth_comparison
from awwl.plotting.curriculum import plot_run_curriculum, plot_weight_profile
from awwl.plotting.grids import make_image_grid
from awwl.plotting.loss_curve import LossCurveSpec, plot_loss_curves
from awwl.plotting.paired_grid import paired_sample_grid
from awwl.plotting.paper_figures import (
    parse_boost_tables,
    plot_convergence,
    plot_correction_value,
    plot_effect_sizes,
    plot_weight_schedule,
)
from awwl.plotting.radar import RadarPlotSpec, RadarStyle, plot_radar
from awwl.plotting.spectrum import plot_spectrum_deviation

__all__ = [
    "AlphaSeries",
    "LossCurveSpec",
    "PowerSeries",
    "RadarPlotSpec",
    "RadarStyle",
    "make_image_grid",
    "paired_sample_grid",
    "parse_boost_tables",
    "plot_ablation",
    "plot_convergence",
    "plot_correction_value",
    "plot_effect_sizes",
    "plot_dreambooth_comparison",
    "plot_loss_curves",
    "plot_radar",
    "plot_run_curriculum",
    "plot_spectrum_deviation",
    "plot_weight_profile",
    "plot_weight_schedule",
]
