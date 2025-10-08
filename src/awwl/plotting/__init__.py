"""Publication-quality figure helpers."""

from __future__ import annotations

from awwl.plotting.ablation import AlphaSeries, PowerSeries, plot_ablation
from awwl.plotting.bar_scatter import plot_dreambooth_comparison
from awwl.plotting.grids import make_image_grid
from awwl.plotting.loss_curve import LossCurveSpec, plot_loss_curves
from awwl.plotting.radar import RadarPlotSpec, RadarStyle, plot_radar
from awwl.plotting.spectrum import plot_spectrum_deviation

__all__ = [
    "AlphaSeries",
    "LossCurveSpec",
    "PowerSeries",
    "RadarPlotSpec",
    "RadarStyle",
    "make_image_grid",
    "plot_ablation",
    "plot_dreambooth_comparison",
    "plot_loss_curves",
    "plot_radar",
    "plot_spectrum_deviation",
]
