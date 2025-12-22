"""Radar-chart visualisation of normalised metrics across methods.

Replaces ``AWWL-Diff/radar_plot.py``. Each axis is a metric, each line a
method; values are normalised so the baseline (MSE) sits at 1.0 and "higher
is better" is consistent across axes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from awwl.plotting._style import apply_style

logger = logging.getLogger(__name__)

MetricDirection = Literal["higher", "lower"]


@dataclass
class RadarStyle:
    """Per-method styling for the radar plot."""

    color: str = "#2F4F4F"
    linestyle: str = "-"
    linewidth: float = 1.8
    marker: str = ""
    markersize: float = 0.0
    alpha: float = 0.9
    zorder: int = 5
    fill: bool = False


@dataclass
class RadarPlotSpec:
    """Inputs to :func:`plot_radar`."""

    metrics: list[str]
    metric_types: list[MetricDirection]
    data: dict[str, list[float]]
    baseline_label: str
    styles: dict[str, RadarStyle] = field(default_factory=dict)


def _radar_factory(num_vars: int, frame: str = "polygon") -> np.ndarray:
    """Register a polar projection that draws polygonal radar grids."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                x, y = line.get_data()
                if x[0] != x[-1]:
                    line.set_data(np.concatenate((x, [x[0]])), np.concatenate((y, [y[0]])))

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=10, fontweight="bold")

        def _gen_axes_patch(self):
            return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            spine = Spine(axes=self, spine_type="circle", path=MplPath.unit_regular_polygon(num_vars))
            spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
            return {"polar": spine}

    register_projection(RadarAxes)
    return theta


def _normalize(raw: list[float], baseline: list[float], metric_types: list[MetricDirection]) -> list[float]:
    out: list[float] = []
    for v, base, mtype in zip(raw, baseline, metric_types, strict=True):
        if mtype == "higher":
            out.append(v / base)
        else:
            out.append(base / v)
    return out


def plot_radar(spec: RadarPlotSpec, *, output_path: str | Path) -> Path:
    """Render the radar chart described by ``spec``."""
    apply_style()
    if spec.baseline_label not in spec.data:
        raise KeyError(f"baseline {spec.baseline_label!r} not in data")
    baseline = spec.data[spec.baseline_label]
    n = len(spec.metrics)
    theta = _radar_factory(n, frame="polygon")

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(projection="radar"))
    fig.subplots_adjust(top=0.88, bottom=0.12)
    ax.grid(color="#AAAAAA", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.plot(theta, [1.0] * n, color="black", linestyle="-", linewidth=0.6, alpha=0.4)

    for label, raw in spec.data.items():
        norm = _normalize(raw, baseline, spec.metric_types)
        style = spec.styles.get(label, RadarStyle())
        ax.plot(
            theta, norm, label=label,
            color=style.color, ls=style.linestyle, lw=style.linewidth,
            marker=style.marker, ms=style.markersize, alpha=style.alpha, zorder=style.zorder,
        )
        if style.fill:
            ax.fill(theta, norm, color=style.color, alpha=0.08)

    ax.set_varlabels(spec.metrics)
    ax.set_ylim(0.8, 1.04)
    ax.set_rgrids([0.85, 0.90, 0.95, 1.0], labels=["0.85", "0.90", "0.95", "1.0"], angle=0, fontsize=8, color="#555555")
    plt.title(f"Relative performance (normalized to {spec.baseline_label} = 1.0)", y=1.08, fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=10)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
