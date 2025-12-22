"""Ablation-study line + bar plot for the Finetune method.

Replaces ``AWWL-Diff/plot.py``. Two panels:

* (a) FID/IS as a function of ``alpha`` (with ``power=2.0``), showing the
  alpha dose-response.
* (b) Bar chart of FID/IS as a function of ``power`` at the best ``alpha``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from awwl.plotting._style import apply_style

logger = logging.getLogger(__name__)


@dataclass
class AlphaSeries:
    """Data for the (a) panel — FID/IS vs alpha."""

    alphas: list[float]
    fid: list[float]
    inception_score: list[float]


@dataclass
class PowerSeries:
    """Data for the (b) panel — FID/IS at different power values."""

    labels: list[str]
    fid: list[float]
    inception_score: list[float]


def plot_ablation(
    *,
    alpha_series: AlphaSeries,
    power_series: PowerSeries,
    baseline_fid: float,
    baseline_is: float,
    output_path: str | Path,
) -> Path:
    """Render the two-panel ablation plot."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.set_title(r"(a) Impact of weight balance ($\alpha$ with $p=2.0$)")
    ax1.set_xlabel(r"Alpha ($\alpha$)")
    ax1.set_ylabel(r"FID $\downarrow$", color="tab:red", fontweight="bold")
    line1 = ax1.plot(alpha_series.alphas, alpha_series.fid, marker="o", color="tab:red", linewidth=2, label="FID")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_xticks(alpha_series.alphas)
    ax1.axhline(y=baseline_fid, color="gray", linestyle="--", alpha=0.7, label="MSE baseline")

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel(r"Inception Score $\uparrow$", color="tab:blue", fontweight="bold")
    line2 = ax1_twin.plot(
        alpha_series.alphas, alpha_series.inception_score,
        marker="s", linestyle="--", color="tab:blue", linewidth=2, label="IS",
    )
    ax1_twin.tick_params(axis="y", labelcolor="tab:blue")
    lines = line1 + line2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left")

    x = np.arange(len(power_series.labels))
    width = 0.35
    ax2.set_title(r"(b) Impact of adaptive schedule ($p$)")
    ax2.set_ylabel("Score")
    ax2.bar(x - width / 2, power_series.fid, width, label=r"FID $\downarrow$", color="#ff9999", edgecolor="black", alpha=0.9)
    ax2.bar(x + width / 2, power_series.inception_score, width, label=r"IS $\uparrow$", color="#66b3ff", edgecolor="black", alpha=0.9)
    ax2.axhline(y=baseline_fid, color="red", linestyle="-", linewidth=1.5, label="MSE FID")
    ax2.axhline(y=baseline_is, color="blue", linestyle="-", linewidth=1.5, label="MSE IS")
    ax2.set_xticks(x)
    ax2.set_xticklabels(power_series.labels)
    ax2.set_ylim(min(min(power_series.inception_score), baseline_is) - 0.5, max(max(power_series.fid), baseline_fid) + 1.0)
    ax2.legend(loc="upper center", ncol=2, framealpha=0.9)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
