"""DreamBooth quantitative-comparison bar + scatter plots.

Replaces ``AWWL/plot.py``. Reads a ``summary_all_models.csv`` produced by
:func:`awwl.evaluation.evaluate_clip_over_models` and writes two figures: a
two-panel bar chart and a 2-D trade-off scatter.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from awwl.plotting._style import apply_style

logger = logging.getLogger(__name__)

_DEFAULT_DISPLAY_NAMES = {
    "adaptive_wavelet": "AWWL (Ours)",
    "mse": "MSE (Baseline)",
    "l1": "L1",
    "charbonnier": "Charbonnier",
    "simple_wavelet": "Static Wavelet",
    "snr_weighted": "SNR-Weighted",
    "perceptual": "Perceptual",
    "vlb": "Variational Lower Bound",
    "huber": "Huber",
}


def plot_dreambooth_comparison(
    *,
    summary_csv: str | Path,
    output_dir: str | Path,
    display_names: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Render the bar-chart and scatter-trade-off figures.

    Args:
        summary_csv: CSV with columns ``model``, ``clip_mean``, ``clip_std``,
            ``image_sim_mean``, ``image_sim_std``.
        output_dir: Folder where PNGs are written.
        display_names: Optional override of the model→pretty-name map.

    Returns:
        ``(bar_path, scatter_path)``.
    """
    apply_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)
    names = {**_DEFAULT_DISPLAY_NAMES, **(display_names or {})}
    df["display_name"] = df["model"].map(names).fillna(df["model"])

    palette = sns.color_palette("colorblind", n_colors=df["display_name"].nunique())

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    sns.barplot(
        data=df, x="display_name", y="clip_mean", hue="display_name",
        palette=palette, legend=False, edgecolor="black", ax=axes[0],
    )
    axes[0].errorbar(df["display_name"], df["clip_mean"], yerr=df["clip_std"], fmt="none", c="black", capsize=4)
    axes[0].set_ylabel(r"Mean CLIP score $\uparrow$")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    sns.barplot(
        data=df, x="display_name", y="image_sim_mean", hue="display_name",
        palette=palette, legend=False, edgecolor="black", ax=axes[1],
    )
    axes[1].errorbar(
        df["display_name"], df["image_sim_mean"], yerr=df["image_sim_std"], fmt="none", c="black", capsize=4
    )
    axes[1].set_ylabel(r"Mean image sim. $\uparrow$")
    axes[1].set_xlabel("Model")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=20, ha="right")
    sns.despine()
    bar_path = out / "comparison_bar.png"
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    for name, color in zip(df["display_name"].unique(), palette, strict=False):
        sub = df[df["display_name"] == name]
        is_ours = "AWWL" in name
        ax.scatter(
            sub["clip_mean"], sub["image_sim_mean"],
            label=name, s=400 if is_ours else 250, color=color,
            edgecolor="#EDB120" if is_ours else "black",
            linewidth=2, alpha=0.9, zorder=3,
        )
    ax.set_xlabel(r"CLIP score (prompt alignment) $\rightarrow$")
    ax.set_ylabel(r"Image similarity (subject fidelity) $\rightarrow$")
    ax.set_title("Performance trade-off")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    ax.legend(title="Models", loc="lower right", frameon=True, fancybox=True)
    sns.despine()
    scatter_path = out / "comparison_scatter.png"
    fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return bar_path, scatter_path
