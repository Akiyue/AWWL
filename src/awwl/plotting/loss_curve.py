"""Smoothed training-loss comparison plot.

Replaces ``AWWL-Diff/plot_losses.py``. Reads each experiment's
``loss_history.json`` and overlays exponentially-smoothed loss curves.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from awwl.plotting._style import apply_style

logger = logging.getLogger(__name__)


@dataclass
class LossCurveSpec:
    """One curve in :func:`plot_loss_curves`."""

    path: str | Path
    label: str
    color: str = "#1f77b4"


def _ema(values: list[float], *, weight: float) -> list[float]:
    """Tensorboard-style exponential moving average."""
    if not values:
        return values
    out: list[float] = []
    last = values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        out.append(last)
    return out


def plot_loss_curves(
    experiments: list[LossCurveSpec],
    *,
    output_path: str | Path,
    smoothing: float = 0.99,
    log_scale: bool = True,
) -> Path:
    """Plot one smoothed loss curve per experiment.

    Args:
        experiments: List of (path-to-folder, label, color) triples. The path
            should contain a ``loss_history.json`` file.
        output_path: PNG to write.
        smoothing: EMA weight (0 = raw, 0.99 = very smooth).
        log_scale: Use log Y axis (recommended when losses span orders of
            magnitude).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for spec in experiments:
        history = Path(spec.path) / "loss_history.json"
        if not history.exists():
            logger.warning("missing %s; skipping", history)
            continue
        with history.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        losses = payload.get("losses", [])
        if not losses:
            logger.warning("%s has no losses; skipping", history)
            continue
        smoothed = _ema(losses, weight=smoothing)
        ax.plot(range(len(smoothed)), smoothed, label=spec.label, color=spec.color, linewidth=1.5)

    ax.set_title("Training-loss convergence")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (smoothed)")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
