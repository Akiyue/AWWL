"""Plot what the loss actually prioritises across the noise schedule.

The paper argues coarse-to-fine qualitatively; this draws the schedule the
objective genuinely applies. It matters most for the learned weightings, where
the curriculum is an *outcome* rather than an assumption — the interesting
result is whether the network rediscovered coarse-to-fine on its own, settled
on something flat, or chose an ordering nobody would have hand-designed.

Two panels:

* **weights vs σ**, one line per sub-band. Reads right-to-left in training
  order (high noise first).
* **total weight vs σ** — the diagnostic for the confound in the published
  equations, where the total drifts from ``α`` to ``1-α`` instead of holding
  constant. A flat line here means the strategy redistributes gradient across
  bands without also rescaling it across timesteps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from awwl.plotting._style import apply_style
from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)

# Structure vs detail: one warm colour for LL, cool shades for the details, so
# the coarse-to-fine hand-off is legible without reading the legend.
_BAND_COLOURS = {
    "ll": "#c0392b",
    "lh": "#2980b9",
    "hl": "#27ae60",
    "hh": "#8e44ad",
}
# Dash by *direction*, not level. LH and HL share a schedule unless one is
# explicitly overridden, so with a shared line style the second curve is drawn
# exactly on top of the first and the figure reads as though a band vanished.
_DIRECTION_DASHES = {
    "ll": "-",
    "lh": "-",
    "hl": (0, (6, 3)),
    "hh": (0, (1.5, 2)),
}
# Levels are then separated by line width, which stays legible under a dash.
_LEVEL_WIDTHS = [2.6, 1.9, 1.4, 1.1]


def _style_for(key: str) -> tuple[str, object, float]:
    direction, level = (key.split("_") + ["0"])[:2] if "_" in key else (key, "0")
    colour = _BAND_COLOURS.get(direction, "#555555")
    dash = _DIRECTION_DASHES.get(direction, "-")
    width = _LEVEL_WIDTHS[min(max(int(level) - 1, 0), len(_LEVEL_WIDTHS) - 1)]
    return colour, dash, width


# Solid curves first, dashed and dotted last, so that where two bands coincide
# the broken line is drawn on top and both remain readable. Alphabetical order
# would hide HL underneath LH exactly whenever they share a schedule.
_DRAW_ORDER = {"ll": 0, "lh": 1, "hl": 2, "hh": 3}


def _draw_order(key: str) -> tuple[int, int]:
    direction, level = (key.split("_") + ["0"])[:2] if "_" in key else (key, "0")
    return _DRAW_ORDER.get(direction, 9), int(level)


def plot_weight_profile(
    profile: dict[str, list[float]],
    sigmas,
    *,
    out_path: str | Path,
    title: str = "Sub-band weighting across the noise schedule",
) -> Path:
    """Draw ``{band: weights}`` against σ and write it to ``out_path``.

    Args:
        profile: As returned by
            :meth:`~awwl.losses.generalized_wavelet.GeneralizedWaveletLoss.weight_profile`.
        sigmas: The σ values the profile was evaluated at.
    """
    import matplotlib.pyplot as plt

    apply_style()
    sigma = np.asarray(sigmas, dtype=float).reshape(-1)

    fig, (ax, ax_total) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    total = np.zeros_like(sigma, dtype=float)
    for key in sorted(profile, key=_draw_order):
        values = np.asarray(profile[key], dtype=float).reshape(-1)
        colour, dash, width = _style_for(key)
        if values.size != sigma.size:
            # GradNorm yields one weight per band, not a curve.
            ax.axhline(values.mean(), label=key, color=colour, linestyle=dash, linewidth=width)
            total += values.mean()
            continue
        ax.plot(sigma, values, label=key, color=colour, linestyle=dash, linewidth=width)
        total += values

    ax.set_ylabel("weight")
    ax.set_title(title)
    ax.legend(ncol=2, frameon=False)
    ax.grid(alpha=0.3)

    ax_total.plot(sigma, total, color="#333333")
    ax_total.set_ylabel("total")
    ax_total.set_xlabel(r"noise level $\sigma_t$   (high noise $\rightarrow$ right)")
    ax_total.grid(alpha=0.3)
    ax_total.set_ylim(0, max(1.15 * float(total.max()), 1.15))
    ax_total.axhline(1.0, color="#999999", linestyle=":", linewidth=1)

    spread = float(total.max() - total.min())
    ax_total.text(
        0.02,
        0.08,
        f"drift = {spread:.3f}" + ("  (constant)" if spread < 1e-3 else "  (varies with t)"),
        transform=ax_total.transAxes,
        fontsize=10,
    )

    fig.tight_layout()
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def plot_run_curriculum(
    run_dir: str | Path,
    *,
    out_path: str | Path | None = None,
    points: int = 101,
) -> Path:
    """Rebuild a run's loss from disk and plot the schedule it ended on.

    Reads ``config.json`` for the objective and, when the loss carries learned
    parameters, the ``loss.pt`` in the newest resume snapshot — so for a
    learned weighting this shows the curriculum *after* training, not the
    initialisation.
    """
    import json

    import torch
    from diffusers import DDPMScheduler

    from awwl.losses import get_loss_function, loss_module

    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    loss_cfg = dict(cfg["loss"])
    name = loss_cfg.pop("name")

    scheduler = DDPMScheduler(
        num_train_timesteps=int(cfg.get("scheduler", {}).get("num_train_timesteps", 1000))
    )
    loss_fn = get_loss_function(name, noise_scheduler=scheduler, **loss_cfg)
    module = loss_module(loss_fn)
    if module is None or not hasattr(module, "weight_profile"):
        raise ValueError(
            f"loss {name!r} has no sub-band weight profile to plot; "
            "use one of the wavelet_* objectives"
        )

    pointer = run / "state" / "latest.json"
    if pointer.exists():
        snapshot = run / "state" / json.loads(pointer.read_text(encoding="utf-8"))["dir"] / "loss.pt"
        if snapshot.exists():
            module.load_state_dict(torch.load(snapshot, map_location="cpu", weights_only=False))
            logger.info("loaded learned loss state from %s", snapshot)

    sigmas = torch.linspace(0.01, 0.99, points)
    profile = module.weight_profile(sigmas)
    out = Path(out_path) if out_path else run / "curriculum.png"
    return plot_weight_profile(profile, sigmas.numpy(), out_path=out, title=f"{name} — {run.name}")
