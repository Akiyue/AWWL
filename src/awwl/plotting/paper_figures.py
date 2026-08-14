"""The paper's argument figures, generated from the results ledger.

Each of these carries a claim that prose states less well:

* :func:`plot_effect_sizes` — every configuration's difference from the
  baseline with its confidence interval. For a replication this is the
  headline: intervals that straddle zero say more than any table of means.
* :func:`plot_correction_value` — what the spectral correction is worth,
  and how far the trained objective falls short of it. The accounting of the
  paper's central section, in one panel.
* :func:`plot_weight_schedule` — the published weights against σ with the
  total, showing it drift from α to 1−α, and the Min-SNR curve it tracks.
* :func:`plot_convergence` — quality against training epoch.

Design constraints, applied throughout:

* One axis per figure. Where two quantities of different scale must be
  compared (weights against Min-SNR) both are normalised to unit mean first,
  which is legitimate because both *are* per-timestep multipliers — a second
  y-axis would not be.
* Identity is never carried by colour alone: every series also differs in
  line style or marker, so the figures survive greyscale printing and
  colour-vision deficiency.
* Categorical hues are assigned in fixed order, never cycled, from a palette
  checked for CVD separation against the light surface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from awwl.plotting._style import apply_style
from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)

# Validated categorical palette (light surface #fcfcfb): lightness band,
# chroma floor, CVD separation and normal-vision floor all pass. Aqua sits
# below 3:1 contrast, so it is only ever used with a direct label or legend
# text beside it, never as the sole carrier of meaning.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#c9c8c2"

# Fixed display order and styling, so a configuration keeps its identity
# across every figure in the paper.
SERIES = {
    "mse": ("MSE", INK, "-", "o"),
    "static_wavelet": ("Static wavelet", ORANGE, "--", "s"),
    "awwl": ("Frequency-aware", BLUE, "-", "D"),
    "awwl_normalized": ("Normalised", VIOLET, "-.", "^"),
    "awwl_norm_matched": ("Budget fixed", AQUA, ":", "v"),
}


def _finish(fig, ax, out_path: str | Path):
    for one in ax if isinstance(ax, (list, tuple)) else [ax]:
        for spine in ("top", "right"):
            one.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            one.spines[spine].set_color(MUTED)
        one.tick_params(colors=MUTED, labelcolor=INK)
        one.grid(True, linestyle=":", linewidth=0.7, color=GRID, alpha=0.9)
        one.set_axisbelow(True)
    fig.tight_layout()
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def plot_effect_sizes(
    rows,
    *,
    metric: str,
    baseline: str,
    out_path: str | Path,
    alpha: float = 0.05,
):
    """Difference from ``baseline`` per configuration, with 95% intervals.

    A filled marker marks a difference that survives Holm correction; hollow
    marks one that does not. Significance is therefore encoded by shape, not by
    colour, and the zero line does the rest of the work: anything whose
    interval crosses it is a result the data cannot distinguish from the
    baseline.
    """
    import matplotlib.pyplot as plt

    from awwl.analysis.stats import _t_critical, compare_to_baseline

    apply_style()
    results = compare_to_baseline(rows, metric=metric, baseline=baseline, alpha=alpha)
    if not results:
        raise ValueError(f"no comparable configurations for {metric!r}")

    results = sorted(results, key=lambda r: r.mean_delta)
    labels, deltas, halves, filled, colours = [], [], [], [], []
    for r in results:
        se = abs(r.mean_delta / r.t_stat) if r.t_stat else 0.0
        label, colour = SERIES.get(r.group, (r.group, BLUE))[:2]
        labels.append(label)
        colours.append(colour)
        deltas.append(r.mean_delta)
        halves.append(_t_critical(r.n_pairs - 1, 0.95) * se)
        filled.append(r.significant)

    y = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(7.2, 0.52 * len(results) + 1.9))

    ax.axvline(0, color=INK, linewidth=1.4, zorder=2)
    ax.errorbar(
        deltas, y, xerr=halves, fmt="none", ecolor=MUTED, elinewidth=1.6,
        capsize=4, capthick=1.4, zorder=3,
    )
    # Each configuration keeps the colour it has in every other figure, so the
    # reader carries one mapping through the paper rather than four.
    for yi, d, is_sig, colour in zip(y, deltas, filled, colours, strict=True):
        ax.plot(
            d, yi, marker="o", markersize=8, zorder=4,
            color=colour if is_sig else "white",
            markeredgecolor=colour, markeredgewidth=1.8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(results) - 0.3)
    ax.set_xlabel(f"$\\Delta$ {metric.upper()} vs. {baseline.upper()}"
                  "   (left of the line is better)")
    ax.set_title(f"Effect on {metric.upper()}, five seeds, 95% CI", loc="left")

    # A legend of two entries would be heavier than the sentence it replaces.
    ax.text(
        0.99, 0.02,
        "filled = significant after Holm correction",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, color=MUTED,
    )
    return _finish(fig, ax, out_path)


def plot_correction_value(
    arms: dict[str, tuple[float, float, float, float]],
    *,
    out_path: str | Path,
):
    """What the spectral correction buys, and what the objective actually gets.

    Args:
        arms: ``{label: (deficit_db, fid, boosted_deficit_db, boosted_fid)}``.

    The arrow from each arm's own point to its post-processed point *is* the
    value of the correction. The baseline's arrow, translated onto the
    frequency-aware arm, is where that arm would sit if the correction were the
    only difference between them; the gap between that and where it does sit is
    the paper's central quantity.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    for key, (d0, f0, d1, f1) in arms.items():
        label, colour, _ls, marker = SERIES.get(key, (key, MUTED, "-", "o"))
        ax.annotate(
            "", xy=(d1, f1), xytext=(d0, f0),
            arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.8,
                            shrinkA=7, shrinkB=7),
        )
        ax.plot(d0, f0, marker=marker, markersize=9, color=colour,
                markeredgecolor="white", markeredgewidth=1.4, label=f"{label}, as trained")
        ax.plot(d1, f1, marker=marker, markersize=9, color="white",
                markeredgecolor=colour, markeredgewidth=2.0,
                label=f"{label}, spectrum corrected")

    if "mse" in arms and "awwl" in arms:
        bd0, bf0, bd1, bf1 = arms["mse"]
        ad0, af0, _ad1, _af1 = arms["awwl"]

        # The baseline's post-processed spectrum lands almost exactly where the
        # trained objective's does unaided, so the vertical gap between them is
        # very nearly a controlled comparison: same spectrum, different FID.
        # No separate "predicted" marker is drawn -- it would sit on top of the
        # baseline's own point and read as clutter rather than as the argument.
        ax.annotate(
            "", xy=(ad0, af0), xytext=(bd1, bf1),
            arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=1.3,
                            linestyle=(0, (4, 2.5))),
        )
        ax.text(
            (ad0 + bd1) / 2 - 0.03, (bf1 + af0) / 2,
            f"{af0 - bf1:+.2f} FID\nat the same spectrum",
            ha="right", va="center", fontsize=11.5, color=INK,
        )
        ax.set_xlim(max(bd0, ad0) + 0.30, min(bd1, ad0) - 0.55)
    else:
        lo = min(min(v[0], v[2]) for v in arms.values())
        hi = max(max(v[0], v[2]) for v in arms.values())
        ax.set_xlim(hi + 0.2, lo - 0.2)

    ax.set_xlabel("High-band spectral deficit (dB)   $\\longrightarrow$ closer to real")
    ax.set_ylabel("FID")
    ax.set_title("What the spectral correction is worth", loc="left")
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", handletextpad=0.6)
    return _finish(fig, ax, out_path)


def plot_weight_schedule(
    *,
    alpha: float,
    power: float,
    out_path: str | Path,
    snr_gamma: float = 5.0,
):
    """The published weights against σ, with the total and the Min-SNR curve.

    The total is the point of the figure: it is not constant, so α also sets a
    timestep schedule. Both the total and the Min-SNR weight are normalised to
    unit mean before plotting — they are the same kind of quantity, a
    per-timestep multiplier, which is what makes one shared axis honest here
    where a second y-axis would not be.
    """
    import matplotlib.pyplot as plt
    import torch
    from diffusers import DDPMScheduler

    from awwl.losses import AdaptiveWaveletLoss
    from awwl.losses.weighting import min_snr_weight

    apply_style()
    # Sampled at the sigma of each training timestep, not uniformly in sigma.
    # The correlation below depends on this choice -- uniform-in-sigma gives
    # 0.87 and a handful of round sigma values gives 0.92 -- and the only
    # defensible grid is the one training actually draws from.
    sigmas = torch.sqrt(1.0 - DDPMScheduler(num_train_timesteps=1000).alphas_cumprod)
    w = AdaptiveWaveletLoss(alpha=alpha, power=power).weights_at(sigmas)
    s = sigmas.numpy()

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(s, w["w_ll"].numpy(), color=BLUE, linewidth=2.2, label="$w_{LL}$ (structure)")
    ax.plot(s, w["w_det"].numpy(), color=ORANGE, linewidth=2.2, linestyle="--",
            label="$w_{det}$ (detail)")

    total = w["total"].numpy()
    ax.plot(s, total / total.mean(), color=INK, linewidth=2.6, linestyle="-",
            label="total (unit mean)")

    snr = min_snr_weight(sigmas, snr_gamma).numpy()
    ax.plot(s, snr / snr.mean(), color=VIOLET, linewidth=2.0, linestyle=(0, (1, 1.6)),
            label=f"Min-SNR $\\gamma={snr_gamma:g}$ (unit mean)")

    corr = float(np.corrcoef(total, snr)[0, 1])
    ax.text(
        0.5, 0.03, f"total vs. Min-SNR:  $r = {corr:+.2f}$",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=11, color=INK,
    )

    ax.set_xlabel("noise level $\\sigma_t$   (high noise $\\longrightarrow$)")
    ax.set_ylabel("weight")
    ax.set_title(f"The weighting at $\\alpha={alpha:g}$, $p={power:g}$", loc="left")
    ax.legend(frameon=False, fontsize=10, ncol=2)
    return _finish(fig, ax, out_path)


def plot_convergence(
    rows,
    *,
    metric: str,
    out_path: str | Path,
    configs=None,
    baseline: str | None = None,
):
    """Quality against training epoch, averaged over seeds.

    With a *baseline*, draws a second panel of the seed-paired difference from
    it. That panel is the one worth reading: over 200 epochs FID falls by tens
    of points while the configurations separate by barely one, so on a shared
    absolute axis every curve collapses onto the same band and the effect the
    paper is about is invisible. The absolute panel stays because it is the
    evidence that training converged at all.
    """
    import matplotlib.pyplot as plt

    from awwl.analysis.stats import summarize_groups

    apply_style()
    epochs = sorted({r["epoch"] for r in rows if r.get("epoch") is not None})
    names = configs or [k for k in SERIES if any(str(r.get("group")) == k for r in rows)]

    def curve(key: str) -> tuple[list[float], list[float], dict[float, dict]]:
        xs, ys, per_seed = [], [], {}
        for e in epochs:
            subset = [r for r in rows if str(r.get("group")) == key and r.get("epoch") == e]
            summary = summarize_groups(subset, metric=metric)
            if summary:
                xs.append(e)
                ys.append(summary[0].mean)
                per_seed[e] = summary[0].by_seed
        return xs, ys, per_seed

    base_seeds: dict[float, dict] = {}
    if baseline is not None:
        _, _, base_seeds = curve(baseline)
        if not base_seeds:
            raise ValueError(f"baseline {baseline!r} has no rows for metric {metric!r}")

    if baseline is None:
        fig, ax_abs = plt.subplots(figsize=(7.2, 4.6))
        axes, ax_rel = [ax_abs], None
    else:
        fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(10.6, 4.3), sharex=True)
        axes = [ax_abs, ax_rel]

    plotted = 0
    for key in names:
        label, colour, style, marker = SERIES.get(key, (key, MUTED, "-", "o"))
        xs, ys, per_seed = curve(key)
        if len(xs) < 2:
            continue
        ax_abs.plot(xs, ys, color=colour, linestyle=style, marker=marker,
                    markersize=6, linewidth=2.0, label=label)
        plotted += 1

        if ax_rel is None or key == baseline:
            continue
        # Pair within seed before differencing: the seed effect is far larger
        # than the effect under test, and differencing means removes none of it.
        dxs, dys = [], []
        for e in xs:
            shared = set(per_seed.get(e, {})) & set(base_seeds.get(e, {}))
            if shared:
                dxs.append(e)
                dys.append(
                    float(np.mean([per_seed[e][s] - base_seeds[e][s] for s in shared]))
                )
        if len(dxs) >= 2:
            ax_rel.plot(dxs, dys, color=colour, linestyle=style, marker=marker,
                        markersize=6, linewidth=2.0, label=label)
    if not plotted:
        raise ValueError("no configuration had more than one evaluated epoch")

    ax_abs.set_ylabel(f"{metric.upper()}  (mean over seeds)")
    ax_abs.set_title(f"{metric.upper()} through training", loc="left")
    for ax in axes:
        ax.set_xlabel("training epoch")

    # The legend lives on the absolute panel because that is the one carrying
    # every series; the baseline appears in no other legend, being identically
    # zero on the right.
    ax_abs.legend(frameon=False, fontsize=10)
    if ax_rel is not None:
        ax_rel.axhline(0, color=INK, linewidth=1.4, zorder=2)
        ax_rel.set_ylabel(f"$\\Delta$ {metric.upper()} vs. {baseline.upper()}")
        ax_rel.set_title("the same data, paired by seed", loc="left")
    return _finish(fig, axes, out_path)


def parse_boost_tables(folder: str | Path) -> dict[str, tuple[float, float, float, float]]:
    """Recover the boost experiment's numbers from ``scripts/boost_test.sh`` output.

    Averages across seeds per configuration. Reads the per-run text tables
    rather than asking for the numbers to be retyped, for the same reason the
    tables are generated: a figure that disagrees with its own data is worse
    than no figure.
    """
    import re

    per_config: dict[str, list[tuple[float, float, float, float]]] = {}
    for path in sorted(Path(folder).glob("*.txt")):
        name = path.stem.rsplit("_s", 1)[0]
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\s*(-?[\d.]+)d\s+(-?[\d.]+)d\s+([\d.]+)\s", line)
            if m:
                rows.append((float(m.group(2)), float(m.group(3))))
        if len(rows) >= 2:
            (d0, f0), (d1, f1) = rows[0], rows[1]
            per_config.setdefault(name, []).append((d0, f0, d1, f1))

    return {
        name: tuple(float(np.mean([r[i] for r in runs])) for i in range(4))  # type: ignore[misc]
        for name, runs in per_config.items()
    }
