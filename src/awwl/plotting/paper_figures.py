"""The paper's figures, generated from the results ledgers.

Every figure in ``paper/figures`` is written by one function here (or in
:mod:`awwl.plotting.spectrum` and :mod:`awwl.plotting.paired_grid`), and
:mod:`awwl.plotting.paper_build` maps each file name to the ledger it reads,
so ``awwl figures --out paper/figures`` rebuilds the whole set.

Design rules, applied throughout:

* Authored at the printed width (see :mod:`awwl.plotting._style`), with no
  titles inside the artwork: the caption carries the claim, and multi-panel
  figures carry only a panel letter and a short descriptive header.
* One colour per training arm, fixed across the paper (:data:`SERIES`).
  Quantities that are not arms (band weights, samplers, operators) are drawn
  in ink and grey with distinct line styles, and an arm without a fixed hue
  falls back to grey, never to the audited objective's blue.
* Identity is never carried by colour alone: every series also differs in line
  style or marker, so the figures survive greyscale printing and colour-vision
  deficiency. Filled markers mean "significant after Holm correction" in every
  figure that uses fill at all.
* Every comparison is paired by training seed and shown with its 95%
  t-interval, because the seed-to-seed spread is as large as most effects in
  this study.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from awwl.plotting._style import apply_style, figure_size, save_figure

logger = logging.getLogger(__name__)

# Validated categorical palette (light surface #fcfcfb, dataviz validator):
# the first four pass every all-pairs check. RED is used only for the
# sum-of-squares control, which never shares a line figure with ORANGE. AQUA
# and RED sit below 3:1 contrast or in the CVD floor band against a neighbour,
# so they are only used with a direct label, legend text or a distinct line
# style beside them.
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#d6d5cf", "#efeee9"

#: Every arm's printed name: a copy of ``NAMES`` in ``scripts/make_tables.py``,
#: the one map that every table, every figure and the text use, so an arm
#: carries one name wherever it is read. The package does not import the
#: script; ``tests/test_paper_figures.py`` checks that the two copies agree.
#: Matplotlib's mathtext renders the TeX. ``\omega_b`` is the effective weight
#: of sub-band ``b`` (each detail band carries ``w_det/3`` when the detail
#: bands are averaged, as in AWWL, and ``w_det`` when they are summed).
DISPLAY_NAMES: dict[str, str] = {
    "mse": r"MSE",
    "l1": r"$L_1$",
    "huber": r"Huber",
    "charbonnier": r"Charbonnier",
    "snr_weighted": r"Raw SNR weighting",
    "min_snr": r"Min-SNR-$\gamma$",
    "focal_frequency": r"FFL",
    "static_wavelet": r"Static wavelet ($p = 0$)",
    "awwl": r"AWWL",
    "awwl_normalized": r"AWWL, $\sum_b \omega_b = 1$",
    "awwl_norm_matched": r"AWWL, $\sum_b \omega_b = 0.5$",
    "awwl_grad_matched_mean": r"AWWL, $\sum_b \omega_b^2$ fixed",
    "awwl_eq7": r"AWWL, detail bands summed",
    "awwl_grad_matched": r"AWWL, detail bands summed, $\sum_b \omega_b^2$ fixed",
    "awwl_timestep": r"AWWL, timestep only",
    "mse_planted": r"MSE, planted change",
    "mse_ema": r"MSE + EMA",
    "awwl_ema": r"AWWL + EMA",
    "awwl_subband": r"Per-sub-band schedules",
    "awwl_spatial": r"Spatially adaptive",
    "awwl_multilevel": r"Multi-level (3 levels)",
    "awwl_learned": r"Learned weighting",
    "awwl_gradnorm": r"GradNorm balancing",
    "awwl_lifting": r"Learned basis (lifting)",
    "awwl_full": r"All axes combined",
}

#: The groups the arms are presented in, in order: a copy of ``BLOCKS`` in
#: ``scripts/make_tables.py`` (checked by the tests). Fig. effects and Fig.
#: pricing-arms use these headers and this order, as the tables do.
BLOCKS: list[tuple[str, tuple[str, ...]]] = [
    ("Conference-paper arms", ("awwl", "static_wavelet", "awwl_eq7")),
    ("Normalisation controls", ("awwl_normalized", "awwl_norm_matched", "awwl_grad_matched_mean",
                                "awwl_grad_matched")),
    ("Timestep schedule alone", ("awwl_timestep",)),
    ("Planted-mechanism control", ("mse_planted",)),
    ("Other weightings", ("min_snr", "snr_weighted", "focal_frequency")),
    ("Pixel losses", ("l1", "huber", "charbonnier")),
    ("Extensions (unnormalised)", ("awwl_subband", "awwl_spatial", "awwl_multilevel", "awwl_learned",
                                   "awwl_gradnorm", "awwl_lifting", "awwl_full")),
    ("With EMA", ("mse_ema", "awwl_ema")),
]

def for_mathtext(text: str) -> str:
    r"""``text`` as matplotlib's mathtext should receive it, reading as TeX does inline.

    Mathtext sets the limits of ``\sum`` beneath it even inside a line of
    text, where TeX sets them to the right; ``{\sum}_b`` gives TeX's inline
    form. Nothing else in the display names needs changing.
    """
    return re.sub(r"\\sum_(\w|\{[^}]*\})\s*", r"{\\sum}_\1\\,", text)


#: Every arm in presentation order: the baseline, then :data:`BLOCKS`.
PRESENTATION_ORDER: tuple[str, ...] = ("mse",) + tuple(k for _t, keys in BLOCKS for k in keys)

# Training arms with a fixed identity across the paper's figures:
# (name, colour, line style, marker); the name is DISPLAY_NAMES'.
SERIES = {
    key: (for_mathtext(DISPLAY_NAMES[key]), colour, dash, marker)
    for key, (colour, dash, marker) in {
        "mse": (INK, "-", "o"),
        "static_wavelet": (ORANGE, "--", "s"),
        "awwl": (BLUE, "-", "D"),
        "awwl_normalized": (VIOLET, "-.", "^"),
        "awwl_norm_matched": (AQUA, ":", "v"),
        "awwl_grad_matched_mean": (RED, (0, (5, 1.5, 1, 1.5, 1, 1.5)), "h"),
    }.items()
}

# Markers for arms without a fixed hue, so that every arm of the DreamBooth
# scatter, where the markers carry identity through the legend, has a marker
# of its own; the same arm has the same marker in every figure. None of these
# repeats a SERIES marker. Other arms without a hue are grey circles.
EXTRA_MARKERS = {"min_snr": "P", "snr_weighted": "X", "l1": "<", "huber": ">",
                 "charbonnier": "p", "awwl_learned": "*"}

#: The one key to marker fill, drawn in every figure where fill carries
#: significance: ``(filled label, open label)``.
FILL_KEY = (r"filled: $p_{\mathrm{Holm}} < 0.05$", r"open: $p_{\mathrm{Holm}} \geq 0.05$")

#: Arms of the forest plot that may be absent: the figure raises for any other
#: arm without rows, which is how the gradient-scale control once went missing
#: from a build unnoticed. The timestep-only arm is drawn only once all its
#: seeds are in (:data:`EFFECT_MIN_SEEDS`).
EFFECT_OPTIONAL: frozenset[str] = frozenset({"awwl_grad_matched_mean", "awwl_timestep", "mse_planted"})

#: The CIFAR-10 comparison family of the forest plot, in its fixed display
#: order: :data:`BLOCKS`, each arm marked ``(key, optional)``.
EFFECT_BLOCKS: list[tuple[str, list[tuple[str, bool]]]] = [
    (title, [(key, key in EFFECT_OPTIONAL) for key in keys]) for title, keys in BLOCKS
]

#: The Holm family of the forest plot: every arm compared with MSE at epoch
#: 199 (a copy of make_tables.py FAMILY_MAIN, checked by the tests). Declared,
#: not read from the ledger: an arm of the family without rows yet enters with
#: p = 1, so a marker's fill does not depend on which jobs have finished.
EFFECT_FAMILY: tuple[str, ...] = (
    "l1", "huber", "charbonnier", "snr_weighted", "min_snr", "focal_frequency",
    "static_wavelet", "awwl", "awwl_normalized", "awwl_norm_matched",
    "awwl_grad_matched_mean", "awwl_eq7", "awwl_grad_matched", "mse_ema",
    "awwl_subband", "awwl_multilevel", "awwl_spatial", "awwl_learned",
    "awwl_gradnorm", "awwl_lifting", "awwl_full",
)

#: Arms compared with their own reference instead of MSE, each a one-test
#: family: AWWL + EMA against MSE + EMA (make_tables.py EMA_PAIR).
EFFECT_PAIRS: dict[str, str] = {"awwl_ema": "mse_ema"}

#: Arms corrected in a Holm family of their own, given as its ``(arm,
#: reference)`` contrasts, the first being the one drawn: the timestep-only
#: arm against MSE, corrected together with AWWL against it
#: (make_tables.py FAMILY_TIMESTEP). Such an arm is drawn only once it has
#: :data:`EFFECT_MIN_SEEDS` seeds, and its label carries a dagger.
EFFECT_OWN_FAMILIES: dict[str, tuple[tuple[str, str], ...]] = {
    "awwl_timestep": (("awwl_timestep", "mse"), ("awwl", "awwl_timestep")),
    # The planted-mechanism control is a one-test family of its own.
    "mse_planted": (("mse_planted", "mse"),),
}
EFFECT_MIN_SEEDS: dict[str, int] = {"awwl_timestep": 5, "mse_planted": 5}


def display_name(group: str, *, flat: bool = False) -> str:
    """The paper's name for a ledger group, ready for mathtext.

    ``flat`` is accepted for callers written when nested rows had shortened
    names; every arm now has one name, printed the same everywhere.
    """
    del flat
    return for_mathtext(DISPLAY_NAMES.get(group, group))


#: Markers that read smaller than a circle at the same size, and the factor
#: that brings them level with it.
MARKER_SCALE: dict[str, float] = {"*": 1.4, "P": 1.05, "X": 1.05}


def marker_size(marker: str, size: float) -> float:
    """``size`` for ``marker``, scaled so that every marker reads equally large."""
    return size * MARKER_SCALE.get(marker, 1.0)


def style_of(group: str) -> tuple[str, object, str]:
    """``(colour, line style, marker)`` for an arm; grey for arms without a hue."""
    if group in SERIES:
        _, colour, dash, marker = SERIES[group]
        return colour, dash, marker
    return MUTED, "-", EXTRA_MARKERS.get(group, "o")


# --------------------------------------------------------------------------
# Shared helpers


def _chrome(ax, *, grid: str | None = "both") -> None:
    """Recessive axes: no top/right spines, grey ticks, light grid."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    if grid:
        ax.grid(True, axis=grid, linestyle="-", linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def _fill_legend(ax, *, colour: str = MUTED, marker: str = "o", size: float = 4.6,
                 labels: tuple[str, str] = FILL_KEY, **kwargs):
    """The key to marker fill: filled is significant after Holm correction, open is not.

    Added to ``ax`` as an artist of its own, so a legend the axes already
    carries stays. The key sits on a white ground so that grid lines do not
    run through it.
    """
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], linestyle="none", marker=marker, markersize=size, color=colour,
                      markerfacecolor=face, markeredgecolor=colour, markeredgewidth=0.9)
               for face in (colour, "white")]
    opts = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
                fontsize=7, handletextpad=0.3, borderaxespad=0.3, borderpad=0.25,
                labelspacing=0.25, handlelength=1.0)
    opts.update(kwargs)
    legend = Legend(ax, handles, list(labels), **opts)
    legend.set_zorder(6)
    ax.add_artist(legend)
    return legend


def _panel(ax, letter: str, header: str = "") -> None:
    """Panel letter and a short descriptive header, above the plotting area."""
    text = f"({letter})" + (f"  {header}" if header else "")
    ax.set_title(text, loc="left", fontsize=9, pad=4)


def _mean_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Mean and t-interval half-width of a set of (paired) values."""
    from awwl.analysis.stats import _t_critical

    v = np.asarray(list(values), dtype=float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), 0.0
    half = _t_critical(len(v) - 1, confidence) * float(v.std(ddof=1)) / np.sqrt(len(v))
    return float(v.mean()), float(half)


def _halfwidth(result) -> float:
    """95% t-interval half-width of a :class:`ComparisonResult`'s mean delta."""
    from awwl.analysis.stats import _t_critical

    if not result.t_stat or not np.isfinite(result.t_stat):
        return 0.0
    se = abs(result.mean_delta / result.t_stat)
    return _t_critical(result.n_pairs - 1, 0.95) * se


def _paired(rows, group: str, baseline: str, metric: str) -> dict:
    """Per-seed ``group - baseline`` differences, over the seeds both share."""
    from awwl.analysis.stats import summarize_groups

    summary = {s.group: s.by_seed for s in summarize_groups(rows, metric=metric)}
    a, b = summary.get(group, {}), summary.get(baseline, {})
    return {s: a[s] - b[s] for s in a if s in b}


def _fmt_signed(value: float, digits: int = 2) -> str:
    """``+0.79`` / ``−0.35``: a true minus sign, as the typeset text uses.

    A value that rounds to zero prints unsigned (``0.00``), as the generated
    tables print it.
    """
    if round(value, digits) == 0:
        return f"{0.0:.{digits}f}"
    return f"{value:+.{digits}f}".replace("-", "−")


def _fmt_offscale(value: float) -> str:
    """An off-axis mean at the tables' precision: two decimals, three
    significant figures from 10 (``+12.3``, ``+306``)."""
    big = abs(value)
    return _fmt_signed(value, 0 if big >= 100 else 1 if big >= 10 else 2)


def _training_sigmas(schedule: str = "linear"):
    """σ_t of the 1000 training timesteps (CIFAR-10 linear or SD scaled-linear)."""
    import torch
    from diffusers import DDPMScheduler

    if schedule == "linear":
        sched = DDPMScheduler(num_train_timesteps=1000)
    elif schedule == "scaled_linear":
        # Stable Diffusion 1.5's training schedule, which the DreamBooth runs use.
        sched = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="scaled_linear",
            beta_start=0.00085, beta_end=0.012,
        )
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return torch.sqrt(1.0 - sched.alphas_cumprod)


def _training_timesteps(n: int) -> np.ndarray:
    """The training timesteps ``t = 1, ..., T`` of Section 3.1 (index ``i`` is ``t = i + 1``)."""
    return np.arange(1, n + 1)


#: Ticks of every training-timestep axis: its two ends, t = 1 and t = T.
TIMESTEP_TICKS = (1, 200, 400, 600, 800, 1000)


def _sigma_axis(ax, sigmas, ticks=(0.2, 0.5, 0.8, 0.9, 0.99)) -> None:
    """A top axis giving σ_t at the plotted training timesteps."""
    s = np.asarray(sigmas, dtype=float)
    t = _training_timesteps(len(s)).astype(float)

    def forward(x):
        return np.interp(x, t, s)

    def inverse(y):
        return np.interp(y, s, t)

    top = ax.secondary_xaxis("top", functions=(forward, inverse))
    top.set_xticks(list(ticks))
    top.set_xticklabels([f"{v:g}" for v in ticks])
    top.tick_params(colors=MUTED, labelcolor=INK, labelsize=7.5, length=2.5, pad=1.5)
    top.spines["top"].set_color(MUTED)
    top.spines["top"].set_linewidth(0.8)
    top.set_xlabel("noise level $\\sigma_t$", fontsize=8, labelpad=2)


# --------------------------------------------------------------------------
# Forest plot of the CIFAR-10 comparison family


def effect_results(
    rows,
    *,
    metric: str,
    baseline: str,
    alpha: float = 0.05,
    blocks: list[tuple[str, list[tuple[str, bool]]]] | None = None,
    family: Sequence[str] | None = None,
    pairs: dict[str, str] | None = None,
    own_families: dict[str, tuple[tuple[str, str], ...]] | None = None,
    min_seeds: dict[str, int] | None = None,
):
    """The comparisons :func:`plot_effect_sizes` draws, Holm-corrected as it draws them.

    Returns ``(results, pairs, own)``: ``{arm: ComparisonResult}`` for every
    arm of ``blocks`` that can be drawn, with ``p_holm`` and ``significant``
    set by the arm's family, and the pairs and own families in force.
    """
    from awwl.analysis.stats import _holm, compare_to_baseline

    blocks = blocks or EFFECT_BLOCKS
    wanted = [g for _, arms in blocks for g, _ in arms]
    pairs = dict(EFFECT_PAIRS if pairs is None else pairs)
    pairs = {a: ref for a, ref in pairs.items() if a in wanted}
    own = dict(EFFECT_OWN_FAMILIES if own_families is None else own_families)
    own = {a: fam for a, fam in own.items() if a in wanted and a not in pairs}
    min_seeds = dict(EFFECT_MIN_SEEDS if min_seeds is None else min_seeds)
    main = [g for g in wanted if g not in pairs and g not in own]
    family = [g for g in (EFFECT_FAMILY if family is None else family)]
    family += [g for g in main if g not in family]
    if not any(str(r.get("group")) == baseline for r in rows):
        raise ValueError(f"baseline {baseline!r} has no rows for {metric!r}")

    def seeds_of(group: str) -> int:
        return len({r.get("seed") for r in rows if str(r.get("group")) == group
                    and isinstance(r.get(metric), (int, float)) and r.get(metric) != -1})

    def against(arm: str, ref: str):
        sub = [r for r in rows if str(r.get("group")) in (arm, ref)]
        if not any(str(r.get("group")) == ref for r in sub):
            return None
        found = [c for c in compare_to_baseline(sub, metric=metric, baseline=ref, alpha=alpha)
                 if c.group == arm]
        return found[0] if found else None

    subset = [r for r in rows if str(r.get("group")) in set(family) | {baseline}]
    results = {r.group: r for r in compare_to_baseline(
        subset, metric=metric, baseline=baseline, alpha=alpha)}
    # Holm over the declared family; an arm without rows counts as p = 1.
    adjusted = _holm([results[g].p_value if g in results else 1.0 for g in family])
    for g, p_adj in zip(family, adjusted, strict=True):
        if g in results:
            results[g].p_holm = p_adj
            results[g].significant = p_adj < alpha
    for arm, ref in pairs.items():
        found = against(arm, ref)
        if found is not None:
            results[arm] = found
    # An arm with its own family: the first contrast is drawn, with the Holm
    # p over the family's contrasts (an unmeasured contrast counts as p = 1).
    for arm, contrasts in own.items():
        tests = [against(a, ref) for a, ref in contrasts]
        if tests[0] is None:
            continue
        adjusted = _holm([t.p_value if t is not None else 1.0 for t in tests])
        tests[0].p_holm = adjusted[0]
        tests[0].significant = adjusted[0] < alpha
        results[arm] = tests[0]
    results = {g: r for g, r in results.items()
               if g in wanted and seeds_of(g) >= min_seeds.get(g, 2)}
    return results, pairs, own


def plot_effect_sizes(
    rows,
    *,
    metric: str,
    baseline: str,
    out_path: str | Path,
    alpha: float = 0.05,
    blocks: list[tuple[str, list[tuple[str, bool]]]] | None = None,
    family: Sequence[str] | None = None,
    pairs: dict[str, str] | None = None,
    own_families: dict[str, tuple[tuple[str, str], ...]] | None = None,
    min_seeds: dict[str, int] | None = None,
    reference: tuple[str, str] | None = ("Extensions (unnormalised)", "awwl_eq7"),
    fraction: float = 0.92,
):
    """Paired difference from ``baseline`` for every arm, with 95% t-intervals.

    Rows come from ``blocks`` (default :data:`EFFECT_BLOCKS`), not from the
    ledger: a required arm with no rows raises, an optional one appears as
    soon as its rows exist. Filled = Holm-adjusted p < ``alpha``; hollow =
    not. The Holm family is ``family`` (default :data:`EFFECT_FAMILY`, the
    generated tables' family), with p = 1 for an arm not yet run, so fills
    match the tables. Arms in ``pairs`` (default :data:`EFFECT_PAIRS`) are
    compared with their own reference, each as a one-test family, and say so
    in their label. Arms in ``own_families`` (default
    :data:`EFFECT_OWN_FAMILIES`) are compared with MSE but corrected within
    their own family of contrasts, and carry a dagger. An arm with fewer seeds
    than ``min_seeds`` (default :data:`EFFECT_MIN_SEEDS`) is left out, as the
    tables leave it out. Arms whose interval is an order of magnitude beyond the
    rest (runs that collapsed or became unstable) are drawn as an arrow at the
    axis edge with their mean printed. Three-seed arms say so in their label.

    ``reference`` draws a dotted rule through one block at another arm's mean:
    the extension arms were trained unnormalised on the detail-summed shape,
    so their like-for-like reference is that arm, not the baseline alone.
    """
    import matplotlib.pyplot as plt

    apply_style()
    blocks = blocks or EFFECT_BLOCKS
    results, pairs, own = effect_results(
        rows, metric=metric, baseline=baseline, alpha=alpha, blocks=blocks, family=family,
        pairs=pairs, own_families=own_families, min_seeds=min_seeds)
    if not results:
        raise ValueError(f"no comparable configurations for {metric!r}")
    missing = [g for _, arms in blocks for g, opt in arms if not opt and g not in results]
    if missing:
        raise ValueError(f"arms in the fixed list have no {metric!r} rows: {missing}")

    # Layout: one header row per block, then its arms, top to bottom.
    entries: list[tuple[str, str | None]] = []  # (label, group or None for a header)
    block_rows: dict[str, list[int]] = {}
    for title, arms in blocks:
        present = [g for g, _ in arms if g in results]
        if not present:
            continue
        entries.append((title, None))
        for g in present:
            n = results[g].n_pairs
            notes = [] if n >= 5 else [f"$n = {n}$"]
            if g in pairs:
                notes.append(f"vs. {display_name(pairs[g])}")
            name = display_name(g)
            if notes and name.endswith(")"):
                name = f"{name[:-1]}; {'; '.join(notes)})"
            elif notes:
                name = f"{name} ({'; '.join(notes)})"
            if g in own:
                name += "$^{\\dagger}$"
            block_rows.setdefault(title, []).append(len(entries))
            entries.append((name, g))
    n_rows = len(entries)
    y_of = {i: n_rows - 1 - i for i in range(n_rows)}
    # A dagger row's family, spelled out beneath the axis (the caption need
    # not repeat it): each own-family arm drawn, with the contrasts it is
    # corrected with.
    def _who(key: str, arm: str) -> str:
        return "this arm" if key == arm else display_name(key)

    counts = {2: "two", 3: "three", 4: "four"}
    family_notes = [
        "$^{\\dagger}$Corrected in its own Holm family of "
        + f"{counts.get(len(own[g]), len(own[g]))} tests: "
        + ", and ".join(f"{_who(a, g)} vs. {_who(r, g)}" for a, r in own[g]) + "."
        for _label, g in entries if g in own]
    note_in = 0.16 * len(family_notes)

    spans = [abs(results[g].mean_delta) + _halfwidth(results[g]) for _, g in entries if g]
    cut = _divergence_cut(spans)
    limit = cut * 1.10 if cut is not None else max(spans) * 1.05
    lower = min(-0.6, min(results[g].mean_delta - _halfwidth(results[g])
                         for _, g in entries if g) - 0.3)

    width = figure_size(fraction)[0]
    height = 0.158 * n_rows + 0.62 + note_in
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axvline(0, color=INK, linewidth=0.8, zorder=2)

    for i, (_label, g) in enumerate(entries):
        if g is None:
            continue
        r = results[g]
        y = y_of[i]
        colour, _dash, marker = style_of(g)
        half = _halfwidth(r)
        if abs(r.mean_delta) - half > limit or abs(r.mean_delta) > limit:
            ax.annotate(
                _fmt_offscale(r.mean_delta),
                xy=(limit, y), xytext=(-16, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.0,
                                shrinkA=1, shrinkB=0, mutation_scale=8),
                annotation_clip=False, zorder=4,
            )
            continue
        ax.errorbar([r.mean_delta], [y], xerr=[half], fmt="none", ecolor=colour,
                    elinewidth=1.0, capsize=2.2, capthick=0.9, zorder=3)
        ax.plot(r.mean_delta, y, marker=marker, markersize=marker_size(marker, 5.2),
                linestyle="none", color=colour if r.significant else "white",
                markeredgecolor=colour, markeredgewidth=1.0, zorder=4)

    if reference is not None and reference[0] in block_rows and reference[1] in results:
        ys = [y_of[i] for i in block_rows[reference[0]]]
        x_ref = results[reference[1]].mean_delta
        ax.plot([x_ref, x_ref], [min(ys) - 0.45, max(ys) + 0.45], color=MUTED,
                linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=1)
        ax.annotate(display_name(reference[1]),
                    xy=(x_ref, max(ys) + 0.45), xytext=(2, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7, color=MUTED, style="italic")

    ax.set_xlim(lower, limit)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.set_yticks([y_of[i] for i in range(n_rows)])
    ax.set_yticklabels([label for label, _ in entries])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"$\\Delta${metric.upper()} vs. {display_name(baseline)} "
                  "(paired, 95% CI; negative is better)")
    _chrome(ax, grid="x")

    # Headers left-aligned in the label column, arms indented beneath them.
    for (_label, g), tick in zip(entries, ax.get_yticklabels(), strict=True):
        if g is None:
            tick.set_fontstyle("italic")
            tick.set_color(MUTED)
    from matplotlib.transforms import ScaledTranslation

    indent_pt = 9.0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    col = max(t.get_window_extent(renderer).width * 72.0 / fig.dpi
              + (0.0 if g is None else indent_pt)
              for t, (_label, g) in zip(ax.get_yticklabels(), entries, strict=True))
    ax.tick_params(axis="y", pad=col + 3)
    shift = ScaledTranslation(indent_pt / 72.0, 0, fig.dpi_scale_trans)
    for tick, (_label, g) in zip(ax.get_yticklabels(), entries, strict=True):
        tick.set_horizontalalignment("left")
        if g is not None:
            tick.set_transform(tick.get_transform() + shift)
    # Thin rules between blocks.
    for i, (_label, g) in enumerate(entries):
        if g is None and i > 0:
            ax.axhline(y_of[i] + 0.5, color=GRID, linewidth=0.6, zorder=0)
    # The key to fill, in the empty top right; colour and marker identify the
    # arm, as in every figure.
    _fill_legend(ax, loc="upper right")
    if not family_notes:
        return save_figure(fig, out_path)
    fig.tight_layout(pad=0.4, rect=(0, note_in / height, 1, 1))
    for k, text in enumerate(family_notes):
        fig.text(4.0 / 72.0 / width, (note_in - (k + 0.55) * 0.16) / height, text,
                 ha="left", va="center", fontsize=7, color=MUTED)
    return save_figure(fig, out_path, layout=False)


def _divergence_cut(magnitudes: list[float]) -> float | None:
    """Where the arms that trained end and the ones that diverged begin.

    Find the largest multiplicative gap in the sorted magnitudes and cut there,
    but only when the gap is a real discontinuity (>= 5x) and cutting hides at
    most a third of the arms. Otherwise return ``None`` and let the axis show
    everything, which is the right answer for a sweep whose arms are all of one
    scale.
    """
    finite = sorted(m for m in magnitudes if m > 0)
    if len(finite) < 4:
        return None
    ratios = [(finite[i + 1] / finite[i], i) for i in range(len(finite) - 1)]
    best_ratio, index = max(ratios)
    hidden = len(finite) - (index + 1)
    if best_ratio < 5.0 or hidden > len(finite) / 3:
        return None
    return finite[index]


# --------------------------------------------------------------------------
# Earlier pricing figures, no longer part of the paper's set (paper_build does
# not build them). Kept as library functions: the package exports them and the
# tests exercise them.


def plot_correction_value(
    arms: dict[str, tuple[float, float, float, float]],
    *,
    out_path: str | Path,
    per_seed: dict[str, dict] | None = None,
    mrg_ci: tuple[float, float] | None = None,
    baseline: str = "mse",
    objective: str = "awwl",
    operator_label: str = "high-band boost",
    show_seeds: bool = False,
    fraction: float = 0.95,
):
    """What the spectral correction buys, and what the objective actually gets.

    Args:
        arms: ``{group: (deficit_db, fid, boosted_deficit_db, boosted_fid)}``,
            seed means; the deficit is in dB *below* the real reference. A NaN
            boosted pair draws the arm as trained only.
        per_seed: optional ``{group: {seed: (d0, f0, d1, f1)}}``, used for the
            MRG's paired interval (and drawn faintly with ``show_seeds``).
        mrg_ci: the MRG's paired 95% interval, if computed elsewhere.

    Both axes run the same way: lower is better. Each arrow joins an arm's
    samples as trained to the same samples after the post-process, so its
    vertical extent is the value of the post-process for that arm. The dashed
    gap joins the post-processed baseline to the objective as trained, which
    sit at nearly the same high-band deficit: that gap is the MRG. The two
    spectra agree in the high-band mean only, which is what the label says.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    fig, ax = plt.subplots(figsize=figure_size(fraction, 1.8))

    order = [k for k in (baseline, objective) if k in arms] + [
        k for k in arms if k not in (baseline, objective)]
    handles = []
    xs, ys = [], []
    for key in order:
        d0, f0, d1, f1 = arms[key]
        colour, _dash, marker = style_of(key)
        boosted = np.isfinite(d1) and np.isfinite(f1)
        if show_seeds and per_seed and key in per_seed:
            pts = np.asarray(list(per_seed[key].values()), dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], linestyle="none", marker=marker, markersize=3,
                    color=colour, alpha=0.3, markeredgewidth=0, zorder=2)
            if boosted:
                ax.plot(pts[:, 2], pts[:, 3], linestyle="none", marker=marker,
                        markersize=3, color="white", markeredgecolor=colour,
                        markeredgewidth=0.6, alpha=0.5, zorder=2)
        name = display_name(key)
        ax.plot(d0, f0, linestyle="none", marker=marker, markersize=6, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        handles.append(Line2D([], [], linestyle="none", marker=marker, markersize=5.5,
                              color=colour, markeredgecolor="white",
                              label=f"{name}, as trained"))
        xs.append(d0)
        ys.append(f0)
        if not boosted:
            continue
        ax.annotate("", xy=(d1, f1), xytext=(d0, f0),
                    arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.2,
                                    shrinkA=5, shrinkB=5, mutation_scale=9), zorder=3)
        ax.plot(d1, f1, linestyle="none", marker=marker, markersize=6, color="white",
                markeredgecolor=colour, markeredgewidth=1.2, zorder=4)
        handles.append(Line2D([], [], linestyle="none", marker=marker, markersize=5.5,
                              color="white", markeredgecolor=colour, markeredgewidth=1.1,
                              label=f"{name}, after {operator_label}"))
        xs.append(d1)
        ys.append(f1)

    if baseline in arms and objective in arms:
        _bd0, _bf0, bd1, bf1 = arms[baseline]
        ad0, af0, _ad1, _af1 = arms[objective]
        mrg = af0 - bf1
        if mrg_ci is None and per_seed and baseline in per_seed and objective in per_seed:
            shared = sorted(set(per_seed[baseline]) & set(per_seed[objective]))
            if len(shared) >= 2:
                values = [per_seed[objective][s][1] - per_seed[baseline][s][3] for s in shared]
                m, h = _mean_ci(values)
                mrg, mrg_ci = m, (m - h, m + h)
        x_gap = 0.5 * (bd1 + ad0)
        ax.annotate("", xy=(x_gap, af0), xytext=(x_gap, bf1),
                    arrowprops=dict(arrowstyle="<|-|>", color=INK, linewidth=0.9,
                                    linestyle=(0, (3, 2)), shrinkA=4, shrinkB=4,
                                    mutation_scale=8), zorder=3)
        label = f"MRG = {_fmt_signed(mrg)} FID"
        if mrg_ci is not None:
            label += f"\n95% CI [{mrg_ci[0]:.2f}, {mrg_ci[1]:.2f}]"
        label += "\nat the same high-band deficit"
        # Right of the dashed gap and above the baseline's arrow, which runs
        # in from the lower right: that corner of the plot is empty.
        ax.annotate(label, xy=(x_gap, bf1 + 0.62 * (af0 - bf1)), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=INK, linespacing=1.25, zorder=5)

    lo, hi = min(xs), max(xs)
    ax.set_xlim(lo - 0.10 * (hi - lo), hi + 0.10 * (hi - lo))
    ylo, yhi = min(ys), max(ys)
    ax.set_ylim(ylo - 0.22 * (yhi - ylo), yhi + 0.12 * (yhi - ylo))
    ax.set_xlabel("High-band spectral deficit (dB below real; lower is better)")
    ax.set_ylabel("FID (lower is better)")
    _chrome(ax)
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower left",
              handletextpad=0.4, borderaxespad=0.3)
    return save_figure(fig, out_path)


def plot_pricing_protocol(
    *,
    baseline_fid: float | None,
    corrected_fid: float,
    trained_fid: float,
    out_path: str | Path,
    delta_db: float | None = None,
    delta_ci: tuple[float, float] | None = None,
    mrg: float | None = None,
    mrg_ci: tuple[float, float] | None = None,
    baseline_name: str = "MSE",
    objective_name: str = "AWWL",
    metric_name: str = "FID",
    statistic: str = "high-band energy",
    fraction: float = 0.95,
):
    """The pricing test as a procedure: two scored paths and the gap between them.

    Every number is an argument, so the diagram cannot drift from the ledger.
    ``mrg`` defaults to ``trained_fid - corrected_fid`` (the paired mean equals
    the difference of means when every seed is paired); pass ``mrg_ci`` to
    print its paired interval, and ``delta_db``/``delta_ci`` to print the
    measured shift that sets the operator.

    Upper path: the baseline's existing samples, post-processed by the
    operator; nothing is trained. Lower path: the objective's samples, scored
    as trained. The dashed branch is what ties the operator to the objective:
    ``delta_A = s(A) - s(B)`` is measured on the two sets of samples.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    apply_style()
    width = figure_size(fraction)[0]
    fig = plt.figure(figsize=(width, 0.44 * width))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 11.55)
    ax.set_ylim(0.1, 5.45)
    ax.axis("off")

    def box(x, y, w, h, text, *, edge, dashed=False, size=8):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
            linewidth=0.9, edgecolor=edge, facecolor="white", zorder=2,
            linestyle=(0, (3, 2)) if dashed else "-",
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                color=INK, zorder=3, linespacing=1.3)

    def arrow(p0, p1, *, colour=MUTED, dashed=False, style="-|>", lw=0.9):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=9, linewidth=lw, color=colour,
            linestyle=(0, (3, 2)) if dashed else "-", zorder=1,
            shrinkA=0, shrinkB=0,
        ))

    blue = SERIES["awwl"][1]
    base_line = f"samples of baseline $B$\n({baseline_name}"
    base_line += f"; {metric_name} {baseline_fid:.2f})" if baseline_fid is not None else ")"
    # Upper path: post-process the baseline's samples.
    box(0.15, 3.85, 2.55, 0.95, base_line, edge=INK)
    box(4.05, 3.85, 3.25, 0.95,
        f"$T_{{\\delta_A}}$: raise the {statistic}\nof every sample by $\\delta_A$", edge=MUTED)
    box(8.55, 3.85, 2.25, 0.95,
        f"$M(T_{{\\delta_A}}(B))$\n{metric_name} {corrected_fid:.2f}", edge=INK)
    arrow((2.76, 4.32), (3.99, 4.32))
    arrow((7.36, 4.32), (8.49, 4.32))
    ax.text(5.675, 5.14, "no training: the baseline's existing samples are filtered",
            ha="center", va="center", fontsize=7.5, color=MUTED, style="italic")

    # Lower path: the objective, scored as it is.
    box(0.15, 0.30, 2.55, 0.95, f"samples of model $A$\n({objective_name}, as trained)",
        edge=blue)
    box(8.55, 0.30, 2.25, 0.95, f"$M(A)$\n{metric_name} {trained_fid:.2f}", edge=blue)
    arrow((2.76, 0.775), (8.49, 0.775), colour=blue)
    ax.text(5.675, 0.60, "scored as trained", ha="center", va="top", fontsize=7.5,
            color=MUTED, style="italic")

    # The measurement that sets the operator, taken on both sets of samples.
    d_text = "measure $\\delta_A = s(A) - s(B)$"
    if delta_db is not None:
        d_text += f"\n{delta_db:.2f} dB"
        if delta_ci is not None:
            d_text += f"  [{delta_ci[0]:.2f}, {delta_ci[1]:.2f}]"
    box(4.05, 2.02, 3.25, 0.95, d_text, edge=MUTED, dashed=True)
    arrow((1.43, 3.79), (3.99, 2.62), dashed=True)
    arrow((1.43, 1.31), (3.99, 2.37), dashed=True)
    arrow((5.675, 3.03), (5.675, 3.79), dashed=True)

    # The quantity the two paths define.
    gap = trained_fid - corrected_fid if mrg is None else mrg
    arrow((9.675, 3.79), (9.675, 1.31), colour=INK, style="<|-|>", lw=1.0)
    m_text = f"MRG\n{_fmt_signed(gap)} {metric_name}"
    if mrg_ci is not None:
        m_text += f"\n[{mrg_ci[0]:.2f}, {mrg_ci[1]:.2f}]"
    ax.text(9.84, 2.55, m_text, ha="left", va="center", fontsize=8, color=INK,
            linespacing=1.3)
    return save_figure(fig, out_path, layout=False)


# --------------------------------------------------------------------------
# The pricing test: teaser, price curve and the test across arms


@dataclass
class Interval:
    """A mean with its 95% interval (paired by seed unless stated)."""

    mean: float
    lo: float
    hi: float
    significant: bool = False

    @classmethod
    def of(cls, res, *, significant: bool = False, sign: float = 1.0) -> Interval | None:
        """From a ``make_tables.Paired`` (``sign=-1`` negates, swapping the ends)."""
        if res is None:
            return None
        lo, hi = sorted((sign * res.lo, sign * res.hi))
        return cls(sign * res.mean, lo, hi, significant)


@dataclass
class Verdict:
    """One application of the pricing test, as the teaser draws it.

    ``value`` is V = M(B) - M(T_A(B)); ``gap`` is M(A) - M(B); ``mrg`` is
    M(A) - M(T_A(B)). All three are paired by seed. ``fid_*`` are the seed
    means of the three scored sets, printed under the axis.
    """

    title: str
    n: int
    fid_b: float
    fid_tb: float
    fid_a: float
    value: Interval
    gap: Interval
    mrg: Interval
    arm: str = "awwl"


def _box(ax, x, y, w, h, text, *, edge=INK, dashed=False, size=7.5, face="white", lw=0.8):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", linewidth=lw,
        edgecolor=edge, facecolor=face, zorder=2, linestyle=(0, (3, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            zorder=3, linespacing=1.25)


def _arrow(ax, p0, p1, *, colour=MUTED, dashed=False, style="-|>", lw=0.8, scale=7):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=scale, linewidth=lw, color=colour,
        linestyle=(0, (3, 2)) if dashed else "-", zorder=1, shrinkA=0, shrinkB=0))


def plot_teaser(
    verdicts: Sequence[Verdict],
    *,
    out_path: str | Path,
    fraction: float = 1.0,
    height_cm: float = 5.6,
    split: float = 0.47,
):
    """The pricing test in one figure: (a) the procedure, (b) its verdicts.

    (a) is symbolic: the baseline's samples ``B`` are scored as they are and
    after ``T_A``, which imposes on them the spectral change ``delta_A = s(A) -
    s(B)`` measured between the objective's samples ``A`` and ``B``; ``A`` is
    scored as trained. ``V`` and the MRG are the two differences marked.

    (b) draws each verdict on its own axis of FID relative to ``M(B)`` (the
    scales differ: the positive control's MRG is several times AWWL's), so
    every group starts at zero and the two differences read as vertical
    distances: ``T_A(B)`` sits ``V`` below ``B``, and ``A`` sits the MRG above
    ``T_A(B)``.
    Whiskers are paired 95% t-intervals of ``M(T_A(B)) - M(B)`` and ``M(A) -
    M(B)``; V and the MRG are printed with their own paired intervals under
    each group. No marker here encodes significance: all are filled.
    """
    import matplotlib.pyplot as plt

    apply_style()
    width = figure_size(fraction)[0]
    height = height_cm / 2.54
    fig = plt.figure(figsize=(width, height))
    w_cm, h_cm = width * 2.54, height_cm

    # ---- (a) the procedure, drawn in centimetres -------------------------
    ax = fig.add_axes((0.0, 0.0, split, 1.0))
    aw = w_cm * split
    ax.set_xlim(0, aw)
    ax.set_ylim(0, h_cm)
    ax.axis("off")
    ax.text(0.05, h_cm - 0.08, "(a)  the pricing test", ha="left", va="top", fontsize=9, color=INK)
    blue = SERIES["awwl"][1]
    r1, r2, r3 = 4.25, 3.00, 0.95          # rows: M(B), M(T_A(B)), M(A)
    x_src, w_src = 0.05, 1.90              # sample boxes
    x_t, w_t = 2.45, 2.25                  # operator and measurement
    x_m, w_m = 5.15, 1.35                  # scores
    x_br = x_m + w_m + 0.14                # brackets
    _box(ax, x_src, r2 - 0.42, w_src, r1 - r2 + 0.84,
         "$B$: baseline’s\nsamples", edge=INK)
    _box(ax, x_src, r3 - 0.42, w_src, 0.84, "$A$: objective’s\nsamples", edge=blue)
    _box(ax, x_t, r2 - 0.40, w_t, 0.80, "$T_A$: impose $\\delta_A$\non $B$’s samples", edge=INK)
    y_d = 1.95
    _box(ax, x_t, y_d - 0.32, w_t, 0.64, "spectral change\n$\\delta_A = s(A) - s(B)$",
         edge=MUTED, dashed=True)
    for y, label, edge in ((r1, "$M(B)$", INK), (r2, "$M(T_A(B))$", INK), (r3, "$M(A)$", blue)):
        _box(ax, x_m, y - 0.26, w_m, 0.52, label, edge=edge, size=8)
    right = x_src + w_src
    _arrow(ax, (right, r1), (x_m, r1))
    ax.text((right + x_m) / 2, r1 + 0.07, "score as generated", ha="center", va="bottom",
            fontsize=7, color=MUTED, style="italic")
    _arrow(ax, (right, r2), (x_t, r2))
    _arrow(ax, (x_t + w_t, r2), (x_m, r2))
    _arrow(ax, (right, r3), (x_m, r3), colour=blue)
    ax.text((right + x_m) / 2, r3 - 0.07, "score as trained", ha="center", va="top",
            fontsize=7, color=MUTED, style="italic")
    # The measurement that sets the operator, taken on both sets of samples.
    _arrow(ax, (right, r2 - 0.30), (x_t, y_d + 0.12), dashed=True)
    _arrow(ax, (right, r3 + 0.30), (x_t, y_d - 0.12), dashed=True)
    _arrow(ax, (x_t + w_t / 2, y_d + 0.32), (x_t + w_t / 2, r2 - 0.40), dashed=True)
    # The two differences the three scores define.
    # One rule with a tick at each score; the two stretches are V and the MRG.
    ax.plot([x_br, x_br], [r1, r3], color=INK, linewidth=0.8, solid_capstyle="butt")
    for y in (r1, r2, r3):
        ax.plot([x_br - 0.10, x_br + 0.10], [y, y], color=INK, linewidth=0.8)
    for y0, y1, label in ((r1, r2, "$V$"), (r2, r3, "MRG")):
        ax.text(x_br + 0.16, (y0 + y1) / 2, label, ha="left", va="center", fontsize=8.5,
                color=INK)
    ax.text(aw / 2 - 0.1, 0.10,
            "$V = M(B) - M(T_A(B))$      $\\mathrm{MRG} = M(A) - M(T_A(B))$",
            ha="center", va="bottom", fontsize=7.5, color=INK)

    # ---- (b) the verdicts --------------------------------------------------
    # One axes per verdict, each on its own FID scale: the positive control's
    # MRG is several times AWWL's, and on a shared axis AWWL's V and MRG would
    # shrink to a few points. Every axes starts at zero (M(B)), so the two
    # differences read as vertical distances within a group.
    left_cm, gap_cm, right_cm, bottom_cm, top_cm = 1.45, 0.70, 0.30, 1.50, 1.20
    step = 1.0
    n_groups = max(len(verdicts), 1)
    group_cm = (w_cm * (1.0 - split) - left_cm - right_cm - gap_cm * (n_groups - 1)) / n_groups

    def ci(iv):
        return f"[{_fmt_signed(iv.lo).lstrip('+')}, {_fmt_signed(iv.hi).lstrip('+')}]"

    axes_b = []
    for g, v in enumerate(verdicts):
        x0 = split + (left_cm + g * (group_cm + gap_cm)) / w_cm
        ax_b = fig.add_axes((x0, bottom_cm / h_cm, group_cm / w_cm,
                             1.0 - (bottom_cm + top_cm) / h_cm))
        axes_b.append(ax_b)
        xb, xt, xa = 0.0, step, 2 * step
        tb = Interval(-v.value.mean, -v.value.hi, -v.value.lo)
        colour_a, _dash, marker_a = style_of(v.arm)
        for x, iv, colour, marker in ((xt, tb, MUTED, "s"), (xa, v.gap, colour_a, marker_a)):
            ax_b.errorbar([x], [iv.mean], yerr=[[iv.mean - iv.lo], [iv.hi - iv.mean]],
                          fmt="none", ecolor=colour, elinewidth=0.9, capsize=2.2,
                          capthick=0.8, zorder=3)
            ax_b.plot(x, iv.mean, marker=marker, markersize=5.2, linestyle="none",
                      color=colour, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax_b.plot(xb, 0.0, marker="o", markersize=5.2, linestyle="none", color=INK,
                  markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        lows, highs = [0.0, tb.lo, v.gap.lo], [0.0, tb.hi, v.gap.hi]
        # V runs from B's level to T_A(B)'s, the MRG from T_A(B)'s to A's,
        # each midway between the two columns it joins, with dotted guides.
        arrows = []
        for xm, y0, y1, label in ((xb + 0.5 * step, 0.0, tb.mean, "$V$"),
                                  (xt + 0.5 * step, tb.mean, v.gap.mean, "MRG")):
            for xs, y in (((xm - 0.5 * step, xm), y0), ((xm, xm + 0.5 * step), y1)):
                ax_b.plot(xs, [y, y], color=MUTED, linewidth=0.6, linestyle=(0, (1, 1.2)),
                          zorder=2)
            ax_b.annotate("", xy=(xm, y1), xytext=(xm, y0),
                          arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=0.8,
                                          mutation_scale=6, shrinkA=0, shrinkB=0), zorder=3)
            arrows.append((xm, y0, y1, label))
        ax_b.text(step, 1.0, v.title, transform=ax_b.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=7.5, color=INK, linespacing=1.2)
        verdict = (f"$V$ = {_fmt_signed(v.value.mean)} {ci(v.value)}\n"
                   f"MRG = {_fmt_signed(v.mrg.mean)} {ci(v.mrg)}")
        ax_b.annotate(verdict, xy=(step, 0), xycoords=("data", "axes fraction"),
                      xytext=(0, -13), textcoords="offset points", ha="center", va="top",
                      fontsize=7.5, color=INK, linespacing=1.3, annotation_clip=False)
        ax_b.axhline(0, color=INK, linewidth=0.8, zorder=1.5)
        ax_b.set_xticks([xb, xt, xa])
        ax_b.set_xticklabels(["$B$", "$T_A(B)$", "$A$"])
        ax_b.tick_params(axis="x", length=0, pad=2)
        ax_b.set_xlim(-0.35, xa + 0.35)
        lo, hi = min(lows), max(highs)
        pad = 0.08 * (hi - lo)
        ax_b.set_ylim(lo - pad, hi + pad)
        from matplotlib.ticker import MaxNLocator

        ax_b.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        if g == 0:
            ax_b.set_ylabel("$M(\\cdot) - M(B)$ (FID)", labelpad=2)
        _chrome(ax_b, grid="y")
        # Arrow labels, left of each arrow. A label sits on the longer stretch
        # of its arrow on either side of zero, clear of the zero rule; an arrow
        # too short to carry one is labelled beyond whichever end has more room
        # before the frame (beyond its tip into a group title would collide).
        fig.canvas.draw()
        to_pt = ax_b.transData.transform
        y_lo, y_hi = ax_b.get_ylim()
        for xm, y0, y1, label in arrows:
            length = abs(to_pt((xm, y1))[1] - to_pt((xm, y0))[1]) * 72.0 / fig.dpi
            if length < 14:
                top_end, bottom_end = max(y0, y1), min(y0, y1)
                up = (y_hi - top_end) >= (bottom_end - y_lo)
                ax_b.annotate(label, xy=(xm, top_end if up else bottom_end),
                              xytext=(0, 3 if up else -3), textcoords="offset points",
                              ha="center", va="bottom" if up else "top", fontsize=8, color=INK,
                              zorder=5)
                continue
            lo_, hi_ = sorted((y0, y1))
            parts = [(lo_, min(hi_, 0.0)), (max(lo_, 0.0), hi_)]
            a, b = max(parts, key=lambda p: p[1] - p[0])
            ax_b.annotate(label, xy=(xm, (a + b) / 2), xytext=(-3, 0), textcoords="offset points",
                          ha="right", va="center", fontsize=8, color=INK, zorder=5)
    fig.text(split + 0.04 / w_cm, 1.0 - 0.08 / h_cm,
             "(b)  the verdict (FID; lower is better)",
             ha="left", va="top", fontsize=9, color=INK)
    return save_figure(fig, out_path, layout=False)


@dataclass
class PricePoint:
    """One operator on the baseline's samples: its value and its spectral change.

    ``value`` is V = M(B) - M(T(B)) with its paired interval (``significant``
    after Holm within the panel); ``shift`` is the measured change of the
    high-band deviation (dB), T(B) minus B, paired by seed.
    """

    label: str
    value: Interval
    shift: Interval
    kind: str = "boost"


def _forest_rows(ax, blocks, *, indent_pt: float = 8.0):
    """Block headers and indented row labels on a forest plot's y axis.

    ``blocks`` is ``[(header, [(label, payload), ...]), ...]``. Sets the tick
    labels (headers italic grey, rows indented under them, a thin rule between
    blocks) and returns ``([(y, payload), ...], width_pt)``: the rows top to
    bottom, and the width of the label column in points, so the caller can
    place the axes with the column inside its own panel.
    """
    from matplotlib.transforms import ScaledTranslation

    entries: list[tuple[str, object, bool]] = []
    for header, rows in blocks:
        if not rows:
            continue
        entries.append((header, None, True))
        entries += [(label, payload, False) for label, payload in rows]
    n = len(entries)
    ys = [n - 1 - i for i in range(n)]
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entries])
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", length=0)
    fig = ax.figure
    for (_l, _p, is_header), tick in zip(entries, ax.get_yticklabels(), strict=True):
        if is_header:
            tick.set_fontstyle("italic")
            tick.set_color(MUTED)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    col = max(t.get_window_extent(renderer).width * 72.0 / fig.dpi + (0.0 if e[2] else indent_pt)
              for t, e in zip(ax.get_yticklabels(), entries, strict=True))
    ax.tick_params(axis="y", pad=col + 3)
    shift = ScaledTranslation(indent_pt / 72.0, 0, fig.dpi_scale_trans)
    for tick, e in zip(ax.get_yticklabels(), entries, strict=True):
        tick.set_horizontalalignment("left")
        if not e[2]:
            tick.set_transform(tick.get_transform() + shift)
    for i, e in enumerate(entries):
        if e[2] and i > 0:
            ax.axhline(ys[i] + 0.5, color=GRID, linewidth=0.6, zorder=0)
    rows = [(y, e[1]) for y, e in zip(ys, entries, strict=True) if not e[2]]
    return rows, col + 3.0


#: Marker of each operator kind in Fig. price-curve (b): ``(colour, marker,
#: size)``. The transplant and the match differ in marker as well as in ink,
#: and the match's pentagon repeats no marker of panel (a) (circles on the
#: boost's curve, a square for the dithered boost, triangles for noise), so no
#: row of (b) reads as a point of (a).
BAND_MARKERS: dict[str, tuple[str, str, float]] = {
    "transplant": (INK, "o", 4.6),
    "match": (MUTED, "p", 5.0),
}


def plot_price_curve(
    curve: Sequence[PricePoint],
    controls: Sequence[PricePoint],
    bands: Sequence[tuple[str, Sequence[PricePoint]]],
    *,
    out_path: str | Path,
    fraction: float = 0.95,
    height_cm: float = 5.9,
    split: float = 0.47,
    legend_labels: dict[str, str] | None = None,
):
    """What the metric pays for a spectral change, and where it pays.

    (a) The value of the unquantised high-band boost against the high-band
    change it measurably produces, one point per dose, with the controls that
    change the spectrum little but move the metric a lot (``controls``: the
    same boost dithered to 8 bits, sub-LSB white noise) marked. (b) The value
    of band-restricted operators, grouped (``bands``: ``[(header, points)]``),
    with the high-band change each makes printed at the right.

    Every value is paired by seed with its 95% t-interval; filled = significant
    after Holm correction within its panel.
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    apply_style()
    width = figure_size(fraction)[0]
    height = height_cm / 2.54
    fig = plt.figure(figsize=(width, height))
    w_cm = width * 2.54
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]

    # ---- (a) the price curve --------------------------------------------
    left, bottom, top = 1.25, 1.05, 0.55
    ax = fig.add_axes((left / w_cm, bottom / height_cm, split - (left + 0.35) / w_cm,
                       1.0 - (bottom + top) / height_cm))
    xs = np.array([p.shift.mean for p in curve])
    ys = np.array([p.value.mean for p in curve])
    ax.plot(xs, ys, color=INK, linewidth=1.0, zorder=3)
    for p in curve:
        ax.errorbar([p.shift.mean], [p.value.mean], yerr=[[p.value.mean - p.value.lo],
                    [p.value.hi - p.value.mean]], fmt="none", ecolor=INK, elinewidth=0.8,
                    capsize=1.6, capthick=0.7, zorder=3)
        ax.plot(p.shift.mean, p.value.mean, marker="o", markersize=4.2, linestyle="none",
                color=INK if p.value.significant else "white", markeredgecolor=INK,
                markeredgewidth=0.9, zorder=4)
    markers = {"dither": "s", "noise": "^"}
    # A kind's legend entry: given, else its first point's label.
    legend_labels = dict(legend_labels or {"noise": "uniform white noise, not rounded"})
    for p in controls:
        marker = markers.get(p.kind, "D")
        ax.errorbar([p.shift.mean], [p.value.mean], yerr=[[p.value.mean - p.value.lo],
                    [p.value.hi - p.value.mean]], xerr=[[p.shift.mean - p.shift.lo],
                    [p.shift.hi - p.shift.mean]], fmt="none", ecolor=MUTED, elinewidth=0.8,
                    capsize=1.6, capthick=0.7, zorder=3)
        ax.plot(p.shift.mean, p.value.mean, marker=marker, markersize=5.0, linestyle="none",
                color=MUTED if p.value.significant else "white", markeredgecolor=MUTED,
                markeredgewidth=0.9, zorder=4)
    # The chosen dose, and what dithering the same filtered images does to it.
    for d in (p for p in controls if p.kind == "dither"):
        same = min(curve, key=lambda p: abs(p.shift.mean - d.shift.mean))
        ax.annotate("", xy=(d.shift.mean, d.value.lo), xytext=(same.shift.mean, same.value.hi),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.7,
                                    linestyle=(0, (2, 1.5)), mutation_scale=6,
                                    shrinkA=2, shrinkB=2), zorder=2)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=INK, linewidth=1.0, marker="o", markersize=4.2,
                      label="high-band boost, unquantised")]
    seen = set()
    for p in controls:
        if p.kind in seen:
            continue
        seen.add(p.kind)
        text = legend_labels.get(p.kind, p.label)
        handles.append(Line2D([], [], linestyle="none", marker=markers.get(p.kind, "D"),
                              markersize=5.0, color=MUTED, label=text))
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="center right",
              bbox_to_anchor=(1.0, 0.62), borderaxespad=0.1, handletextpad=0.5,
              labelspacing=0.35, handlelength=1.8)
    ax.axhline(0, color=INK, linewidth=0.8, zorder=1)
    ax.set_xlim(0, max(xs.max(), max((p.shift.hi for p in controls), default=0)) * 1.05)
    top_v = max(ys.max(), max((p.value.hi for p in controls), default=0))
    ax.set_ylim(-0.04 * top_v, top_v * 1.10)
    # Short direct labels on the noise controls; the legend says what each
    # marker is. A label sits right of its marker unless it would run into
    # another control's marker (and so read as that point's label); it then
    # sits above its marker with the amplitude over the unit.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clear_px = 5.0 / 72.0 * fig.dpi
    for p in (p for p in controls if p.kind == "noise"):
        label = ax.annotate(p.label, xy=(p.shift.mean, p.value.mean), xytext=(5, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=7.5, color=INK, path_effects=halo, zorder=5)
        box = label.get_window_extent(renderer).padded(clear_px)
        if any(box.contains(*ax.transData.transform((q.shift.mean, q.value.mean)))
               for q in controls if q is not p):
            label.remove()
            ax.annotate(p.label.replace(" ", "\n", 1), xy=(p.shift.mean, p.value.hi),
                        xytext=(2, 2), textcoords="offset points", ha="left", va="bottom",
                        fontsize=7.5, color=INK, linespacing=1.05, path_effects=halo,
                        zorder=5)
    ax.set_xlabel("measured high-band change of $B$’s samples (dB)")
    ax.set_ylabel("value $V$ (FID)")
    _chrome(ax)
    fig.text(0.04 / w_cm, 1.0 - 0.08 / height_cm, "(a)  the price of the raised-cosine boost",
             ha="left", va="top", fontsize=9, color=INK)

    # ---- (b) value by band ----------------------------------------------
    col_cm = 0.95   # the printed high-band change, right of the axes
    x_right = 1.0 - col_cm / w_cm
    ax_b = fig.add_axes((split + 0.2, bottom / height_cm, x_right - split - 0.2,
                         1.0 - (bottom + top) / height_cm))
    rows, label_pt = _forest_rows(ax_b, [(h, [(p.label, p) for p in pts]) for h, pts in bands])
    x_left = split + 0.10 / w_cm + label_pt / 72.0 / width
    ax_b.set_position((x_left, bottom / height_cm, x_right - x_left,
                       1.0 - (bottom + top) / height_cm))
    ax_b.axvline(0, color=INK, linewidth=0.8, zorder=2)
    lo = min(0.0, min(p.value.lo for _y, p in rows))
    hi = max(p.value.hi for _y, p in rows)
    for y, p in rows:
        colour, marker, size = BAND_MARKERS.get(p.kind, (MUTED, "D", 4.0))
        ax_b.errorbar([p.value.mean], [y], xerr=[[p.value.mean - p.value.lo],
                      [p.value.hi - p.value.mean]], fmt="none", ecolor=colour, elinewidth=0.9,
                      capsize=2.0, capthick=0.8, zorder=3)
        ax_b.plot(p.value.mean, y, marker=marker, markersize=size, linestyle="none",
                  color=colour if p.value.significant else "white", markeredgecolor=colour,
                  markeredgewidth=0.9, zorder=4)
        ax_b.annotate(f"{_fmt_signed(p.shift.mean)}", xy=(1.0, y),
                      xycoords=("axes fraction", "data"), xytext=(col_cm * 72 / 2.54 - 1, 0),
                      textcoords="offset points", ha="right", va="center", fontsize=7.5,
                      color=INK, annotation_clip=False)
    fig.text(1.0 - 0.03 / w_cm, 1.0 - 0.12 / height_cm, "high-band\nchange (dB)",
             ha="right", va="top", fontsize=7, color=MUTED, linespacing=1.1)
    span = hi - lo
    ax_b.set_xlim(lo - 0.06 * span, hi + 0.06 * span)
    from matplotlib.ticker import MaxNLocator

    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax_b.set_xlabel("value $V$ (FID, 95% CI)")
    _chrome(ax_b, grid="x")
    fig.text(split + 0.04 / w_cm, 1.0 - 0.08 / height_cm, "(b)  value by band",
             ha="left", va="top", fontsize=9, color=INK)
    # One key for both panels (Holm runs within each), in the free corner
    # beneath panel (b)'s row labels, clear of (a)'s axis label.
    fig.canvas.draw()
    label_right = ax.xaxis.label.get_window_extent(fig.canvas.get_renderer()).x1
    _fill_legend(ax_b, colour=INK, size=4.2, loc="lower left",
                 bbox_to_anchor=(label_right / (width * fig.dpi) + 0.2 / w_cm, 0.0),
                 bbox_transform=fig.transFigure, borderaxespad=0.15)
    return save_figure(fig, out_path, layout=False)


#: The arms of the pricing family (make_tables.py FAMILY_PRICING, checked by
#: the tests): every arm transplanted onto MSE's samples. The three arms that
#: diverged (raw SNR weighting, multi-level, GradNorm) are not priced.
PRICING_FAMILY: tuple[str, ...] = (
    "awwl", "static_wavelet", "awwl_normalized", "awwl_norm_matched", "awwl_eq7",
    "awwl_grad_matched", "awwl_grad_matched_mean", "focal_frequency", "min_snr", "huber", "awwl_ema",
    "mse_ema", "awwl_subband", "awwl_spatial", "awwl_learned", "awwl_lifting",
    "awwl_full", "l1", "charbonnier",
)

#: The positive-control rows of Fig. pricing-arms: the 800-epoch MSE runs at
#: the long schedule's FID optimum (epoch 399) and, drawn as secondary, at its
#: end (epoch 799), each priced against the same runs' epoch-199 samples.
POSITIVE_ROWS: tuple[str, ...] = ("positive_control", "positive_control_end")

#: The rows of Fig. pricing-arms, in display order: the pricing family under
#: the :data:`BLOCKS` headers and in their order (as in Fig. effects and Table
#: pricing-arms), then the positive control (:data:`POSITIVE_ROWS`). Every arm
#: of the family appears exactly once; the builder checks this, so an arm
#: cannot drop out unnoticed.
PRICING_BLOCKS: list[tuple[str, list[str]]] = [
    (title, [k for k in keys if k in PRICING_FAMILY]) for title, keys in BLOCKS
] + [("Positive control (MSE + EMA, 800-epoch runs)", list(POSITIVE_ROWS))]

#: What an arm of Fig. pricing-arms is priced against, where Fig. effects
#: compares it with something else: ``B`` is MSE without EMA in every row,
#: whereas Fig. effects pairs AWWL + EMA with MSE + EMA (:data:`EFFECT_PAIRS`).
#: The label says so, as Fig. effects' label says what it pairs with.
PRICING_AGAINST: dict[str, str] = {"awwl_ema": "mse", "mse_ema": "mse"}

#: Rows drawn on a shaded band, with the role in their label.
#: The positive control's role is its block's header, so its labels add none.
PRICING_HIGHLIGHT = {"positive_control": "", "positive_control_end": "",
                     "static_wavelet": "null control"}


@dataclass
class PricedArm:
    """One row of the forest plot: an arm's V and MRG, paired by seed."""

    key: str
    n: int
    value: Interval
    mrg: Interval
    style: str = ""   # the arm whose colour and marker the row uses
    secondary: bool = False   # drawn lighter and smaller: a row reported for completeness


def plot_pricing_arms(
    blocks: Sequence[tuple[str, Sequence[PricedArm]]],
    *,
    out_path: str | Path,
    names: dict[str, str] | None = None,
    highlight: dict[str, str] | None = None,
    value_floor: float = -0.5,
    mrg_ceiling: float = 7.9,
    fraction: float = 0.9,
):
    """The pricing test across arms: V and the MRG side by side, per arm.

    Two columns of 95% t-intervals share the rows: (a) the value V of each
    arm's spectral change, transplanted onto the baseline's samples; (b) the
    arm's MRG. Filled = significant after Holm correction (the families are
    the builder's). Rows in ``highlight`` sit on a shaded band with their role
    in the label. An interval beyond ``value_floor`` (a) or ``mrg_ceiling``
    (b) is drawn as an arrow at the axis edge with its mean printed, as in
    Fig. effects; no row is dropped. A ``secondary`` row is drawn in grey with
    a smaller marker, so it reads as reported alongside the row above it.
    """
    import matplotlib.pyplot as plt

    apply_style()
    names = names or {}
    highlight = PRICING_HIGHLIGHT if highlight is None else highlight
    labelled = []
    for header, arms in blocks:
        rows = []
        for arm in arms:
            name = names.get(arm.key, display_name(arm.key))
            notes = ([highlight[arm.key]] if highlight.get(arm.key) else []) + (
                [f"$n = {arm.n}$"] if arm.n < 5 else []) + (
                [f"vs. {display_name(PRICING_AGAINST[arm.key])}"] if arm.key in PRICING_AGAINST else [])
            if notes:
                name = (f"{name[:-1]}; {'; '.join(notes)})" if name.endswith(")")
                        else f"{name} ({'; '.join(notes)})")
            rows.append((name, arm))
        labelled.append((header, rows))
    n_rows = sum(len(r) + 1 for _h, r in labelled if r)

    width = figure_size(fraction)[0]
    height = 0.150 * n_rows + 0.62
    fig = plt.figure(figsize=(width, height))
    bottom_in, top_in, gap_in, right_in = 0.42, 0.20, 0.22, 0.08
    probe = fig.add_axes((0.5, bottom_in / height, 0.2, 1 - (bottom_in + top_in) / height))
    rows, label_pt = _forest_rows(probe, labelled)
    x0 = 0.04 / width * 2.54 / 2.54 + label_pt / 72.0 / width
    avail = 1.0 - right_in / width - x0 - gap_in / width
    ax_v = probe
    ax_v.set_position((x0, bottom_in / height, avail * 0.44, 1 - (bottom_in + top_in) / height))
    ax_m = fig.add_axes((x0 + avail * 0.44 + gap_in / width, bottom_in / height, avail * 0.56,
                         1 - (bottom_in + top_in) / height), sharey=ax_v)
    ax_m.tick_params(axis="y", left=False, labelleft=False)

    def column(ax, get, lo_clip, hi_clip, xlabel, letter):
        ax.axvline(0, color=INK, linewidth=0.8, zorder=2)
        lows, highs = [], []
        for y, arm in rows:
            iv = get(arm)
            if arm.key in highlight:
                ax.axhspan(y - 0.5, y + 0.5, color=SHADE, zorder=0, linewidth=0)
            colour, _dash, marker = style_of(arm.style or arm.key)
            size, width = (4.8, 0.9)
            if arm.secondary:
                colour, size, width = MUTED, 3.9, 0.8
            below = lo_clip is not None and iv.lo < lo_clip
            above = hi_clip is not None and iv.hi > hi_clip
            if below or above:
                edge, sign = (lo_clip, -1) if below else (hi_clip, 1)
                ax.annotate(_fmt_offscale(iv.mean), xy=(edge, y), xytext=(-sign * 14, 0),
                            textcoords="offset points", ha="left" if sign < 0 else "right",
                            va="center", fontsize=7.5, color=INK,
                            arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.0,
                                            shrinkA=1, shrinkB=0, mutation_scale=7),
                            annotation_clip=False, zorder=4)
                continue
            lows.append(iv.lo)
            highs.append(iv.hi)
            ax.errorbar([iv.mean], [y], xerr=[[iv.mean - iv.lo], [iv.hi - iv.mean]], fmt="none",
                        ecolor=colour, elinewidth=width, capsize=2.0, capthick=0.8, zorder=3)
            ax.plot(iv.mean, y, marker=marker, markersize=marker_size(marker, size),
                    linestyle="none", color=colour if iv.significant else "white",
                    markeredgecolor=colour, markeredgewidth=width, zorder=4)
        lo = min(lows + [0.0]) if lo_clip is None else lo_clip
        hi = max(highs + [0.0]) if hi_clip is None else hi_clip
        span = hi - lo
        ax.set_xlim(lo - (0.06 * span if lo_clip is None else 0.0),
                    hi + (0.06 * span if hi_clip is None else 0.0))
        ax.set_xlabel(xlabel, labelpad=2)
        _chrome(ax, grid="x")
        ax.set_title(f"({letter})", loc="left", fontsize=9, pad=3)

    column(ax_v, lambda a: a.value, value_floor, None,
           "value $V$ of the spectral change (FID)", "a")
    column(ax_m, lambda a: a.mrg, None, mrg_ceiling, "MRG (FID)", "b")
    # The key to fill sits in the bottom left corner, beneath the row labels:
    # neither column has room for it clear of every marker.
    _fill_legend(ax_v, loc="lower left", bbox_to_anchor=(0.0, 0.0),
                 bbox_transform=fig.transFigure, borderaxespad=0.15)
    # The block rules drawn on the label axes also belong across the MRG axes.
    for line in ax_v.lines:
        if line.get_linewidth() == 0.6 and line.get_color() == GRID:
            ax_m.axhline(line.get_ydata()[0], color=GRID, linewidth=0.6, zorder=0)
    return save_figure(fig, out_path, layout=False)


# --------------------------------------------------------------------------
# The weighting itself


def plot_weight_schedule(
    *,
    alpha: float,
    power: float,
    out_path: str | Path,
    snr_gamma: float = 5.0,
    second: tuple[float, float, str] | None = (0.8, 2.0, "scaled_linear"),
    fraction: float = 0.95,
):
    """AWWL's band weights and the timestep schedule they imply.

    (a) The band weights at ``alpha``/``power`` and their sum, against the
    training timestep ``t``, which training samples uniformly; the top axis
    gives σ_t. On the linear schedule 60% of timesteps have σ_t > 0.9, which a
    σ axis would hide. (b) The sum as a per-timestep multiplier (unit mean
    over the training timesteps) at this optimum and at ``second`` (``(alpha,
    power, schedule)``, the DreamBooth optimum on Stable Diffusion's
    schedule), each against the Min-SNR-γ weight on its own schedule, with the
    correlation printed in the legend. Both multipliers are the same kind of
    quantity, which is what makes one axis legitimate here.
    """
    import matplotlib.pyplot as plt

    from awwl.losses import AdaptiveWaveletLoss
    from awwl.losses.weighting import min_snr_weight

    apply_style()
    sig = _training_sigmas("linear")
    t = _training_timesteps(len(sig))
    w = AdaptiveWaveletLoss(alpha=alpha, power=power).weights_at(sig)
    w_ll, w_det, total = w["w_ll"].numpy(), w["w_det"].numpy(), w["total"].numpy()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figure_size(fraction, 2.35))
    ax_a.plot(t, w_ll, color=MUTED, linewidth=1.2, linestyle="-",
              label="$w_{\\mathrm{LL}}$ (LL band)")
    ax_a.plot(t, w_det, color=MUTED, linewidth=1.2, linestyle="--",
              label="$w_{\\mathrm{det}}$ (on the detail-band mean)")
    ax_a.plot(t, total, color=BLUE, linewidth=1.5,
              label="$S = w_{\\mathrm{LL}} + w_{\\mathrm{det}}$")
    ax_a.axhline(total.mean(), color=BLUE, linewidth=0.6, linestyle=(0, (1, 1.5)))
    ax_a.annotate(f"training mean {total.mean():.4f}", xy=(t[-1], total.mean()),
                  xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                  fontsize=7, color=MUTED)
    ax_a.set_ylabel("band weight")
    ax_a.set_ylim(0, max(total.max(), w_det.max()) * 1.08)
    ax_a.legend(frameon=False, fontsize=7.5, loc="upper right", borderaxespad=0.2)

    snr = min_snr_weight(sig, snr_gamma).numpy()
    r1 = float(np.corrcoef(total, snr)[0, 1])
    ax_b.axhline(1.0, color=GRID, linewidth=0.9)
    ax_b.plot(t, total / total.mean(), color=BLUE, linewidth=1.5,
              label=f"$S$, CIFAR-10: $r={_fmt_signed(r1)}$")
    snr2 = None
    if second is not None:
        # Each optimum is correlated with the Min-SNR weight on its own
        # training schedule, and each of those is drawn in its optimum's ink.
        # They share a plateau (1.10 and 1.09 at gamma = 5) but reach it at
        # different timesteps (t = 131 on the linear schedule, 148 on SD's).
        a2, p2, sched2 = second
        sig2 = _training_sigmas(sched2)
        tot2 = AdaptiveWaveletLoss(alpha=a2, power=p2).weights_at(sig2)["total"].numpy()
        snr2 = min_snr_weight(sig2, snr_gamma).numpy()
        r2 = float(np.corrcoef(tot2, snr2)[0, 1])
        ax_b.plot(t, tot2 / tot2.mean(), color=MUTED, linewidth=1.5, linestyle="--",
                  label=f"$S$, DreamBooth: $r={_fmt_signed(r2)}$")
    ax_b.plot(t, snr / snr.mean(), color=INK, linewidth=1.0, linestyle=(0, (1, 1.3)),
              label="Min-SNR-$\\gamma$" + (", CIFAR-10 schedule" if snr2 is not None else ""))
    if snr2 is not None:
        ax_b.plot(t, snr2 / snr2.mean(), color=MUTED, linewidth=1.0,
                  linestyle=(0, (4, 1.5, 1, 1.5)),
                  label="Min-SNR-$\\gamma$, Stable Diffusion schedule")
    ax_b.set_ylabel("relative to training mean")
    ymax = max(ax_b.get_ylim()[1], 1.0)
    ax_b.set_ylim(0, ymax * 1.6)
    ax_b.legend(frameon=False, fontsize=7.5, loc="upper right", borderaxespad=0.2,
                handlelength=2.4, labelspacing=0.35)

    for ax in (ax_a, ax_b):
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(list(TIMESTEP_TICKS))
        ax.set_xlabel("training timestep $t$ (sampled uniformly)")
        _chrome(ax)
    _sigma_axis(ax_a, sig)
    # Only panel (a) carries the sigma axis (the DreamBooth curve in (b) is on
    # another schedule), so the panel headers go on one line above both.
    headers = [(ax_a, f"(a)  band weights, $\\alpha={alpha:g}$, $p={power:g}$"),
               (ax_b, "(b)  per-timestep multiplier")]
    return _save_with_headers(fig, headers, out_path)


def _save_with_headers(fig, headers, out_path, *, header_pt: float = 13.0):
    """Lay out, then write panel headers on one line at the top of the figure."""
    height_pt = fig.get_size_inches()[1] * 72.0
    top = 1.0 - header_pt / height_pt
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, top))
    for ax, text in headers:
        x0 = ax.get_position().x0
        fig.text(x0, 1.0 - 3.0 / height_pt, text, ha="left", va="top", fontsize=9,
                 color=INK)
    return save_figure(fig, out_path, layout=False)


def plot_gradient_scale(
    *,
    alpha: float,
    power: float,
    out_path: str | Path,
    sum_target: float = 0.5,
    square_target: float | None = None,
    show_square_control: bool = True,
    fraction: float = 1.0,
):
    """What each timestep control pins, and what it leaves free.

    (a) ``sum_b omega_b``, the per-timestep scale of the loss; (b)
    ``sum_b omega_b^2``, which sets the per-timestep squared gradient norm when
    the band residuals have equal norm (Eq. gradscale). Both against the
    training timestep, each relative to AWWL's mean over the training
    timesteps, so a flat line at 1 means "AWWL's average, at every timestep".

    Detail bands are averaged, as in AWWL, so the three detail bands carry
    ``w_det/3`` each. The summed-weight control pins ``sum_b omega_b`` at
    ``sum_target``; the sum-of-squares control pins ``sum_b omega_b^2`` at
    ``square_target`` (default: AWWL's training mean, 0.0530 at alpha=0.2,
    p=1; ``show_square_control=False`` omits it).
    Under AdamW a constant rescaling of the
    loss leaves the update unchanged (up to epsilon and clipping), so only the
    shape of each curve over t matters, not its level.
    """
    import matplotlib.pyplot as plt

    from awwl.losses.adaptive_wavelet import AdaptiveWaveletLoss

    apply_style()
    sig = _training_sigmas("linear")
    t = _training_timesteps(len(sig))
    parts = AdaptiveWaveletLoss(alpha=alpha, power=power).weights_at(sig)
    w_ll, w_det = parts["w_ll"].numpy().astype(float), parts["w_det"].numpy().astype(float)

    def norms(ll, det):
        # detail_reduction="mean": each of the three detail bands carries det/3.
        return ll + det, ll**2 + 3 * (det / 3) ** 2

    pub_sum, pub_sq = norms(w_ll, w_det)
    sum_scale = sum_target / pub_sum
    sq_target = float(pub_sq.mean()) if square_target is None else square_target
    sq_scale = np.sqrt(sq_target / pub_sq)
    # Legend names are the arms' display names and nothing else (one name per
    # arm in every figure); the levels are the ones the controls were trained
    # at, and panel (b) shows the sum of squares held at AWWL's mean (1 on the
    # relative axis).
    curves = [
        ("awwl", display_name("awwl"), norms(w_ll, w_det)),
        ("awwl_norm_matched", display_name("awwl_norm_matched") if sum_target == 0.5
         else for_mathtext(f"AWWL, $\\sum_b \\omega_b = {sum_target:g}$"),
         norms(w_ll * sum_scale, w_det * sum_scale)),
        ("awwl_grad_matched_mean", display_name("awwl_grad_matched_mean"),
         norms(w_ll * sq_scale, w_det * sq_scale)),
    ]
    if not show_square_control:
        curves = curves[:2]

    fig, axes = plt.subplots(1, 2, figsize=figure_size(fraction, 2.35), sharex=True)
    headers = [
        ("a", for_mathtext("$\\sum_b \\omega_b$ (loss scale per timestep)"), 0, pub_sum.mean()),
        ("b", for_mathtext("$\\sum_b \\omega_b^2$ (squared-gradient scale per timestep)"), 1,
         pub_sq.mean()),
    ]
    for ax, (letter, header, which, ref) in zip(axes, headers, strict=True):
        ax.axhline(1.0, color=GRID, linewidth=0.9)
        for key, label, pair in curves:
            colour, dash, _marker = style_of(key)
            ax.plot(t, pair[which] / ref, color=colour, linestyle=dash,
                    linewidth=1.5 if key == "awwl" else 1.3, label=label)
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(list(TIMESTEP_TICKS))
        ax.set_xlabel("training timestep $t$ (sampled uniformly)")
        ax.set_ylim(0, None)
        _panel(ax, letter, header)
        _chrome(ax)
        _sigma_axis(ax, sig)
    axes[0].set_ylabel("relative to AWWL’s training mean")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right", borderaxespad=0.2)
    return save_figure(fig, out_path)


# --------------------------------------------------------------------------
# Training dynamics


def plot_convergence(
    rows,
    *,
    metric: str,
    out_path: str | Path,
    configs=None,
    baseline: str | None = None,
    interval: str = "bars",
    fraction: float = 1.0,
):
    """Quality against training epoch; with a baseline, also the paired gap.

    (a) Seed means of ``metric`` per arm, evidence that every arm converged.
    (b) With ``baseline``: the mean of the per-seed differences from it at
    each evaluated epoch, with its 95% t-interval (``interval="bars"`` draws
    error bars, dodged between arms; ``"band"`` a shaded ribbon). Ticks sit on
    the evaluated epochs, and zero is always inside the axis. The paired
    markers are hollow: no correction is applied across checkpoints, and a
    filled marker means "significant after Holm" everywhere else in the paper.
    """
    import matplotlib.pyplot as plt

    from awwl.analysis.stats import summarize_groups

    apply_style()
    epochs = sorted({r["epoch"] for r in rows if r.get("epoch") is not None})
    names = list(configs) if configs else [
        k for k in SERIES if any(str(r.get("group")) == k for r in rows)]

    def means(key):
        xs, ys = [], []
        for e in epochs:
            sub = [r for r in rows if str(r.get("group")) == key and r.get("epoch") == e]
            s = summarize_groups(sub, metric=metric)
            if s:
                xs.append(e)
                ys.append(s[0].mean)
        return xs, ys

    if baseline is not None and not means(baseline)[0]:
        raise ValueError(f"baseline {baseline!r} has no rows for metric {metric!r}")

    if baseline is None:
        fig, ax_abs = plt.subplots(figsize=figure_size(fraction * 0.6, 1.5))
        ax_rel = None
        axes = [ax_abs]
    else:
        fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=figure_size(fraction, 2.45))
        axes = [ax_abs, ax_rel]

    plotted = 0
    others = [k for k in names if k != baseline]
    spread = (epochs[-1] - epochs[0]) if len(epochs) > 1 else 1.0
    offsets = {k: (i - (len(others) - 1) / 2) * 0.018 * spread for i, k in enumerate(others)}
    lows, highs = [0.0], [0.0]
    for key in names:
        colour, dash, marker = style_of(key)
        xs, ys = means(key)
        if len(xs) < 2:
            continue
        ax_abs.plot(xs, ys, color=colour, linestyle=dash, marker=marker, markersize=4.5,
                    linewidth=1.2, label=display_name(key),
                    markeredgecolor="white" if key != baseline else colour,
                    markeredgewidth=0.5, zorder=3 if key == baseline else 4)
        plotted += 1
        if ax_rel is None or key == baseline:
            continue
        dx, dm, dh = [], [], []
        for e in xs:
            sub = [r for r in rows if r.get("epoch") == e]
            diffs = _paired(sub, key, baseline, metric)
            if len(diffs) >= 2:
                m, h = _mean_ci(diffs.values())
                dx.append(e)
                dm.append(m)
                dh.append(h)
        if len(dx) < 2:
            continue
        dx_, dm_, dh_ = np.asarray(dx, float), np.asarray(dm), np.asarray(dh)
        lows.append(float((dm_ - dh_).min()))
        highs.append(float((dm_ + dh_).max()))
        if interval == "band":
            ax_rel.fill_between(dx_, dm_ - dh_, dm_ + dh_, color=colour, alpha=0.16,
                                linewidth=0, zorder=2)
            ax_rel.plot(dx_, dm_, color=colour, linestyle=dash, marker=marker,
                        markersize=4.5, linewidth=1.2, label=display_name(key),
                        markerfacecolor="white", markeredgecolor=colour,
                        markeredgewidth=1.0, zorder=4)
        else:
            xo = dx_ + offsets.get(key, 0.0)
            ax_rel.errorbar(xo, dm_, yerr=dh_, fmt="none", ecolor=colour, elinewidth=0.9,
                            capsize=2.0, capthick=0.8, zorder=3)
            ax_rel.plot(xo, dm_, color=colour, linestyle=dash, marker=marker,
                        markersize=4.5, linewidth=1.0, label=display_name(key),
                        markerfacecolor="white", markeredgecolor=colour,
                        markeredgewidth=1.0, zorder=4)
    if not plotted:
        raise ValueError("no configuration had more than one evaluated epoch")

    ax_abs.set_ylabel(f"{metric.upper()} (mean over seeds)")
    ax_abs.legend(frameon=False, fontsize=7.5, loc="upper right", borderaxespad=0.2)
    for ax in axes:
        ax.set_xticks(epochs)
        ax.set_xlabel("training epoch")
        pad = 0.04 * spread
        ax.set_xlim(epochs[0] - pad, epochs[-1] + pad)
        _chrome(ax)
    if ax_rel is not None:
        _panel(ax_abs, "a", "seed means")
        # Open markers: no correction across checkpoints (filled means
        # "significant after Holm" in every other figure).
        _panel(ax_rel, "b", "paired difference, 95% CI; open: uncorrected")
        ax_rel.axhline(0, color=INK, linewidth=0.8, zorder=2)
        lo, hi = min(lows), max(highs)
        margin = 0.08 * (hi - lo if hi > lo else 1.0)
        ax_rel.set_ylim(lo - margin, hi + margin)
        ax_rel.set_ylabel(f"$\\Delta${metric.upper()} vs. {display_name(baseline)}")
    return save_figure(fig, out_path)


def plot_budget_sampler(
    points: dict[str, list[tuple[str, float, float, bool, int]]],
    *,
    out_path: str | Path,
    fraction: float = 0.72,
):
    """The objective's paired FID gap at two budgets, under two samplers.

    Args:
        points: ``{sampler: [(budget_label, mean, halfwidth, significant, n), ...]}``,
            in budget order. Filled = significant after Holm correction.

    The samplers are not arms, so they are drawn in ink and grey with distinct
    line styles and markers rather than in arm colours.
    """
    import matplotlib.pyplot as plt

    apply_style()
    styles = [(INK, "-", "o"), (MUTED, "--", "s"), (INK, ":", "^"), (MUTED, "-.", "v")]
    fig, ax = plt.subplots(figsize=figure_size(fraction, 1.6))
    ax.axhline(0, color=INK, linewidth=0.8, zorder=2)
    labels: list[str] = []
    n_samplers = len(points)
    for i, (sampler, pts) in enumerate(points.items()):
        colour, dash, marker = styles[i % len(styles)]
        off = (i - (n_samplers - 1) / 2) * 0.07
        xs = np.arange(len(pts)) + off
        means = [p[1] for p in pts]
        ax.plot(xs, means, color=colour, linestyle=dash, linewidth=1.0, zorder=3,
                label=sampler)
        for x, (_label, m, h, sig, _n) in zip(xs, pts, strict=True):
            ax.errorbar([x], [m], yerr=[h], fmt="none", ecolor=colour, elinewidth=0.9,
                        capsize=2.2, capthick=0.8, zorder=3)
            ax.plot(x, m, marker=marker, markersize=5, linestyle="none",
                    color=colour if sig else "white", markeredgecolor=colour,
                    markeredgewidth=1.0, zorder=4)
            # Outside the dodge horizontally; vertically, away from the line
            # segment on that side, so the number never sits on its own line.
            j = int(round(x - off))
            right = off >= 0
            nb = (j + 1) if right else (j - 1)
            dy = (5 if pts[nb][1] < m else -5) if 0 <= nb < len(pts) else 0
            ax.annotate(_fmt_signed(m), xy=(x, m), xytext=(5 if right else -5, dy),
                        textcoords="offset points", ha="left" if right else "right",
                        va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
                        fontsize=7.5, color=INK)
        if not labels:
            labels = [f"{p[0]}\n($n = {p[4]}$)" for p in pts]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, linespacing=1.15)
    ax.set_xlim(-0.45, len(labels) - 0.55)
    ax.set_ylabel("$\\Delta$FID, AWWL vs. MSE (paired, 95% CI)")
    # Each sampler's entry shows its line and its marker (drawn open: fill
    # has its own key, bottom left).
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, linestyle=d, linewidth=1.0, marker=m, markersize=5,
                      markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.0, label=name)
               for name, (c, d, m) in zip(points, styles, strict=False)]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right", borderaxespad=0.2)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -0.1 * (hi - lo)), hi)
    _chrome(ax, grid="y")
    _fill_legend(ax, colour=INK, loc="lower left", fontsize=7.5)
    return save_figure(fig, out_path)


def plot_deficit_versus_metric(
    *,
    epochs: list[int],
    deficit: dict[str, list[list[float]]],
    fid: dict[str, list[list[float]]],
    out_path: str | Path,
    fraction: float = 0.82,
):
    """The high-band deficit and FID through a longer run, in their own units.

    Args:
        epochs: evaluated checkpoints.
        deficit: ``{group: [per-seed deficits (dB below real) at each epoch]}``.
        fid: ``{group: [per-seed FID at each epoch]}``.

    Two panels, not one indexed axis: a percentage change of a decibel
    quantity next to a percentage change of FID invites the reader to compare
    the two numbers, which have nothing in common. Means with ±1 s.d. over
    seeds.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, (ax_d, ax_f) = plt.subplots(1, 2, figsize=figure_size(fraction, 2.0), sharex=True)
    for ax, data, ylabel, letter, header in (
        (ax_d, deficit, "dB below real", "a", "high-band deficit, mean $\\pm$ s.d."),
        (ax_f, fid, "FID", "b", "FID, mean $\\pm$ s.d."),
    ):
        for group, per_epoch in data.items():
            colour, dash, marker = style_of(group)
            m = np.array([np.mean(v) for v in per_epoch])
            sd = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in per_epoch])
            ax.errorbar(epochs, m, yerr=sd, fmt="none", ecolor=colour, elinewidth=0.9,
                        capsize=2.0, capthick=0.8, zorder=3)
            ax.plot(epochs, m, color=colour, linestyle=dash, marker=marker, markersize=4.5,
                    linewidth=1.2, markeredgecolor="white", markeredgewidth=0.5,
                    label=display_name(group), zorder=4)
        ax.set_xticks(epochs)
        ax.set_xlabel("training epoch")
        ax.set_ylabel(ylabel)
        _panel(ax, letter, header)
        _chrome(ax)
    ax_d.legend(frameon=False, fontsize=7.5, loc="best", borderaxespad=0.2)
    return save_figure(fig, out_path)


# --------------------------------------------------------------------------
# Second tasks


def _place_labels(ax, points: dict[str, tuple[float, float]], hard: list, soft: list,
                  fontsize: float = 7.5, *, softer: list | None = None) -> None:
    """Greedy label placement around each point.

    Candidates are tried at growing distances in a fixed order of directions.
    A label must stay inside the axes and clear every placed label and every
    ``hard`` box (markers); the first pass also asks it to clear the ``soft``
    boxes (interval whiskers) and the ``softer`` ones (zero rules), the second
    only the ``soft`` ones, and the third neither. Labels carry a thin
    white halo, so a whisker that does pass behind one does not strike it
    through. Beyond the nearest ring a hairline leader joins label and point.
    """
    import matplotlib.patheffects as pe
    from matplotlib.text import Text

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(renderer)
    placed: list = []
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    radii = (6, 10, 15, 21, 28, 36, 46)
    angles = (0, 180, 30, 150, 330, 210, 60, 120, 300, 240, 90, 270)

    def attempt(name, x, y, obstacles):
        for rad in radii:
            for ang in angles:
                dx, dy = rad * np.cos(np.radians(ang)), rad * np.sin(np.radians(ang))
                ha = "left" if dx > 1 else ("right" if dx < -1 else "center")
                va = "bottom" if dy > 1 else ("top" if dy < -1 else "center")
                txt = ax.annotate(
                    name, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=fontsize, color=INK, zorder=6,
                    path_effects=halo,
                    arrowprops=(dict(arrowstyle="-", color=MUTED, linewidth=0.5,
                                     shrinkA=0, shrinkB=3) if rad > radii[0] else None),
                )
                # The text alone: an Annotation's own extent includes its
                # leader, which always touches the point being labelled.
                txt.update_positions(renderer)
                bb = Text.get_window_extent(txt, renderer).expanded(1.03, 1.10)
                inside = (bb.x0 >= axbox.x0 and bb.x1 <= axbox.x1
                          and bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1)
                if inside and not any(bb.overlaps(o) for o in obstacles + placed):
                    placed.append(bb)
                    return True
                txt.remove()
        return False

    for name, (x, y) in points.items():
        softer_ = softer or []
        # A long name may also be set on two lines, broken after its comma.
        variants = [name]
        if ", " in name and len(name) > 16:
            head, _, tail = name.partition(", ")
            variants.append(f"{head},\n{tail}")
        tiers = [hard + soft + softer_, hard + soft, hard]
        if any(attempt(v, x, y, obstacles) for obstacles in tiers for v in variants):
            continue
        ax.annotate(name, xy=(x, y), xytext=(6, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=fontsize, color=INK, zorder=6,
                    path_effects=halo)
        logger.warning("label %r could not be placed without overlap", name)


def plot_tradeoff(
    rows,
    *,
    x_metric: str,
    y_metric: str,
    baseline: str,
    out_path: str | Path,
    x_label: str | None = None,
    y_label: str | None = None,
    fraction: float = 0.88,
    aspect: float = 1.45,
    alpha: float = 0.05,
    labels: str = "legend",
    legend_loc: str = "upper right",
    legend_columns: int = 2,
    legend_room: float = 0.14,
):
    """Two metrics' paired differences from the baseline, one point per arm.

    The baseline sits at the origin, so "better than the baseline on both"
    is the upper-right quadrant. Each arm is drawn at its mean paired
    difference with a 95% t-interval on each axis. Filled = significant after
    Holm correction on either metric.

    ``labels="legend"`` (the default) names the arms in a legend at
    ``legend_loc``, by the colour and marker each arm has in every figure: the
    arms crowd into one corner with intervals that cross one another, so
    leader lines from direct labels would cross whiskers whatever their
    placement. The legend closes with the key to fill. ``labels="direct"``
    places a label beside each point instead.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox

    from awwl.analysis.stats import compare_to_baseline

    if labels not in ("legend", "direct"):
        raise ValueError(f"labels must be 'legend' or 'direct', not {labels!r}")
    apply_style()
    res_x = {r.group: r for r in compare_to_baseline(rows, metric=x_metric,
                                                       baseline=baseline, alpha=alpha)}
    res_y = {r.group: r for r in compare_to_baseline(rows, metric=y_metric,
                                                       baseline=baseline, alpha=alpha)}
    groups = [g for g in res_x if g in res_y]
    if not groups:
        raise ValueError(f"baseline {baseline!r} missing or nothing to compare")
    order = [g for g in PRESENTATION_ORDER if g in groups] + sorted(
        g for g in groups if g not in PRESENTATION_ORDER)

    fig, ax = plt.subplots(figsize=figure_size(fraction, aspect))
    ax.axvline(0, color=INK, linewidth=0.8, zorder=2)
    ax.axhline(0, color=INK, linewidth=0.8, zorder=2)
    colour_b, _d, marker_b = style_of(baseline)
    ax.plot(0, 0, marker=marker_b, markersize=5.5, color=colour_b, linestyle="none",
            markeredgecolor="white", zorder=5)
    xs, ys = [0.0], [0.0]
    whiskers = []
    handles = []
    for g in order:
        rx, ry = res_x[g], res_y[g]
        hx, hy = _halfwidth(rx), _halfwidth(ry)
        colour, _dash, marker = style_of(g)
        sig = rx.significant or ry.significant
        ax.errorbar([rx.mean_delta], [ry.mean_delta], xerr=[hx], yerr=[hy], fmt="none",
                    ecolor=colour, elinewidth=0.7, alpha=0.8, capsize=0, zorder=3)
        ax.plot(rx.mean_delta, ry.mean_delta, marker=marker, markersize=marker_size(marker, 5.5),
                linestyle="none", color=colour if sig else "white",
                markeredgecolor=colour, markeredgewidth=1.0, zorder=5)
        xs += [rx.mean_delta - hx, rx.mean_delta + hx]
        ys += [ry.mean_delta - hy, ry.mean_delta + hy]
        whiskers.append((rx.mean_delta, ry.mean_delta, hx, hy))
        # Identity only: every entry is drawn open, and fill has its own key.
        handles.append(Line2D([], [], linestyle="none", marker=marker,
                              markersize=marker_size(marker, 5.0),
                              markerfacecolor="white", markeredgecolor=colour,
                              markeredgewidth=1.0, label=display_name(g)))
    xr, yr = max(xs) - min(xs), max(ys) - min(ys)
    # The legend sits in the upper right: a little more room on the right keeps
    # it clear of the vertical zero rule.
    ax.set_xlim(min(xs) - 0.06 * xr, max(xs) + (legend_room if labels == "legend" else 0.06) * xr)
    ax.set_ylim(min(ys) - 0.06 * yr, max(ys) + 0.06 * yr)
    ax.set_xlabel(x_label or f"$\\Delta$ {x_metric} vs. {display_name(baseline)}")
    ax.set_ylabel(y_label or f"$\\Delta$ {y_metric} vs. {display_name(baseline)}")
    _chrome(ax)

    if labels == "legend":
        handles += [
            Line2D([], [], linestyle="none", marker="o", markersize=5.0, markerfacecolor=MUTED,
                   markeredgecolor=MUTED, markeredgewidth=1.0,
                   label=FILL_KEY[0]),
        ]
        legend = ax.legend(handles=handles, loc=legend_loc, ncol=legend_columns, frameon=True,
                           facecolor="white", edgecolor="none", framealpha=1.0, fontsize=7,
                           handletextpad=0.3, columnspacing=0.7, labelspacing=0.25,
                           borderaxespad=0.25, borderpad=0.25, handlelength=1.0)
        legend.set_zorder(6)
        return save_figure(fig, out_path)

    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    to_px = ax.transData.transform
    hard, soft = [], []
    for g in [baseline] + order:
        x = res_x[g].mean_delta if g != baseline else 0.0
        y = res_y[g].mean_delta if g != baseline else 0.0
        px, py = to_px((x, y))
        hard.append(Bbox.from_extents(px - 9, py - 9, px + 9, py + 9))
    for x, y, hx, hy in whiskers:
        (x0, y0), (x1, y1) = to_px((x - hx, y)), to_px((x + hx, y))
        soft.append(Bbox.from_extents(x0, y0 - 2, x1, y1 + 2))
        (x0, y0), (x1, y1) = to_px((x, y - hy)), to_px((x, y + hy))
        soft.append(Bbox.from_extents(x0 - 2, y0, x1 + 2, y1))
    # A label is better off the zero rules too, but a label that has to cross
    # something should cross a rule before it crosses a whisker.
    box = ax.get_window_extent()
    ox, oy = to_px((0.0, 0.0))
    rules = [Bbox.from_extents(box.x0, oy - 2, box.x1, oy + 2),
             Bbox.from_extents(ox - 2, box.y0, ox + 2, box.y1)]
    # The baseline is the origin by construction (the axis labels say so), so
    # it carries no label of its own to be confused with a neighbour's.
    names = {display_name(g, flat=True): (res_x[g].mean_delta, res_y[g].mean_delta)
             for g in order}
    # Points crowded into the upper-left quadrant (better on similarity, worse
    # on CLIP score) are labelled in a column beside the cluster, in the order
    # of their heights, so that no two leaders cross; the rest are placed
    # around their points.
    crowded = {n: xy for n, xy in names.items() if xy[0] < 0 < xy[1]}
    if len(crowded) >= 3:
        _column_labels(ax, crowded, whiskers, hard)
        names = {n: xy for n, xy in names.items() if n not in crowded}
    _place_labels(ax, names, hard, soft, softer=rules)
    return save_figure(fig, out_path, layout=False)


def _column_labels(ax, points: dict[str, tuple[float, float]], whiskers, hard,
                   fontsize: float = 7.5) -> None:
    """Labels for a cluster of points, stacked in a column right of the cluster.

    The column starts just right of the cluster's rightmost whisker and runs
    downwards one line per label; each label is joined to its point by a
    straight hairline leader that leaves the label at its left-hand middle, so
    the leaders drawn are the segments scored here. Among a few placements of
    the column and every order of the labels, the one chosen is, in order of
    priority: the one whose leaders cross each other least, then pass over
    fewest other markers, then cross fewest other arms' whiskers, then are
    shortest. No two leaders cross where that is possible.
    """
    import itertools

    import matplotlib.patheffects as pe

    fig = ax.figure
    to_px = ax.transData.transform
    from_px = ax.transData.inverted().transform
    names = list(points)
    px = {n: tuple(to_px(points[n])) for n in names}
    own = {n: [(x, y, hx, hy) for x, y, hx, hy in whiskers
               if abs(x - points[n][0]) < 1e-12 and abs(y - points[n][1]) < 1e-12] for n in names}
    right = max(to_px((x + hx, 0.0))[0] for n in names for x, _y, hx, _hy in own[n]) \
        if any(own.values()) else max(p[0] for p in px.values())
    line_px = fontsize * 1.35 * fig.dpi / 72.0
    box = ax.get_window_extent()
    centre = max(p[1] for p in px.values()) + 0.5 * line_px * (len(names) - 1)
    marks = [((b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0) for b in hard]
    radius = max(((b.x1 - b.x0) / 2.0 for b in hard), default=9.0)
    bars = []   # every whisker as a pixel segment, with the point it belongs to
    for x, y, hx, hy in whiskers:
        centre_px = tuple(to_px((x, y)))
        bars.append((centre_px, tuple(to_px((x - hx, y))), tuple(to_px((x + hx, y)))))
        bars.append((centre_px, tuple(to_px((x, y - hy))), tuple(to_px((x, y + hy)))))

    def crosses(a, b, c, d) -> bool:
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)

    def distance(p, q, r) -> float:
        """Distance from ``r`` to the segment ``p``-``q``."""
        dx, dy = q[0] - p[0], q[1] - p[1]
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((r[0] - p[0]) * dx + (r[1] - p[1]) * dy)
                                                / (dx * dx + dy * dy)))
        return float(np.hypot(p[0] + t * dx - r[0], p[1] + t * dy - r[1]))

    def near(a, b) -> bool:
        return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6

    def cost(order, slots):
        segs = [(px[n], slots[i]) for i, n in enumerate(order)]
        n_cross = sum(crosses(*s, *t) for s, t in itertools.combinations(segs, 2))
        n_marks = sum(1 for p, q in segs for m in marks
                      if not near(m, p) and distance(p, q, m) < radius + 2.0)
        n_bars = sum(1 for p, q in segs for c, a, b in bars if not near(c, p) and crosses(p, q, a, b))
        return n_cross, n_marks, n_bars, sum(float(np.hypot(q[0] - p[0], q[1] - p[1])) for p, q in segs)

    tops = [min(centre + shift * line_px, box.y1 - line_px / 2)
            for shift in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    fitting = [t for t in tops if t - (len(names) - 1) * line_px >= box.y0 + line_px / 2]
    # A column too tall for the axes at every shift is still placed (from the
    # unshifted top, running below the frame) rather than dropped.
    tops = fitting or [tops[2]]
    best = None
    for dx in (6.0, 18.0, 32.0):
        for top in tops:
            slots = [(right + dx, top - i * line_px) for i in range(len(names))]
            candidates = (itertools.permutations(names) if len(names) <= 7
                          else [sorted(names, key=lambda n: -points[n][1])])
            for order in candidates:
                c = cost(order, slots)
                if best is None or c < best[0]:
                    best = (c, order, slots)
    _cost, order, slots = best
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    for i, name in enumerate(order):
        tx, ty = from_px(slots[i])
        ax.annotate(name, xy=points[name], xytext=(tx, ty), textcoords="data", ha="left",
                    va="center", fontsize=fontsize, color=INK, zorder=6, path_effects=halo,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5, shrinkA=1,
                                    shrinkB=3, relpos=(0.0, 0.5)))


def plot_null_panel(
    rows,
    *,
    metrics: list[tuple[str, str, bool]],
    baseline: str,
    out_path: str | Path,
    order: list[str] | None = None,
    fraction: float = 1.0,
    alpha: float = 0.05,
):
    """Small multiples of paired differences, one panel per metric.

    Args:
        metrics: ``[(ledger_key, axis label, lower_is_better), ...]``.
        order: arm order top to bottom (default: the order of
            :data:`PRESENTATION_ORDER`, which is the generated tables' order).

    Each marker is the mean paired difference over seeds with its 95%
    t-interval; filled = significant after Holm correction within the metric.
    Each panel has its own scale because the metrics are not commensurable,
    and small metrics are scaled by a power of ten stated in the axis label.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from awwl.analysis.stats import compare_to_baseline

    apply_style()
    first = {r.group for r in compare_to_baseline(rows, metric=metrics[0][0],
                                                   baseline=baseline, alpha=alpha)}
    if order is None:
        order = [g for g in PRESENTATION_ORDER if g in first] + sorted(first - set(PRESENTATION_ORDER))
    order = [g for g in order if g in first]

    fig, axes = plt.subplots(1, len(metrics), figsize=figure_size(fraction, 3.3),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    ys = {g: len(order) - 1 - i for i, g in enumerate(order)}
    for k, (ax, (metric, label, lower_better)) in enumerate(zip(axes, metrics, strict=True)):
        res = {r.group: r for r in compare_to_baseline(rows, metric=metric,
                                                        baseline=baseline, alpha=alpha)}
        spans = [abs(res[g].mean_delta) + _halfwidth(res[g]) for g in order if g in res]
        biggest = max(spans, default=1.0)
        exp = 0
        while biggest and biggest * 10**exp < 1.0:
            exp += 1
        mult = 10**exp
        ax.axvline(0, color=INK, linewidth=0.8, zorder=2)
        for g in order:
            r = res.get(g)
            if r is None:
                continue
            colour, _dash, marker = style_of(g)
            ax.errorbar([r.mean_delta * mult], [ys[g]], xerr=[_halfwidth(r) * mult],
                        fmt="none", ecolor=colour, elinewidth=0.9, capsize=2.0,
                        capthick=0.8, zorder=3)
            ax.plot(r.mean_delta * mult, ys[g], marker=marker, markersize=marker_size(marker, 5),
                    linestyle="none", color=colour if r.significant else "white",
                    markeredgecolor=colour, markeredgewidth=1.0, zorder=4)
        # "PSNR (dB)" scaled by 10^-2 reads "ΔPSNR (10^-2 dB)": the unit of the axis.
        name, _, unit = label.partition(" (")
        unit = unit.rstrip(")")
        power = f"$10^{{-{exp}}}$" if exp else ""
        units = " ".join(u for u in (power, unit) if u)
        arrow = "better $\\leftarrow$" if lower_better else "$\\rightarrow$ better"
        ax.set_xlabel(f"$\\Delta${name}" + (f" ({units})" if units else "") + f"\n{arrow}",
                      linespacing=1.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        lim = max(abs(v) for v in ax.get_xlim())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.tick_params(axis="y", length=0)
        _chrome(ax, grid="x")
        _panel(ax, "abcdefgh"[k])
    axes[0].set_yticks([ys[g] for g in order])
    axes[0].set_yticklabels([display_name(g, flat=True) for g in order])
    # The key to fill, beneath the panels (Holm within each metric).
    from matplotlib.lines import Line2D

    key = [Line2D([], [], linestyle="none", marker="o", markersize=4.6, markerfacecolor=face,
                  markeredgecolor=MUTED, markeredgewidth=0.9) for face in (MUTED, "white")]
    legend = fig.legend(key, [FILL_KEY[0] + " within the metric", FILL_KEY[1]], loc="lower center",
                        ncol=2, frameon=False, fontsize=7.5, handletextpad=0.3, columnspacing=1.5,
                        borderaxespad=0.2, handlelength=1.0)
    fig.canvas.draw()
    height_px = legend.get_window_extent(fig.canvas.get_renderer()).height
    fig.tight_layout(pad=0.4, rect=(0, (height_px + 0.04 * fig.dpi)
                                    / (fig.get_size_inches()[1] * fig.dpi), 1, 1))
    return save_figure(fig, out_path, layout=False)


# --------------------------------------------------------------------------
# Readers


def parse_boost_tables(
    folder: str | Path, *, per_seed: bool = False,
) -> dict[str, tuple[float, float, float, float]] | dict[str, dict[int, tuple]]:
    """Recover the boost experiment's numbers from ``scripts/boost_test.sh`` output.

    Each ``<config>_s<seed>.txt`` holds a table whose first two rows are the
    unmodified samples and the boosted samples: ``(deficit_db, fid)`` each.
    Returns seed means per configuration, or with ``per_seed=True``
    ``{config: {seed: (d0, f0, d1, f1)}}``. Reads the per-run text tables
    rather than asking for the numbers to be retyped: a figure that disagrees
    with its own data is worse than no figure.
    """
    import re

    runs: dict[str, dict[int, tuple[float, float, float, float]]] = {}
    for path in sorted(Path(folder).glob("*.txt")):
        name, _, seed_text = path.stem.rpartition("_s")
        if not name:
            continue
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\s*(-?[\d.]+)d\s+(-?[\d.]+)d\s+([\d.]+)\s", line)
            if m:
                rows.append((float(m.group(2)), float(m.group(3))))
        if len(rows) >= 2:
            (d0, f0), (d1, f1) = rows[0], rows[1]
            try:
                seed = int(seed_text)
            except ValueError:
                seed = len(runs.get(name, {}))
            runs.setdefault(name, {})[seed] = (d0, f0, d1, f1)

    if per_seed:
        return runs
    return {
        name: tuple(float(np.mean([r[i] for r in by_seed.values()])) for i in range(4))  # type: ignore[misc]
        for name, by_seed in runs.items()
    }
