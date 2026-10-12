"""Generate the paper's result tables directly from the results ledgers.

Every quantitative table of the manuscript is emitted by this script (or, for
the spectral tables, by ``make_band_table.py``) and pulled in with ``\\input``.
Rebuilding after new runs land updates every table at once; a ledger that is
missing produces a table whose cells say so (``\\textbf{??}``) rather than a
stale number.

Usage::

    python scripts/make_tables.py            # every table, default ledgers

Ledgers read (all optional; a missing one only disables its tables):

``runs/phase0/results.jsonl``            CIFAR-10 matrix, DDIM-100
``runs/phase0_ddpm1000/results.jsonl``   the same weights, ancestral DDPM-1000
``runs/converged/results.jsonl``         800-epoch EMA runs, DDIM-100
``runs/converged_ddpm1000/results.jsonl`` the same, DDPM-1000
``runs/dreambooth_seeds/results.jsonl``  DreamBooth fine-tunes
``runs/restoration_cifar/results.jsonl`` Gaussian denoising
``runs/pricing/results.jsonl``           FID pricing of spectral operators

Statistical conventions, used by every builder here and by
``make_numbers.py``:

* ``±`` is the across-seed **sample** standard deviation (``ddof = 1``).
* A comparison is paired by training-seed ID (never by list position); the
  interval is the 95% Student-t interval of the mean paired difference and
  ``p`` the two-sided paired t-test.
* Holm--Bonferroni correction runs over a **fixed, declared family** of arms
  (the ``FAMILY_*`` constants below). The family does not grow with the
  ledger: an arm of the family that has not been run enters the correction
  with ``p = 1``, so a result's corrected ``p`` does not depend on which other
  jobs happen to have finished. Raw ``p`` is printed next to the corrected one.
  A contrast belongs to exactly one family; a table that repeats a contrast of
  another family prints that family's corrected ``p`` and marks it.
* Bold marks ``p_Holm < 0.05`` and nothing else, and every table that bolds a
  value says so in its caption (:data:`BOLD_NOTE`).
* Precision: an FID gap or MRG prints with two decimals and the value of an
  operator (a small, precisely estimated FID change) with three; from 10 the
  number keeps three significant figures (:func:`adaptive`). A cell's
  estimate and interval bounds share the estimate's precision.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awwl.analysis.results import load_results  # noqa: E402

# =============================================================================
# Display names
# =============================================================================

#: The one display name of every arm. Every table written here and by
#: ``make_band_table.py``, and every figure (``awwl.plotting.paper_figures``
#: keeps a copy, which ``tests/test_paper_figures.py`` checks against this
#: map), prints an arm under exactly this name; the manuscript's text uses the
#: same names. ``\omega_b`` is the effective weight of sub-band ``b`` (each
#: detail band carries ``w_det/3`` when the detail bands are averaged, as in
#: the released AWWL, and ``w_det`` when they are summed).
NAMES: dict[str, str] = {
    "mse": r"MSE",
    "l1": r"$L_1$",
    "huber": r"Huber",
    "charbonnier": r"Charbonnier",
    # The *raw* ratio a_t/(1-a_t), unclamped; not Min-SNR-gamma.
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

#: The groups the arms are presented in, in order. Every table that lists
#: arms, Fig. effects and Fig. pricing-arms use these groups, these headers
#: and this order (the figures keep a copy, checked by the tests), so a row
#: sits in the same place wherever it is read. MSE, the baseline, precedes the
#: groups where it is a row at all.
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

#: Presentation order, ``(key, name)``: the baseline, then the blocks.
ROWS: list[tuple[str, str]] = [("mse", NAMES["mse"])] + [
    (k, NAMES[k]) for _title, keys in BLOCKS for k in keys]

#: Kept for callers written when nested rows had shortened labels; every arm
#: now has one name, so this is the same map.
FLAT: dict[str, str] = NAMES

#: Position of each arm in the presentation order, for sorting.
ORDER: dict[str, int] = {k: i for i, (k, _name) in enumerate(ROWS)}

# What each extension turns on (Table extensions).
AXIS: dict[str, str] = {
    "awwl_subband": r"separate schedule per detail band",
    "awwl_spatial": r"weights vary over position",
    "awwl_multilevel": r"three-level decomposition",
    "awwl_learned": r"$\sigma$-conditioned learned weights",
    "awwl_gradnorm": r"GradNorm sets the weights",
    "awwl_lifting": r"learned lifting basis",
    "awwl_full": r"learned weights, spatial, lifting",
}

# How the three detail bands of a level combine. The released AWWL averages
# them (each carries w_det/3); the conference paper's equation as printed sums
# them (each carries w_det). The rational weighting behind the ``wavelet_*``
# arms gives each detail band the full w_det unless ``share_detail_budget`` is
# set, whatever the config's ``detail_reduction`` field says, so this is
# recorded here from the code rather than read from the ledger. The
# multi-level arm sets ``share_detail_budget`` (each of its nine detail bands
# carries w_det/9) and so is not in this set; neither are the learned and
# GradNorm arms, which do not use the rational weighting.
SUMMED_DETAIL = {
    "awwl_eq7", "awwl_grad_matched", "awwl_subband", "awwl_spatial", "awwl_lifting",
}

# =============================================================================
# Comparison families (fixed; see the module docstring)
# =============================================================================

BASELINE = "mse"

#: Every arm of the CIFAR-10 matrix compared against MSE at epoch 199,
#: DDIM-100. The same list is the family for FID, IS, KID, precision and
#: recall (each metric corrected separately). An arm of the family without
#: rows counts with p = 1. ``awwl_ema`` is not here: it is compared with
#: ``mse_ema``.
FAMILY_MAIN: tuple[str, ...] = (
    "l1", "huber", "charbonnier", "snr_weighted", "min_snr", "focal_frequency",
    "static_wavelet", "awwl", "awwl_normalized", "awwl_norm_matched",
    "awwl_grad_matched_mean", "awwl_eq7", "awwl_grad_matched", "mse_ema",
    "awwl_subband", "awwl_multilevel", "awwl_spatial", "awwl_learned",
    "awwl_gradnorm", "awwl_lifting", "awwl_full",
)

#: The EMA pair: AWWL + EMA against MSE + EMA, a one-test family.
EMA_PAIR: tuple[str, str] = ("awwl_ema", "mse_ema")

#: Confound contrasts, each arm against the reference sharing its detail
#: reduction (Table confound, right-hand columns). Holm over these four.
FAMILY_CONFOUND: tuple[tuple[str, str], ...] = (
    ("awwl_normalized", "awwl"),
    ("awwl_norm_matched", "awwl"),
    ("awwl_grad_matched_mean", "awwl"),
    ("awwl_grad_matched", "awwl_eq7"),
)

#: The extensions against their parent, the summed-detail unnormalised AWWL.
EXTENSION_PARENT = "awwl_eq7"
FAMILY_EXTENSIONS: tuple[str, ...] = (
    "awwl_subband", "awwl_spatial", "awwl_multilevel", "awwl_learned",
    "awwl_gradnorm", "awwl_lifting", "awwl_full",
)

#: AWWL's timestep component alone (``awwl_timestep``, S(sigma) times MSE)
#: and what AWWL adds to it: a two-test family of its own, not part of
#: FAMILY_MAIN (make_numbers.py ``dTimestep``/``dAWWLVsTimestep``). Its rows
#: enter a table or figure only once all seeds of the design are in.
TIMESTEP_ARM = "awwl_timestep"
FAMILY_TIMESTEP: tuple[tuple[str, str], ...] = ((TIMESTEP_ARM, "mse"), ("awwl", TIMESTEP_ARM))
TIMESTEP_SEEDS = 5

#: The ancestral-sampler check (Table sampler).
FAMILY_DDPM: tuple[str, ...] = ("static_wavelet", "awwl")

#: Table budget's operating points, ``(budget, sampler)``, in row order. Only
#: the two 800-epoch contrasts form the Budget family; each 200-epoch row
#: repeats a test of another family (FAMILY_MAIN, the EMA pair, FAMILY_DDPM)
#: and carries that family's corrected p (:func:`budget_points`).
BUDGET_POINTS: tuple[tuple[str, str], ...] = (
    ("200 epochs, no EMA", "DDIM-100"),
    ("200 epochs, EMA", "DDIM-100"),
    ("800 epochs, EMA", "DDIM-100"),
    ("200 epochs, no EMA", "DDPM-1000"),
    ("800 epochs, EMA", "DDPM-1000"),
)
FAMILY_BUDGET: tuple[int, ...] = (2, 4)

#: DreamBooth, per metric.
FAMILY_DREAMBOOTH: tuple[str, ...] = (
    "l1", "huber", "charbonnier", "snr_weighted", "min_snr", "static_wavelet",
    "awwl", "awwl_normalized", "awwl_learned",
)

#: Gaussian denoising, per metric.
FAMILY_RESTORATION: tuple[str, ...] = (
    "l1", "charbonnier", "static_wavelet", "awwl", "awwl_normalized",
)

#: The pricing test applied across arms (Table pricing-arms): every arm the
#: pricing job transplants onto MSE's samples.
FAMILY_PRICING: tuple[str, ...] = (
    "awwl", "static_wavelet", "awwl_normalized", "awwl_norm_matched", "awwl_eq7",
    "awwl_grad_matched", "awwl_grad_matched_mean", "focal_frequency", "min_snr", "huber", "awwl_ema",
    "mse_ema", "awwl_subband", "awwl_spatial", "awwl_learned", "awwl_lifting",
    "awwl_full", "l1", "charbonnier",
)

# =============================================================================
# The conference paper's CIFAR-10 table
# =============================================================================

#: Single-run values of Table 2 of the conference paper \citep{thao2026awwl}
#: (200 epochs, ancestral DDPM-1000 sampling, no EMA), as recorded in
#: README.md ("Table 2") and docs/PHASE0.md: all seven rows, FID and Inception
#: Score. These are external inputs, the one place a published number enters
#: the build. The conference table's "SNR-Weighted" row is the raw SNR
#: weighting of the released code (``snr_weighted``;
#: ``awwl.losses.analytic.snr_weighted_loss``); its "Perceptual" row was not
#: retrained (configs/pipeline/phase0.yaml gives the reasons).
CONFERENCE_CIFAR10: dict[str, dict[str, float]] = {
    "mse": {"fid": 16.68, "is_mean": 7.78},
    "l1": {"fid": 19.70, "is_mean": 7.69},
    "huber": {"fid": 20.13, "is_mean": 7.62},
    "perceptual": {"fid": 20.02, "is_mean": 7.70},
    "snr_weighted": {"fid": 19.83, "is_mean": 7.58},
    "static_wavelet": {"fid": 16.55, "is_mean": 7.81},
    "awwl": {"fid": 16.62, "is_mean": 7.95},
}

#: The conference table's rows, in its order.
CONFERENCE_ROWS: tuple[str, ...] = tuple(CONFERENCE_CIFAR10)

#: Names of conference rows that are not arms of the matrix (no NAMES entry).
CONFERENCE_ONLY_NAMES: dict[str, str] = {"perceptual": "Perceptual (VGG)"}

#: The chosen boost of the pricing test (a design parameter, not a measurement).
BOOST_DOSE = 0.41

PLACEHOLDER = r"\textbf{??}"

PREAMBLE = (
    "% Generated by scripts/make_tables.py --- do not edit by hand.\n"
    "% Regenerate after every sweep; the paper reads these directly.\n"
)

# =============================================================================
# Statistics
# =============================================================================


def sample_sd(values: Sequence[float]) -> float:
    """Sample standard deviation (``ddof = 1``); 0 for fewer than two values."""
    values = list(values)
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def t_crit(df: int, confidence: float = 0.95) -> float:
    from scipy import stats

    return float(stats.t.ppf(0.5 + confidence / 2.0, df)) if df > 0 else math.nan


@dataclass
class Paired:
    """A paired-by-seed (or one-sample) comparison."""

    n: int
    mean: float
    sd: float
    lo: float
    hi: float
    t: float
    p: float
    wilcoxon: float | None
    seeds: tuple
    deltas: tuple
    p_holm: float | None = None

    @property
    def half(self) -> float:
        return (self.hi - self.lo) / 2.0


def one_sample(deltas: Sequence[float], seeds: Sequence = ()) -> Paired | None:
    """Mean, sample SD, 95% t-interval and two-sided t-test of ``deltas``."""
    from scipy import stats

    d = [float(x) for x in deltas]
    n = len(d)
    if n < 2:
        return None
    mean = sum(d) / n
    sd = sample_sd(d)
    se = sd / math.sqrt(n)
    if sd == 0.0:
        t, p = (0.0, 1.0) if mean == 0.0 else (math.copysign(math.inf, mean), 0.0)
    else:
        t = mean / se
        p = float(2.0 * stats.t.sf(abs(t), n - 1))
    half = t_crit(n - 1) * se
    try:
        w = float(stats.wilcoxon(d).pvalue) if any(x != 0 for x in d) else 1.0
    except ValueError:
        w = None
    return Paired(n, mean, sd, mean - half, mean + half, t, p, w, tuple(seeds), tuple(d))


def paired(a: Mapping[Any, float], b: Mapping[Any, float]) -> Paired | None:
    """``a - b`` over the seeds both contain, keyed by seed ID."""
    shared = sorted(set(a) & set(b), key=str)
    if len(shared) < 2:
        return None
    return one_sample([a[s] - b[s] for s in shared], shared)


def one_sided_p(res: Paired, null: float, *, greater: bool) -> float:
    """p of H1: mean > null (``greater``) or mean < null, from the same t."""
    from scipy import stats

    se = res.sd / math.sqrt(res.n)
    if se == 0.0:
        return 0.0 if (res.mean > null) == greater else 1.0
    t = (res.mean - null) / se
    return float(stats.t.sf(t, res.n - 1) if greater else stats.t.cdf(t, res.n - 1))


def prediction_interval(res: Paired, confidence: float = 0.95) -> tuple[float, float]:
    """Interval for the difference one further paired run would show."""
    half = t_crit(res.n - 1, confidence) * res.sd * math.sqrt(1.0 + 1.0 / res.n)
    return res.mean - half, res.mean + half


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm--Bonferroni step-down adjusted p-values, in input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    out = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        out[idx] = running
    return out


def apply_family(results: Mapping[str, Paired | None], family: Sequence) -> None:
    """Fill ``p_holm`` for every family member present; absent ones count as p = 1."""
    ps = [results[k].p if results.get(k) is not None else 1.0 for k in family]
    for key, adj in zip(family, holm(ps), strict=True):
        if results.get(key) is not None:
            results[key].p_holm = adj


# =============================================================================
# Formatting
# =============================================================================


def fnum(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return PLACEHOLDER
    s = f"{x:.{digits}f}"
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def fsigned(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return PLACEHOLDER
    s = fnum(x, digits)
    return s if s.startswith("-") or float(s) == 0.0 else "+" + s


def fbound(x: float, digits: int = 2) -> str:
    """An interval bound; one that would round to zero gains digits until it does not.

    ``[0.000, 0.005]`` next to ``p = 0.075`` reads as an interval that
    excludes zero; the bound is ``-0.0004``, and that is what is printed.
    """
    d = digits
    while x != 0 and round(x, d) == 0 and d < digits + 3:
        d += 1
    return fnum(x, d)


def fci(lo: float | None, hi: float | None, digits: int = 2) -> str:
    if lo is None or hi is None:
        return PLACEHOLDER
    return "$[" + fbound(lo, digits) + ", " + fbound(hi, digits) + "]$"


def fp(p: float | None) -> str:
    """A p-value for math mode: 4 decimals below 0.01, 3 below 0.1, else 2."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return PLACEHOLDER
    if p < 1e-4:
        return r"<10^{-4}"
    if p < 0.01:
        return f"{p:.4f}"
    if p < 0.1:
        return f"{p:.3f}"
    return f"{p:.2f}"


def fp_wilcoxon(p: float | None) -> str:
    """Wilcoxon p to four decimals, so its floor prints as 0.0625, not 0.062."""
    if p is None:
        return "---"
    return "$" + (f"{p:.4f}" if p < 0.1 else f"{p:.2f}") + "$"


def fp_cell(p: float | None, bold: bool = False) -> str:
    if p is None:
        return "---"
    text = fp(p)
    return r"$\mathbf{" + text + "}$" if bold else "$" + text + "$"


_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve")


def cw(n: int) -> str:
    """A count as prose: words up to twelve, digits above."""
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)


def fint(n: int) -> str:
    return r"\num{" + str(int(n)) + "}"


def fmean_sd(values: Sequence[float] | None, digits: int = 2, scale: float = 1.0) -> str:
    if not values:
        return PLACEHOLDER if values is None else "---"
    v = [x * scale for x in values]
    return "$" + fnum(statistics.mean(v), digits) + r" \pm " + fnum(sample_sd(v), digits) + "$"


def adaptive(digits: int, *values: float) -> int:
    """Decimals for a quantity of this magnitude: ``digits`` below 10, then
    three significant figures (one decimal from 10, none from 100).

    The one precision rule of every table: an FID difference prints with two
    decimals, a 300-point gap needs no 0.01, and a cell's estimate and bounds
    share the estimate's precision (pass the estimate alone).
    """
    big = max((abs(v) for v in values if v is not None and not math.isnan(v)), default=0.0)
    if big >= 100:
        return 0
    if big >= 10:
        return min(digits, 1)
    return digits


def fcell_ci(res: Paired | None, digits: int = 2) -> str:
    """``+0.42 [0.10, 0.74]`` in one cell, bounds at the estimate's precision."""
    if res is None:
        return PLACEHOLDER
    d = adaptive(digits, res.mean)
    return "$" + fsigned(res.mean, d) + r"$\;" + fci(res.lo, res.hi, d)


def fdiff(res: Paired | None, digits: int = 2) -> str:
    """A paired difference's estimate alone, at the precision of :func:`fcell_ci`."""
    if res is None:
        return PLACEHOLDER
    return "$" + fsigned(res.mean, adaptive(digits, res.mean)) + "$"


def fci_of(res: Paired | None, digits: int = 2) -> str:
    """A paired difference's interval alone, at the precision of :func:`fcell_ci`."""
    if res is None:
        return PLACEHOLDER
    return fci(res.lo, res.hi, adaptive(digits, res.mean))


#: Appended to the caption of every table that sets a p-value in bold.
BOLD_NOTE = r" Bold: $p_{\mathrm{Holm}} < 0.05$ within the stated family."


def bold_note(body: Sequence[str]) -> str:
    """:data:`BOLD_NOTE` if any line of a table body sets a value in bold."""
    return BOLD_NOTE if any(r"\mathbf{" in line for line in body) else ""


def name_column(width: str) -> str:
    """A first (name) column of at most ``width``: ragged right, and a name too
    long for it continues on a second line, indented, instead of widening the
    table past the text block (``array`` syntax; siunitx loads the package)."""
    return (r">{\raggedright\arraybackslash\hangindent=1em\hangafter=1}p{" + width + "}")


def block_lines(keys: Sequence[str], ncols: int, row: Callable[[str], str | None],
                *, lead: Sequence[str] = ()) -> list[str]:
    """Table rows grouped under the :data:`BLOCKS` headers, in that order.

    ``row(key)`` returns a row (or ``None`` to leave the arm out); ``lead``
    arms (the baseline) come first, without a header. A block with no row
    prints no header.
    """
    out: list[str] = []
    for key in lead:
        line = row(key) if key in keys else None
        if line:
            out.append(line)
    for title, block in BLOCKS:
        lines = [line for line in (row(k) for k in block if k in keys) if line]
        if not lines:
            continue
        if out:
            out.append(r"\addlinespace[2pt]")
        out.append(r"\multicolumn{" + str(ncols) + r"}{l}{\emph{" + title + r"}} \\")
        out += lines
    return out


def seed_mark(n: int, expected: int = 5) -> str:
    return "" if n >= expected else r"\,\textsuperscript{" + str(n) + "}"


# =============================================================================
# Ledgers
# =============================================================================


def load_eval(path: Path | str, epoch: int | None = None) -> list[dict]:
    """Evaluation rows of a ledger (optionally at one epoch); [] if absent."""
    path = Path(path)
    if not path.exists():
        return []
    rows = load_results(path, kind="eval")
    if epoch is not None:
        rows = [r for r in rows if r.get("epoch") == epoch]
    return rows


def load_train(path: Path | str) -> list[dict]:
    path = Path(path)
    return load_results(path, kind="train") if path.exists() else []


def _valid(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) != -1.0 \
        and not math.isnan(float(value))


def by_seed(rows: Sequence[dict], group: str, metric: str, scale: float = 1.0) -> dict[Any, float]:
    """``{seed: value}`` for one group; a later row for the same seed wins.

    Rows without the metric, or with the evaluators' failure sentinel -1, are
    skipped, as in :func:`awwl.analysis.stats.summarize_groups`.
    """
    out: dict[Any, float] = {}
    for row in rows:
        if str(row.get("group")) != group:
            continue
        value = row.get(metric)
        if _valid(value):
            out[row.get("seed")] = float(value) * scale
    return out


def groups_of(rows: Sequence[dict]) -> set[str]:
    return {str(r.get("group")) for r in rows}


def compare_family(
    rows: Sequence[dict], metric: str, family: Sequence[str], baseline: str = BASELINE,
    scale: float = 1.0,
) -> dict[str, Paired | None]:
    """Each family arm against ``baseline``, Holm-corrected over the family."""
    base = by_seed(rows, baseline, metric, scale)
    out = {arm: paired(by_seed(rows, arm, metric, scale), base) for arm in family}
    apply_family(out, family)
    return out


def contrasts(
    rows: Sequence[dict], metric: str, pairs: Sequence[tuple[str, str]], scale: float = 1.0,
) -> dict[tuple[str, str], Paired | None]:
    out = {(a, r): paired(by_seed(rows, a, metric, scale), by_seed(rows, r, metric, scale))
           for a, r in pairs}
    apply_family(out, list(pairs))
    return out


def run_config(root: Path, arm: str, seed: int = 1) -> dict:
    path = root / f"{arm}_s{seed}" / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def lower_is_better(metric: str) -> bool:
    return metric.startswith(("fid", "kid", "lpips", "niqe"))


# =============================================================================
# Pricing ledger
# =============================================================================

PRICING_EPOCH = {"cifar10": 199, "cifar10_long": 799, "celeba64": 199}
#: Seeds a pricing measurement must cover before any number is drawn from it.
#: The ledger fills while the job runs; a mean over the first two seeds would
#: otherwise reach the paper looking like a result.
PRICING_SEEDS = {"cifar10": 5, "cifar10_long": 3, "celeba64": 5}
_PRICING_KEY = ("dataset", "op", "dose_db", "target", "target_epoch", "bands", "source", "source_epoch",
                "source_folder", "features", "seed", "quantize")


def load_pricing(path: Path | str) -> list[dict]:
    """Rows of ``runs/pricing/results.jsonl``, de-duplicated (last row wins).

    Re-running a pricing step appends duplicates; the key is every field that
    identifies a measurement.
    """
    path = Path(path)
    if not path.exists():
        return []
    keyed: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        dose = row.get("dose_db")
        # Rows written before target_epoch existed used the dataset's epoch.
        if row.get("op") == "transplant" and row.get("target_epoch") is None:
            row["target_epoch"] = PRICING_EPOCH.get(row.get("dataset"))
        key = tuple(round(dose, 6) if k == "dose_db" and dose is not None else row.get(k)
                    for k in _PRICING_KEY)
        keyed[key] = row
    return list(keyed.values())


def high_before(row: dict) -> float:
    return float(row["dev_before"][2])


def high_after(row: dict) -> float:
    return float(row["dev_after"][2])


def price(
    rows: Sequence[dict], dataset: str, op: str, *, source: str = BASELINE,
    target: str | None = None, dose: float | None = None, quantize: str = "float",
    source_epoch: int | None = None, field: str | Callable[[dict], float] = "fid",
    expected: int | None = None, bands: str | None = None, target_epoch: int | None = None,
    source_folder: str | None = None, features: str | None = None,
) -> dict[Any, float]:
    """``{seed: value}`` of one pricing measurement, or ``{}`` while incomplete.

    ``expected`` is the number of seeds the measurement must cover (default:
    ``PRICING_SEEDS`` of the dataset); fewer rows than that return nothing, so
    every consumer shows a placeholder rather than a partial mean.
    """
    epoch = PRICING_EPOCH.get(dataset) if source_epoch is None else source_epoch
    get = field if callable(field) else (lambda r: r.get(field))
    out: dict[Any, float] = {}
    for r in rows:
        if r.get("dataset") != dataset or r.get("op") != op:
            continue
        if r.get("source", BASELINE) != source or r.get("quantize", "float") != quantize:
            continue
        if epoch is not None and r.get("source_epoch") not in (None, epoch):
            continue
        if op == "transplant" and r.get("target") != target:
            continue
        # A band-restricted operator is a different measurement from the
        # full one; rows written before the field existed are full.
        if r.get("bands") != bands:
            continue
        if r.get("source_folder") != source_folder:
            continue
        # Rows written before the field existed are Inception-V3 (None).
        if r.get("features") != features:
            continue
        if op == "transplant":
            want = PRICING_EPOCH.get(dataset) if target_epoch is None else target_epoch
            if r.get("target_epoch", PRICING_EPOCH.get(dataset)) != want:
                continue
        if op in ("boost", "noise") and (r.get("dose_db") is None or dose is None
                                         or abs(float(r["dose_db"]) - dose) > 1e-6):
            continue
        try:
            value = get(r)
        except (KeyError, IndexError, TypeError):
            continue
        if _valid(value):
            out[r.get("seed")] = float(value)
    need = PRICING_SEEDS.get(dataset) if expected is None else expected
    if need and len(out) < need:
        return {}
    return out


def boost_doses(rows: Sequence[dict], dataset: str = "cifar10", source: str = BASELINE) -> list[float]:
    return sorted({float(r["dose_db"]) for r in rows
                   if r.get("dataset") == dataset and r.get("op") == "boost"
                   and r.get("source", BASELINE) == source and r.get("quantize", "float") == "float"
                   and r.get("dose_db") is not None})


def value_of(before: Mapping, after: Mapping) -> Paired | None:
    """Value of an operator: M(B) - M(T(B)), paired by seed (positive = better)."""
    return paired(before, after)


def mrg(fid_a: Mapping, after: Mapping) -> Paired | None:
    """Mechanism residual gap M(A) - M(T(B)), paired by seed."""
    return paired(fid_a, after)


# =============================================================================
# Builders: CIFAR-10 matrix
# =============================================================================

METRICS: list[tuple[str, str, int, float]] = [
    ("fid", r"FID $\downarrow$", 2, 1.0),
    ("is_mean", r"IS $\uparrow$", 3, 1.0),
    ("kid_tf", r"KID$\times 10^{3}$ $\downarrow$", 2, 1e3),
    ("precision", r"Prec. $\uparrow$", 3, 1.0),
    ("recall", r"Rec. $\uparrow$", 3, 1.0),
]


def shown(rows: Sequence[dict], arm: str, metric: str = "fid") -> bool:
    """Whether an arm's rows are shown: any rows, or for the timestep-only
    arm (its own family) all :data:`TIMESTEP_SEEDS` seeds."""
    n = len(by_seed(rows, arm, metric))
    return n >= (TIMESTEP_SEEDS if arm == TIMESTEP_ARM else 1)


def build_main_table(rows: Sequence[dict], epoch: int, expected_seeds: int = 5) -> str:
    keys = [k for k, _ in ROWS if k in groups_of(rows) and shown(rows, k)]
    uneven = False

    def row(key: str) -> str:
        nonlocal uneven
        n = len(by_seed(rows, key, "fid"))
        cells = [NAMES[key], str(n)]
        for metric, _, digits, scale in METRICS:
            vals = by_seed(rows, key, metric, scale)
            if not vals:
                cells.append("---")
                continue
            mark = seed_mark(len(vals), n) if len(vals) < n else ""
            uneven |= bool(mark)
            cells.append(fmean_sd(list(vals.values()), digits) + mark)
        return " & ".join(cells) + r" \\"

    body = block_lines(keys, len(METRICS) + 2, row, lead=["mse"])
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Every metric for every objective of the CIFAR-10 matrix: 200 epochs, epoch "
        + str(epoch) + r", DDIM-100, \num{10000} samples per run. Mean $\pm$ sample standard deviation over the "
        r"$n$ training seeds" + ("; a superscript gives the number of seeds where a "
                                  r"metric has fewer" if uneven else "") + r". KID is from "
        r"\texttt{torch-fidelity}; Appendix~\ref{app:supp-kid} explains why the hand-written "
        r"estimator is not reported.}",
        r"\label{tab:full}",
        r"\begin{tabular}{" + name_column("4.6cm") + "r" + "r" * len(METRICS) + "}",
        r"\toprule",
        "Objective & $n$ & " + " & ".join(label for _, label, _, _ in METRICS) + r" \\",
        r"\midrule",
        *body,
    ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_comparison_table(rows: Sequence[dict], metric: str, title: str, *,
                           digits: int = 3, scale: float = 1.0) -> str:
    res = compare_family(rows, metric, FAMILY_MAIN, scale=scale)
    own: dict[str, Paired] = {}
    ema = paired(by_seed(rows, EMA_PAIR[0], metric, scale), by_seed(rows, EMA_PAIR[1], metric, scale))
    if ema is not None:
        ema.p_holm = ema.p
        own[EMA_PAIR[0]] = ema
    # The timestep-only arm: against MSE, corrected within its own two-test
    # family; shown only once every seed of the design is in.
    if shown(rows, TIMESTEP_ARM, metric):
        tt = contrasts(rows, metric, FAMILY_TIMESTEP, scale=scale).get(FAMILY_TIMESTEP[0])
        if tt is not None:
            own[TIMESTEP_ARM] = tt
    lower = lower_is_better(metric)
    missing = [a for a in FAMILY_MAIN if res.get(a) is None]

    def row(key: str) -> str | None:
        r = own.get(key) or (res.get(key) if key in FAMILY_MAIN else None)
        if r is None:
            return None
        sig = r.p_holm is not None and r.p_holm < 0.05
        label = NAMES[key] + (r" vs.\ " + NAMES[EMA_PAIR[1]] if key == EMA_PAIR[0] else "")
        holm_cell = fp_cell(r.p_holm, sig)
        if key in own:
            holm_cell += r"$^{\dagger}$"
        return (f"{label} & {r.n} & {fdiff(r, digits)} & {fci_of(r, digits)} & "
                f"{fp_cell(r.p)} & {holm_cell} & {fp_wilcoxon(r.wilcoxon)} " + r"\\")

    body = block_lines([k for k, _ in ROWS], 7, row)
    note = ""
    if missing:
        note = (" Declared but not yet in the ledger, and counted with $p = 1$: "
                + "; ".join(NAMES[a] for a in missing) + ".")
    own_note = r" $^{\dagger}$A family of its own: " + "; ".join(
        ([r"AWWL + EMA is paired with MSE + EMA (one test)"] if EMA_PAIR[0] in own else [])
        + ([r"the timestep-only arm is corrected with AWWL against it, the two-test "
            r"decomposition family (Section~\ref{sec:confound})"]
           if TIMESTEP_ARM in own else [])) + "." if own else ""
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Paired-by-seed comparison against MSE on " + title + r", CIFAR-10, "
        r"epoch 199, DDIM-100. $\Delta$ is the mean difference (objective $-$ MSE) over "
        r"the $n$ shared training seeds, with its 95\% $t$-interval; $p$ is the two-sided "
        r"paired $t$-test, raw and Holm--Bonferroni corrected over the fixed family of "
        + str(len(FAMILY_MAIN)) + r" arms (every row without a dagger)." + note + own_note
        + r" $p_{\mathrm{W}}$ is the two-sided Wilcoxon signed-rank test, whose smallest "
        r"attainable value is $0.0625$ at five pairs and $0.25$ at three."
        + bold_note(body) + " "
        + (r"Positive $\Delta$ is worse." if lower else r"Negative $\Delta$ is worse.") + "}",
        r"\label{tab:compare-" + metric.replace("_", "") + "}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Objective & $n$ & $\Delta$ & 95\% CI & $p$ & $p_{\mathrm{Holm}}$ & "
        r"$p_{\mathrm{W}}$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


#: The metrics of Table compare-other: ``(metric, header, digits, scale)``.
OTHER_METRICS: list[tuple[str, str, int, float]] = [
    ("is_mean", r"IS $\uparrow$", 3, 1.0),
    ("kid_tf", r"KID$\times 10^{3}$ $\downarrow$", 2, 1e3),
    ("precision", r"Precision $\uparrow$", 4, 1.0),
    ("recall", r"Recall $\uparrow$", 4, 1.0),
]


def _metric_results(rows: Sequence[dict], metric: str, scale: float) -> dict[str, Paired]:
    """Every arm against MSE on one metric, with the corrected p of its family.

    The main arms are corrected over FAMILY_MAIN; AWWL + EMA is paired with
    MSE + EMA (a one-test family); the timestep-only arm, once every seed is
    in, carries the corrected p of FAMILY_TIMESTEP. The last two are the
    arms :func:`build_comparison_table` marks with a dagger.
    """
    res = {k: v for k, v in compare_family(rows, metric, FAMILY_MAIN, scale=scale).items() if v}
    ema = paired(by_seed(rows, EMA_PAIR[0], metric, scale), by_seed(rows, EMA_PAIR[1], metric, scale))
    if ema is not None:
        ema.p_holm = ema.p
        res[EMA_PAIR[0]] = ema
    if shown(rows, TIMESTEP_ARM, metric):
        tt = contrasts(rows, metric, FAMILY_TIMESTEP, scale=scale).get(FAMILY_TIMESTEP[0])
        if tt is not None:
            res[TIMESTEP_ARM] = tt
    return res


#: Layout of Table compare-other: one landscape table, so that every arm's
#: name fits on one line and each metric block stands apart. The name column,
#: the space either side of a column within a metric block, the gap that
#: replaces it between blocks, and the type size.
COMPARE_OTHER_NAME_WIDTH = "6.15cm"
COMPARE_OTHER_TABCOLSEP = "2.8pt"
COMPARE_OTHER_BLOCK_GAP = "10pt"
COMPARE_OTHER_SIZE = r"\footnotesize"


def build_compare_other_table(rows: Sequence[dict]) -> str:
    """Table compare-other: every arm against MSE on IS, KID, precision and recall.

    One table in place of the four per-metric ones (which are still written,
    with their intervals and Wilcoxon tests): the mean paired difference, raw
    and corrected p, per metric, under the :data:`BLOCKS` headers. Thirteen
    columns do not fit the text width at a legible size, so the table is set
    on its own page, turned a quarter (``graphicx``'s ``\rotatebox``; no
    package beyond the manuscript's own).
    """
    per = {m: _metric_results(rows, m, scale) for m, _h, _d, scale in OTHER_METRICS}
    own = {EMA_PAIR[0], TIMESTEP_ARM}
    ncols = 1 + 3 * len(OTHER_METRICS)
    n_of = {k: max((r.n for r in (per[m].get(k) for m in per) if r), default=0) for k in NAMES}
    expected = max(n_of.values(), default=0)

    def row(key: str) -> str | None:
        if not any(per[m].get(key) for m in per):
            return None
        # The seed count rides on the name, as in Table full: a column of
        # "5"s would cost width. The own-family dagger rides on each
        # corrected p, as in Table compare-fid.
        label = NAMES[key] + seed_mark(n_of[key], expected) + (
            r" vs.\ " + NAMES[EMA_PAIR[1]] if key == EMA_PAIR[0] else "")
        mark = r"$^{\dagger}$" if key in own else ""
        cells = [label]
        for m, _h, digits, _s in OTHER_METRICS:
            r = per[m].get(key)
            if r is None:
                cells += ["---"] * 3
                continue
            cells += [fdiff(r, digits), fp_cell(r.p),
                      fp_cell(r.p_holm, r.p_holm is not None and r.p_holm < 0.05) + mark]
        return " & ".join(cells) + r" \\"

    body = block_lines([k for k, _ in ROWS], ncols, row)
    missing = sorted({a for m in per for a in FAMILY_MAIN if a not in per[m]}, key=ORDER.get)
    note = ""
    if missing:
        note = (" Declared but not yet in the ledger, and counted with $p = 1$: "
                + "; ".join(NAMES[a] for a in missing) + ".")
    daggers = [text for key, text in (
        (EMA_PAIR[0], r"AWWL + EMA is paired with MSE + EMA (one test)"),
        (TIMESTEP_ARM, r"the timestep-only arm is corrected with AWWL against it on the same "
         r"metric, the two-test decomposition family of Table~\ref{tab:confound}"),
    ) if any(key in per[m] for m in per)]
    own_note = (r" $^{\dagger}$A family of its own: " + "; ".join(daggers) + ".") if daggers else ""
    header = " & ".join(r"\multicolumn{3}{c}{" + h + "}" for _m, h, _d, _s in OTHER_METRICS)
    rules = "".join(r"\cmidrule(lr){" + f"{2 + 3 * i}-{4 + 3 * i}" + "}"
                    for i in range(len(OTHER_METRICS)))
    # Metric blocks are set apart by a wider gap than the columns within a
    # block, so a row reads as four groups of three numbers, not as one run.
    columns = name_column(COMPARE_OTHER_NAME_WIDTH) + (r"@{\hspace{" + COMPARE_OTHER_BLOCK_GAP
                                                        + "}}").join(["rrr"] * len(OTHER_METRICS))
    lines = [
        PREAMBLE,
        r"\begin{table}[p]",
        r"\centering",
        r"\noindent\rotatebox{90}{\begin{minipage}{\textheight}%",
        r"\centering",
        COMPARE_OTHER_SIZE,
        r"\setlength{\tabcolsep}{" + COMPARE_OTHER_TABCOLSEP + "}",
        r"\caption{Paired-by-seed comparison against MSE on Inception Score, KID, precision and "
        r"recall, CIFAR-10, epoch 199, DDIM-100 (\num{10000} samples per run). $\Delta$ is the "
        r"mean difference (objective $-$ MSE) over the shared training seeds, "
        + cw(expected) + r" unless a superscript gives the number; $p$ is the "
        r"two-sided paired $t$-test, raw and Holm--Bonferroni corrected, within each metric, "
        r"over the fixed family of " + str(len(FAMILY_MAIN)) + r" arms (every row whose "
        r"$p_{\mathrm{Holm}}$ carries no dagger)." + note + own_note + r" KID is from "
        r"\texttt{torch-fidelity} (Appendix~\ref{app:supp-kid}). An upward arrow means higher "
        r"is better, so a negative $\Delta$ is worse on IS, precision and recall and a positive "
        r"one on KID. FID, with intervals, is in Table~\ref{tab:compare-fid}." + bold_note(body) + "}",
        r"\label{tab:compare-other}",
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        r" & " + header + r" \\",
        rules,
        r"Objective & " + " & ".join(
            [r"$\Delta$ & $p$ & $p_{\mathrm{Holm}}$"] * len(OTHER_METRICS)) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{minipage}}%",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_convergence_table(rows: Sequence[dict], epochs: Sequence[int]) -> str:
    """AWWL against MSE at every evaluated checkpoint of the 200-epoch runs."""
    body = []
    for e in epochs:
        sub = [r for r in rows if r.get("epoch") == e]
        m, a = by_seed(sub, "mse", "fid"), by_seed(sub, "awwl", "fid")
        res = paired(a, m)
        if res is None:
            continue
        ahead = sum(1 for d in res.deltas if d < 0)
        body.append(
            f"{e} & {fmean_sd(list(m.values()))} & {fmean_sd(list(a.values()))} & "
            f"{fcell_ci(res, 2)} & {fp_cell(res.p)} & {ahead} of {res.n} " + r"\\"
        )
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{FID through training, CIFAR-10, DDIM-100, five seeds. Mean $\pm$ "
        r"sample standard deviation over seeds; $\Delta$ is AWWL $-$ MSE paired by seed "
        r"with its 95\% $t$-interval and two-sided paired $t$-test $p$, uncorrected. The "
        r"last column counts the seeds at which AWWL has the lower FID.}",
        r"\label{tab:convergence}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Epoch & MSE & AWWL & $\Delta$ [95\% CI] & $p$ & AWWL ahead \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _replication_arms() -> list[tuple[str, str]]:
    return [(a, NAMES[a]) for a in ("mse", "static_wavelet", "awwl")]


def _replication_block(ddim_rows: Sequence[dict], ddpm_rows: Sequence[dict], metric: str,
                       digits: int) -> list[str]:
    """Rows of one metric of the replication: the published single runs and the
    five-seed means under each sampler, then each wavelet arm's difference from
    MSE with its interval and its raw and Holm-corrected p (FAMILY_DDPM under
    DDPM-1000, FAMILY_MAIN under DDIM-100)."""
    arms = _replication_arms()
    body = []
    for arm, label in arms:
        pub = CONFERENCE_CIFAR10.get(arm, {}).get(metric)
        cells = ["---" if pub is None else "$" + fnum(pub, 2) + "$"]
        for rows in (ddpm_rows, ddim_rows):
            vals = by_seed(rows, arm, metric)
            cells.append(fmean_sd(list(vals.values()), digits) if len(vals) >= 2 else "---")
        body.append(label + " & " + " & ".join(cells) + r" \\")
    ddpm = compare_family(ddpm_rows, metric, FAMILY_DDPM)
    ddim = compare_family(ddim_rows, metric, FAMILY_MAIN)
    body.append(r"\addlinespace[2pt]")
    for arm, label in arms[1:]:
        pub_d = CONFERENCE_CIFAR10[arm][metric] - CONFERENCE_CIFAR10["mse"][metric]
        rd, rm = ddpm.get(arm), ddim.get(arm)
        body.append(label + r" $-$ MSE & " + " & ".join(
            ["$" + fsigned(pub_d, 2) + "$"] + [fcell_ci(r, digits) if r else "---" for r in (rd, rm)])
            + r" \\")
        body.append(r"\quad $p$ (raw; Holm) & & " + " & ".join(_p_pair(r) for r in (rd, rm)) + r" \\")
    return body


def build_replication_table(ddim_rows: Sequence[dict], ddpm_rows: Sequence[dict]) -> str:
    """The conference paper's single-run FID claims against the five-seed results.

    The Inception Score results, the one-sided tests against the published
    differences and the prediction intervals are in Table replication-details
    (:func:`build_replication_details_table`; make_numbers.py ``pPub*``, ``pi*``).
    """
    body = _replication_block(ddim_rows, ddpm_rows, "fid", 2)
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{The conference paper's CIFAR-10 FID results \citep{thao2026awwl} against "
        r"the replication. Published values are single runs (ancestral DDPM-1000 "
        r"sampling, no EMA, as documented with the released code). Replicated values "
        r"are mean $\pm$ sample standard deviation over five training seeds, the same "
        r"trained weights sampled with DDPM-1000 and with DDIM-100 (\num{10000} samples "
        r"each). Differences are paired by seed, with 95\% $t$-intervals and two-sided "
        r"paired $t$-tests; Holm correction is over the " + cw(len(FAMILY_DDPM))
        + r" arms shown for DDPM-1000 and over the " + str(len(FAMILY_MAIN))
        + r"-arm family of Table~\ref{tab:compare-fid} for DDIM-100." + bold_note(body)
        + r" Inception Score, the tests against the published differences and the "
        r"prediction intervals are in Table~\ref{tab:replication-details}.}",
        r"\label{tab:replication}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r" & Published & \multicolumn{2}{c}{Replicated, five seeds} \\",
        r"\cmidrule(lr){3-4}",
        r" & (one run) & DDPM-1000 & DDIM-100 \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_replication_details_table(ddim_rows: Sequence[dict], ddpm_rows: Sequence[dict]) -> str:
    """Table replication-details: every row of the conference paper's CIFAR-10
    table against the five-seed results, on FID and Inception Score; each arm's
    difference from MSE; the one-sided tests of each published difference; and
    the 95% prediction interval for one further paired run, under both samplers.

    The same computations as make_numbers.py ``pPub*`` and ``pi*``: the
    one-sided p is that of H1 "the true difference is less favourable than
    published" (greater for FID, smaller for IS), uncorrected; the prediction
    interval is :func:`prediction_interval` of the paired differences. Under
    DDPM-1000 only MSE and the two wavelet arms were sampled (FAMILY_DDPM);
    under DDIM-100 each arm carries the corrected p of FAMILY_MAIN, as in
    Table compare-fid.
    """
    metrics = (("fid", 2, True), ("is_mean", 3, False))
    samplers = (ddpm_rows, ddim_rows)
    fams = {m: (compare_family(ddpm_rows, m, FAMILY_DDPM), compare_family(ddim_rows, m, FAMILY_MAIN))
            for m, _d, _l in metrics}
    trained = [a for a in CONFERENCE_ROWS if a in NAMES]
    compared = [a for a in trained if a != BASELINE]

    def name(arm: str) -> str:
        return NAMES.get(arm) or CONFERENCE_ONLY_NAMES.get(arm, arm)

    def pub_d(arm: str, metric: str) -> float:
        return CONFERENCE_CIFAR10[arm][metric] - CONFERENCE_CIFAR10[BASELINE][metric]

    def results(arm: str, metric: str) -> list[Paired | None]:
        return [fams[metric][0].get(arm), fams[metric][1].get(arm)]

    means = []
    for arm in CONFERENCE_ROWS:
        cells = [name(arm)]
        for metric, digits, _lower in metrics:
            cells.append("$" + fnum(CONFERENCE_CIFAR10[arm][metric], 2) + "$")
            for rows in samplers:
                vals = list(by_seed(rows, arm, metric).values())
                # A mean in the hundreds (a run that did not train) needs no decimals.
                d = 0 if vals and abs(statistics.mean(vals)) >= 100 else digits
                cells.append(fmean_sd(vals, d) if len(vals) >= 2 else "not run")
        means.append(" & ".join(cells) + r" \\")

    diffs, tests, intervals = [], [], []
    # Arm and metric where the replication is more favourable than published
    # and one run could not show the published difference: the one-sided test
    # returns a p near 1 there, which the caption must not leave unexplained.
    favourable: dict[str, None] = {}
    for arm in compared:
        # The panel headers say that every row is a difference from MSE.
        label = name(arm)
        d_cells, c_cells, p_cells = [label], [r"\quad 95\% CI"], [r"\quad $p$ (raw; Holm)"]
        t_cells, i_cells = [label], [label]
        for metric, digits, lower in metrics:
            pub = "$" + fsigned(pub_d(arm, metric), 2) + "$"
            d_cells.append(pub)
            c_cells.append("")
            p_cells.append("")
            t_cells.append(pub)
            i_cells.append(pub)
            for r in results(arm, metric):
                # Estimate, interval and tests on three lines: side by side
                # they would not fit six columns into the text width.
                d_cells.append(fdiff(r, digits) if r else "---")
                c_cells.append(fci_of(r, digits) if r else "---")
                p_cells.append(_p_pair(r))
                t_cells.append("---" if r is None else fp_cell(one_sided_p(r, pub_d(arm, metric),
                                                                         greater=lower)))
                if r is None:
                    i_cells.append("---")
                else:
                    lo, hi = prediction_interval(r)
                    i_cells.append(fci(lo, hi, adaptive(digits, r.mean)))
                    if (one_sided_p(r, pub_d(arm, metric), greater=lower) > 0.5
                            and not lo <= pub_d(arm, metric) <= hi):
                        favourable[label + " on " + ("FID" if metric == "fid"
                                                     else "Inception Score")] = None
        diffs += [" & ".join(cells) + r" \\" for cells in (d_cells, c_cells, p_cells)]
        tests.append(" & ".join(t_cells) + r" \\")
        intervals.append(" & ".join(i_cells) + r" \\")

    def head(text: str) -> str:
        return r"\multicolumn{7}{l}{\emph{" + text + r"}} \\"

    body = [head(r"Published single run; replicated mean $\pm$ sample standard deviation over "
                 r"five seeds"), *means,
            r"\midrule", head(r"Difference from MSE, paired by seed, with its 95\% $t$-interval"),
            *diffs,
            r"\midrule", head(r"One-sided test of the published difference from MSE, $p$"), *tests,
            r"\midrule", head(r"95\% prediction interval for the difference from MSE of one further "
                               r"paired run"), *intervals]
    missing = [name(a) for a in CONFERENCE_ROWS if a not in NAMES]
    snr = NAMES.get("snr_weighted")
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Replication details: every row of the CIFAR-10 table of the conference paper "
        r"\citep{thao2026awwl} against the replication, on FID and Inception Score, epoch 199, "
        r"\num{10000} samples per run. Published values are single runs (DDPM-1000, no EMA). "
        r"Replicated values come from five training seeds per arm, the same weights sampled with "
        r"DDPM-1000, as published, and with DDIM-100; only MSE and the two wavelet arms were "
        r"sampled with DDPM-1000. " + (snr + r" is the SNR-weighted row: the released code's "
        r"unclamped weighting, which did not train stably in any seed. " if snr else "")
        + ("The " + " and ".join(missing) + r" row was not retrained "
           r"(Appendix~\ref{app:setup-arms}). " if missing else "")
        + r"Differences are paired by seed, with 95\% $t$-intervals and two-sided paired "
        r"$t$-tests, Holm-corrected over the " + cw(len(FAMILY_DDPM)) + r" wavelet arms for "
        r"DDPM-1000 (as in Table~\ref{tab:replication}) and over the " + str(len(FAMILY_MAIN))
        + r"-arm family of Table~\ref{tab:compare-fid} for DDIM-100." + bold_note(body)
        + r" The one-sided test asks whether the published difference is the arm's mean effect: "
        r"its $p$ is that of the alternative that the true difference is less favourable than "
        r"published (larger for FID, smaller for Inception Score), uncorrected. The prediction "
        r"interval asks whether it could be one run: it is the 95\% interval for the paired "
        r"difference that one further run would show. A one-sided $p$ near 1 says only that the "
        r"replicated difference is not less favourable than published; a replication more "
        r"favourable than published"
        + (" (" + ", ".join(favourable) + ")" if favourable else "")
        + r" shows in the prediction interval instead. In the lower three panels the Published "
        r"columns repeat the published difference.}",
        r"\label{tab:replication-details}",
        r"\begin{tabular}{" + name_column("3.2cm") + r"rrr@{\hspace{9pt}}rrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{FID $\downarrow$} & \multicolumn{3}{c}{Inception Score $\uparrow$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & Published & DDPM-1000 & DDIM-100 & Published & DDPM-1000 & DDIM-100 \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _p_pair(r: Paired | None) -> str:
    if r is None:
        return "---"
    return fp_cell(r.p) + "; " + fp_cell(r.p_holm, (r.p_holm or 1) < 0.05)


def _constraint(arm: str, root: Path) -> str:
    """What a normalised arm holds fixed over sigma, read from its run configuration."""
    cfg = run_config(root, arm).get("loss", {})
    if not cfg.get("normalize_weights"):
        return r"none"
    scale = cfg.get("normalize_scale")
    scale = 1.0 if scale is None else float(scale)
    order = float(cfg.get("normalize_order") or 1.0)
    power = "^2" if order == 2.0 else ""
    return r"$\sum_b \omega_b" + power + " = " + f"{scale:g}" + "$"


def _confound_table(rows: Sequence[dict], root: Path, *, arms: Sequence[str], reference: str,
                    label: str, caption: str, ref_header: str | None = None) -> str:
    """One panel of normalisation controls against MSE and against their reference.

    Against MSE the family is FAMILY_MAIN; against the reference it is
    FAMILY_CONFOUND. Used for Table confound-summed; Table confound, which
    also carries the timestep-only arm and its own family, has its own
    builder (:func:`build_confound_table`).
    """
    vs_mse = compare_family(rows, "fid", FAMILY_MAIN)
    vs_ref = contrasts(rows, "fid", FAMILY_CONFOUND)
    ref_of = dict(FAMILY_CONFOUND)
    have = groups_of(rows)
    body = []
    levels = []
    for arm in arms:
        n = len(by_seed(rows, arm, "fid"))
        if arm not in have or n < 3:
            continue
        m = vs_mse.get(arm)
        c = vs_ref.get((arm, ref_of[arm])) if arm in ref_of else None
        level = _constraint(arm, root)
        if level != "none" and level not in NAMES[arm]:
            levels.append(NAMES[arm] + ": " + level)
        cells = [NAMES[arm], str(n)]
        cells += ["---"] * 3 if m is None else [
            fcell_ci(m, 2), fp_cell(m.p), fp_cell(m.p_holm, (m.p_holm or 1) < 0.05)]
        if arm == reference:
            cells += [r"\emph{reference}", "", ""]
        else:
            cells += ["---"] * 3 if c is None else [
                fcell_ci(c, 2), fp_cell(c.p), fp_cell(c.p_holm, (c.p_holm or 1) < 0.05)]
        body.append(" & ".join(cells) + r" \\")
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{" + caption + (" Levels read from the run configurations: " + "; ".join(levels)
                                  + "." if levels else "") + bold_note(body) + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + name_column("3.3cm") + "rrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Against MSE} & \multicolumn{3}{c}{"
        + (ref_header or "Against " + NAMES[reference]) + r"} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Arm & $n$ & $\Delta$ [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & "
        r"$\Delta$ [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


_CONFOUND_STATS = (
    r"$\Delta$ is the paired-by-seed FID difference with its 95\% $t$-interval over the $n$ "
    r"seeds; $p$ is the two-sided paired $t$-test, raw and Holm-corrected. Against MSE the "
    r"Holm family is the " + str(len(FAMILY_MAIN)) + r"-arm family of "
    r"Table~\ref{tab:compare-fid}; against the reference it is the "
    + cw(len(FAMILY_CONFOUND)) + r" confound contrasts of Tables~\ref{tab:confound} "
    r"and~\ref{tab:confound-summed}."
)


#: Rows of Table confound, in order: the two arms without a frequency shape,
#: then AWWL and the three controls that keep its shape and fix a
#: residual-independent measure of its timestep weight.
CONFOUND_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Without a frequency shape", ("mse", TIMESTEP_ARM)),
    ("With AWWL's frequency shape", ("awwl", "awwl_normalized", "awwl_norm_matched",
                                     "awwl_grad_matched_mean")),
)

#: Whether Table confound carries the 2x2 summary beneath its rows. Set to
#: False to drop the lower panel (and its caption sentences) if the main text
#: needs the space; nothing else depends on it.
CONFOUND_WITH_GRID = True

#: The 2x2 summary beneath Table confound: ``(schedule, shape) -> arm``. The
#: schedule is the timestep total S(sigma_t) = sum_b omega_b, either AWWL's or
#: held constant; the constant-S cell with the shape is the control at
#: sum_b omega_b = 0.5 (make_numbers.py ``levelSumFixed``, the level the text
#: of Sec. confound-results motivates), not the repeat at 1. The
#: sum-of-squares control holds sum_b omega_b^2, not S, and so has no cell.
#: The panel is descriptive: no interaction is tested.
CONFOUND_GRID: dict[tuple[str, str], str] = {
    ("constant", "absent"): "mse",
    ("constant", "present"): "awwl_norm_matched",
    ("varying", "absent"): TIMESTEP_ARM,
    ("varying", "present"): "awwl",
}


def confound_results(rows: Sequence[dict]) -> dict[str, tuple[Paired | None, Paired | None, str]]:
    """``{arm: (against MSE, against AWWL, family mark)}`` for Table confound.

    Against MSE the family is FAMILY_MAIN and against AWWL FAMILY_CONFOUND,
    except for the timestep-only arm, whose two contrasts (against MSE, and
    AWWL against it) are FAMILY_TIMESTEP, corrected on their own; its
    against-AWWL entry is printed as arm - AWWL like every other row (the
    negative of AWWL - timestep only, with the same two-sided p). The mark is
    a dagger for that arm. An arm without rows, and the timestep-only arm
    before all :data:`TIMESTEP_SEEDS` seeds are in, maps to ``(None, None, "")``.
    """
    vs_mse = compare_family(rows, "fid", FAMILY_MAIN)
    vs_ref = contrasts(rows, "fid", FAMILY_CONFOUND)
    ref_of = dict(FAMILY_CONFOUND)
    out: dict[str, tuple[Paired | None, Paired | None, str]] = {}
    for _title, arms in CONFOUND_BLOCKS:
        for arm in arms:
            if arm == TIMESTEP_ARM:
                if not shown(rows, arm):
                    out[arm] = (None, None, "")
                    continue
                tt = contrasts(rows, "fid", FAMILY_TIMESTEP)
                a, b = tt.get(FAMILY_TIMESTEP[0]), tt.get(FAMILY_TIMESTEP[1])
                flipped = None if b is None else Paired(
                    b.n, -b.mean, b.sd, -b.hi, -b.lo, -b.t, b.p, b.wilcoxon, b.seeds,
                    tuple(-d for d in b.deltas), b.p_holm)
                out[arm] = (a, flipped, r"$^{\dagger}$")
            elif arm == BASELINE:
                out[arm] = (None, None, "")
            else:
                c = vs_ref.get((arm, ref_of[arm])) if arm in ref_of else None
                out[arm] = (vs_mse.get(arm), c, "")
    return out


def _confound_grid(res: Mapping[str, tuple], rows: Sequence[dict]) -> list[str]:
    """The 2x2 summary tabular of Table confound (schedule x frequency shape)."""
    def cell(arm: str) -> str:
        if arm == BASELINE:
            return NAMES[arm] + r"\newline $0$ (reference)"
        m = res.get(arm, (None,))[0]
        if m is None:
            return NAMES[arm] + r"\newline " + PLACEHOLDER
        return NAMES[arm] + res[arm][2] + r"\newline " + fcell_ci(m, 2)

    if not CONFOUND_WITH_GRID or any(res.get(a, (None,))[0] is None
                                     for a in CONFOUND_GRID.values() if a != BASELINE):
        return []
    col = r">{\raggedright\arraybackslash}p{4.1cm}"
    return [
        r"\par\medskip",
        r"\begin{tabular}{@{}l" + col + col + "@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Frequency shape} \\",
        r"\cmidrule(lr){2-3}",
        r"Timestep total $S(\sigma_t)$ & absent & AWWL's \\",
        r"\midrule",
        r"held constant & " + cell(CONFOUND_GRID[("constant", "absent")]) + " & "
        + cell(CONFOUND_GRID[("constant", "present")]) + r" \\",
        r"\addlinespace[2pt]",
        r"AWWL's & " + cell(CONFOUND_GRID[("varying", "absent")]) + " & "
        + cell(CONFOUND_GRID[("varying", "present")]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]


def build_confound_table(rows: Sequence[dict], root: Path = ROOT / "runs" / "phase0") -> str:
    """Table confound: MSE, AWWL's timestep component alone, AWWL and the
    normalisation controls on AWWL as released (detail bands averaged), each
    against MSE and against AWWL, with the schedule x shape summary beneath."""
    res = confound_results(rows)
    have = groups_of(rows)
    body: list[str] = []
    levels: list[str] = []
    dagger = False
    for title, arms in CONFOUND_BLOCKS:
        lines = []
        for arm in arms:
            n = len(by_seed(rows, arm, "fid"))
            if arm not in have or n < 2 or (arm == TIMESTEP_ARM and res[arm][0] is None):
                continue
            m, c, mark = res[arm]
            dagger |= bool(mark)
            cells = [NAMES[arm], str(n)]
            if arm == BASELINE:
                cells += [r"\emph{reference}", "", "", "", "", ""]
            else:
                cells += ["---"] * 3 if m is None else [
                    fcell_ci(m, 2), fp_cell(m.p), fp_cell(m.p_holm, (m.p_holm or 1) < 0.05) + mark]
                if arm == "awwl":
                    cells += [r"\emph{reference}", "", ""]
                else:
                    cells += ["---"] * 3 if c is None else [
                        fcell_ci(c, 2), fp_cell(c.p), fp_cell(c.p_holm, (c.p_holm or 1) < 0.05) + mark]
                level = _constraint(arm, root) if arm.startswith("awwl_") and arm != TIMESTEP_ARM else "none"
                if level != "none" and level not in NAMES[arm]:
                    levels.append(level)
            lines.append(" & ".join(cells) + r" \\")
        if lines:
            if body:
                body.append(r"\addlinespace[2pt]")
            body += [r"\multicolumn{8}{l}{\emph{" + title + r"}} \\", *lines]
    grid = _confound_grid(res, rows)
    timestep = dagger   # the arm's row is in the table
    caption = (
        r"The timestep schedule and the normalisation controls of AWWL (released form, detail "
        r"bands averaged), CIFAR-10, epoch 199, DDIM-100. The band weights are $\omega_b = "
        r"(w_{\mathrm{LL}}, w_{\det}/3, w_{\det}/3, w_{\det}/3)$, and $S(\sigma_t) = \sum_b "
        r"\omega_b = w_{\mathrm{LL}} + w_{\det}$ is AWWL's total weight. "
        + ((r"The timestep-only arm trains the pixel MSE multiplied by $S(\sigma_t)$, AWWL's "
            r"timestep component alone (Proposition~\ref{prop:confound}); ") if timestep else "")
        + r"each normalisation "
        r"control multiplies the AWWL loss by a function of $\sigma_t$ alone and holds the "
        r"stated sum fixed over $\sigma_t$, the sum of squares at AWWL's mean over the "
        r"training timesteps"
        + (", " + "; ".join(levels) if levels else "") + r". $\Delta$ is the "
        r"paired-by-seed FID difference, the arm minus MSE or minus AWWL, with its 95\% $t$-interval "
        r"over the $n$ seeds; $p$ is the two-sided paired $t$-test, raw and Holm-corrected. "
        r"Holm families: against MSE, the " + str(len(FAMILY_MAIN)) + r"-arm family of "
        r"Table~\ref{tab:compare-fid}; against AWWL, the " + cw(len(FAMILY_CONFOUND))
        + r" confound contrasts of this table and Table~\ref{tab:confound-summed}"
        + ((r"; $^{\dagger}$for the timestep-only arm, its own " + cw(len(FAMILY_TIMESTEP))
            + r"-test family (this arm against MSE, and AWWL against this arm), corrected "
            r"separately from both")
           if dagger else "") + "." + bold_note(body)
    )
    if grid:
        caption += (
            r" Lower panel: the difference from MSE of four of the arms, arranged by whether "
            r"$S(\sigma_t)$ is AWWL's or held constant and whether AWWL's frequency shape is "
            r"present. Only $S$ is held constant: the effective timestep weight $\Lambda(t)$ "
            r"is not (Propositions~\ref{prop:confound}(b) and~\ref{prop:residual}). The "
            r"sum-of-squares control holds $\sum_b \omega_b^2$ rather than $S$ and has no cell.")
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{" + caption + "}",
        r"\label{tab:confound}",
        r"\begin{tabular}{" + name_column("3.3cm") + "rrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Against MSE} & \multicolumn{3}{c}{Against AWWL} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Arm & $n$ & $\Delta$ [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & "
        r"$\Delta$ [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        *grid,
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_confound_summed_table(rows: Sequence[dict], root: Path = ROOT / "runs" / "phase0") -> str:
    """The normalisation control on the summed-detail variant (Supplementary results)."""
    return _confound_table(
        rows, root, reference="awwl_eq7", label="tab:confound-summed",
        arms=["awwl_eq7", "awwl_grad_matched"], ref_header="Against its parent (first row)",
        caption=r"Normalisation control on the summed-detail variant of AWWL (each detail "
        r"band carries $w_{\det}$), CIFAR-10, epoch 199, DDIM-100. The band weights are "
        r"$\omega_b = (w_{\mathrm{LL}}, w_{\det}, w_{\det}, w_{\det})$; the control "
        r"holds $\sum_b \omega_b^2$ at the variant's own mean over the training timesteps. "
        + _CONFOUND_STATS,
    )


def build_extensions_table(rows: Sequence[dict]) -> str:
    vs_mse = compare_family(rows, "fid", FAMILY_MAIN)
    parent = by_seed(rows, EXTENSION_PARENT, "fid")
    vs_parent = {a: paired(by_seed(rows, a, "fid"), parent) for a in FAMILY_EXTENSIONS}
    apply_family(vs_parent, FAMILY_EXTENSIONS)

    def row(arm: str, label: str, ref: Paired | None, right: str | None = None) -> str:
        n = len(by_seed(rows, arm, "fid"))
        m = vs_mse.get(arm)
        cells = [label, str(n)]
        cells += ["---"] * 3 if m is None else [
            fcell_ci(m, 2), fp_cell(m.p), fp_cell(m.p_holm, (m.p_holm or 1) < 0.05)]
        if right is not None:
            cells += [right, "", ""]
        elif ref is None:
            cells += ["---", "", ""]
        else:
            cells += [fcell_ci(ref, 2), fp_cell(ref.p), fp_cell(ref.p_holm, (ref.p_holm or 1) < 0.05)]
        return " & ".join(cells) + r" \\"

    body = [r"\multicolumn{8}{l}{\emph{Reference}} \\"]
    if by_seed(rows, "awwl", "fid"):
        body.append(row("awwl", NAMES["awwl"], None, right=""))
    body.append(row(EXTENSION_PARENT, NAMES[EXTENSION_PARENT], None, right=r"\emph{parent}"))
    body.append(r"\midrule")
    for arm in sorted(FAMILY_EXTENSIONS, key=ORDER.get):
        if by_seed(rows, arm, "fid"):
            body.append(row(arm, NAMES[arm], vs_parent.get(arm)))
    axes = "; ".join(NAMES[a] + ": " + AXIS[a] for a in sorted(FAMILY_EXTENSIONS, key=ORDER.get))
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{The method extensions, CIFAR-10, epoch 199, DDIM-100; $n$ is the number "
        r"of seeds. No extension normalises its band weights. The per-sub-band, spatially "
        r"adaptive and lifting arms keep AWWL's rational weighting with every "
        r"detail band at the full $w_{\mathrm{det}}$, the summed form; the multi-level arm "
        r"divides $w_{\mathrm{det}}$ among its nine detail bands; the learned-weighting, GradNorm "
        r"and all-axes arms set the band weights themselves. Every extension is compared with the "
        r"unnormalised AWWL with detail bands summed (second row, the parent of the "
        r"summed-form arms), not with the normalised variants of "
        r"Table~\ref{tab:confound}. $\Delta$ is the "
        r"paired-by-seed FID difference with its 95\% $t$-interval, over the seeds both "
        r"arms share; $p$ is the two-sided paired $t$-test, raw and Holm-corrected: "
        r"against MSE over the " + str(len(FAMILY_MAIN)) + r"-arm family of "
        r"Table~\ref{tab:compare-fid}, against the parent over the "
        + cw(len(FAMILY_EXTENSIONS)) + r" extensions." + bold_note(body) + " Axes turned on: " + axes + ".}",
        r"\label{tab:extensions}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Against MSE} & \multicolumn{3}{c}{Against the parent} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Arm & $n$ & $\Delta$ [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & "
        r"$\Delta$ [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# =============================================================================
# Builders: other tasks, samplers, budget
# =============================================================================


def build_dreambooth_table(rows: Sequence[dict]) -> str:
    """Both DreamBooth metrics in one table, ordered by CLIP-score rank."""
    metrics = ("clip_score", "similarity")
    groups = sorted(groups_of(rows))
    means = {m: {g: statistics.mean(by_seed(rows, g, m).values()) for g in groups if by_seed(rows, g, m)}
             for m in metrics}
    ranks = {m: {g: i + 1 for i, g in enumerate(sorted(means[m], key=lambda g: -means[m][g]))}
             for m in metrics}
    tests = {m: compare_family(rows, m, FAMILY_DREAMBOOTH) for m in metrics}
    n_images = sorted({int(r["n_images"]) for r in rows if _valid(r.get("n_images"))})
    seeds = sorted({len(by_seed(rows, g, "clip_score")) for g in groups})
    img = fint(n_images[0]) if len(n_images) == 1 else "varying numbers of"
    n_seeds = cw(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}--{seeds[-1]}"
    body = []
    for g in sorted(means["clip_score"], key=lambda g: ranks["clip_score"][g]):
        cells = []
        for m in metrics:
            if g not in means[m]:
                cells += ["---"] * 3
                continue
            t = tests[m].get(g)
            p_cell = "---" if t is None else fp_cell(t.p_holm, (t.p_holm or 1) < 0.05)
            cells += [f"${means[m][g]:.4f}$", str(ranks[m][g]), p_cell]
        body.append(NAMES.get(g, g) + " & " + " & ".join(cells) + r" \\")
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{DreamBooth fine-tuning of Stable Diffusion~1.5 on one subject, "
        + n_seeds + r" seeds per objective, " + img + r" images per run. Rank is over the "
        + cw(len(means["clip_score"])) + r" objectives, the baseline included, on each "
        r"metric; $p$ is the two-sided paired-by-seed $t$-test against MSE, Holm-corrected "
        r"over the " + cw(len(FAMILY_DREAMBOOTH)) + r" objectives within each metric."
        + bold_note(body) + r" The published claim is that AWWL is the only method in "
        r"the top two on both metrics.}",
        r"\label{tab:dreambooth}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{CLIP score $\uparrow$} & "
        r"\multicolumn{3}{c}{Subject similarity $\uparrow$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Objective & Mean & Rank & $p_{\mathrm{Holm}}$ & Mean & Rank & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


RESTORATION_METRICS: list[tuple[str, str, str]] = [
    ("psnr", r"PSNR $\uparrow$", "{:.3f}"),
    ("ssim", r"SSIM $\uparrow$", "{:.4f}"),
    ("lpips", r"LPIPS $\downarrow$", "{:.5f}"),
    ("niqe", r"NIQE-feat.\ dist.\ $\downarrow$", "{:.4f}"),
]


def restoration_tests(rows: Sequence[dict]) -> dict[str, dict[str, Paired | None]]:
    return {m: compare_family(rows, m, FAMILY_RESTORATION) for m, _, _ in RESTORATION_METRICS}


def build_restoration_table(rows: Sequence[dict], epoch: int,
                            root: Path = ROOT / "runs" / "restoration_cifar") -> str:
    r"""Denoising, four metrics, means only; the tests are summarised in the caption."""
    keys = [k for k, _ in ROWS if k in groups_of(rows)]
    seeds = sorted({len(by_seed(rows, g, "psnr")) for g in groups_of(rows)} - {0})
    tests = restoration_tests(rows)
    best = min(((t.p, t.p_holm, m, a) for m, d in tests.items() for a, t in d.items() if t),
               default=None)
    summary = ""
    if best is not None:
        name = {"psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS",
                "niqe": "NIQE-feature distance"}[best[2]]
        summary = (r" Paired differences with 95\% intervals are plotted in "
                   r"Fig.~\ref{fig:restoration}; the smallest raw $p$ is $" + fp(best[0]) + r"$ ("
                   + NAMES.get(best[3], best[3]) + ", " + name + r"), $" + fp(best[1])
                   + r"$ after Holm correction over the " + cw(len(FAMILY_RESTORATION))
                   + r" arms within that metric.")
    alphas = sorted({run_config(root, k).get("loss", {}).get("alpha") for k in keys
                     if k.startswith("awwl") or k == "static_wavelet"} - {None})
    setting = ""
    if len(alphas) == 1:
        setting = r" The AWWL arms use $\alpha = " + f"{alphas[0]:g}" + r"$."

    def row(key: str) -> str:
        cells = []
        for metric, _, fmt in RESTORATION_METRICS:
            vals = by_seed(rows, key, metric)
            cells.append("---" if not vals else "$" + fmt.format(statistics.mean(vals.values())) + "$")
        return NAMES[key] + " & " + " & ".join(cells) + r" \\"

    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{Gaussian denoising on CIFAR-10, "
        + (cw(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}--{seeds[-1]}") + r" seeds, epoch "
        + fint(epoch) + r"; means over seeds." + setting + r" Each arm is paired by seed with "
        r"MSE (two-sided paired $t$-test)." + summary + r" NIQE-feature distance is the "
        r"distance of the NIQE features from a pristine model fitted on the clean test "
        r"images. The degradation is drawn once and reused across every arm.}",
        r"\label{tab:restoration}",
        r"\begin{tabular}{l" + "r" * len(RESTORATION_METRICS) + "}",
        r"\toprule",
        "Objective & " + " & ".join(label for _, label, _ in RESTORATION_METRICS) + r" \\",
        r"\midrule",
        *[row(k) for k in keys],
    ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_sampler_table(ddim_rows: Sequence[dict], ddpm_rows: Sequence[dict]) -> str:
    fid = compare_family(ddpm_rows, "fid", FAMILY_DDPM)
    kid = compare_family(ddpm_rows, "kid_tf", FAMILY_DDPM, scale=1e3)
    body = []
    for key in [BASELINE, *FAMILY_DDPM]:
        b = by_seed(ddpm_rows, key, "fid")
        if not b:
            continue
        a = by_seed(ddim_rows, key, "fid")
        cells = [NAMES[key], fmean_sd(list(a.values())) if a else "---", fmean_sd(list(b.values()))]
        f, k = fid.get(key), kid.get(key)
        cells += ["---"] * 3 if f is None else [
            fcell_ci(f, 2), fp_cell(f.p), fp_cell(f.p_holm, (f.p_holm or 1) < 0.05)]
        cells += ["---"] * 2 if k is None else [
            "$" + fsigned(k.mean, 2) + "$", fp_cell(k.p_holm, (k.p_holm or 1) < 0.05)]
        body.append(" & ".join(cells) + r" \\")
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{The same trained weights under two samplers, five seeds, "
        r"\num{10000} samples per run, fixed sampling seed. FID is mean $\pm$ sample "
        r"standard deviation. $\Delta$ (objective $-$ MSE, paired by seed, with its 95\% "
        r"$t$-interval) and $p$ refer to DDPM-1000; Holm correction is over the "
        + cw(len(FAMILY_DDPM)) + r" arms shown, separately for FID and KID "
        r"(KID from \texttt{torch-fidelity}, $\times 10^{3}$)." + bold_note(body) + "}",
        r"\label{tab:sampler}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{FID} & \multicolumn{3}{c}{FID, DDPM-1000} & "
        r"\multicolumn{2}{c}{KID, DDPM-1000} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        r"Objective & DDIM-100 & DDPM-1000 & $\Delta$ [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & "
        r"$\Delta$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def budget_points(short_rows, short_ema_rows, long_rows, short_anc, long_anc):
    """The operating points of Table budget: ``(budget, sampler, mse, awwl, result)``.

    One tuple per point of :data:`BUDGET_POINTS` that has at least two seeds
    of each arm, in that order. ``result`` is AWWL - MSE paired by seed; its
    ``p_holm`` is the Budget family's (the two 800-epoch points,
    :data:`FAMILY_BUDGET`) for those points and, for each 200-epoch point, the
    corrected p of the family that contrast belongs to: FAMILY_MAIN (no EMA,
    DDIM-100), the one-test EMA pair, FAMILY_DDPM (DDPM-1000). A contrast is
    thus corrected once, and prints with the same p_Holm in every table.
    """
    specs = [
        (short_rows, "mse", "awwl"),
        (short_ema_rows, "mse_ema", "awwl_ema"),
        (long_rows, "mse", "awwl"),
        (short_anc, "mse", "awwl"),
        (long_anc, "mse", "awwl"),
    ]
    results: dict[int, Paired | None] = {}
    arms: dict[int, tuple[dict, dict]] = {}
    for i, (rows, b, a) in enumerate(specs):
        mb, ma = by_seed(rows or [], b, "fid"), by_seed(rows or [], a, "fid")
        results[i] = paired(ma, mb) if len(mb) >= 2 and len(ma) >= 2 else None
        arms[i] = (mb, ma)
    # The Budget family: the two 800-epoch points, whether or not each is measured.
    apply_family(results, list(FAMILY_BUDGET))
    # The 200-epoch points carry the corrected p of their own families; the
    # EMA pair is a one-test family, so its corrected p is its raw p.
    if results[0] is not None:
        own = compare_family(short_rows or [], "fid", FAMILY_MAIN).get("awwl")
        results[0].p_holm = own.p_holm if own is not None else None
    if results[1] is not None:
        results[1].p_holm = results[1].p
    if results[3] is not None:
        own = compare_family(short_anc or [], "fid", FAMILY_DDPM).get("awwl")
        results[3].p_holm = own.p_holm if own is not None else None
    return [(BUDGET_POINTS[i][0], BUDGET_POINTS[i][1], arms[i][0], arms[i][1], res)
            for i, res in results.items() if res is not None]


#: Where each 200-epoch point of Table budget is corrected (its home family),
#: with the row it names in the caption.
BUDGET_HOME: dict[int, str] = {
    0: r"the main family of Table~\ref{tab:compare-fid} (no EMA, DDIM-100)",
    1: r"the one-test EMA family (EMA, DDIM-100)",
    3: r"the sampler family of Table~\ref{tab:replication} (DDPM-1000)",
}


def build_budget_table(points) -> str:
    body = []
    repeated = False
    for budget, sampler, mb, ma, res in points:
        index = BUDGET_POINTS.index((budget, sampler))
        mark = r"$^{\dagger}$" if index in BUDGET_HOME else ""
        repeated |= bool(mark)
        cells = [budget, sampler, str(res.n), fmean_sd(list(mb.values())), fmean_sd(list(ma.values())),
                 fcell_ci(res, 2), fp_cell(res.p), fp_cell(res.p_holm, (res.p_holm or 1) < 0.05) + mark]
        body.append(" & ".join(cells) + r" \\")
    note = ""
    if repeated:
        homes = [BUDGET_HOME[i] for i in sorted(BUDGET_HOME)]
        note = (r" $^{\dagger}$The 200-epoch rows repeat tests of other families and carry "
                r"those families' corrected $p$: " + ", ".join(homes[:-1]) + " and " + homes[-1] + ".")
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{AWWL against MSE at two training budgets and two samplers; $n$ is the "
        r"number of seeds. FID is mean $\pm$ sample standard deviation over seeds. Each "
        r"row is one operating point, and the comparison within a row is the meaningful "
        r"one: $\Delta$ is the FID gap AWWL $-$ MSE paired by seed with its 95\% "
        r"$t$-interval; $p$ is the two-sided paired $t$-test, raw and Holm-corrected, for "
        r"the 800-epoch rows over the " + cw(len(FAMILY_BUDGET)) + r" 800-epoch operating "
        r"points." + note + bold_note(body) + r" The 800-epoch runs use EMA and a cosine "
        r"schedule stretched to their length.}",
        r"\label{tab:budget}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Budget & Sampler & $n$ & MSE & AWWL & $\Delta$ [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# =============================================================================
# Builders: pricing
# =============================================================================

_PENDING_NOTE = (r" Cells marked \textbf{??} await rows of the pricing ledger "
                 r"(\texttt{runs/pricing/results.jsonl}).")


def _fid_cell(values: Mapping | None, digits: int = 3) -> str:
    if not values or len(values) < 2:
        return PLACEHOLDER
    return fmean_sd(list(values.values()), digits)


def _deficit_cell(values: Mapping | None, digits: int = 2) -> str:
    """Positive high-band deficit D = -deviation, mean +- SD."""
    if not values or len(values) < 2:
        return PLACEHOLDER
    return fmean_sd([-v for v in values.values()], digits)


def _pending(*cells: str) -> bool:
    return any(PLACEHOLDER in c for c in cells)


def build_worth_table(pr: Sequence[dict]) -> str:
    """The band boost applied to each arm's own samples (Table worth)."""
    body = []
    pending = False
    for arm, label in (("mse", "MSE"), ("awwl", "AWWL")):
        ident = price(pr, "cifar10", "identity", source=arm)
        dev = price(pr, "cifar10", "identity", source=arm, field=high_before)
        boosted = price(pr, "cifar10", "boost", source=arm, dose=BOOST_DOSE)
        # Same sign as every other pricing table: value = M(B) - M(T(B)).
        val = value_of(ident, boosted)
        cells = [_deficit_cell(dev), _fid_cell(ident), _fid_cell(boosted), fcell_ci(val, 3)]
        pending |= _pending(*cells)
        body.append(label + " & " + " & ".join(cells) + r" \\")
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{The high-band boost of $+" + f"{BOOST_DOSE:g}" + r"$~dB applied as a "
        r"Fourier post-process to each arm's own samples (CIFAR-10, epoch 199, DDIM-100, "
        r"\num{10000} samples per seed, five seeds), scored without 8-bit quantisation. "
        r"The deficit is the high-band spectral deficit of the samples as generated (dB, "
        r"positive means too little energy). FID is mean $\pm$ sample standard deviation. The "
        r"value of the boost is the FID it removes, $M(\text{as generated}) - "
        r"M(\text{boosted})$ ($M$ is FID), paired by seed, with its 95\% $t$-interval." + (_PENDING_NOTE if pending else "")
        + "}",
        r"\label{tab:worth}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & Deficit (dB) & FID & FID after boost & Value [95\% CI] \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_operators_table(pr: Sequence[dict], phase0_rows: Sequence[dict]) -> str:
    """Several operators on MSE's samples, and AWWL as trained (Table operators)."""
    ident = price(pr, "cifar10", "identity")
    specs = [
        (r"nothing (as generated)", dict(op="identity")),
        (r"high-band boost, $+" + f"{BOOST_DOSE:g}" + r"$~dB", dict(op="boost", dose=BOOST_DOSE)),
        (r"\quad the same, rounded to 8 bits with dither",
         dict(op="boost", dose=BOOST_DOSE, quantize="dither")),
        (r"AWWL's radial profile transplanted", dict(op="transplant", target="awwl")),
        (r"full radial-spectrum match", dict(op="match")),
        (r"\quad the same, rounded to 8 bits", dict(op="match", quantize="round")),
    ]
    body = []
    pending = False
    for label, kw in specs:
        op = kw.pop("op")
        fid = price(pr, "cifar10", op, **kw)
        dev = price(pr, "cifar10", op, field=high_after, **kw)
        val = None if op == "identity" else value_of(ident, fid)
        cells = [_fid_cell(fid), "---" if op == "identity" else fcell_ci(val, 3), _deficit_cell(dev)]
        pending |= _pending(*cells)
        body.append(label + " & " + " & ".join(cells) + r" \\")
    a = by_seed(phase0_rows, "awwl", "fid")
    a_dev = price(pr, "cifar10", "identity", source="awwl", field=high_before)
    cells = [_fid_cell(a), "---", _deficit_cell(a_dev)]
    pending |= _pending(*cells)
    body += [r"\midrule", r"\emph{AWWL as trained} & " + " & ".join(cells) + r" \\"]
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{Spectral operators applied to MSE's own samples (CIFAR-10, epoch 199, "
        r"DDIM-100, five seeds). Unless marked, the filtered images are scored in floating "
        r"point, without 8-bit quantisation. FID is mean $\pm$ sample standard deviation. "
        r"The value of an operator is $M(B) - M(T(B))$, the FID it removes ($M$ is FID), "
        r"paired by seed, "
        r"with its 95\% $t$-interval. The deficit is the high-band deficit after the operator "
        r"(dB). "
        r"The transplant imposes, seed by seed, the radial profile of the AWWL run with the "
        r"same seed; the match imposes the reference set's profile and so removes the mean "
        r"radial deficit, not per-image or anisotropic differences." + (_PENDING_NOTE if pending else "")
        + "}",
        r"\label{tab:operators}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Applied to MSE's samples & FID & Value [95\% CI] & Deficit after (dB) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


#: The columns of Table accounting: ``(key, training, operator)``. The key
#: names the column in code; the two header rows print the training budget
#: and the operator.
ACCOUNTING_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("boost200", "200", r"boost, $+" + f"{BOOST_DOSE:g}" + r"$~dB"),
    ("transplant200", "200", r"transplant"),
    ("transplant800", "800", r"transplant"),
)


def accounting_columns(pr, phase0_rows, long_rows):
    """(key, M(B), M(T(B)), M(A)) per accounting column (:data:`ACCOUNTING_COLUMNS`);
    values are {seed: fid}."""
    return [
        ("boost200", price(pr, "cifar10", "identity"),
         price(pr, "cifar10", "boost", dose=BOOST_DOSE), by_seed(phase0_rows, "awwl", "fid")),
        ("transplant200", price(pr, "cifar10", "identity"),
         price(pr, "cifar10", "transplant", target="awwl"), by_seed(phase0_rows, "awwl", "fid")),
        ("transplant800", price(pr, "cifar10_long", "identity"),
         price(pr, "cifar10_long", "transplant", target="awwl"), by_seed(long_rows, "awwl", "fid")),
    ]


def build_accounting_table(pr: Sequence[dict], phase0_rows: Sequence[dict], long_rows: Sequence[dict]) -> str:
    """Table accounting. Bold means ``p_Holm < 0.05`` in every table, so an
    MRG is bold only when it is significant after correction; only the
    200-epoch transplant belongs to a family (FAMILY_PRICING, whose corrected
    p is printed beneath it), so the other two MRGs are never bold."""
    cols = accounting_columns(pr, phase0_rows, long_rows)
    awwl_priced = pricing_arms(pr, phase0_rows).get("awwl", {}).get("mrg")
    holm_of = {"transplant200": awwl_priced.p_holm if awwl_priced is not None else None}

    def mean_cell(v):
        return PLACEHOLDER if not v or len(v) < 2 else "$" + fnum(statistics.mean(v.values()), 3) + "$"

    rows_out = {k: [] for k in ("b", "tb", "val", "a", "gap", "mrg", "p", "holm")}
    for title, b, tb, a in cols:
        val, gap, m = value_of(b, tb), paired(a, b), mrg(a, tb)
        rows_out["b"].append(mean_cell(b))
        rows_out["tb"].append(mean_cell(tb))
        rows_out["val"].append(fcell_ci(val, 3) if val else PLACEHOLDER)
        rows_out["a"].append(mean_cell(a))
        rows_out["gap"].append(fcell_ci(gap, 2) if gap else PLACEHOLDER)
        h = holm_of.get(title)
        significant = h is not None and h < 0.05
        mrg_cell = fcell_ci(m, 2) if m else PLACEHOLDER
        if m and significant:
            d = adaptive(2, m.mean)
            mrg_cell = (r"$\mathbf{" + fsigned(m.mean, d) + r"}$\;$\mathbf{[" + fbound(m.lo, d) + ", "
                        + fbound(m.hi, d) + "]}$")
        rows_out["mrg"].append(mrg_cell)
        rows_out["p"].append(fp_cell(m.p) if m else PLACEHOLDER)
        if title in holm_of:
            rows_out["holm"].append(PLACEHOLDER if h is None else fp_cell(h, significant))
        else:
            rows_out["holm"].append("---")
    pending = any(_pending(c) for v in rows_out.values() for c in v)
    labels = {
        "b": r"MSE, $M(B)$",
        "tb": r"MSE after $T$, $M(T(B))$",
        "val": r"Value, $M(B) - M(T(B))$",
        "a": r"AWWL, $M(A)$",
        "gap": r"Gap, $M(A) - M(B)$",
        "mrg": r"MRG, $M(A) - M(T(B))$",
        "p": r"\quad $p$ (MRG $= 0$)",
        "holm": r"\quad $p_{\mathrm{Holm}}$ (pricing family)",
    }
    body = []
    for key in ("b", "tb", "val", "a", "gap"):
        body.append(labels[key] + " & " + " & ".join(rows_out[key]) + r" \\")
    body.append(r"\midrule")
    for key in ("mrg", "p", "holm"):
        body.append(labels[key] + " & " + " & ".join(rows_out[key]) + r" \\")
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{The accounting of the pricing test; $M$ is FID. $B$ is MSE, $A$ is AWWL, and $T$ "
        r"is the operator applied to $B$'s own samples: the $+" + f"{BOOST_DOSE:g}" + r"$~dB "
        r"high-band boost, or the transplant of the AWWL run's radial profile onto the MSE run "
        r"with the same seed. The two 200-epoch columns use the main runs at epoch 199 (five "
        r"seeds, no EMA); the 800-epoch column the EMA runs at their last checkpoint, epoch 799 "
        r"(three seeds). Means over seeds; differences are paired by seed with 95\% "
        r"$t$-intervals, and $p$ is the two-sided paired $t$-test of MRG $= 0$. By construction "
        r"MRG is the FID gap plus the value of the correction. Only the 200-epoch transplant "
        r"belongs to a correction family: its $p_{\mathrm{Holm}}$ is over the "
        + str(len(FAMILY_PRICING)) + r" arms of Table~\ref{tab:pricing-arms}, and an MRG is set "
        r"in bold only if that corrected $p$ is below $0.05$." + bold_note(body)
        + (_PENDING_NOTE if pending else "") + "}",
        r"\label{tab:accounting}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{200-epoch runs, epoch 199} & 800-epoch runs, epoch 799 \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}",
        r"Operator $T$ & " + " & ".join(op for _k, _t, op in ACCOUNTING_COLUMNS) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def pricing_arms(pr: Sequence[dict], phase0_rows: Sequence[dict]) -> dict[str, dict]:
    """Per arm: transplant value, FID gap and MRG (Holm over FAMILY_PRICING)."""
    ident = price(pr, "cifar10", "identity")
    targets = {r.get("target") for r in pr if r.get("dataset") == "cifar10"
               and r.get("op") == "transplant" and r.get("source", BASELINE) == BASELINE
               and r.get("quantize", "float") == "float"}
    arms = list(FAMILY_PRICING) + sorted(t for t in targets if t and t not in FAMILY_PRICING)
    out: dict[str, dict] = {}
    mrgs: dict[str, Paired | None] = {}
    for arm in arms:
        a = by_seed(phase0_rows, arm, "fid")
        # A three-seed arm can only be transplanted at its three seeds.
        need = min(PRICING_SEEDS["cifar10"], len(a)) or PRICING_SEEDS["cifar10"]
        tb = price(pr, "cifar10", "transplant", target=arm, expected=need)
        delta_high = paired(
            price(pr, "cifar10", "transplant", target=arm, field=high_after, expected=need),
            price(pr, "cifar10", "transplant", target=arm, field=high_before, expected=need))
        out[arm] = {"value": value_of(ident, tb), "gap": paired(a, ident), "mrg": mrg(a, tb),
                    "delta_high": delta_high, "n": len(set(tb) & set(a) & set(ident))}
        mrgs[arm] = out[arm]["mrg"]
    apply_family(mrgs, FAMILY_PRICING)
    return out


def _mean_cell(res: Paired | None, digits: int) -> str:
    return "---" if res is None else "$" + fsigned(res.mean, adaptive(digits, res.mean)) + "$"


#: Cached radial profiles of every sample set (all samples), written by
#: ``scripts/price_fid.py`` and ``scripts/make_band_table.py``: read, never written, here.
PROFILE_CACHE = ROOT / "runs" / "pricing" / "cache"


def achieved_high_change(arm: str, seeds: Sequence, dataset: str = "cifar10",
                         epoch: int = 199) -> float | None:
    """Mean over ``seeds`` of the achieved high-band change ``s(A) - s(B)`` (dB).

    ``A`` is ``arm``'s samples and ``B`` MSE's of the same seed, from the cached
    profiles (the numbers of Table bands). ``None`` when a profile is missing.
    """
    import numpy as np

    from awwl.evaluation.price_fid import band_deviations

    base = PROFILE_CACHE / dataset
    ref_path = base / "ref_profile.npy"
    if not ref_path.exists() or not seeds:
        return None
    ref = np.load(ref_path)
    changes = []
    for seed in seeds:
        a = base / "profiles" / f"{arm}_s{seed}_ep{epoch}.npy"
        b = base / "profiles" / f"{BASELINE}_s{seed}_ep{epoch}.npy"
        if not (a.exists() and b.exists()):
            return None
        changes.append(float(band_deviations(np.load(a), ref)[2] - band_deviations(np.load(b), ref)[2]))
    return sum(changes) / len(changes)


def _measure_back_note(data: Mapping[str, dict]) -> str:
    """How far the change measured back on ``T_A(B)`` is from the achieved one.

    Clipping to the pixel range after the operator (Appendix scoring) makes
    them differ; the note gives the largest difference among the other arms
    and names the arm that differs most when it stands out.
    """
    gaps = []
    for arm, d in data.items():
        dh = d.get("delta_high")
        if dh is None:
            continue
        achieved = achieved_high_change(arm, dh.seeds)
        if achieved is None:
            return ""
        gaps.append((abs(dh.mean - achieved), arm, dh.mean, achieved))
    if len(gaps) < 2:
        return ""
    gaps.sort(reverse=True)
    (worst, arm, back, achieved), rest = gaps[0], gaps[1:]
    second = rest[0][0]
    text = (r" Clipping to the pixel range makes $\delta_{\mathrm{high}}$ differ from the achieved "
            r"change $s(A) - s(B)$ (Table~\ref{tab:bands} for the arms listed there; "
            r"Appendix~\ref{app:scoring})")
    if worst > 2 * second and round(worst, 2) > round(second, 2):
        return (text + r" by at most $" + fnum(max(second, 0.005), 2) + r"$~dB, except for "
                + NAMES.get(arm, arm) + r" ($" + fsigned(back, 2) + r"$ against $" + fsigned(achieved, 2)
                + r"$~dB).")
    return text + r" by at most $" + fnum(max(worst, 0.005), 2) + r"$~dB."


def build_pricing_arms_table(pr: Sequence[dict], phase0_rows: Sequence[dict]) -> str:
    """Table pricing-arms: the arms of FAMILY_PRICING under the :data:`BLOCKS`
    headers and in their order (the order of Table compare-fid and Fig.
    pricing-arms), then any priced arm outside the family."""
    data = pricing_arms(pr, phase0_rows)
    missing = [arm for arm in FAMILY_PRICING if data[arm]["mrg"] is None]

    def cells(arm: str, holm: bool) -> str:
        d, m = data[arm], data[arm]["mrg"]
        return " & ".join([
            NAMES.get(arm, arm), str(d["n"]),
            "$" + fsigned(d["delta_high"].mean, 2) + "$" if d["delta_high"] else "---",
            _mean_cell(d["gap"], 2), _mean_cell(d["value"], 3),
            fcell_ci(m, 2), fp_cell(m.p),
            fp_cell(m.p_holm, (m.p_holm or 1) < 0.05) if holm else "---",
        ]) + r" \\"

    body = block_lines(list(FAMILY_PRICING), 8,
                       lambda arm: cells(arm, True) if data[arm]["mrg"] is not None else None)
    extra = [cells(arm, False) for arm, d in sorted(data.items(), key=lambda kv: ORDER.get(kv[0], 99))
             if arm not in FAMILY_PRICING and d["mrg"] is not None]
    if not body:
        body = [r"\multicolumn{8}{c}{" + PLACEHOLDER + r"} \\"]
    if extra:
        body += [r"\addlinespace[2pt]", r"\multicolumn{8}{l}{\emph{Outside the declared family}} \\",
                 *extra]
    note = ""
    if missing:
        note = (r" Not yet measured, and counted with $p = 1$ in the correction: "
                + str(len(missing)) + " of the " + str(len(FAMILY_PRICING)) + " arms." + _PENDING_NOTE)
    ema = [a for a in (EMA_PAIR[0], EMA_PAIR[1]) if a in FAMILY_PRICING and data[a]["mrg"] is not None]
    ema_note = ""
    if ema:
        ema_note = (r" $B$ is MSE without EMA for every arm, the two EMA arms included, so the "
                    + NAMES[EMA_PAIR[1]] + r" row measures weight averaging rather than an "
                    r"objective, and " + NAMES[EMA_PAIR[0]] + r" is priced against MSE, not against "
                    r"the " + NAMES[EMA_PAIR[1]] + r" of its EMA-matched comparison in "
                    r"Table~\ref{tab:compare-fid}.")
    # Which arms the family holds: the stably trained arms of the main family,
    # and any priced arm from outside it (AWWL + EMA).
    stable = [a for a in FAMILY_MAIN if a in FAMILY_PRICING]
    unstable = [NAMES[a] for a in FAMILY_MAIN if a not in FAMILY_PRICING]
    added = [NAMES[a] for a in FAMILY_PRICING if a not in FAMILY_MAIN]
    members = (": the " + str(len(stable)) + r" arms of the main family of "
               r"Table~\ref{tab:compare-fid} that trained stably"
               + (" (" + ", ".join(unstable[:-1]) + " and " + unstable[-1] + r" did not and are not "
                  r"priced)" if len(unstable) > 1 else "")
               + ("" if not added else " and " + " and ".join(added)))
    back_note = _measure_back_note({a: data[a] for a in data if data[a]["mrg"] is not None})
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{The pricing test applied to every trained arm $A$ (CIFAR-10, epoch 199, "
        r"DDIM-100). $T_A$ transplants the radial profile of $A$'s run onto the MSE run "
        r"with the same seed, $B$; $\delta_{\mathrm{high}}$ is the resulting change in the "
        r"high-band deviation, measured back on $T_A(B)$ (dB). With $M$ the FID, value $= M(B) - M(T_A(B))$, gap "
        r"$= M(A) - M(B)$ and MRG $= M(A) - M(T_A(B))$, each a mean of paired-by-seed "
        r"differences over the $n$ seeds with all three measurements; the MRG carries its "
        r"95\% $t$-interval (the gap's is in Table~\ref{tab:compare-fid}). $p$ is the "
        r"two-sided paired $t$-test of MRG $= 0$, raw and Holm-corrected over the fixed family "
        r"of the " + str(len(FAMILY_PRICING)) + r" arms listed" + members + "."
        + ema_note + back_note + note + bold_note(body) + "}",
        r"\label{tab:pricing-arms}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Arm $A$ & $n$ & $\delta_{\mathrm{high}}$ & Gap & Value & "
        r"MRG [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_price_curve_table(pr: Sequence[dict]) -> str:
    """The value of the band boost against its dose, and the CelebA-64 prices."""
    ident = price(pr, "cifar10", "identity")
    body = []
    for dose in boost_doses(pr) or [BOOST_DOSE]:
        fid = price(pr, "cifar10", "boost", dose=dose)
        dev = price(pr, "cifar10", "boost", dose=dose, field=high_after)
        val = value_of(ident, fid)
        body.append(f"$+{dose:g}$ dB & " + " & ".join(
            [_fid_cell(fid), fcell_ci(val, 3), _deficit_cell(dev)]) + r" \\")
    celeba = [
        ("nothing (as generated)", dict(op="identity")),
        (r"boost, $+" + f"{BOOST_DOSE:g}" + r"$ dB", dict(op="boost", dose=BOOST_DOSE)),
        (r"boost, $+1.25$ dB", dict(op="boost", dose=1.25)),
        ("full radial-spectrum match", dict(op="match")),
    ]
    c_ident = price(pr, "celeba64", "identity")
    body += [r"\midrule", r"\multicolumn{4}{l}{\emph{CelebA-64, MSE, epoch 199}} \\"]
    for label, kw in celeba:
        op = kw.pop("op")
        fid = price(pr, "celeba64", op, **kw)
        dev = price(pr, "celeba64", op, field=high_after if op != "identity" else high_before, **kw)
        val = None if op == "identity" else value_of(c_ident, fid)
        body.append(label + " & " + " & ".join(
            [_fid_cell(fid), "---" if op == "identity" else fcell_ci(val, 3), _deficit_cell(dev)]) + r" \\")
    pending = any(PLACEHOLDER in line for line in body)
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{The value of the high-band boost as a function of its dose, on MSE's "
        r"samples (CIFAR-10, epoch 199, DDIM-100, five seeds, scored in floating point), "
        r"and the prices on CelebA-64. FID is mean $\pm$ sample standard deviation; value "
        r"$= M(B) - M(T(B))$ ($M$ is FID), paired by seed with its 95\% $t$-interval; "
        r"the deficit is the "
        r"high-band deficit after the operator (dB)." + (_PENDING_NOTE if pending else "") + "}",
        r"\label{tab:price-curve}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Operator on MSE's samples & FID & Value [95\% CI] & Deficit after (dB) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_features_table(pr: Sequence[dict], phase0_rows: Sequence[dict]) -> str:
    """The pricing test in two feature spaces: Inception-V3 (FID) and DINOv2 (FD_DINOv2).

    Every DINOv2 number comes from rows scored with features = "dinov2",
    including A's own distance (its identity row), so nothing mixes spaces.
    The Inception column uses the same pricing rows, and the run ledger for
    arms scored only there.
    """
    def one(values: Mapping | None, digits: int) -> str:
        if not values:
            return PLACEHOLDER
        v = next(iter(values.values()))
        return "$" + fsigned(v, digits) + "$"

    def space(features):
        F = {} if features is None else {"features": features}
        B = price(pr, "cifar10", "identity", **F)

        def arm_values(arm):
            if features is None:
                return by_seed(phase0_rows, arm, "fid")
            return price(pr, "cifar10", "identity", source=arm, **F)

        A = arm_values("awwl")
        T = price(pr, "cifar10", "transplant", target="awwl", **F)
        cells = {
            "base": _fid_cell(B, 1 if features else 3),
            "gap": fcell_ci(paired(A, B) if A and B else None, 2),
            "value": fcell_ci(value_of(B, T) if B and T else None, 2),
            "mrg": fcell_ci(mrg(A, T) if A and T else None, 2),
        }
        for band in ("mid", "high"):
            tb = price(pr, "cifar10", "transplant", target="awwl", bands=band, **F)
            cells["v_" + band] = fcell_ci(value_of(B, tb) if B and tb else None, 2)
        for key, kw in (("boost", dict(op="boost", dose=BOOST_DOSE)), ("match", dict(op="match")),
                        ("noise05", dict(op="noise", dose=0.5)), ("noise1", dict(op="noise", dose=1.0))):
            op = kw.pop("op")
            after = price(pr, "cifar10", op, **kw, **F)
            cells[key] = fcell_ci(value_of(B, after) if B and after else None, 2)
        rt = price(pr, "cifar10", "identity", source="real_test", source_folder="data/cifar10_test_png",
                   expected=1, **F)
        rn = price(pr, "cifar10", "noise", dose=0.5, source="real_test", source_folder="data/cifar10_test_png",
                   expected=1, **F)
        cells["real05"] = one({0: next(iter(rt.values())) - next(iter(rn.values()))} if rt and rn else None, 2)
        B399 = price(pr, "cifar10_long", "identity", source_epoch=399, **F)
        t399 = price(pr, "cifar10_long", "transplant", target="mse", source_epoch=199, target_epoch=399, **F)
        cells["pos"] = fcell_ci(mrg(B399, t399) if B399 and t399 else None, 2)
        for key, arm in (("gap_timestep", "awwl_timestep"), ("gap_sumfixed", "awwl_norm_matched")):
            a = arm_values(arm)
            cells[key] = fcell_ci(paired(a, B) if a and B else None, 2)
        Tp = price(pr, "cifar10", "transplant", target="mse_planted", **F)
        Ap = arm_values("mse_planted")
        cells["planted"] = fcell_ci(mrg(Ap, Tp) if Ap and Tp else None, 2)
        Tf = price(pr, "cifar10", "transplant", target="focal_frequency", **F)
        Af = arm_values("focal_frequency")
        cells["ffl_v"] = fcell_ci(value_of(B, Tf) if B and Tf else None, 2)
        cells["ffl_mrg"] = fcell_ci(mrg(Af, Tf) if Af and Tf else None, 2)
        Bc = price(pr, "celeba64", "identity", **F)
        for key, kw in (("c_match", dict(op="match")), ("c_noise", dict(op="noise", dose=0.5))):
            op = kw.pop("op")
            after = price(pr, "celeba64", op, **kw, **F)
            cells[key] = fcell_ci(value_of(Bc, after) if Bc and after else None, 2)
        return cells

    inc, dino = space(None), space("dinov2")
    rows = [
        ("base", r"MSE, $M(B)$ (mean $\pm$ SD)"),
        ("gap", r"AWWL's gap to MSE, $D = M(A) - M(B)$"),
        ("value", r"Value of AWWL's change, $V$ (transplant)"),
        ("mrg", r"$\mathrm{MRG} = M(A) - M(T_A(B))$"),
        ("v_mid", r"$V$, mid third only"),
        ("v_high", r"$V$, high third only"),
        ("boost", r"Value of the $+" + f"{BOOST_DOSE:g}" + r"$~dB boost"),
        ("match", r"Value of the full radial match"),
        ("noise05", r"Value of white noise, $\pm0.5$~LSB"),
        ("noise1", r"Value of white noise, $\pm1$~LSB"),
        ("real05", r"The same $\pm0.5$~LSB on held-out real images\textsuperscript{a}"),
        ("pos", r"Positive control, MRG (epoch 399 vs.\ 199)\textsuperscript{b}"),
        ("planted", r"Planted-mechanism control, MRG"),
        ("gap_timestep", r"AWWL, timestep only: gap to MSE"),
        ("gap_sumfixed", r"AWWL, $\sum_b w_b = 0.5$: gap to MSE"),
        ("ffl_v", r"FFL: value of its change, $V$"),
        ("ffl_mrg", r"FFL: MRG"),
        ("c_match", r"CelebA-64: value of the full radial match"),
        ("c_noise", r"CelebA-64: value of white noise, $\pm0.5$~LSB"),
    ]
    groups = (("base", "gap", "value", "mrg"),
              ("v_mid", "v_high", "boost", "match", "noise05", "noise1", "real05"),
              ("pos", "planted"), ("gap_timestep", "gap_sumfixed"), ("ffl_v", "ffl_mrg"), ("c_match", "c_noise"))
    labels = dict(rows)
    body = []
    for g, keys in enumerate(groups):
        if g:
            body.append(r"\midrule")
        body += [f"{labels[key]} & {inc[key]} & {dino[key]} \\\\" for key in keys]
    pending = any(PLACEHOLDER in line for line in body)
    lines = [
        PREAMBLE,
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{The pricing test in two feature spaces: FID (Inception-V3 pool features) and "
        r"FD$_{\text{DINOv2}}$ (Fr\'echet distance in DINOv2 ViT-L/14 features), both scored in "
        r"floating point on the same samples. CIFAR-10, epoch 199, DDIM-100, five seeds unless "
        r"stated; value $= M(B) - M(T(B))$, gaps and MRG paired by seed with 95\% $t$-intervals "
        r"(uncorrected); positive values are improvements for a value and costs for a gap or MRG. "
        r"\textsuperscript{a}One measurement of the 10{,}000 CIFAR-10 test images against the "
        r"training images; negative means the noise raises the distance. \textsuperscript{b}Three "
        r"seeds of the 800-epoch MSE run." + (_PENDING_NOTE if pending else "") + "}",
        r"\label{tab:features}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & FID & FD$_{\text{DINOv2}}$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# =============================================================================
# Driver
# =============================================================================

DEFAULTS = {
    "ledger": ROOT / "runs" / "phase0" / "results.jsonl",
    "ddpm1000": ROOT / "runs" / "phase0_ddpm1000" / "results.jsonl",
    "converged": ROOT / "runs" / "converged" / "results.jsonl",
    "converged_ddpm1000": ROOT / "runs" / "converged_ddpm1000" / "results.jsonl",
    "dreambooth": ROOT / "runs" / "dreambooth_seeds" / "results.jsonl",
    "restoration": ROOT / "runs" / "restoration_cifar" / "results.jsonl",
    "pricing": ROOT / "runs" / "pricing" / "results.jsonl",
}
CONVERGENCE_EPOCHS = (39, 79, 119, 159, 199)


def build_all(
    *, ledger: Path, ddpm1000: Path, converged: Path, converged_ddpm1000: Path,
    dreambooth: Path, restoration: Path, pricing: Path, epoch: int = 199,
    converged_epoch: int = 799, restoration_epoch: int = 99, seeds: int = 5,
) -> dict[str, str]:
    """Every table, as ``{file name: LaTeX}``; tables whose ledger is absent are skipped."""
    written: dict[str, str] = {}
    all_rows = load_eval(ledger)
    rows = [r for r in all_rows if r.get("epoch") == epoch]
    ddpm_rows = load_eval(ddpm1000, epoch)
    long_rows = load_eval(converged, converged_epoch)
    long_anc = load_eval(converged_ddpm1000, converged_epoch)
    pr = load_pricing(pricing)

    if rows:
        written["full.tex"] = build_main_table(rows, epoch, seeds)
        written["compare_fid.tex"] = build_comparison_table(rows, "fid", "FID", digits=2)
        written["compare_is.tex"] = build_comparison_table(rows, "is_mean", "Inception Score", digits=3)
        written["compare_kid.tex"] = build_comparison_table(
            rows, "kid_tf", r"KID ($\times 10^{3}$, \texttt{torch-fidelity})", digits=2, scale=1e3)
        written["compare_precision.tex"] = build_comparison_table(rows, "precision", "precision", digits=4)
        written["compare_recall.tex"] = build_comparison_table(rows, "recall", "recall", digits=4)
        written["compare_other.tex"] = build_compare_other_table(rows)
        written["convergence.tex"] = build_convergence_table(all_rows, CONVERGENCE_EPOCHS)
        written["confound.tex"] = build_confound_table(rows, ledger.parent)
        written["confound_summed.tex"] = build_confound_summed_table(rows, ledger.parent)
        written["extensions.tex"] = build_extensions_table(rows)
        written["worth.tex"] = build_worth_table(pr)
        written["operators.tex"] = build_operators_table(pr, rows)
        written["accounting.tex"] = build_accounting_table(pr, rows, long_rows)
        written["pricing_arms.tex"] = build_pricing_arms_table(pr, rows)
        written["price_curve.tex"] = build_price_curve_table(pr)
        written["features.tex"] = build_features_table(pr, rows)
        if ddpm_rows:
            written["replication.tex"] = build_replication_table(rows, ddpm_rows)
            written["replication_details.tex"] = build_replication_details_table(rows, ddpm_rows)
            written["sampler.tex"] = build_sampler_table(rows, ddpm_rows)
        if long_rows:
            written["budget.tex"] = build_budget_table(
                budget_points(rows, rows, long_rows, ddpm_rows, long_anc))
    db_rows = load_eval(dreambooth)
    if db_rows:
        written["dreambooth.tex"] = build_dreambooth_table(db_rows)
    r_rows = load_eval(restoration, restoration_epoch)
    if r_rows:
        written["restoration.tex"] = build_restoration_table(r_rows, restoration_epoch)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, default in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=Path, default=default)
    parser.add_argument("--epoch", type=int, default=199)
    parser.add_argument("--converged-epoch", type=int, default=799)
    parser.add_argument("--restoration-epoch", type=int, default=99)
    parser.add_argument("--seeds", type=int, default=5, help="Expected seeds per arm.")
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "tables")
    args = parser.parse_args()

    written = build_all(
        ledger=args.ledger, ddpm1000=args.ddpm1000, converged=args.converged,
        converged_ddpm1000=args.converged_ddpm1000, dreambooth=args.dreambooth,
        restoration=args.restoration, pricing=args.pricing, epoch=args.epoch,
        converged_epoch=args.converged_epoch, restoration_epoch=args.restoration_epoch,
        seeds=args.seeds,
    )
    if not written:
        print("nothing to write")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    for name, body in written.items():
        (args.out / name).write_text(body, encoding="utf-8")
        print(f"wrote {args.out / name}")

    rows = load_eval(args.ledger, args.epoch)
    if rows:
        groups = groups_of(rows)
        unknown = sorted(g for g in groups if g not in NAMES)
        if unknown:
            print(f"\nnot in the display order, so omitted: {', '.join(unknown)}")
        missing = [k for k in FAMILY_MAIN if k not in groups]
        if missing:
            print(f"declared in FAMILY_MAIN but not in the ledger (counted with p = 1): {', '.join(missing)}")
    pr = load_pricing(args.pricing)
    print(f"pricing ledger: {len(pr)} measurement(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
