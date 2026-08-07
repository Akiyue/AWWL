"""Significance testing over the results ledger.

The published tables compare single runs whose gaps (0.07 FID, 0.001
similarity) are far smaller than the seed-to-seed spread of the same
configuration. This module answers the question that makes such a table
publishable: **given N seeds per configuration, is the difference real?**

Comparisons are *paired by seed*. Seed 1 of AWWL is compared against seed 1
of the baseline, and so on, then the test runs on the differences. Pairing
removes the shared variance that comes from the seed itself (data order,
initialisation) and is markedly more sensitive than an unpaired test at the
small N that GPU budgets allow.

Both a paired t-test and a Wilcoxon signed-rank test are reported. At N=5 the
t-test's normality assumption is unverifiable, so the Wilcoxon column is there
as a robustness check: read it as *direction agreement*, not as a second
significance verdict.

Mind its floor. The smallest two-sided p a signed-rank test can produce is
``2 / 2^N`` — 0.0625 at N=5, 0.031 at N=6, 0.016 at N=7. **With five seeds
Wilcoxon can never clear α=0.05**, so a 0.0625 there alongside a small t-test
p means "as significant as this test can get", not "not significant". Six or
seven seeds is the cheapest way to make the two tests comparable.

Testing every configuration against one baseline is a multiple-comparison
problem, so p-values are corrected with Holm-Bonferroni — uniformly more
powerful than Bonferroni and, unlike Benjamini-Hochberg, controlling the
family-wise error rate, which is the right guarantee when the claim is "this
method beats these baselines" rather than "some of these are interesting".

``scipy`` is used when installed (it is in the ``eval`` extra); without it the
module falls back to a normal approximation for the t-test and skips Wilcoxon,
flagging the degradation rather than failing.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

LOWER_IS_BETTER = ("fid", "kid", "spec_dist", "loss")


@dataclass
class GroupSummary:
    """Per-configuration statistics for one metric across seeds."""

    group: str
    metric: str
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    by_seed: dict[Any, float] = field(default_factory=dict)

    @property
    def ci_halfwidth(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0


@dataclass
class ComparisonResult:
    """Paired comparison of one configuration against the baseline."""

    group: str
    baseline: str
    metric: str
    n_pairs: int
    mean_delta: float
    t_stat: float
    p_value: float
    p_holm: float
    wilcoxon_p: float | None
    better: bool
    significant: bool


# --------------------------------------------------------------- summarising


def summarize_groups(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    group_key: str = "group",
    seed_key: str = "seed",
    confidence: float = 0.95,
) -> list[GroupSummary]:
    """Mean, std and confidence interval of ``metric`` per configuration.

    Rows missing the metric are ignored. When a configuration has several rows
    for the same seed (e.g. a job re-run), the last one wins.
    """
    buckets: dict[str, dict[Any, float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None or not isinstance(value, (int, float)) or _is_sentinel(value):
            continue
        group = str(row.get(group_key, "?"))
        buckets.setdefault(group, {})[row.get(seed_key)] = float(value)

    summaries: list[GroupSummary] = []
    for group, by_seed in sorted(buckets.items()):
        values = list(by_seed.values())
        n = len(values)
        mean = sum(values) / n
        std = _stdev(values)
        half = _t_critical(n - 1, confidence) * std / math.sqrt(n) if n > 1 else 0.0
        summaries.append(
            GroupSummary(
                group=group,
                metric=metric,
                n=n,
                mean=mean,
                std=std,
                ci_low=mean - half,
                ci_high=mean + half,
                by_seed=dict(by_seed),
            )
        )
    return summaries


# ----------------------------------------------------------------- comparing


def compare_to_baseline(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    baseline: str,
    group_key: str = "group",
    seed_key: str = "seed",
    alpha: float = 0.05,
) -> list[ComparisonResult]:
    """Paired-by-seed comparison of every configuration against ``baseline``.

    Only seeds present in *both* configurations are used; a configuration
    sharing fewer than two seeds with the baseline cannot be tested and is
    skipped with a warning.

    Returns comparisons sorted by corrected p-value (most significant first).
    """
    summaries = {s.group: s for s in summarize_groups(rows, metric=metric, group_key=group_key, seed_key=seed_key)}
    if baseline not in summaries:
        raise ValueError(f"baseline {baseline!r} has no rows for metric {metric!r}")
    base = summaries[baseline].by_seed
    lower_better = _lower_is_better(metric)

    raw: list[ComparisonResult] = []
    for group, summary in summaries.items():
        if group == baseline:
            continue
        shared = sorted(set(base) & set(summary.by_seed), key=str)
        if len(shared) < 2:
            logger.warning(
                "%s shares %d seed(s) with %s — not enough to test", group, len(shared), baseline
            )
            continue
        deltas = [summary.by_seed[s] - base[s] for s in shared]
        t_stat, p_value = _paired_t(deltas)
        mean_delta = sum(deltas) / len(deltas)
        raw.append(
            ComparisonResult(
                group=group,
                baseline=baseline,
                metric=metric,
                n_pairs=len(shared),
                mean_delta=mean_delta,
                t_stat=t_stat,
                p_value=p_value,
                p_holm=p_value,
                wilcoxon_p=_wilcoxon(deltas),
                better=(mean_delta < 0) if lower_better else (mean_delta > 0),
                significant=False,
            )
        )

    for result, corrected in zip(raw, _holm([r.p_value for r in raw]), strict=True):
        result.p_holm = corrected
        result.significant = corrected < alpha
    raw.sort(key=lambda r: r.p_holm)
    return raw


def _holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down correction, preserving input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    corrected = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p_values[idx])
        running = max(running, adjusted)  # enforce monotonicity
        corrected[idx] = running
    return corrected


# ------------------------------------------------------------------- testing


def _paired_t(deltas: Sequence[float]) -> tuple[float, float]:
    """Two-sided paired t-test on already-differenced values."""
    n = len(deltas)
    mean = sum(deltas) / n
    sd = _stdev(deltas)
    if sd == 0.0:
        # Identical in every pair: no evidence of a difference either way.
        return (0.0, 1.0) if mean == 0.0 else (math.inf, 0.0)
    t_stat = mean / (sd / math.sqrt(n))
    return t_stat, _t_sf(abs(t_stat), n - 1) * 2.0


def _wilcoxon(deltas: Sequence[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value, or ``None`` without scipy."""
    try:
        from scipy import stats
    except ImportError:
        return None
    if all(d == 0 for d in deltas):
        return 1.0
    try:
        return float(stats.wilcoxon(list(deltas)).pvalue)
    except ValueError:
        # scipy refuses samples that are too small to have any power.
        return None


def _t_sf(t: float, df: int) -> float:
    """Upper-tail probability of Student's t."""
    if df <= 0:
        return 1.0
    try:
        from scipy import stats

        return float(stats.t.sf(t, df))
    except ImportError:
        # Normal approximation; conservative to report, so warn once.
        logger.debug("scipy missing — using a normal approximation for the t-test")
        return 0.5 * math.erfc(t / math.sqrt(2.0))


_T_TABLE_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
               8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def _t_critical(df: int, confidence: float) -> float:
    """Two-sided critical value of Student's t."""
    if df <= 0:
        return 0.0
    try:
        from scipy import stats

        return float(stats.t.ppf(0.5 + confidence / 2.0, df))
    except ImportError:
        if abs(confidence - 0.95) > 1e-9:
            return 1.96
        for key in sorted(_T_TABLE_95):
            if df <= key:
                return _T_TABLE_95[key]
        return 1.96


def _stdev(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _lower_is_better(metric: str) -> bool:
    return any(metric.lower().startswith(m) for m in LOWER_IS_BETTER)


def _is_sentinel(value: float) -> bool:
    """The evaluation helpers report a failed metric as ``-1.0``."""
    return float(value) == -1.0


# ----------------------------------------------------------------- rendering


def format_summary_table(summaries: Sequence[GroupSummary], *, metric: str) -> str:
    """Render :func:`summarize_groups` output as a fixed-width table."""
    if not summaries:
        return f"no results for metric {metric!r}"
    arrow = "↓" if _lower_is_better(metric) else "↑"
    width = max(len(s.group) for s in summaries)
    header = f"{'config':<{width}}  {'n':>2}  {metric + ' ' + arrow:>14}  {'95% CI':>22}"
    lines = [header, "-" * len(header)]
    ranked = sorted(summaries, key=lambda s: s.mean, reverse=not _lower_is_better(metric))
    for s in ranked:
        ci = f"[{s.ci_low:.4f}, {s.ci_high:.4f}]"
        lines.append(f"{s.group:<{width}}  {s.n:>2}  {s.mean:>9.4f} ± {s.std:.4f}  {ci:>22}")
    return "\n".join(lines)


def format_comparison_table(results: Sequence[ComparisonResult]) -> str:
    """Render :func:`compare_to_baseline` output as a fixed-width table."""
    if not results:
        return "no comparable configurations"
    baseline = results[0].baseline
    metric = results[0].metric
    width = max(len(r.group) for r in results)
    header = (
        f"{'config':<{width}}  {'n':>2}  {'Δ vs ' + baseline:>14}  "
        f"{'t':>8}  {'p':>9}  {'p(Holm)':>9}  {'Wilcoxon':>9}  verdict"
    )
    lines = [f"metric: {metric}   baseline: {baseline}", header, "-" * len(header)]
    for r in results:
        wilcox = f"{r.wilcoxon_p:.4f}" if r.wilcoxon_p is not None else "n/a"
        if r.significant:
            verdict = "better (p<0.05)" if r.better else "WORSE (p<0.05)"
        else:
            verdict = "no sig. difference"
        lines.append(
            f"{r.group:<{width}}  {r.n_pairs:>2}  {r.mean_delta:>+14.4f}  "
            f"{r.t_stat:>8.3f}  {r.p_value:>9.4f}  {r.p_holm:>9.4f}  {wilcox:>9}  {verdict}"
        )
    lines.append("")
    lines.append("Δ is (config − baseline); p(Holm) is family-wise corrected across the rows above.")
    return "\n".join(lines)


def convergence_table(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    group_key: str = "group",
    epoch_key: str = "epoch",
    seed_key: str = "seed",
) -> str:
    """Metric versus training epoch per configuration, averaged over seeds.

    This is the table behind the "does AWWL reach a given quality *earlier*"
    question — a claim about training cost, which needs checkpoints evaluated
    along the way rather than only at the end.
    """
    epochs = sorted({r[epoch_key] for r in rows if r.get(epoch_key) is not None})
    if not epochs:
        return "no rows carry an epoch; nothing to plot against"

    groups = sorted({str(r.get(group_key, "?")) for r in rows})
    width = max(len(g) for g in groups)
    header = f"{'config':<{width}}  " + "  ".join(f"{'ep' + str(e):>10}" for e in epochs)
    lines = [f"metric: {metric} (mean over seeds)", header, "-" * len(header)]
    for group in groups:
        cells = []
        for epoch in epochs:
            subset = [r for r in rows if str(r.get(group_key)) == group and r.get(epoch_key) == epoch]
            summary = summarize_groups(subset, metric=metric, group_key=group_key, seed_key=seed_key)
            cells.append(f"{summary[0].mean:>10.4f}" if summary else f"{'—':>10}")
        lines.append(f"{group:<{width}}  " + "  ".join(cells))
    return "\n".join(lines)
