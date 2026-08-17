"""What has actually been measured, per manifest, per group, per seed.

The job queue and the results ledger answer different questions and can
disagree. A job marked ``done`` whose ledger row never landed is the dangerous
case: the queue reports success, the table silently averages over four seeds
instead of five, and nothing anywhere says so. This module reports both and
puts the disagreement in its own column.

The ledger is the authority on what a paper may claim: a row exists only if a
metric was computed and written. Queue state is reported alongside because it
explains *why* a row is missing -- not yet run, running now, or failed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from awwl.pipeline.manifest import build_jobs, load_manifest


@dataclass
class GroupStatus:
    """Coverage for one configuration of one manifest."""

    group: str
    tier: int
    planned_seeds: set[Any] = field(default_factory=set)
    trained_seeds: set[Any] = field(default_factory=set)
    evaluated_seeds: set[Any] = field(default_factory=set)
    metrics: set[str] = field(default_factory=set)
    queue: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.planned_seeds) and self.evaluated_seeds >= self.planned_seeds

    @property
    def started(self) -> bool:
        return bool(self.trained_seeds or self.evaluated_seeds)


# Metrics a CIFAR-10 arm needs before it can appear in the paper's tables.
# Absence of one is invisible in a per-group seed count, which is exactly how a
# half-evaluated arm reaches a table.
EXPECTED_METRICS = {
    "finetune": ("fid", "is_mean", "kid_tf"),
    "dreambooth": ("clip_score", "similarity"),
}


_SEED_RE = re.compile(r"_s(\d+)")


def _seed_in(job_id: str) -> str | None:
    """The seed encoded in an experiment name, e.g. ``mse_s3`` -> ``\"3\"``."""
    found = _SEED_RE.findall(job_id)
    return found[-1] if found else None


# Lines that appear inside every traceback and identify nothing.
_NOISE = ("Traceback (most recent call last)", "During handling of", "The above exception")


def _strip_shutdown_noise(error: str) -> str:
    """Drop ``Exception ignored in:`` blocks, which are teardown, not failure.

    Python prints these while finalising the interpreter, *after* whatever
    actually went wrong. On this project the ``multiprocess`` package emits
    ``AttributeError: '_thread.RLock' object has no attribute
    '_recursion_count'`` on every exit under Python 3.12 -- successful runs
    included -- so reading the last line of stderr reports it as the cause of
    19 failures and buries the real one.

    Everything from the first such block onward is dropped: they only occur at
    shutdown, so nothing that matters follows.
    """
    lines = error.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Exception ignored"):
            return "\n".join(lines[:i])
    return error


def _last_meaningful_line(error: str, *, width: int = 160) -> str:
    """The line of a captured stderr that says what went wrong.

    Python puts the exception type and message last, so the final non-empty
    line of the real output is almost always the useful one; source echoes and
    frame headers above it are not. Truncated, because these are printed one
    per distinct failure and a wrapped traceback defeats the purpose.
    """
    for line in reversed(_strip_shutdown_noise(error).strip().splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith(("File \"", "^", "~")) or stripped in _NOISE:
            continue
        return stripped[:width]
    return "(process died without a traceback -- killed, or output truncated)"


def collect_status(
    manifest_path: str | Path,
    *,
    ledger_override: Path | None = None,
) -> tuple[str, str, list[GroupStatus], dict[str, int]]:
    """Cross-reference a manifest's plan against the ledger and the job queue.

    Returns:
        ``(name, method, per-group statuses, queue counts)``. Queue counts are
        empty when the pipeline has never been started.
    """
    from awwl.analysis.results import load_results

    manifest = load_manifest(manifest_path)
    name = str(manifest["name"])
    method = str(manifest.get("method", "finetune"))
    output_root = Path(manifest["output_root"])
    ledger = Path(ledger_override or manifest.get("ledger", output_root / "results.jsonl"))

    statuses: dict[str, GroupStatus] = {}
    for job in build_jobs(manifest):
        st = statuses.setdefault(job.group_id, GroupStatus(job.group_id, job.tier))
        # Only train jobs are counted, one per seed. Eval job ids carry an
        # epoch suffix (``:eval:mse_s1:199``), so counting those would multiply
        # the seed total by the number of evaluated checkpoints.
        if job.kind == "train":
            seed = _seed_in(job.job_id)
            if seed is not None:
                st.planned_seeds.add(seed)

    # Ledger: the authority on what exists to be written about.
    if ledger.exists():
        for row in load_results(ledger):
            group = str(row.get("group") or "")
            st = statuses.get(group)
            if st is None:
                continue
            seed = str(row.get("seed"))
            if row.get("kind") == "train":
                st.trained_seeds.add(seed)
            else:
                st.evaluated_seeds.add(seed)
                st.metrics.update(
                    k for k in EXPECTED_METRICS.get(method, ()) if row.get(k) is not None
                )

    # Queue: the explanation for anything the ledger is missing.
    counts: dict[str, int] = {}
    from awwl.pipeline.store import JobStore, store_path

    db = store_path(output_root)
    if db.exists():
        store = JobStore(db)
        counts = store.counts()
        for job in store.list_jobs():
            st = statuses.get(job.group_id)
            if st is not None:
                st.queue[job.status] += 1
                if job.error:
                    st.errors.append(job.error)

    ordered = sorted(statuses.values(), key=lambda s: (s.tier, s.group))
    return name, method, ordered, counts


def format_status(
    name: str,
    method: str,
    statuses: list[GroupStatus],
    counts: dict[str, int],
    *,
    show_errors: bool = False,
) -> str:
    """Render one manifest's coverage as a table plus a verdict."""
    expected = EXPECTED_METRICS.get(method, ())
    lines = [f"## {name}  ({method})", ""]
    if not statuses:
        return "\n".join(lines + ["  no experiments in this manifest", ""])

    header = f"  {'tier':<5} {'group':<22} {'train':>7} {'eval':>7}  {'metrics':<22} status"
    lines += [header, "  " + "-" * (len(header) - 2)]

    for st in statuses:
        planned = len(st.planned_seeds)
        train = f"{len(st.trained_seeds)}/{planned}"
        ev = f"{len(st.evaluated_seeds)}/{planned}"

        missing = [m for m in expected if m not in st.metrics]
        if not st.started:
            metric_cell, verdict = "-", "NOT STARTED"
        elif st.complete and not missing:
            metric_cell, verdict = "all", "complete"
        else:
            metric_cell = ",".join(sorted(st.metrics)) or "-"
            if missing and st.evaluated_seeds:
                verdict = "missing " + ",".join(missing)
            else:
                verdict = "partial"

        # Queue state earns a mention only when it explains the gap.
        if st.queue.get("failed"):
            verdict = f"{st.queue['failed']} FAILED"
        elif st.queue.get("running"):
            verdict += f" ({st.queue['running']} running)"

        lines.append(
            f"  {st.tier:<5} {st.group:<22} {train:>7} {ev:>7}  {metric_cell:<22} {verdict}"
        )

    lines.append("")

    # Why things failed, deduplicated. A sweep fails the same way 5 or 25 times
    # over, so the distinct reasons are short even when the failure count is
    # not -- and printing them here is the difference between "some jobs
    # failed" and knowing what to fix.
    reasons: dict[str, list[str]] = {}
    samples: dict[str, str] = {}
    for st in statuses:
        for err in st.errors:
            reason = _last_meaningful_line(err)
            reasons.setdefault(reason, []).append(st.group)
            samples.setdefault(reason, err)
    if reasons:
        lines.append("  failures, by reason:")
        for reason, groups in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            unique = sorted(set(groups))
            lines.append(f"    {len(groups):>3}x  {reason}")
            lines.append(f"         in: {', '.join(unique)}")
            if show_errors:
                lines.append("         ---- full stderr of one such job ----")
                lines += [f"         {ln}" for ln in samples[reason].splitlines()]
                lines.append("         ---- end ----")
        lines.append("")

    if counts:
        lines.append("  queue: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    done = sum(1 for s in statuses if s.complete)
    lines.append(f"  usable for the paper: {done}/{len(statuses)} configurations")

    # The disagreement that a seed count alone hides.
    ghosts = [s.group for s in statuses if s.queue.get("done") and not s.evaluated_seeds]
    if ghosts:
        lines += [
            "",
            "  WARNING: queue reports done but no ledger rows exist for: "
            + ", ".join(ghosts),
            "  Those runs cannot be written about. Check the eval logs.",
        ]
    lines.append("")
    return "\n".join(lines)


def status_report(manifests: list[Path], *, show_errors: bool = False) -> str:
    """Full report across every manifest handed in."""
    out = ["# Experiment coverage", ""]
    for path in manifests:
        try:
            name, method, statuses, counts = collect_status(path)
        except Exception as exc:  # a broken manifest must not hide the others
            out += [f"## {path}", "", f"  could not read: {exc}", ""]
            continue
        out.append(format_status(name, method, statuses, counts, show_errors=show_errors))
    return "\n".join(out)
