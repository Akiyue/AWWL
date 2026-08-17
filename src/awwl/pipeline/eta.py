"""How long the rest of a sweep will take, measured from how long it has taken.

Estimating from step counts and FLOPs is guesswork on a shared machine. The
queue already records when every finished job started and stopped, so the same
work's observed duration on this hardware is available and is the only figure
worth quoting.

Medians rather than means: a job that was reclaimed after a stale heartbeat,
or that shared a GPU with someone else's process, is an outlier and there are
enough of them in a long sweep to move a mean noticeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass
class Estimate:
    """Remaining work, and what it is based on."""

    per_kind: dict[str, tuple[int, float]]  # kind -> (jobs remaining, median seconds)
    workers: int
    sampled: dict[str, int]  # kind -> how many finished jobs the median came from

    @property
    def gpu_seconds(self) -> float:
        return sum(n * seconds for n, seconds in self.per_kind.values())

    @property
    def wall_seconds(self) -> float:
        """Assumes the queue keeps every worker busy, which it does until the tail."""
        return self.gpu_seconds / max(self.workers, 1)

    @property
    def unmeasured(self) -> list[str]:
        """Kinds with work remaining but no finished job to estimate from."""
        return [k for k, (n, seconds) in self.per_kind.items() if n and seconds == 0.0]


def estimate(
    pending: dict[str, int],
    durations: dict[str, list[float]],
    *,
    workers: int = 1,
) -> Estimate:
    """Combine what is left with how long each kind has taken."""
    per_kind: dict[str, tuple[int, float]] = {}
    sampled: dict[str, int] = {}
    for kind, count in pending.items():
        observed = durations.get(kind, [])
        per_kind[kind] = (count, median(observed) if observed else 0.0)
        sampled[kind] = len(observed)
    return Estimate(per_kind=per_kind, workers=workers, sampled=sampled)


def humanise(seconds: float) -> str:
    """A duration a reader can act on: minutes, hours, or days and hours."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days ({hours:.0f} h)"


def format_estimate(est: Estimate) -> list[str]:
    """Report lines, ordered by which kind dominates the remaining time."""
    if not est.gpu_seconds:
        return []

    lines = ["  time remaining, from measured job durations:"]
    ordered = sorted(est.per_kind.items(), key=lambda kv: -kv[1][0] * kv[1][1])
    for kind, (count, seconds) in ordered:
        if not count:
            continue
        if seconds == 0.0:
            lines.append(f"    {count:>4} {kind:<7}  --  never run yet, cannot estimate")
            continue
        share = count * seconds
        lines.append(
            f"    {count:>4} {kind:<7}  {humanise(seconds)} each"
            f"  ->  {humanise(share)}  (n={est.sampled.get(kind, 0)})"
        )
    lines.append(
        f"    total {humanise(est.gpu_seconds)} of GPU time"
        f"  ->  {humanise(est.wall_seconds)} on {est.workers} worker(s)"
    )
    if est.unmeasured:
        lines.append(
            "    excludes " + ", ".join(est.unmeasured) + ": no finished job to measure"
        )
    return lines
