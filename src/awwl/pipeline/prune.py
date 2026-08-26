"""Reclaim disk from model weights whose results are already recorded.

A sweep's weights are the bulk of its footprint and the least of its value: a
DreamBooth UNet is ~3.4 GB and there are 50 of them, while the row it produced
is a few hundred bytes in the ledger. Once a run has been evaluated, its
weights are only needed to evaluate it *again*.

The rule this module enforces is the one a hurried ``rm -rf`` breaks: never
delete weights for a run that has no results. The ledger is consulted first,
per experiment, and anything unevaluated is left alone and reported as such.

Deleting is opt-in. The default run reports what it *would* free.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from awwl.pipeline.manifest import build_jobs, load_manifest

# Folders holding model weights, relative to a run directory. Samples and
# configs are left alone: samples are needed to re-score without a GPU, and a
# config is what makes a run identifiable at all.
WEIGHT_DIRS = ("unet",)
WEIGHT_GLOBS = ("checkpoint-*",)


@dataclass
class Reclaimable:
    """One run's weights, and whether they may go."""

    exp: str
    group: str
    paths: list[Path]
    bytes_used: int
    evaluated: bool

    @property
    def gigabytes(self) -> float:
        return self.bytes_used / 1e9


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _weight_paths(run_dir: Path) -> list[Path]:
    found = [run_dir / name for name in WEIGHT_DIRS]
    for pattern in WEIGHT_GLOBS:
        found.extend(run_dir.glob(pattern))
    return [p for p in found if p.is_dir()]


def survey(manifest_path: str | Path) -> tuple[str, list[Reclaimable]]:
    """What each run's weights cost, and whether its results are safely stored."""
    from awwl.analysis.results import load_results

    manifest = load_manifest(manifest_path)
    name = str(manifest["name"])
    output_root = Path(manifest["output_root"])
    run_root = Path(manifest.get("run_root", output_root))
    ledger = Path(manifest.get("ledger", output_root / "results.jsonl"))

    evaluated: set[str] = set()
    if ledger.exists():
        for row in load_results(ledger):
            if row.get("kind") != "train" and row.get("exp"):
                evaluated.add(str(row["exp"]))

    seen: dict[str, str] = {}
    for job in build_jobs(manifest):
        exp = job.payload.get("exp")
        if exp:
            seen.setdefault(str(exp), job.group_id)

    out: list[Reclaimable] = []
    for exp, group in sorted(seen.items()):
        run_dir = run_root / exp
        paths = _weight_paths(run_dir)
        if not paths:
            continue
        out.append(
            Reclaimable(
                exp=exp,
                group=group,
                paths=paths,
                bytes_used=sum(_dir_size(p) for p in paths),
                evaluated=exp in evaluated,
            )
        )
    return name, out


def format_survey(name: str, items: list[Reclaimable]) -> str:
    """Report what can be freed, and what is being kept back and why."""
    safe = [i for i in items if i.evaluated]
    held = [i for i in items if not i.evaluated]

    lines = [f"## {name}", ""]
    if not items:
        return "\n".join(lines + ["  no model weights on disk", ""])

    lines.append(f"  {'run':<28} {'size':>9}  state")
    lines.append("  " + "-" * 48)
    for item in items:
        state = "evaluated -- can be freed" if item.evaluated else "NO RESULTS -- keeping"
        lines.append(f"  {item.exp:<28} {item.gigabytes:>7.1f}G  {state}")

    lines += [
        "",
        f"  reclaimable: {sum(i.gigabytes for i in safe):.1f} G across {len(safe)} run(s)",
    ]
    if held:
        lines.append(
            f"  kept: {sum(i.gigabytes for i in held):.1f} G across {len(held)} run(s) "
            "with no ledger row -- deleting these would lose the run"
        )
    lines.append("")
    return "\n".join(lines)


def prune(items: list[Reclaimable]) -> tuple[int, int]:
    """Delete the weights of evaluated runs. Returns ``(runs, bytes freed)``."""
    runs = 0
    freed = 0
    for item in items:
        if not item.evaluated:
            continue
        for path in item.paths:
            shutil.rmtree(path)
        runs += 1
        freed += item.bytes_used
    return runs, freed
