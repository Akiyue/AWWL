"""Rebuild sampling checkpoints from state snapshots, in bulk.

The DDPM-1000 sampler check re-samples the tier-1 runs, but the disk-cleanup
prune deleted every evaluated run's ``checkpoint-*`` folder. Samples and
metrics survive; the samplable weights do not. Each run's crash-recovery
snapshot under ``state/`` still carries the same UNet at the same epoch, and
:func:`~awwl.methods.finetune.inference.rebuild_checkpoint_from_state`
reconstitutes a loadable pipeline from it.

This script walks the runs the ddpm1000 manifest needs and rebuilds each
checkpoint in place (``<run>/checkpoint-<epoch>``, the path the deleted one
had), so no manifest change is required. Idempotent: an existing checkpoint
is left alone.

Usage::

    python scripts/rebuild_checkpoints.py \
        --root runs/phase0 --groups mse,static_wavelet,awwl --seeds 1,2,3,4,5

Verify afterwards that every run now has a checkpoint-199 before re-running
the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from awwl.methods.finetune.inference import rebuild_checkpoint_from_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/phase0"))
    parser.add_argument("--groups", default="mse,static_wavelet,awwl")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    failures: list[str] = []
    for group in groups:
        for seed in seeds:
            run = args.root / f"{group}_s{seed}"
            try:
                target = rebuild_checkpoint_from_state(run)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                failures.append(f"{run.name}: {exc}")
                print(f"FAIL {run.name}: {exc}")
                continue
            print(f"ok   {run.name}: {target}")
    if failures:
        print(f"\n{len(failures)} run(s) could not be rebuilt")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
