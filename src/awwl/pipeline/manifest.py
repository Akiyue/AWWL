"""Expand an experiment manifest into a dependency-ordered list of jobs.

A manifest describes a sweep declaratively; this module turns it into the
concrete ``train → sample → eval`` chains the runner executes. Example::

    name: phase0
    base_config: configs/finetune.yaml
    output_root: ./runs/phase0
    ledger: ./runs/phase0/results.jsonl
    real_images: ./data/cifar10_train_png

    defaults:
      seeds: [1, 2, 3, 4, 5]
      eval_epochs: [199]
      sample: {num_samples: 10000, sampler: ddim, steps: 100, batch_size: 256}

    experiments:
      - group: mse
        tier: 1
        overrides: {loss.name: mse}

Every experiment is crossed with its seeds, giving one run directory per
``(group, seed)``. Job ids are derived from that tuple rather than
auto-incremented, so re-expanding an unchanged manifest produces exactly the
same ids and :meth:`JobStore.add_jobs` recognises the work as already queued.

``tier`` orders the sweep: the runner drains tier 1 before touching tier 2.
Put the experiments that decide whether the hypothesis survives in tier 1 —
there is no point spending a week on ablations of an effect that turns out to
be seed noise.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from awwl.core.exceptions import ConfigError
from awwl.pipeline.store import Job
from awwl.utils.io import load_yaml

logger = logging.getLogger(__name__)


def _cli(*args: str) -> list[str]:
    """An argv that invokes this interpreter's ``awwl`` CLI."""
    return [sys.executable, "-m", "awwl.cli", *args]


def _override_args(overrides: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in overrides.items():
        out += ["--override", f"{key}={value}"]
    return out


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a manifest YAML."""
    manifest = load_yaml(path)
    for required in ("name", "base_config", "output_root", "experiments"):
        if required not in manifest:
            raise ConfigError(f"manifest {path} is missing required key {required!r}")
    if not isinstance(manifest["experiments"], list) or not manifest["experiments"]:
        raise ConfigError(f"manifest {path}: 'experiments' must be a non-empty list")
    return manifest


def build_jobs(manifest: dict[str, Any], *, manifest_dir: Path | None = None) -> list[Job]:
    """Expand a loaded manifest into jobs.

    Args:
        manifest: Result of :func:`load_manifest`.
        manifest_dir: Directory the manifest was loaded from; relative paths
            inside it resolve against the current working directory, which is
            the repository root in normal use.

    Returns:
        Jobs in dependency order (every job appears after the one it needs).
    """
    del manifest_dir  # paths are repo-root relative by convention

    name = str(manifest["name"])
    base_config = str(manifest["base_config"])
    method = str(manifest.get("method", "finetune"))
    output_root = Path(manifest["output_root"])
    ledger = str(manifest.get("ledger", output_root / "results.jsonl"))
    real_images = manifest.get("real_images")
    if method not in ("finetune", "dreambooth"):
        raise ConfigError(f"manifest {name}: unsupported method {method!r}")

    defaults = manifest.get("defaults", {}) or {}
    default_seeds = list(defaults.get("seeds", [42]))
    default_epochs = list(defaults.get("eval_epochs", []))
    default_sample = dict(defaults.get("sample", {}) or {})
    default_overrides = dict(defaults.get("overrides", {}) or {})

    jobs: list[Job] = []
    for spec in manifest["experiments"]:
        group = str(spec["group"])
        tier = int(spec.get("tier", 1))
        seeds = list(spec.get("seeds", default_seeds))
        eval_epochs = list(spec.get("eval_epochs", default_epochs))
        sample_cfg = {**default_sample, **(spec.get("sample", {}) or {})}
        overrides = {**default_overrides, **(spec.get("overrides", {}) or {})}

        for seed in seeds:
            exp = f"{group}_s{seed}"
            run_dir = output_root / exp
            train_id = f"{name}:train:{exp}"

            train_overrides = {
                **overrides,
                "seed": seed,
                "output.dir": str(output_root),
                "output.name": exp,
                "output.group": group,
                "output.ledger": ledger,
            }
            jobs.append(
                Job(
                    job_id=train_id,
                    pipeline=name,
                    kind="train",
                    group_id=group,
                    tier=tier,
                    depends_on=None,
                    payload={
                        "argv": _cli("train", "--config", base_config, *_override_args(train_overrides)),
                        "run_dir": str(run_dir),
                        "exp": exp,
                        "seed": seed,
                    },
                    status="pending",
                    attempts=0,
                )
            )

            if method == "dreambooth":
                if real_images:
                    jobs.append(
                        Job(
                            job_id=f"{name}:eval:{exp}",
                            pipeline=name,
                            kind="eval",
                            group_id=group,
                            tier=tier,
                            depends_on=train_id,
                            payload={
                                "argv": _cli(
                                    "eval-dreambooth",
                                    "--run-dir", str(run_dir),
                                    "--real", str(real_images),
                                    "--ledger", ledger,
                                    "--num-images", str(sample_cfg.get("num_samples", 20)),
                                    "--steps", str(sample_cfg.get("steps", 50)),
                                    *(
                                        ["--prompts", str(sample_cfg["prompts"])]
                                        if sample_cfg.get("prompts")
                                        else []
                                    ),
                                ),
                                "run_dir": str(run_dir),
                                "exp": exp,
                            },
                            status="pending",
                            attempts=0,
                        )
                    )
                continue

            for epoch in eval_epochs:
                checkpoint = run_dir / f"checkpoint-{epoch}"
                samples_dir = run_dir / "samples" / f"ep{epoch}"
                sample_id = f"{name}:sample:{exp}:{epoch}"
                jobs.append(
                    Job(
                        job_id=sample_id,
                        pipeline=name,
                        kind="sample",
                        group_id=group,
                        tier=tier,
                        depends_on=train_id,
                        payload={
                            "argv": _cli(
                                "infer",
                                "--method", "finetune",
                                "--weights", str(checkpoint),
                                "--output-dir", str(samples_dir),
                                "--num-samples", str(sample_cfg.get("num_samples", 10000)),
                                "--sampler", str(sample_cfg.get("sampler", "ddim")),
                                "--steps", str(sample_cfg.get("steps", 100)),
                                "--batch-size", str(sample_cfg.get("batch_size", 256)),
                                "--sample-seed", str(sample_cfg.get("seed", 12345)),
                            ),
                            "run_dir": str(run_dir),
                            "exp": exp,
                            "epoch": epoch,
                        },
                        status="pending",
                        attempts=0,
                    )
                )

                if not real_images:
                    continue
                jobs.append(
                    Job(
                        job_id=f"{name}:eval:{exp}:{epoch}",
                        pipeline=name,
                        kind="eval",
                        group_id=group,
                        tier=tier,
                        depends_on=sample_id,
                        payload={
                            "argv": _cli(
                                "eval-samples",
                                "--run-dir", str(run_dir),
                                "--samples", str(samples_dir),
                                "--real", str(real_images),
                                "--epoch", str(epoch),
                                "--ledger", ledger,
                            ),
                            "run_dir": str(run_dir),
                            "exp": exp,
                            "epoch": epoch,
                        },
                        status="pending",
                        attempts=0,
                    )
                )

    logger.info("manifest %s expands to %d jobs", name, len(jobs))
    return jobs
