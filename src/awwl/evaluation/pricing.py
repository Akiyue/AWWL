"""Price a spectral correction on a DreamBooth run.

The CIFAR-10 half of this argument asks: the objective corrects the spectrum by
some amount, so what is that amount worth? It answers by applying the identical
correction to the baseline's own samples as a Fourier post-process and reading
the change in FID.

DreamBooth is scored by CLIP rather than FID, so the same question needs its
own path -- but it is the same question, and answering it on a second task is
what stops the paper's central claim resting on one dataset.

The correction is measured, not assumed: the deficit of a run's samples against
the instance images it was trained on is what sets the boost, so each arm is
priced at the correction it actually achieves.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from awwl.evaluation.sensitivity import attenuate_folder
from awwl.evaluation.spectrum import band_deviations, radial_profile
from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class PricedRun:
    """One run's spectrum and metrics, before and after the post-process."""

    exp: str
    deficit_db: float
    boosted_deficit_db: float
    clip_score: float
    clip_score_boosted: float
    similarity: float
    similarity_boosted: float
    n_images: int
    boost_db: float

    def as_row(self) -> dict[str, Any]:
        return {
            "exp": self.exp,
            "kind": "pricing",
            "deficit_db": self.deficit_db,
            "boosted_deficit_db": self.boosted_deficit_db,
            "clip_score": self.clip_score,
            "clip_score_boosted": self.clip_score_boosted,
            "similarity": self.similarity,
            "similarity_boosted": self.similarity_boosted,
            "n_images": self.n_images,
            "boost_db": self.boost_db,
        }


def _prompt_dirs(run_dir: Path) -> list[Path]:
    return sorted(p for p in (run_dir / "samples").glob("prompt*") if p.is_dir())


def high_band_deficit(sample_dirs: list[Path], real_dir: Path, *, max_images: int) -> float:
    """Signed high-band deviation of these samples from the real subject images.

    Negative means the samples carry too little high-frequency energy, which is
    the failure the objective under audit sets out to correct. Returned as a
    positive deficit in dB for readability, matching how the CIFAR-10 side
    reports it.
    """
    real = radial_profile(real_dir, max_images=max_images)
    if real is None:
        raise ValueError(f"no images to profile in {real_dir}")

    profiles = [radial_profile(d, max_images=max_images) for d in sample_dirs]
    usable = [p for p in profiles if p is not None]
    if not usable:
        raise ValueError("no sample images to profile")

    # Average the per-prompt profiles: the three prompts are the same model
    # measured on different content, not three different models.
    n = min(min(len(p) for p in usable), len(real))
    mean_profile = sum(p[:n] for p in usable) / len(usable)
    _, _, high = band_deviations(mean_profile, real[:n], bands=3)
    return -float(high)


def price_run(
    run_dir: str | Path,
    *,
    prompts: list[str],
    score_folder,
    real_dir: str | Path | None = None,
    boost_db: float | None = None,
    work: str | Path = "runs/db_pricing",
    max_images: int = 10000,
) -> PricedRun:
    """Measure this run's correction, then buy it back on its own samples.

    Args:
        score_folder: ``(folder, prompt) -> (clip_scores, similarities)``. Passed
            in rather than imported so the CLIP model is loaded once by the
            caller across many runs, and so this is testable without weights.
        boost_db: Correction to apply. Defaults to the deficit this run
            actually shows, which is the quantity the question is about.

    Returns:
        The run's spectrum and both metrics, before and after.
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    real = Path(real_dir or cfg["data"]["instance_data_dir"])
    if not real.is_dir():
        raise ValueError(f"reference images not found: {real}")

    dirs = _prompt_dirs(run_dir)
    if not dirs:
        raise ValueError(f"no sample folders under {run_dir / 'samples'}")
    if len(dirs) != len(prompts):
        raise ValueError(
            f"{len(dirs)} sample folder(s) against {len(prompts)} prompt(s); "
            "each prompt directory must have the prompt that produced it"
        )

    deficit = high_band_deficit(dirs, real, max_images=max_images)
    boost = float(boost_db) if boost_db is not None else deficit
    logger.info("%s: high-band deficit %.3f dB, boosting by %.3f dB", run_dir.name, deficit, boost)

    work = ensure_dir(Path(work) / run_dir.name)
    boosted_dirs = [
        # Negative dB boosts. Dither is on by default and matters here for the
        # same reason it does on CIFAR-10: without it the unboosted arm is the
        # only one with a clean spectrum.
        attenuate_folder(d, work / d.name, db=-boost)
        for d in dirs
    ]

    clip_plain: list[float] = []
    clip_boost: list[float] = []
    sim_plain: list[float] = []
    sim_boost: list[float] = []
    for plain, boosted, prompt in zip(dirs, boosted_dirs, prompts, strict=True):
        c, s = score_folder(plain, prompt)
        clip_plain += c
        sim_plain += s
        c, s = score_folder(boosted, prompt)
        clip_boost += c
        sim_boost += s

    mean = lambda xs: float(sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
    return PricedRun(
        exp=run_dir.name,
        deficit_db=deficit,
        boosted_deficit_db=high_band_deficit(boosted_dirs, real, max_images=max_images),
        clip_score=mean(clip_plain),
        clip_score_boosted=mean(clip_boost),
        similarity=mean(sim_plain),
        similarity_boosted=mean(sim_boost),
        n_images=len(clip_plain),
        boost_db=boost,
    )


def format_pricing_table(runs: list[PricedRun]) -> str:
    """One row per run: what the correction was, and what it bought."""
    if not runs:
        return "no runs priced"

    header = (
        f"{'run':<24} {'deficit':>9} {'boosted':>9} "
        f"{'CLIP':>8} {'CLIP+':>8} {'dCLIP':>8} "
        f"{'sim':>8} {'sim+':>8} {'dsim':>8}"
    )
    lines = [
        "What the spectral correction is worth on DreamBooth.",
        "'+' is the same samples after the Fourier post-process; d is the change.",
        "",
        header,
        "-" * len(header),
    ]
    for r in runs:
        lines.append(
            f"{r.exp:<24} {r.deficit_db:>8.3f}d {r.boosted_deficit_db:>8.3f}d "
            f"{r.clip_score:>8.4f} {r.clip_score_boosted:>8.4f} "
            f"{r.clip_score_boosted - r.clip_score:>+8.4f} "
            f"{r.similarity:>8.4f} {r.similarity_boosted:>8.4f} "
            f"{r.similarity_boosted - r.similarity:>+8.4f}"
        )
    lines += [
        "",
        "Read it the way the CIFAR-10 table is read: if post-processing the",
        "baseline's samples reaches what the trained objective reaches, the",
        "training-time machinery is not paying for itself on this task either.",
    ]
    return "\n".join(lines)
