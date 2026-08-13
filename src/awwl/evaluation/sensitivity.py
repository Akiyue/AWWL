"""How large a spectral deviation must be before FID notices it.

The replication established that a frequency-aware loss measurably corrects
the sample power spectrum — roughly 0.5 dB in the high band — while FID, IS
and KID do not move. That is an observation. The obvious reviewer question is
the one it does not answer: *how much should FID have moved?*

This measures that directly, and needs no trained model. Take real images,
attenuate their high-frequency content by a known amount, and score them
against a disjoint set of real images. The result is a calibration curve:
FID as a function of spectral deviation in dB.

With it, the argument becomes causal rather than circumstantial. If FID only
departs from its finite-sample floor beyond some threshold, then any method
whose spectral correction is smaller than that threshold **cannot** show a
gain on FID, whatever else it does right. That statement covers every
frequency-aware objective, not one paper's loss.

The attenuation targets the same band the reporting uses (the top third of
radial frequency). The requested dB is only a knob: what goes on the curve's
x-axis is the deviation **measured back** from the filtered images, because a
gentle transition deliberately spreads the attenuation and a requested 4 dB
does not land at 4 dB.

**Quantisation floor.** Rungs below roughly 1 dB are not resolvable. At
0 dB the mask is exactly 1, so the FFT round-trip returns the original
integers untouched; at any non-zero attenuation the result is fractional and
rounding to 8-bit adds white noise, which lifts the measured high band by
~0.7 dB regardless of how little was actually removed. That is why 0.25 dB
and 0.5 dB both come back as 0.73 dB with identical FID. Read the curve from
its reliable rungs (>= ~1.7 dB) and extrapolate with the fitted power law;
to probe below the floor, filter model samples directly with ``source=``
instead, where the comparison is against real images and no round-trip is
needed on the reference side.

One honest caveat, which belongs in the paper rather than in a footnote.
Removing high-frequency energy from an image leaves a spatial signature, and
an FFT filter's signature is not the same as a diffusion model's
under-production. On a hard step edge this filter overshoots by ~0.2% of the
value range at 0.5 dB and ~3% at 8 dB; natural images have softer edges and
fare better, but some of any measured FID rise is attributable to the filter
rather than to the missing energy alone. The curve therefore calibrates *a*
spectral deficit, not *the* one a model produces — which makes it a
conservative bound: the real threshold is at least this high.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from awwl.evaluation.spectrum import band_deviations, radial_profile
from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)

_VALID_EXTS = (".png", ".jpg", ".jpeg")


@dataclass
class SensitivityPoint:
    """One rung of the calibration curve."""

    requested_db: float
    measured_db: float
    metrics: dict[str, float] = field(default_factory=dict)


def _radial_gain(shape: tuple[int, int], *, db: float, cutoff: float, width: float) -> np.ndarray:
    """Multiplicative FFT mask attenuating above ``cutoff`` by ``db`` decibels.

    The edge is a raised cosine rather than a step, and a wide one. Measured on
    a step edge, widening the transition from 0.15 to 0.5 halves the Gibbs
    overshoot, while the shape of the edge alone buys only ~7% over a hard
    cutoff — so the width does the work, not the cosine.

    The cost is that the attenuation is no longer confined to the top third,
    so the achieved deviation falls short of ``db``. That is fine and
    deliberate: callers report the deviation they measure back from the
    filtered images, never the value requested here.
    """
    h, w = shape
    ys, xs = np.indices((h, w))
    cy, cx = h // 2, w // 2
    # Normalise so 1.0 is the Nyquist radius along the shorter axis.
    radius = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / (min(h, w) / 2.0)

    ramp = np.clip((radius - cutoff) / max(width, 1e-6), 0.0, 1.0)
    ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)  # smoothstep in [0, 1]
    return np.power(10.0, -(db / 20.0) * ramp)


def attenuate_folder(
    source: str | Path,
    destination: str | Path,
    *,
    db: float,
    files: list[Path] | None = None,
    cutoff: float = 0.667,
    width: float = 0.5,
    dither: float = 0.5,
) -> Path:
    """Write copies of ``source`` images with the high band scaled by ``db``.

    Negative ``db`` boosts rather than attenuates.

    Args:
        dither: Uniform noise in ±``dither`` least-significant bits added
            before rounding to 8-bit.

    Dither is on by default and it is not cosmetic. At ``db=0`` the mask is
    exactly 1, so the FFT round-trip returns the original integers and no
    rounding occurs; at any other rung the result is fractional and rounding
    injects white noise. White noise is flat, so it lands mostly in the high
    band — the very quantity being manipulated. Without dither the zero rung
    is the only one with a clean spectrum, and part of every measured change
    is quantisation rather than the filter. Dithering every rung identically,
    zero included, makes the comparison isolate the spectral change.
    """
    out = ensure_dir(destination)
    paths = files if files is not None else _list_images(source)
    # Fixed seed: the dither must not be a source of run-to-run variation in
    # a measurement whose whole purpose is resolving small differences.
    rng = np.random.default_rng(0)

    gain: np.ndarray | None = None
    for index, path in enumerate(tqdm(paths, desc=f"filter {db:g}dB", leave=False)):
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
        if gain is None:
            gain = _radial_gain(image.shape[:2], db=db, cutoff=cutoff, width=width)

        channels = []
        for c in range(3):
            spectrum = np.fft.fftshift(np.fft.fft2(image[:, :, c]))
            filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * gain))
            channels.append(np.real(filtered))
        stacked = np.stack(channels, axis=-1)
        if dither:
            stacked = stacked + rng.uniform(-dither, dither, stacked.shape)
        stacked = np.clip(np.round(stacked), 0, 255).astype(np.uint8)
        Image.fromarray(stacked).save(out / f"{index:06d}.png")
    return out


def _list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in _VALID_EXTS)


def split_reference(
    folder: str | Path, *, count: int
) -> tuple[list[Path], list[Path]]:
    """Two disjoint halves of a real-image folder.

    Disjoint matters. Scoring filtered images against the very images they came
    from would put the zero rung at an unreachably low FID and make every other
    rung look enormous by comparison. Two independent real samples give the
    honest finite-sample floor that the curve should be read against.
    """
    files = _list_images(folder)
    needed = 2 * count
    if len(files) < needed:
        raise ValueError(
            f"need {needed} images to build two disjoint sets of {count}, found {len(files)}"
        )
    # Interleave rather than slice: consecutive files can share a class or
    # capture condition, and a contiguous split would make the two halves
    # differ for reasons unrelated to filtering.
    return files[0:needed:2], files[1:needed:2]


def measure_sensitivity(
    *,
    real_folder: str | Path,
    work_dir: str | Path,
    deltas: list[float],
    count: int = 10000,
    keep_images: bool = False,
    advanced: bool = True,
    source: str | Path | None = None,
) -> list[SensitivityPoint]:
    """Score progressively high-frequency-attenuated images.

    Args:
        real_folder: Reference images. Without ``source`` this is also the
            origin of the filtered set, split into two disjoint halves.
        work_dir: Scratch space for the filtered copies.
        deltas: Attenuations in dB. Negative values *boost* the high band,
            which is how a model's own samples can be given the correction a
            frequency-aware loss achieves and scored for it.
        count: Images per side.
        keep_images: Retain the filtered folders for inspection.
        advanced: Also compute KID / precision / recall (slower).
        source: Filter this folder instead of half of ``real_folder``. Use a
            model's sample folder to measure sensitivity **at the operating
            point** — the calibration on real images sits at FID ~5, while
            models sit far higher, and FID's response to an added perturbation
            need not be the same at both distances.
    """
    from awwl.evaluation.advanced_metrics import compute_advanced_metrics
    from awwl.evaluation.fid_is import compute_fid_is

    work = ensure_dir(work_dir)

    if source is not None:
        reference_dir = Path(real_folder)
        test_files = _list_images(source)[:count]
        logger.info("filtering %d images from %s against %s", len(test_files), source, reference_dir)
    else:
        reference_files, test_files = split_reference(real_folder, count=count)
        reference_dir = work / "reference"
        if not reference_dir.exists() or len(list(reference_dir.glob("*.png"))) < count:
            logger.info("materialising the reference half (%d images)", count)
            attenuate_folder(real_folder, reference_dir, db=0.0, files=reference_files)

    reference_profile = radial_profile(reference_dir, max_images=2000)

    points: list[SensitivityPoint] = []
    for db in deltas:
        folder = work / f"delta_{db:g}dB".replace(".", "p")
        logger.info("=== %.3g dB", db)
        attenuate_folder(source or real_folder, folder, db=db, files=test_files)

        # Verify the filter did what was asked, in the same band the study
        # reports; a requested value that lands elsewhere moves the threshold.
        measured = 0.0
        profile = radial_profile(folder, max_images=2000)
        if profile is not None and reference_profile is not None:
            measured = -band_deviations(profile, reference_profile)[-1]

        metrics = dict(
            compute_fid_is(fake_folder=folder, real_folder=reference_dir, exp_name=f"{db:g}dB")
        )
        if advanced:
            metrics.update(
                compute_advanced_metrics(
                    real_folder=reference_dir, fake_folder=folder, exp_name=f"{db:g}dB",
                    max_images=count,
                )
            )
        points.append(SensitivityPoint(requested_db=db, measured_db=measured, metrics=metrics))
        logger.info("%.3g dB requested, %.3g dB measured, FID %.3f", db, measured, metrics.get("fid", -1))

        if not keep_images:
            for f in folder.glob("*.png"):
                f.unlink()
            folder.rmdir()
    return points


def format_sensitivity_table(
    points: list[SensitivityPoint],
    *,
    reference_effect_db: float | None = None,
) -> str:
    """Render the calibration curve, and locate an effect size on it.

    Args:
        reference_effect_db: The spectral correction a method actually
            achieves. Marking it on the curve is the whole point — it converts
            "FID did not move" into "FID could not have moved".
    """
    if not points:
        return "no measurements"

    floor = points[0].metrics.get("fid", float("nan"))
    header = (
        f"{'requested':>10}  {'measured':>10}  {'FID':>9}  {'ΔFID':>9}  "
        f"{'IS':>7}  {'KID':>9}"
    )
    lines = [
        "FID response to a controlled high-frequency deficit in *real* images",
        "(both sides are real photographs, so any rise is caused by the filter alone)",
        "",
        header,
        "-" * len(header),
    ]
    for point in points:
        fid = point.metrics.get("fid", float("nan"))
        lines.append(
            f"{point.requested_db:>9.3g}d  {point.measured_db:>9.3g}d  {fid:>9.3f}  "
            f"{fid - floor:>+9.3f}  {point.metrics.get('is_mean', float('nan')):>7.3f}  "
            f"{point.metrics.get('kid_tf', point.metrics.get('kid', float('nan'))):>9.5f}"
        )

    lines.append("")
    lines.append(f"floor (0 dB, two disjoint real sets): FID {floor:.3f}")

    if reference_effect_db is not None:
        below = [p for p in points if p.measured_db <= reference_effect_db]
        bracket = below[-1] if below else points[0]
        lines.append("")
        lines.append(
            f"a method correcting {reference_effect_db:g} dB sits at or below the "
            f"{bracket.measured_db:.3g} dB rung, where ΔFID is "
            f"{bracket.metrics.get('fid', float('nan')) - floor:+.3f}."
        )
        lines.append(
            "If that is within the floor's own noise, no frequency-aware method "
            "operating at this scale can register on FID — which is a statement "
            "about the metric, not about any one loss."
        )
    return "\n".join(lines)
