"""Radial FFT power-spectrum profiling.

Replaces the analysis half of ``AWWL-Diff/spectrum_plot.py`` (the plotting
half lives under :mod:`awwl.plotting.spectrum`).

The scalar ``spec_dist`` reported by
:func:`awwl.evaluation.advanced_metrics.compute_advanced_metrics` says *how
far* a model's spectrum sits from the real one, but not *where*. That
distinction decides whether a frequency-aware loss is doing what it claims:
the whole premise is a correction at **high** frequencies, so an improvement
concentrated at low frequencies would mean the mechanism is real but is not
the advertised one. :func:`band_deviations` splits the profile into low / mid
/ high thirds so the claim can be checked rather than assumed.

Profiles are in **decibels** (``20·log10|F|``). Earlier revisions used a
natural log, which is 2.303x larger and is not dB; ``spec_dist`` values
recorded in the results ledger before this change are in those legacy units
and are 5.3x larger (the metric squares the profile difference). Do not mix
the two in one table — re-score with ``FULL=1 bash scripts/reeval_samples.sh``
to bring old runs onto the dB scale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

_VALID_EXTS = (".png", ".jpg", ".jpeg")


def radial_profile(folder: str | Path, *, max_images: int = 10000) -> np.ndarray | None:
    """Average radially-binned FFT log-magnitude across images in ``folder``.

    Returns:
        A 1-D NumPy array in **decibels** (``20·log10|F|``) of length
        ``min(image_height, image_width)//2 + 1``, or ``None`` if the folder
        is empty.
    """
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.suffix.lower() in _VALID_EXTS][:max_images]
    if not files:
        return None

    profiles: list[np.ndarray] = []
    for f in tqdm(files, desc=f"fft {folder.name}", leave=False):
        try:
            data = np.array(Image.open(f).convert("L"))
        except Exception as exc:
            logger.warning("could not read %s: %s", f, exc)
            continue
        fft = np.fft.fftshift(np.fft.fft2(data))
        # 20*log10 = decibels. The original used a natural log, which is
        # 2.303x larger and is not dB -- a reader seeing '20 log' assumes
        # dB and would read the effect as more than twice its real size.
        psd = 20 * np.log10(np.abs(fft) + 1e-8)

        h, w = data.shape
        ys, xs = np.indices((h, w))
        center = np.array([h // 2, w // 2])
        r = np.sqrt((xs - center[1]) ** 2 + (ys - center[0]) ** 2).astype(int)
        tbin = np.bincount(r.ravel(), psd.ravel())
        nr = np.bincount(r.ravel())
        profiles.append(tbin / (nr + 1e-8))

    if not profiles:
        return None
    min_len = min(len(p) for p in profiles)
    profiles = [p[:min_len] for p in profiles]
    return np.mean(profiles, axis=0)


def band_deviations(
    profile: np.ndarray,
    real: np.ndarray,
    *,
    bands: int = 3,
) -> list[float]:
    """Mean signed deviation from ``real`` in ``bands`` equal frequency bins.

    Returns one value per bin, ordered low → high frequency. Signed, not
    absolute: the sign says whether the model has *too little* energy at that
    frequency (negative — the usual over-smoothing failure) or too much.
    """
    n = min(len(profile), len(real))
    delta = np.asarray(profile[:n], dtype=float) - np.asarray(real[:n], dtype=float)
    edges = np.linspace(0, n, bands + 1).astype(int)
    return [float(delta[a:b].mean()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:], strict=True)]


def profiles_by_config(
    root: str | Path,
    *,
    configs: list[str],
    seeds: list[int],
    epoch: int,
    max_images: int = 2000,
) -> dict[str, np.ndarray]:
    """Radial profile per configuration, averaged over its seeds.

    Averaging profiles across seeds rather than pooling all images keeps each
    seed weighted equally, matching how every other metric in the study is
    aggregated.
    """
    root = Path(root)
    out: dict[str, np.ndarray] = {}
    for config in configs:
        per_seed: list[np.ndarray] = []
        for seed in seeds:
            folder = root / f"{config}_s{seed}" / "samples" / f"ep{epoch}"
            if not folder.is_dir():
                logger.warning("no samples for %s seed %s at %s", config, seed, folder)
                continue
            profile = radial_profile(folder, max_images=max_images)
            if profile is not None:
                per_seed.append(profile)
        if not per_seed:
            logger.warning("no usable samples for config %s; skipping", config)
            continue
        min_len = min(len(p) for p in per_seed)
        out[config] = np.mean([p[:min_len] for p in per_seed], axis=0)
        logger.info("%s: averaged %d seed profile(s)", config, len(per_seed))
    return out


def format_band_table(
    real: np.ndarray,
    profiles: dict[str, np.ndarray],
    *,
    bands: int = 3,
) -> str:
    """Render per-band spectral deviation as a fixed-width table."""
    if not profiles:
        return "no profiles to report"
    names = ["low", "mid", "high"] if bands == 3 else [f"b{i}" for i in range(bands)]
    width = max(len(c) for c in profiles)
    header = f"{'config':<{width}}  " + "  ".join(f"{n:>10}" for n in names) + f"  {'|total|':>10}"
    lines = [
        "signed deviation from the real spectrum (0 = match; negative = too little energy)",
        header,
        "-" * len(header),
    ]
    for config, profile in profiles.items():
        deltas = band_deviations(profile, real, bands=bands)
        total = sum(abs(d) for d in deltas)
        cells = "  ".join(f"{d:>+10.3f}" for d in deltas)
        lines.append(f"{config:<{width}}  {cells}  {total:>10.3f}")
    lines.append("")
    lines.append(
        "A frequency-aware loss claims to correct the HIGH band. If its gain sits "
        "in the low band instead, the effect is real but is not the advertised "
        "mechanism."
    )
    return "\n".join(lines)
