"""Price a spectral intervention in FID, without quantising the intervention.

The pricing test applies an objective's stated mechanism to the baseline's own
samples and scores the result. Two details of how that was first done bias the
answer, and this module removes both.

**Quantisation.** A Fourier filter returns fractional pixel values. Rounding
them back to 8-bit, with or without dither, adds a white error field, and a
white field lands mostly in the high band -- the quantity being manipulated.
:mod:`awwl.evaluation.sensitivity` documents the size of it: on real CIFAR-10
images rounding alone lifts the measured high band by about 0.7 dB. Here the
filtered images stay in float32 and go straight into clean-fid's own resizer
and Inception extractor, which already work in float (clean-fid's "clean" mode
converts every image to float32 before resizing). For an unmodified folder the
features, and so the FID, are identical to ``cleanfid.compute_fid``'s; the
test suite checks that on a real run.

**Subsampling.** The spectral statistic was computed from the first 2000 files
in filesystem order. Here every sample and every reference image is used, in
sorted order, so the statistic is a property of the set rather than of how the
directory happens to be laid out.

The statistic ``s`` is the one the paper reports: per image, the luminance
(ITU-R 601-2 weights, as PIL's ``"L"`` but without rounding) is Fourier
transformed, ``20 log10 |F|`` is averaged within integer-radius bins
``r = floor(sqrt(dx^2 + dy^2))`` about the ``fftshift`` centre, and the
profiles are averaged over images. Bins are split by index into three equal
thirds; on 32x32 that is radii 0-6 (low), 7-14 (mid) and 15-22 (high).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_EXTS = (".png", ".jpg", ".jpeg")
#: ITU-R 601-2 luma weights -- the ones PIL's ``convert("L")`` uses.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


# --------------------------------------------------------------------------
# Images


def list_images(folder: str | Path) -> list[Path]:
    """Image files in ``folder``, sorted by name so the order is reproducible."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _EXTS)


def load_images(folder: str | Path, *, limit: int | None = None) -> np.ndarray:
    """Load a folder as a ``(N, H, W, 3)`` uint8 array, in sorted file order."""
    from PIL import Image

    files = list_images(folder)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    first = np.asarray(Image.open(files[0]).convert("RGB"))
    out = np.empty((len(files), *first.shape), dtype=np.uint8)
    out[0] = first
    for i, f in enumerate(files[1:], start=1):
        out[i] = np.asarray(Image.open(f).convert("RGB"))
    return out


# --------------------------------------------------------------------------
# The spectral statistic


def radial_bins(h: int, w: int) -> np.ndarray:
    """Integer radius of every FFT coefficient about the ``fftshift`` centre."""
    ys, xs = np.indices((h, w))
    return np.sqrt((xs - w // 2) ** 2 + (ys - h // 2) ** 2).astype(int)


def radial_profile(images: np.ndarray, *, batch: int = 1024) -> np.ndarray:
    """Mean over images of the radially binned luminance log-magnitude, in dB.

    ``images`` is ``(N, H, W, 3)`` in [0, 255], uint8 or float. Returns one
    value per integer radius.
    """
    n, h, w, _ = images.shape
    bins = radial_bins(h, w).ravel()
    counts = np.bincount(bins)
    total = np.zeros(len(counts), dtype=np.float64)
    for start in range(0, n, batch):
        x = images[start : start + batch].astype(np.float64) @ LUMA
        spec = np.fft.fftshift(np.fft.fft2(x), axes=(-2, -1))
        db = 20.0 * np.log10(np.abs(spec) + 1e-8).reshape(len(x), -1)
        # Per image: mean within each radius; then summed over the batch.
        per_bin = np.stack([np.bincount(bins, row, minlength=len(counts)) for row in db])
        total += (per_bin / counts).sum(axis=0)
    return total / n


def band_deviations(profile: np.ndarray, reference: np.ndarray, *, bands: int = 3) -> list[float]:
    """Mean signed deviation from ``reference`` in ``bands`` equal index bins."""
    m = min(len(profile), len(reference))
    delta = np.asarray(profile[:m]) - np.asarray(reference[:m])
    edges = np.linspace(0, m, bands + 1).astype(int)
    return [float(delta[a:b].mean()) for a, b in zip(edges[:-1], edges[1:], strict=True)]


# --------------------------------------------------------------------------
# Operators. Each returns float32 images; ``finalize`` decides what is scored.


def raised_cosine_gain(h: int, w: int, *, db: float, cutoff: float = 0.667, width: float = 0.5) -> np.ndarray:
    """The band boost: a smooth gain rising to ``db`` decibels above ``cutoff``.

    Identical in shape to :func:`awwl.evaluation.sensitivity._radial_gain`
    (radius normalised so 1.0 is the Nyquist radius of the shorter axis, a
    raised-cosine ramp of ``width`` starting at ``cutoff``), with the sign
    reversed so positive ``db`` boosts.
    """
    ys, xs = np.indices((h, w))
    radius = np.sqrt((xs - w // 2) ** 2 + (ys - h // 2) ** 2) / (min(h, w) / 2.0)
    ramp = np.clip((radius - cutoff) / max(width, 1e-6), 0.0, 1.0)
    ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)
    return np.power(10.0, (db / 20.0) * ramp)


def profile_gain(target_db: np.ndarray, source_db: np.ndarray, h: int, w: int) -> np.ndarray:
    """A 2-D gain that moves a set with profile ``source_db`` onto ``target_db``.

    One multiplicative gain per integer radius, ``10 ** ((target - source) / 20)``,
    the same for every image and every colour channel.
    """
    m = min(len(target_db), len(source_db))
    per_bin = np.power(10.0, (np.asarray(target_db[:m]) - np.asarray(source_db[:m])) / 20.0)
    bins = np.clip(radial_bins(h, w), 0, m - 1)
    return per_bin[bins]


def apply_gain(images: np.ndarray, gain: np.ndarray, *, batch: int = 1024) -> np.ndarray:
    """Multiply every channel of every image's centred spectrum by ``gain``."""
    out = np.empty(images.shape, dtype=np.float32)
    g = np.fft.ifftshift(gain)
    for start in range(0, len(images), batch):
        x = images[start : start + batch].astype(np.float64)
        f = np.fft.fft2(x, axes=(1, 2)) * g[None, :, :, None]
        out[start : start + batch] = np.real(np.fft.ifft2(f, axes=(1, 2)))
    return out


def finalize(images: np.ndarray, *, quantize: str, seed: int = 0) -> np.ndarray:
    """Clip to the valid range and, optionally, quantise as an 8-bit file would.

    ``"float"`` clips only. ``"round"`` also rounds. ``"dither"`` adds
    uniform noise in +/-0.5 LSB before rounding, with a fixed seed -- the
    policy of :func:`awwl.evaluation.sensitivity.attenuate_folder`, kept so the
    original measurement can be reproduced and its quantisation share isolated.
    """
    x = np.clip(images.astype(np.float32), 0.0, 255.0)
    if quantize == "float":
        return x
    if quantize == "dither":
        rng = np.random.default_rng(seed)
        x = x + rng.uniform(-0.5, 0.5, x.shape).astype(np.float32)
    elif quantize != "round":
        raise ValueError(f"unknown quantize policy {quantize!r}")
    return np.clip(np.rint(x), 0.0, 255.0).astype(np.float32)


# --------------------------------------------------------------------------
# Scoring


class _ArrayDataset:
    """clean-fid's ResizeDataset, over an in-memory array instead of files."""

    def __init__(self, images: np.ndarray):
        from cleanfid.resize import build_resizer

        self.images = images
        self.fn_resize = build_resizer("clean")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        import torch

        # clean mode casts to float32 before resizing, so a uint8 image and
        # the same values as float32 give bit-identical output.
        resized = self.fn_resize(np.asarray(self.images[i], dtype=np.float32))
        return torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))


@dataclass
class Scorer:
    """Inception features, FID and KID, exactly as clean-fid computes them."""

    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8

    def __post_init__(self) -> None:
        import torch
        from cleanfid.features import build_feature_extractor

        self._torch_device = torch.device(self.device)
        self.model = build_feature_extractor("clean", self._torch_device, use_dataparallel=False)

    def features(self, images: np.ndarray) -> np.ndarray:
        import torch

        loader = torch.utils.data.DataLoader(
            _ArrayDataset(images), batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=self.num_workers,
        )
        feats = []
        with torch.no_grad():
            for batch in loader:
                feats.append(self.model(batch.to(self._torch_device)).detach().cpu().numpy())
        return np.concatenate(feats)

    def folder_features(self, folder: str | Path) -> np.ndarray:
        """Features of a folder through clean-fid's own file path."""
        from cleanfid import fid as cleanfid

        return cleanfid.get_folder_features(
            str(folder), self.model, num_workers=self.num_workers,
            batch_size=self.batch_size, device=self._torch_device, mode="clean", verbose=False,
        )


def fid_from_features(feats: np.ndarray, ref: np.ndarray) -> float:
    from cleanfid import fid as cleanfid

    from awwl.evaluation.fid_is import _patch_scipy_sqrtm_disp

    _patch_scipy_sqrtm_disp()
    return float(cleanfid.fid_from_feats(feats, ref))


def kid_from_features(feats: np.ndarray, ref: np.ndarray) -> float:
    from cleanfid import fid as cleanfid

    return float(cleanfid.kernel_distance(feats, ref))


class DinoScorer:
    """Frechet distance in DINOv2 ViT-L/14 features (FD_DINOv2).

    Stein et al. (2023) recommend DINOv2 over Inception-V3 as the feature space
    for evaluating generative models; pricing in both spaces shows which
    conclusions are properties of Inception features. Images are resized to
    224x224 bicubically in floating point (the same path for original and
    filtered images, so no re-quantisation enters either), normalised with the
    ImageNet mean and standard deviation, and represented by the model's
    normalised CLS token. Absolute values therefore differ slightly from the
    reference implementation, which resizes 8-bit PIL images; comparisons
    within this study do not.
    """

    def __init__(self, device: str = "cuda", batch_size: int = 128) -> None:
        import torch

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True)
        self.model = self.model.to(self.device).eval()
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def features(self, images: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        out = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                x = torch.from_numpy(np.asarray(images[start : start + self.batch_size], dtype=np.float32))
                x = x.permute(0, 3, 1, 2).to(self.device) / 255.0
                x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
                x = (x - self._mean) / self._std
                out.append(self.model(x).float().cpu().numpy())
        return np.concatenate(out)

    def folder_features(self, folder: str | Path) -> np.ndarray:
        return self.features(load_images(folder))
