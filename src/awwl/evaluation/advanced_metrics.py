"""KID + Precision/Recall + spectral distance evaluation.

Replaces ``AWWL-Diff/advanced.py``. Uses an InceptionV3 head to extract
2048-d features, then:

* KID: polynomial-kernel MMD between feature sets (manual implementation,
  no extra dependency beyond NumPy).
* Precision / Recall: via ``prdc.compute_prdc``.
* Radial spectral distance: averaged radial FFT magnitude of grayscale
  images, MSE between real and fake profiles.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

_VALID_EXTS = (".png", ".jpg")


def _list_images(folder: str | Path) -> list[Path]:
    """Sorted image paths in ``folder``, resolved so relative inputs are safe."""
    root = Path(folder).resolve()
    return sorted(p for p in root.iterdir() if p.suffix.lower() in _VALID_EXTS)


class _ImagePathDataset(Dataset):
    """Simple list-of-files dataset that resizes to 299×299 for Inception."""

    def __init__(self, folder: str | Path, *, max_imgs: int = 10000) -> None:
        # `iterdir()` already yields folder-prefixed paths. Re-joining them
        # onto the folder used to be harmless only because pathlib discards the
        # left operand when the right is absolute — with a *relative* folder it
        # produced 'data/x/data/x/00000.png' and a FileNotFoundError.
        self._files = _list_images(folder)[:max_imgs]
        self._transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._transform(Image.open(self._files[idx]).convert("RGB"))


def _extract_features(folder: str | Path, *, device: str, batch_size: int = 64, max_imgs: int = 10000) -> np.ndarray:
    """Run InceptionV3 (sans final FC) over ``folder`` and return ``(N, 2048)``."""
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    loader = DataLoader(_ImagePathDataset(folder, max_imgs=max_imgs), batch_size=batch_size, num_workers=4)
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"inception {Path(folder).name}", leave=False):
            feats.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _polynomial_kernel(X: np.ndarray, Y: np.ndarray, *, degree: int = 3, coef0: float = 1.0) -> np.ndarray:
    gamma = 1.0 / X.shape[1]
    return (X @ Y.T * gamma + coef0) ** degree


def _kid_mmd(X: np.ndarray, Y: np.ndarray) -> float:
    """Polynomial-kernel MMD^2, the unbiased KID estimator."""
    m, n = X.shape[0], Y.shape[0]
    K_xx = _polynomial_kernel(X, X)
    K_yy = _polynomial_kernel(Y, Y)
    K_xy = _polynomial_kernel(X, Y)
    return float(
        (np.sum(K_xx) - np.trace(K_xx)) / (m * (m - 1))
        + (np.sum(K_yy) - np.trace(K_yy)) / (n * (n - 1))
        - 2 * np.sum(K_xy) / (m * n)
    )


def _radial_spectrum(folder: str | Path, *, device: str, batch_size: int = 64, max_imgs: int = 2000) -> np.ndarray:
    """Average radial FFT magnitude profile, in decibels, over grayscale images."""
    files = _list_images(folder)[:max_imgs]
    accumulated: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    r_flat: torch.Tensor | None = None

    for i in tqdm(range(0, len(files), batch_size), desc="spectrum-fft", leave=False):
        batch = [np.array(Image.open(p).convert("L")) for p in files[i : i + batch_size]]
        tensor = torch.tensor(np.array(batch), dtype=torch.float32, device=device)
        fft = torch.fft.fftshift(torch.fft.fft2(tensor))
        # Decibels, matching awwl.evaluation.spectrum. A natural log here
        # would make the two implementations disagree by 2.303x.
        mag = 20 * torch.log10(torch.abs(fft) + 1e-8)

        if accumulated is None:
            h, w = tensor.shape[1:]
            ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            cy, cx = h // 2, w // 2
            r = torch.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).long()
            max_r = int(r.max().item()) + 1
            accumulated = torch.zeros(max_r, device=device)
            counts = torch.zeros(max_r, device=device)
            r_flat = r.reshape(-1)
            ones = torch.ones_like(r_flat, dtype=torch.float32)
        for j in range(tensor.shape[0]):
            accumulated.index_add_(0, r_flat, mag[j].reshape(-1))
            counts.index_add_(0, r_flat, ones)
    if accumulated is None or counts is None:
        return np.zeros(1, dtype=np.float32)
    return (accumulated / (counts + 1e-8)).cpu().numpy()


def compute_advanced_metrics(
    *,
    real_folder: str | Path,
    fake_folder: str | Path,
    log_file: str | Path | None = None,
    exp_name: str = "unknown",
    batch_size: int = 64,
    max_images: int = 10000,
) -> dict[str, float]:
    """Compute KID, Precision, Recall, spectral distance for one fake folder.

    Returns:
        Dict with keys ``kid``, ``precision``, ``recall``, ``spec_dist``.
        Failed metrics surface as ``-1.0``.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("extracting Inception features (real)…")
    real_feats = _extract_features(real_folder, device=device, batch_size=batch_size, max_imgs=max_images)
    logger.info("extracting Inception features (fake)…")
    fake_feats = _extract_features(fake_folder, device=device, batch_size=batch_size, max_imgs=max_images)

    kid = _kid_mmd(real_feats, fake_feats)

    precision = -1.0
    recall = -1.0
    try:
        import prdc

        metrics = prdc.compute_prdc(real_features=real_feats, fake_features=fake_feats, nearest_k=5)
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
    except Exception as exc:
        logger.error("PRDC failed: %s", exc)

    spec_dist = -1.0
    try:
        s_real = _radial_spectrum(real_folder, device=device, batch_size=batch_size)
        s_fake = _radial_spectrum(fake_folder, device=device, batch_size=batch_size)
        m = min(len(s_real), len(s_fake))
        spec_dist = float(np.mean((s_real[:m] - s_fake[:m]) ** 2))
    except Exception as exc:
        logger.error("spectral distance failed: %s", exc)

    if log_file is not None:
        _append_log(
            Path(log_file),
            exp_name=exp_name,
            kid=kid,
            precision=precision,
            recall=recall,
            spec_dist=spec_dist,
        )

    return {"kid": kid, "precision": precision, "recall": recall, "spec_dist": spec_dist}


def _append_log(
    path: Path, *, exp_name: str, kid: float, precision: float, recall: float, spec_dist: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            f.write(f"{'Experiment':<30} | {'KID':<10} | {'Prec':<10} | {'Recall':<10} | {'SpecDist':<10}\n")
            f.write("-" * 85 + "\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(
            f"{exp_name:<30} | {kid:.6f}   | {precision:.4f}     | {recall:.4f}     | {spec_dist:.6f}\n"
        )
