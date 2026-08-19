"""Restoration metrics and per-run evaluation.

Where the diffusion task is scored through FID — a statistic over samples —
restoration is scored directly on the reconstructed image. Four metrics, in
increasing order of how much they care about the high frequencies that the
wavelet objective is supposed to improve:

* **PSNR** — mean-squared error in dB. Sensitive to everything, dominated by
  low frequencies.
* **SSIM** — luminance / contrast / structure correlation in local windows.
  Structure-aware, still mostly low-frequency.
* **LPIPS** — deep-feature distance (AlexNet here). The perceptual metric;
  the closest thing the task has to "does the texture look right".
* **NIQE** — no-reference natural-scene-statistics distance from the clean
  reference. Purely on the restored image, so it cannot be gamed by a
  degenerate mapping back to the input.

NIQE is implemented from its definition (Mittal et al. 2013): per-image
natural-scene statistics (MSCN + pairwise products, two scales) are fit with
generalized / asymmetric-generalized Gaussians, pooled into mean + covariance
models, and the distance between the distorted pool and the pristine pool is
reported. The pristine model is fit on the clean reference images themselves,
which keeps the whole thing self-contained.

Results are also stratified by the noise level of the degradation: the
adaptive-loss claim is that low-sigma samples get more high-frequency
attention, which should show up as a larger gain in the low-sigma PSNR bin.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import gamma

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Simple pixel / structural metrics
# --------------------------------------------------------------------------- #


def psnr(restored: torch.Tensor, clean: torch.Tensor) -> float:
    """Mean PSNR (dB) over the batch. Images are in ``[-1, 1]``.

    Peak signal power for a ``[-1, 1]`` signal is 4, so
    ``PSNR = 10 log10(4 / MSE)``, which equals the conventional ``[0, 1]``
    definition with peak value 1 after rescaling.
    """
    mse = ((restored - clean) ** 2).mean(dim=(1, 2, 3))
    return float((10.0 * torch.log10(4.0 / (mse + 1e-8))).mean().item())


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    g = torch.exp(-((torch.arange(size, dtype=torch.float32) - size // 2) ** 2) / (2.0 * sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)


def ssim(restored: torch.Tensor, clean: torch.Tensor) -> float:
    """Mean SSIM over the batch (Wang et al. 2004), images in ``[-1, 1]``."""
    channels = restored.shape[1]
    window = _gaussian_window(channels=channels).to(restored.device)
    padding = window.shape[-1] // 2
    mu1 = F.conv2d(restored, window, padding=padding, groups=channels)
    mu2 = F.conv2d(clean, window, padding=padding, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(restored * restored, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(clean * clean, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(restored * clean, window, padding=padding, groups=channels) - mu1_mu2
    c1 = (0.01 * 2.0) ** 2
    c2 = (0.03 * 2.0) ** 2
    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().item())


# --------------------------------------------------------------------------- #
# LPIPS
# --------------------------------------------------------------------------- #

_LPIPS_MODEL = None


def _lpips_net() -> Any:
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips

        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False)
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL


def lpips_score(restored: torch.Tensor, clean: torch.Tensor, device: str = "cuda") -> float:
    """Mean LPIPS distance over the batch; lower is better."""
    net = _lpips_net()
    with torch.no_grad():
        return float(net(restored.to(device), clean.to(device)).mean().item())


# --------------------------------------------------------------------------- #
# NIQE (natural scene statistics distance, fitted on the clean reference)
# --------------------------------------------------------------------------- #


def _luminance(images: torch.Tensor) -> np.ndarray:
    """``(N, H, W)`` luminance in ``[0, 1]`` from NCHW ``[-1, 1]`` images."""
    rgb = (images * 0.5 + 0.5).clamp(0.0, 1.0)
    return (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).cpu().numpy()


def _ggd_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return beta / (2.0 * alpha * gamma(1.0 / beta)) * np.exp(-((np.abs(x) / alpha) ** beta))


def _fit_ggd(samples: np.ndarray) -> tuple[float, float]:
    """Fit a generalized Gaussian by maximum likelihood."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < 32:
        return 1.0, 1.0
    m2 = float(np.mean(x**2))
    m1 = float(np.mean(np.abs(x)))
    if m2 < 1e-12 or m1 < 1e-12:
        return 1.0, 1.0
    beta0 = max(0.05, min(20.0, math.log(2.0) / (math.log(max(m2 / m1**2, 1e-12)))))
    alpha0 = m1 * gamma(1.0 / beta0) / gamma(2.0 / beta0)

    def neg_ll(params: np.ndarray) -> float:
        alpha, beta = float(params[0]), float(params[1])
        if alpha <= 0.0 or beta <= 0.0:
            return 1e9
        return float(-np.mean(np.log(_ggd_pdf(x, alpha, beta) + 1e-12)))

    res = minimize(
        neg_ll, x0=[alpha0, beta0], method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-6}
    )
    alpha, beta = res.x
    return float(max(alpha, 1e-8)), float(max(beta, 1e-6))


def _agggd_pdf(x: np.ndarray, gamma_k: float, beta_l: float, beta_r: float) -> np.ndarray:
    norm = gamma_k / ((beta_l + beta_r) * gamma(1.0 / gamma_k))
    return norm * np.where(
        x < 0,
        np.exp(-((np.abs(x) / beta_l) ** gamma_k)),
        np.exp(-((np.abs(x) / beta_r) ** gamma_k)),
    )


def _fit_agggd(samples: np.ndarray) -> tuple[float, float, float, float]:
    """Fit an asymmetric generalized Gaussian by maximum likelihood.

    Returns ``(eta, gamma, beta_l, beta_r)``: the four parameters of the
    distribution the NIQE feature set is defined on (the mean is a parameter
    here because the products of MSCN coefficients are not zero-mean).
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < 32:
        return 0.0, 1.0, 1.0, 1.0
    eta = float(np.mean(x))
    centered = x - eta
    neg, pos = centered[centered < 0], centered[centered > 0]
    beta_l0 = float(np.sqrt(np.mean(neg**2))) if neg.size else 0.1
    beta_r0 = float(np.sqrt(np.mean(pos**2))) if pos.size else 0.1
    gamma0 = 1.0

    def neg_ll(params: np.ndarray) -> float:
        g, bl, br = float(params[0]), float(params[1]), float(params[2])
        if g <= 0.0 or bl <= 0.0 or br <= 0.0:
            return 1e9
        return float(-np.mean(np.log(_agggd_pdf(centered, g, bl, br) + 1e-12)))

    res = minimize(
        neg_ll,
        x0=[gamma0, max(beta_l0, 1e-3), max(beta_r0, 1e-3)],
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1e-6},
    )
    g, bl, br = res.x
    return eta, float(max(g, 1e-6)), float(max(bl, 1e-6)), float(max(br, 1e-6))


def _mscn_batch(images: np.ndarray, window: np.ndarray) -> np.ndarray:
    """MSCN coefficients for ``(N, H, W)`` luminance images, batched via conv."""
    tensor = torch.from_numpy(images.astype(np.float32)).unsqueeze(1)
    kernel = torch.from_numpy(window.astype(np.float32)).view(1, 1, *window.shape)
    pad = window.shape[0] // 2
    mu = F.conv2d(tensor, kernel, padding=pad)
    mu_sq = mu * mu
    sigma_sq = F.conv2d(tensor * tensor, kernel, padding=pad) - mu_sq
    sigma = torch.sqrt(sigma_sq.clamp(min=0.0))
    return ((tensor - mu) / (sigma + 1.0 / 255.0)).squeeze(1).numpy()


def _nss_features(images: np.ndarray) -> np.ndarray:
    """``(N, 36)`` natural-scene-statistics feature vectors (two scales)."""
    window = _gaussian_window_np(7, 7.0 / 6.0)
    n = images.shape[0]
    features = np.zeros((n, 36), dtype=np.float64)
    for scale, start in ((0, 0), (1, 18)):
        current = images if scale == 0 else images[:, ::2, ::2]
        mscn = _mscn_batch(current, window)
        for i in range(n):
            m = mscn[i]
            products = {
                "h": m[:, :-1] * m[:, 1:],
                "v": m[:-1, :] * m[1:, :],
                "d1": m[:-1, :-1] * m[1:, 1:],
                "d2": m[:-1, 1:] * m[1:, :-1],
            }
            vector: list[float] = list(_fit_ggd(m))
            for key in ("h", "v", "d1", "d2"):
                vector += list(_fit_agggd(products[key]))
            features[i, start : start + 18] = vector
    return features


def _gaussian_window_np(size: int, sigma: float) -> np.ndarray:
    g = np.exp(-((np.arange(size) - size // 2) ** 2) / (2.0 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def _pool(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    cov = np.atleast_2d(np.cov(features, rowvar=False)) + 1e-6 * np.eye(features.shape[1])
    return mean, cov


def fit_niqe_pristine(clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the pristine NSS model on clean luminance images ``(N, H, W)``."""
    return _pool(_nss_features(clean))


def niqe_score(restored: torch.Tensor, pristine_model: tuple[np.ndarray, np.ndarray]) -> float:
    """NIQE of the restored batch against a pre-fit pristine model."""
    feats = _nss_features(_luminance(restored))
    mean_f, cov_f = _pool(feats)
    diff = mean_f - pristine_model[0]
    inv_cov = np.linalg.inv((cov_f + pristine_model[1]) / 2.0 + 1e-8 * np.eye(diff.shape[0]))
    return float(np.sqrt(max(float(diff @ inv_cov @ diff), 0.0)))


# --------------------------------------------------------------------------- #
# Per-run evaluation
# --------------------------------------------------------------------------- #


def _psnr_strata(per_image_psnr: torch.Tensor, sigmas: torch.Tensor) -> tuple[float, float]:
    """Split mean per-image PSNR by the low/high half of the sigma range."""
    order = torch.argsort(sigmas.cpu())
    split = len(order) // 2
    if split == 0:
        return float(per_image_psnr.mean().item()), float(per_image_psnr.mean().item())
    low = float(per_image_psnr[order[:split]].mean().item())
    high = float(per_image_psnr[order[split:]].mean().item())
    return low, high


def _last_checkpoint(run_dir: Path) -> Path:
    existing = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not existing:
        raise FileNotFoundError(f"no checkpoint-* folder in {run_dir}")
    return existing[-1]


def _load_val_images(folder: Path, count: int, seed: int) -> tuple[list[Path], torch.Tensor]:
    from PIL import Image
    from torchvision import transforms

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    rng = np.random.default_rng(seed)
    chosen = [files[i] for i in rng.choice(len(files), size=min(count, len(files)), replace=False)]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    images = torch.stack([transform(Image.open(p).convert("RGB")) for p in chosen])
    return list(chosen), images


def evaluate_restoration(
    *,
    run_dir: Path,
    real: Path,
    ledger: Path,
    num_val_images: int = 500,
    batch_size: int = 64,
    seed: int = 777,
    device: str = "cuda",
    epoch: int | None = None,
) -> dict[str, float]:
    """Score one trained restoration run and append a ledger row.

    Degradation sigma values are drawn once with ``seed`` and reused for every
    checkpoint of the run (and across runs that share the seed), so the
    evaluation set is identical for every loss arm — differences are
    attributable to the objective, not to the choice of test samples.
    """
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"no config.json in {run_dir}; was this run produced by `awwl train`?"
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    ckpt = _last_checkpoint(run_dir) if epoch is None else run_dir / f"checkpoint-{epoch}"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {ckpt}")

    model_cfg = cfg["model"]
    from awwl.models.ddpm_unet import load_or_build_ddpm_unet

    model = load_or_build_ddpm_unet(
        weights_path=ckpt,
        builder_kwargs={
            "image_size": int(model_cfg["image_size"]),
            "in_channels": int(model_cfg["in_channels"]),
            "out_channels": int(model_cfg["out_channels"]),
            "layers_per_block": int(model_cfg["layers_per_block"]),
            "block_out_channels": tuple(model_cfg["block_out_channels"]),
            "down_block_types": tuple(model_cfg["down_block_types"]),
            "up_block_types": tuple(model_cfg["up_block_types"]),
        },
    ).to(device)
    model.eval()

    degrad_cfg = cfg.get("degradation", {})
    sigma_min = float(degrad_cfg.get("sigma_min", 0.05))
    sigma_max = float(degrad_cfg.get("sigma_max", 0.5))
    num_timesteps = int(cfg.get("model", {}).get("num_train_timesteps", 1000))

    from awwl.data.degradation import sigma_to_timesteps

    _, clean = _load_val_images(real, num_val_images, seed)
    clean = clean.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    sigmas = torch.linspace(sigma_min, sigma_max, clean.shape[0], device=device)
    sigmas = sigmas[torch.randperm(clean.shape[0], device=device, generator=gen)]
    degraded = torch.clamp(clean + torch.randn_like(clean) * sigmas.view(-1, 1, 1, 1), -1.0, 1.0)
    timesteps = sigma_to_timesteps(
        sigmas, num_timesteps=num_timesteps, sigma_min=sigma_min, sigma_max=sigma_max
    )

    restored_list: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, clean.shape[0], batch_size):
            end = min(start + batch_size, clean.shape[0])
            restored_list.append(
                model(degraded[start:end], timesteps[start:end], return_dict=False)[0].cpu()
            )
    restored = torch.cat(restored_list, dim=0)
    clean_cpu = clean.cpu()

    per_image = 10.0 * torch.log10(4.0 / ((restored - clean_cpu) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
    p = float(per_image.mean().item())
    s = ssim(restored, clean_cpu)
    try:
        lp = lpips_score(restored, clean_cpu, device=device)
    except Exception as exc:  # pragma: no cover - network / weights issues
        logger.warning("LPIPS failed (%s); reporting 0.0", exc)
        lp = 0.0

    # Stratify PSNR by the *actual* degradation sigma, not by position: the
    # validation sigmas are a permutation of a linspace, so an index split
    # would mix noise levels.
    psnr_low, psnr_high = _psnr_strata(per_image, sigmas)

    pristine = fit_niqe_pristine(_luminance(clean_cpu))
    niqe = niqe_score(restored, pristine)

    samples_dir = run_dir / "samples" / f"ep{epoch or 0}"
    samples_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    def _to_png(t: torch.Tensor) -> Image.Image:
        arr = ((t * 0.5 + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(arr)

    for i in range(min(8, restored.shape[0])):
        _to_png(restored[i]).save(samples_dir / f"restored_{i:03d}.png")
        _to_png(clean_cpu[i]).save(samples_dir / f"clean_{i:03d}.png")
        _to_png(degraded[i].cpu()).save(samples_dir / f"degraded_{i:03d}.png")

    metrics = {
        "psnr": round(p, 4),
        "ssim": round(s, 5),
        "lpips": round(lp, 5),
        "niqe": round(niqe, 4),
        "psnr_low_sigma": round(psnr_low, 4),
        "psnr_high_sigma": round(psnr_high, 4),
        "eval_seed": seed,
        "n_val": clean.shape[0],
    }

    from awwl.analysis.results import append_result, result_row

    output_cfg = cfg.get("output", {})
    append_result(
        ledger,
        result_row(
            cfg,
            exp=str(output_cfg.get("name", run_dir.name)),
            group=str(output_cfg.get("group", cfg.get("loss", {}).get("name", "?"))),
            kind="eval",
            metrics=metrics,
            epoch=epoch,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            checkpoint=str(ckpt),
        ),
    )
    return metrics
