"""Tests for the image-restoration reframing.

The reframing stands or falls on three things: the degradation really does
add noise of the requested sigma (and the loss sees that same sigma), the
restoration trainer runs end-to-end on CPU with the shared crash-safe
checkpointing, and the evaluation metrics behave like metrics — PSNR/SSIM
improve when the prediction gets closer to the clean image, and NIQE/LPIPS
move in the expected direction.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from awwl.data.degradation import add_noise, sample_sigmas, sigma_to_timesteps
from awwl.evaluation.restoration import (
    fit_niqe_pristine,
    lpips_score,
    niqe_score,
    psnr,
    ssim,
    _luminance,
    _nss_features,
    _psnr_strata,
)


# ------------------------------------------------------------------ degradation


def test_sample_sigmas_in_range_and_shape():
    torch.manual_seed(0)
    sig = sample_sigmas(16, sigma_min=0.05, sigma_max=0.5)
    assert sig.shape == (16,)
    assert sig.min() >= 0.05 and sig.max() <= 0.5
    assert sig.dtype == torch.float32


def test_add_noise_has_expected_magnitude():
    torch.manual_seed(0)
    clean = torch.zeros(2, 3, 32, 32)
    sig = torch.tensor([0.1, 0.4])
    noisy = add_noise(clean, sig)
    # Empirical std of the added noise should track the requested sigma.
    assert abs(noisy[0].std().item() - 0.1) < 0.02
    assert abs(noisy[1].std().item() - 0.4) < 0.04


def test_add_noise_clamps_to_dynamic_range():
    clean = torch.ones(1, 3, 8, 8) * 0.9
    noisy = add_noise(clean, torch.tensor([5.0]))
    assert noisy.max() <= 1.0 and noisy.min() >= -1.0


def test_sigma_to_timesteps_maps_endpoints():
    t = sigma_to_timesteps(
        torch.tensor([0.05, 0.275, 0.5]), num_timesteps=1000, sigma_min=0.05, sigma_max=0.5
    )
    assert t[0].item() == 0
    assert t[-1].item() == 999
    assert t[1].item() == 500  # midpoint of the sigma range


# ------------------------------------------------------------------ loss wiring


def test_restoration_loss_accepts_sigmas_for_wavelet():
    from awwl.methods.restoration.trainer import restoration_loss

    loss_fn = restoration_loss({"loss": {"name": "adaptive_wavelet", "alpha": 0.5, "power": 1.0}})
    pred = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)
    out = loss_fn(pred, target, sigmas=torch.tensor([0.2, 0.4]))
    assert out.ndim == 0
    assert out.item() > 0


def test_restoration_loss_pixel_baselines_ignore_sigmas():
    from awwl.methods.restoration.trainer import restoration_loss

    for name in ("mse", "l1", "huber", "charbonnier"):
        loss_fn = restoration_loss({"loss": {"name": name}})
        pred = torch.randn(2, 3, 8, 8)
        target = torch.randn(2, 3, 8, 8)
        out = loss_fn(pred, target, sigmas=torch.randn(2))
        assert out.ndim == 0


def test_restoration_loss_rejects_unsupported():
    from awwl.methods.restoration.trainer import restoration_loss

    with pytest.raises(ValueError):
        restoration_loss({"loss": {"name": "snr_weighted"}})


# ------------------------------------------------------------------ metrics


def test_psnr_perfect_match_is_infinite():
    x = torch.randn(2, 3, 32, 32)
    # The 1e-8 MSE floor caps the reading at ~86 dB; anything above 80 is a
    # numerically perfect match.
    assert psnr(x, x) > 80.0
    assert psnr(x, x + 0.0) == psnr(x, x)


def test_psnr_worse_with_noise():
    x = torch.randn(2, 3, 32, 32)
    noisy = x + 0.3 * torch.randn_like(x)
    assert psnr(x, noisy) < psnr(x, x)


def test_ssim_perfect_match_is_one():
    x = torch.randn(2, 3, 32, 32)
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-4)


def test_ssim_worse_with_noise():
    x = torch.randn(2, 3, 32, 32)
    noisy = x + 0.3 * torch.randn_like(x)
    assert ssim(x, noisy) < ssim(x, x)


def test_lpips_worse_with_noise():
    x = torch.randn(1, 3, 32, 32)
    noisy = x + 0.5 * torch.randn_like(x)
    lp_clean = lpips_score(x, x, device="cpu")
    lp_noisy = lpips_score(x, noisy, device="cpu")
    assert lp_noisy > lp_clean


def test_niqe_self_consistency():
    torch.manual_seed(0)
    clean = torch.randn(4, 3, 32, 32)
    lum = _luminance(clean)
    pristine = fit_niqe_pristine(lum)
    d_clean = niqe_score(clean, pristine)
    noisy = clean + 0.4 * torch.randn_like(clean)
    d_noisy = niqe_score(noisy, pristine)
    assert d_noisy > d_clean  # noise moves the restored distribution away


def test_psnr_strata_split_by_actual_sigma():
    """The low/high-sigma PSNR bins must follow the sigma value, not position."""
    torch.manual_seed(0)
    per_image = torch.tensor([30.0, 20.0, 25.0, 10.0])
    sigmas = torch.tensor([0.1, 0.5, 0.2, 0.9])
    low, high = _psnr_strata(per_image, sigmas)
    # Sorted by sigma: [0.1, 0.2, 0.5, 0.9] -> low {30, 25}, high {20, 10}.
    assert low == pytest.approx(27.5)
    assert high == pytest.approx(15.0)


def test_nss_features_are_36d():
    x = torch.randn(2, 3, 16, 16)
    feats = _nss_features(_luminance(x))
    assert feats.shape == (2, 36)
    assert np.isfinite(feats).all()


# ------------------------------------------------------------------ trainer smoke


class _RandomImages(torch.utils.data.Dataset):
    """Stand-in for the real dataset: a handful of 8x8 images in [-1, 1]."""

    def __init__(self, n: int = 8) -> None:
        torch.manual_seed(0)
        self._items = [torch.randn(3, 8, 8).clamp(-1, 1) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"images": self._items[idx]}


@pytest.fixture
def patched_loader(monkeypatch):
    """Replace the real dataloader so the smoke test needs no download."""

    def _build(**kwargs):
        return torch.utils.data.DataLoader(_RandomImages(), batch_size=4)

    from awwl.methods.restoration import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "build_image_dataloader", _build)


def _cfg(tmp_path) -> dict:
    return {
        "seed": 1,
        "method": "restoration",
        "precision": {"mixed_precision": "no"},
        "model": {
            "image_size": 8,
            "in_channels": 3,
            "out_channels": 3,
            "layers_per_block": 1,
            "block_out_channels": [32, 32],
            "down_block_types": ["DownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "UpBlock2D"],
            "num_train_timesteps": 50,
            "model_weights_path": None,
        },
        "data": {"dataset_name": "synthetic", "image_size": 8, "batch_size": 4, "num_workers": 0},
        "degradation": {"sigma_min": 0.05, "sigma_max": 0.3},
        "train": {
            "num_epochs": 2,
            "learning_rate": 1e-3,
            "lr_warmup_steps": 1,
            "save_model_epochs": 1,
            "grad_clip_norm": 1.0,
            "resume": True,
            "save_state_epochs": 1,
            "keep_last_states": 2,
            "use_ema": False,
        },
        "loss": {
            "name": "adaptive_wavelet",
            "alpha": 0.5,
            "power": 1.0,
            "wavelet_type": "db1",
            "levels": 1,
        },
        "output": {"dir": str(tmp_path / "runs"), "name": "smoke", "group": "awwl"},
    }


def test_train_restoration_smoke(tmp_path, patched_loader):
    """One pass through the real loop must leave the artefacts the sweep needs."""
    from awwl.methods.restoration.trainer import train_restoration

    cfg = _cfg(tmp_path)
    out = train_restoration(cfg)
    run_dir = tmp_path / "runs" / "smoke"

    assert out.exists() and out.name == "checkpoint-1"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "state" / "latest.json").exists(), "no resume point was written"
    assert (run_dir / "loss_history.json").exists()

    config_json = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_json["method"] == "restoration"
    assert config_json["degradation"]["sigma_max"] == 0.3
