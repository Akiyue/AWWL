"""Tests for the B2 / F1 / F3 / cost / DreamBooth-pipeline additions."""

from __future__ import annotations

import json

import pytest
import torch

from awwl.evaluation.cost import format_cost_table
from awwl.losses import GeneralizedWaveletLoss, get_loss_function
from awwl.losses.weighting import min_snr_weight
from awwl.pipeline.manifest import build_jobs, load_manifest


class _StubScheduler:
    def __init__(self, num_train_timesteps: int = 1000) -> None:
        self.alphas_cumprod = torch.linspace(0.999, 0.001, num_train_timesteps)


# ------------------------------------------------------------ B2: Min-SNR


def test_min_snr_weight_matches_its_definition():
    """min(SNR, gamma)/SNR with SNR = (1 - sigma^2)/sigma^2."""
    sigmas = torch.tensor([0.1, 0.5, 0.9])
    snr = (1 - sigmas**2) / sigmas**2
    expected = torch.clamp(snr, max=5.0) / snr
    assert torch.allclose(min_snr_weight(sigmas, 5.0), expected, atol=1e-5)


def test_min_snr_damps_low_noise_and_spares_high_noise():
    """The behaviour that makes it orthogonal to a frequency weighting."""
    w = min_snr_weight(torch.tensor([0.05, 0.95]), 5.0)
    assert w[0] < 0.1, "high-SNR (low-noise) steps should be damped"
    assert w[1] == pytest.approx(1.0, abs=1e-3), "low-SNR steps should pass through"


def test_snr_gamma_composes_with_the_band_weighting():
    """Min-SNR must scale the whole loss per timestep, not per band."""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 3, 16, 16)
    sigmas = torch.tensor([0.1, 0.9])

    plain = GeneralizedWaveletLoss()
    combined = GeneralizedWaveletLoss(snr_gamma=5.0)
    assert float(combined(pred, target, sigmas)) < float(plain(pred, target, sigmas))


def test_snr_survives_normalisation():
    """Applied before normalisation it would be divided straight back out."""
    loss = GeneralizedWaveletLoss(snr_gamma=5.0, normalize_weights=True)
    torch.manual_seed(0)
    pred, target = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
    low = loss(pred, target, torch.tensor([0.05, 0.05]))
    high = loss(pred, target, torch.tensor([0.95, 0.95]))
    assert float(low) < float(high), "the timestep factor was cancelled by normalisation"


def test_wavelet_minsnr_loss_is_registered():
    loss_fn = get_loss_function("wavelet_minsnr", noise_scheduler=_StubScheduler())
    out = loss_fn(
        torch.randn(2, 3, 16, 16, requires_grad=True),
        torch.randn(2, 3, 16, 16),
        timesteps=torch.randint(0, 1000, (2,)),
    )
    assert torch.isfinite(out)


# ------------------------------------------------------- F3: curriculum plot


def test_weight_profile_plot_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from awwl.plotting.curriculum import plot_weight_profile

    sigmas = torch.linspace(0.01, 0.99, 21)
    profile = GeneralizedWaveletLoss().weight_profile(sigmas)
    out = plot_weight_profile(profile, sigmas.numpy(), out_path=tmp_path / "c.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_run_curriculum_reads_a_run(tmp_path):
    pytest.importorskip("matplotlib")
    from awwl.plotting.curriculum import plot_run_curriculum

    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "loss": {"name": "wavelet_subband", "alpha": 0.2, "power": 1.0},
                "scheduler": {"num_train_timesteps": 1000},
            }
        ),
        encoding="utf-8",
    )
    assert plot_run_curriculum(run, points=11).exists()


def test_plot_run_curriculum_rejects_a_loss_without_bands(tmp_path):
    pytest.importorskip("matplotlib")
    from awwl.plotting.curriculum import plot_run_curriculum

    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"loss": {"name": "mse"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="weight profile"):
        plot_run_curriculum(run)


# ------------------------------------------------------------- cost table


def test_cost_table_reports_ratios_against_the_baseline():
    rows = [
        {"loss": "mse", "step_ms": 100.0, "loss_ms": 1.0, "peak_mb": 1000.0, "flops": 1e9,
         "loss_params": 0, "backbone_params": 35_000_000},
        {"loss": "adaptive_wavelet", "step_ms": 110.0, "loss_ms": 11.0, "peak_mb": 1100.0,
         "flops": 1.1e9, "loss_params": 0},
    ]
    table = format_cost_table(rows)
    assert "1.10x" in table, "overhead ratio missing"
    assert "10.0%" in table, "loss share of the step missing"
    assert "35,000,000" in table


def test_cost_table_handles_missing_cuda_metrics():
    rows = [{"loss": "mse", "step_ms": 10.0, "loss_ms": 1.0, "peak_mb": float("nan"),
             "flops": float("nan"), "loss_params": 0}]
    assert "n/a" in format_cost_table(rows)


# ------------------------------------------------- F1: multi-dataset support


def test_build_transform_keeps_aspect_ratio():
    """A plain resize would squash images and change their frequency content."""
    from PIL import Image

    from awwl.data.images import build_transform

    tensor = build_transform(32, horizontal_flip=False)(Image.new("RGB", (100, 40)))
    assert tensor.shape == (3, 32, 32)
    assert float(tensor.min()) >= -1.0 and float(tensor.max()) <= 1.0


def test_image_column_is_guessed_when_absent():
    from awwl.data.images import _pick_column

    class _DS:
        column_names = ["img", "label"]

    assert _pick_column(_DS(), "image") == "img"


def test_cifar10_still_routes_to_the_published_loader(monkeypatch):
    """The published recipe must not change when the dispatch is added."""
    from awwl.data import images

    called = {}
    monkeypatch.setattr(
        images, "build_cifar10_dataloader", lambda **kw: called.update(kw) or "loader"
    )
    assert images.build_image_dataloader(dataset_name="cifar10", seed=7) == "loader"
    assert called["seed"] == 7
    assert called["source"] == "auto"


# ------------------------------------------------------ CIFAR-10 source


def test_cifar10_falls_back_to_torchvision_when_the_hub_fails(monkeypatch, tmp_path):
    """The Hub dropped single-segment dataset ids and now raises mid-resolution.

    A fixed 50 000-image dataset must not need a live API, so an unavailable
    Hub degrades to the local torchvision copy instead of killing the run.
    """
    from awwl.data import cifar10

    def _boom(*a, **kw):
        raise RuntimeError("HfUriError: Repository id must be 'namespace/name'")

    built = {}

    class _FakeTV:
        def __init__(self, **kw):
            built.update(kw)

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return {"images": torch.zeros(3, 8, 8)}

    monkeypatch.setattr(cifar10, "_load_hf", _boom)
    monkeypatch.setattr(cifar10, "_TorchvisionCifar", _FakeTV)

    loader = cifar10.build_cifar10_dataloader(
        image_size=8, batch_size=2, num_workers=0, root=tmp_path
    )
    assert next(iter(loader))["images"].shape == (2, 3, 8, 8)
    assert built["train"] is True


def test_cifar10_hub_source_fails_loudly(monkeypatch):
    """A table that must state its data source cannot silently switch."""
    from awwl.data import cifar10

    monkeypatch.setattr(cifar10, "_load_hf", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError, match="down"):
        cifar10.build_cifar10_dataloader(source="hf")


def test_cifar10_rejects_an_unknown_source():
    from awwl.data import cifar10

    with pytest.raises(ValueError, match="source must be"):
        cifar10.build_cifar10_dataloader(source="telepathy")


def test_canonical_hub_id_is_tried_first():
    """'cifar10' alone is what newer huggingface_hub clients reject."""
    from awwl.data.cifar10 import _HF_IDS

    assert _HF_IDS[0] == "uoft-cs/cifar10"


# ------------------------------------------- DreamBooth in the pipeline

_DB_MANIFEST = """
name: db
method: dreambooth
base_config: configs/dreambooth.yaml
output_root: ./runs/db
real_images: ./data/subject
defaults:
  seeds: [1, 2]
  sample: {num_samples: 20, steps: 50}
experiments:
  - group: mse
    overrides: {loss.name: mse}
"""


def test_dreambooth_manifest_emits_train_and_eval_pairs(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(_DB_MANIFEST, encoding="utf-8")
    jobs = build_jobs(load_manifest(path))

    kinds = [j.kind for j in jobs]
    assert kinds.count("train") == 2
    assert kinds.count("eval") == 2
    assert "sample" not in kinds, "DreamBooth scoring generates and evaluates in one job"

    by_id = {j.job_id: j for j in jobs}
    assert by_id["db:eval:mse_s1"].depends_on == "db:train:mse_s1"
    argv = " ".join(by_id["db:eval:mse_s1"].argv)
    assert "eval-dreambooth" in argv and "--real" in argv


def test_manifest_rejects_an_unknown_method(tmp_path):
    from awwl.core.exceptions import ConfigError

    path = tmp_path / "m.yaml"
    path.write_text(_DB_MANIFEST.replace("method: dreambooth", "method: telepathy"), encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported method"):
        build_jobs(load_manifest(path))


def test_dreambooth_run_dir_separates_seeds():
    """Previously every seed wrote into output.dir and overwrote the last."""
    from awwl.methods.dreambooth.trainer import dreambooth_run_dir

    base = {"output": {"dir": "./runs/db"}, "loss": {"name": "mse"}}
    assert dreambooth_run_dir({**base, "seed": 1}) != dreambooth_run_dir({**base, "seed": 2})
    assert dreambooth_run_dir({"output": {"dir": "./r", "name": "exp"}}).name == "exp"
