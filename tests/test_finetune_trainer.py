"""End-to-end trainer tests on a synthetic dataset.

The pieces added for the replication sweep — EMA, state snapshots, auto-resume,
the results ledger — only interact inside the real training loop, so unit tests
of each in isolation would not catch a mis-wiring between them. These run the
actual :func:`train_finetune` with the CIFAR-10 loader swapped for random
tensors, so they need no download, no GPU and no network.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from awwl.analysis.results import load_results
from awwl.methods.finetune import trainer as trainer_mod
from awwl.methods.finetune.trainer import run_dir_for, train_finetune


class _RandomImages(Dataset):
    """Stand-in for CIFAR-10: a handful of 8x8 images in [-1, 1]."""

    def __init__(self, n: int = 8) -> None:
        torch.manual_seed(0)
        self._items = [torch.randn(3, 8, 8).clamp(-1, 1) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"images": self._items[idx]}


@pytest.fixture
def patched_loader(monkeypatch):
    """Replace the CIFAR-10 dataloader; assert the seed is threaded through."""
    seen: dict[str, object] = {}

    def _build(**kwargs):
        seen.update(kwargs)
        return DataLoader(_RandomImages(), batch_size=4)

    monkeypatch.setattr(trainer_mod, "build_image_dataloader", _build)
    return seen


def _cfg(tmp_path, **overrides) -> dict:
    cfg = {
        "seed": 7,
        "method": "finetune",
        "precision": {"mixed_precision": "no"},
        "model": {
            "image_size": 8,
            "in_channels": 3,
            "out_channels": 3,
            "layers_per_block": 1,
            "block_out_channels": [32, 32],
            "down_block_types": ["DownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "UpBlock2D"],
            "model_weights_path": None,
        },
        "data": {"dataset_name": "synthetic", "image_size": 8, "batch_size": 4, "num_workers": 0},
        "scheduler": {"num_train_timesteps": 50},
        "train": {
            "num_epochs": 2,
            "learning_rate": 1e-4,
            "lr_warmup_steps": 1,
            "save_model_epochs": 1,
            "save_state_epochs": 1,
            "grad_clip_norm": 1.0,
        },
        "loss": {"name": "adaptive_wavelet", "alpha": 0.2, "power": 1.0, "wavelet_type": "db1"},
        "output": {"dir": str(tmp_path / "runs"), "name": "exp", "group": "awwl"},
    }
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        cfg[section][field] = value
    return cfg


def test_training_produces_checkpoint_state_and_ledger(tmp_path, patched_loader):
    """One pass through the real loop must leave every artefact the sweep needs."""
    cfg = _cfg(tmp_path)
    final = train_finetune(cfg)
    run_dir = run_dir_for(cfg)

    assert final.exists() and final.name == "checkpoint-1"
    assert (run_dir / "config.json").exists(), "eval jobs read identity from config.json"
    assert (run_dir / "state" / "latest.json").exists(), "no resume point was written"
    assert (run_dir / "loss_history.json").exists()

    (row,) = load_results(run_dir / "results.jsonl", kind="train")
    assert row["seed"] == 7
    assert row["group"] == "awwl"
    assert row["alpha"] == 0.2
    assert row["global_step"] == 4, "2 epochs x 2 batches"

    assert patched_loader["seed"] == 7, "dataloader shuffling was left unseeded"


def test_interrupted_run_resumes_instead_of_restarting(tmp_path, patched_loader):
    """The headline guarantee, exercised through the real trainer."""
    cfg = _cfg(tmp_path)
    train_finetune(cfg)
    run_dir = run_dir_for(cfg)

    first = json.loads((run_dir / "state" / "latest.json").read_text(encoding="utf-8"))
    assert first["epoch"] == 1 and first["global_step"] == 4

    # Re-enter as a restarted process would, asking for more epochs.
    cfg_more = _cfg(tmp_path)
    cfg_more["train"]["num_epochs"] = 4
    train_finetune(cfg_more)

    second = json.loads((run_dir / "state" / "latest.json").read_text(encoding="utf-8"))
    assert second["epoch"] == 3
    assert second["global_step"] == 8, (
        f"expected 4 more steps on top of 4, got {second['global_step']} — "
        "training restarted from scratch instead of resuming"
    )


def test_completed_run_is_a_no_op(tmp_path, patched_loader):
    """Re-running a finished job must not retrain it; the sweep relies on this."""
    cfg = _cfg(tmp_path)
    train_finetune(cfg)
    run_dir = run_dir_for(cfg)
    before = json.loads((run_dir / "state" / "latest.json").read_text(encoding="utf-8"))

    final = train_finetune(_cfg(tmp_path))
    after = json.loads((run_dir / "state" / "latest.json").read_text(encoding="utf-8"))

    assert final.name == "checkpoint-1"
    assert after["global_step"] == before["global_step"]


def test_resume_can_be_disabled(tmp_path, patched_loader):
    cfg = _cfg(tmp_path)
    train_finetune(cfg)

    cfg_fresh = _cfg(tmp_path)
    cfg_fresh["train"]["num_epochs"] = 4
    cfg_fresh["train"]["resume"] = False
    train_finetune(cfg_fresh)

    state = json.loads(
        (run_dir_for(cfg) / "state" / "latest.json").read_text(encoding="utf-8")
    )
    assert state["global_step"] == 8, "with resume off, 4 epochs means 8 steps from zero"


def test_ema_produces_different_exported_weights(tmp_path, patched_loader):
    """EMA must actually reach the exported checkpoint, not just the shadow copy."""
    from diffusers import UNet2DModel

    plain_cfg = _cfg(tmp_path)
    plain_cfg["output"]["name"] = "plain"
    plain = train_finetune(plain_cfg)

    ema_cfg = _cfg(tmp_path)
    ema_cfg["output"]["name"] = "with_ema"
    ema_cfg["train"]["use_ema"] = True
    ema_cfg["train"]["ema_decay"] = 0.5  # move fast enough to differ in 4 steps
    ema_path = train_finetune(ema_cfg)

    plain_unet = UNet2DModel.from_pretrained(plain, subfolder="unet")
    ema_unet = UNet2DModel.from_pretrained(ema_path, subfolder="unet")
    differs = any(
        not torch.allclose(a, b)
        for a, b in zip(plain_unet.state_dict().values(), ema_unet.state_dict().values(), strict=True)
    )
    assert differs, "exported checkpoint is identical with and without EMA"

    assert (run_dir_for(ema_cfg) / "state" / "latest.json").exists()
    snapshot = json.loads(
        (run_dir_for(ema_cfg) / "state" / "latest.json").read_text(encoding="utf-8")
    )
    state_dir = run_dir_for(ema_cfg) / "state" / snapshot["dir"]
    assert (state_dir / "ema.pt").exists(), "EMA shadow was not snapshotted; a resume would lose it"


@pytest.mark.parametrize(
    "loss_cfg",
    [
        {"name": "mse"},
        {"name": "snr_weighted"},
        {"name": "adaptive_wavelet", "alpha": 0.2, "power": 1.0, "normalize_weights": True},
        {"name": "adaptive_wavelet", "alpha": 0.2, "power": 1.0, "detail_reduction": "sum"},
        {"name": "adaptive_wavelet", "alpha": 0.2, "power": 0.0},
    ],
)
def test_every_sweep_loss_trains(tmp_path, patched_loader, loss_cfg):
    """Each loss the phase-0 manifest selects must survive a real training step."""
    cfg = _cfg(tmp_path)
    cfg["loss"] = loss_cfg
    cfg["output"]["name"] = loss_cfg["name"] + str(loss_cfg.get("power", ""))
    assert train_finetune(cfg).exists()


@pytest.mark.parametrize(
    "loss_cfg",
    [
        {"name": "wavelet_subband", "alpha": 0.2, "power": 1.0, "direction_powers": {"hh": 2.0}},
        {"name": "wavelet_spatial", "alpha": 0.2, "power": 1.0, "strength": 1.0},
        {"name": "wavelet_learned", "conditioned": True},
        {"name": "wavelet_learned", "conditioned": False},
        {"name": "wavelet_gradnorm"},
        {"name": "wavelet_lifting"},
    ],
)
def test_extension_losses_train_through_the_real_loop(tmp_path, patched_loader, loss_cfg):
    """A1-A4 must survive the trainer, not just a standalone forward pass."""
    cfg = _cfg(tmp_path)
    cfg["loss"] = loss_cfg
    cfg["output"]["name"] = loss_cfg["name"] + str(loss_cfg.get("conditioned", ""))
    assert train_finetune(cfg).exists()


@pytest.mark.parametrize(
    "loss_cfg",
    [
        {"name": "wavelet_learned", "conditioned": False},
        {"name": "wavelet_gradnorm"},
        {"name": "wavelet_lifting"},
    ],
)
def test_learned_loss_parameters_actually_change(tmp_path, patched_loader, loss_cfg):
    """The failure this guards against is silent.

    If the loss's parameters never reach the optimiser, training still runs and
    still converges — it just quietly becomes a fixed-weight objective frozen
    at initialisation. The experiment would look successful while testing
    nothing, so assert the weights moved.
    """
    from diffusers import DDPMScheduler

    from awwl.losses import get_loss_function, trainable_loss_parameters

    reference = get_loss_function(
        loss_cfg["name"],
        noise_scheduler=DDPMScheduler(num_train_timesteps=50),
        **{k: v for k, v in loss_cfg.items() if k != "name"},
    )
    before = [p.detach().clone() for p in trainable_loss_parameters(reference)]
    assert before, "this loss was expected to carry learnable parameters"

    cfg = _cfg(tmp_path)
    cfg["loss"] = loss_cfg
    cfg["train"]["loss_learning_rate"] = 0.05  # move visibly within 4 steps
    cfg["output"]["name"] = "learn_" + loss_cfg["name"]
    train_finetune(cfg)

    state = torch.load(
        run_dir_for(cfg) / "state" / _latest_dir(cfg) / "loss.pt",
        map_location="cpu",
        weights_only=False,
    )
    after = [v for v in state.values() if isinstance(v, torch.Tensor) and v.is_floating_point()]
    assert any(
        not torch.allclose(a, b) for a, b in zip(before, after, strict=False)
    ), "loss parameters are unchanged after training — they never reached the optimiser"


def test_learned_loss_state_survives_resume(tmp_path, patched_loader):
    """A resumed run must not reset the learned schedule to its initialisation."""
    cfg = _cfg(tmp_path)
    cfg["loss"] = {"name": "wavelet_learned", "conditioned": False}
    cfg["train"]["loss_learning_rate"] = 0.05
    train_finetune(cfg)

    first = torch.load(
        run_dir_for(cfg) / "state" / _latest_dir(cfg) / "loss.pt",
        map_location="cpu",
        weights_only=False,
    )["weighting.log_vars"].clone()

    cfg_more = _cfg(tmp_path)
    cfg_more["loss"] = {"name": "wavelet_learned", "conditioned": False}
    cfg_more["train"]["loss_learning_rate"] = 0.05
    cfg_more["train"]["num_epochs"] = 4
    train_finetune(cfg_more)

    second = torch.load(
        run_dir_for(cfg) / "state" / _latest_dir(cfg) / "loss.pt",
        map_location="cpu",
        weights_only=False,
    )["weighting.log_vars"]
    assert not torch.allclose(first, second), "training continued but the loss state did not"


def _latest_dir(cfg) -> str:
    pointer = run_dir_for(cfg) / "state" / "latest.json"
    return json.loads(pointer.read_text(encoding="utf-8"))["dir"]


def test_run_dir_separates_seeds_without_an_explicit_name(tmp_path):
    """Two seeds must not share a directory, or they resume from each other."""
    base = {"output": {"dir": "./runs/finetune"}, "loss": {"name": "mse"}, "model": {}}
    a = run_dir_for({**base, "seed": 1})
    b = run_dir_for({**base, "seed": 2})
    assert a != b
