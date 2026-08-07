"""Checkpoint save/resume tests — the guarantee that a killed run continues."""

from __future__ import annotations

import json

import pytest
import torch
from diffusers import UNet2DModel

from awwl.training.checkpointing import CheckpointManager


def _tiny_model() -> UNet2DModel:
    """The smallest UNet2DModel that still exercises save/from_pretrained."""
    return UNet2DModel(
        sample_size=8,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        # 32 channels, not fewer: the default GroupNorm uses 32 groups.
        block_out_channels=(32, 32),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )


@pytest.fixture
def trio():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return model, optimizer, lr_scheduler


def _take_a_step(model, optimizer, lr_scheduler) -> None:
    """One real optimisation step, so optimiser moments become non-trivial."""
    out = model(torch.randn(1, 3, 8, 8), torch.tensor([0])).sample
    out.pow(2).mean().backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()


def test_nothing_to_resume_from_returns_none(tmp_path, trio):
    model, optimizer, _ = trio
    manager = CheckpointManager(tmp_path)
    assert manager.latest_dir() is None
    assert manager.load(model=model, optimizer=optimizer) is None


def test_roundtrip_restores_weights_and_step(tmp_path, trio):
    model, optimizer, lr_scheduler = trio
    manager = CheckpointManager(tmp_path)
    _take_a_step(model, optimizer, lr_scheduler)

    manager.save(
        epoch=4, global_step=1234, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
    )
    saved = {k: v.clone() for k, v in model.state_dict().items()}

    # Simulate the crash: fresh objects, as a restarted process would build.
    fresh_model = _tiny_model()
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_sched = torch.optim.lr_scheduler.StepLR(fresh_opt, step_size=1, gamma=0.5)

    resumed = CheckpointManager(tmp_path).load(
        model=fresh_model, optimizer=fresh_opt, lr_scheduler=fresh_sched
    )
    assert resumed is not None
    assert resumed.epoch == 4
    assert resumed.global_step == 1234
    for key, value in fresh_model.state_dict().items():
        assert torch.allclose(value, saved[key]), f"weight {key} not restored"
    assert fresh_sched.state_dict()["last_epoch"] == lr_scheduler.state_dict()["last_epoch"]


def test_optimizer_moments_survive(tmp_path, trio):
    """Without this, a 'resumed' run silently restarts Adam from zero moments."""
    model, optimizer, lr_scheduler = trio
    _take_a_step(model, optimizer, lr_scheduler)
    CheckpointManager(tmp_path).save(
        epoch=0, global_step=1, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
    )

    fresh_model = _tiny_model()
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    CheckpointManager(tmp_path).load(model=fresh_model, optimizer=fresh_opt)

    states = list(fresh_opt.state_dict()["state"].values())
    assert states, "optimizer state was not restored"
    assert all(s["step"] >= 1 for s in states)


def test_pruning_keeps_only_recent_states(tmp_path, trio):
    model, optimizer, _ = trio
    manager = CheckpointManager(tmp_path, keep_last=2)
    for step in (1, 2, 3):
        manager.save(epoch=step, global_step=step, model=model, optimizer=optimizer)
    snapshots = sorted(p.name for p in (tmp_path / "state").glob("step-*") if p.is_dir())
    assert snapshots == ["step-000000002", "step-000000003"]


def test_latest_survives_a_torn_snapshot(tmp_path, trio):
    """A crash mid-write must leave the previous snapshot usable."""
    model, optimizer, _ = trio
    manager = CheckpointManager(tmp_path, keep_last=5)
    manager.save(epoch=0, global_step=10, model=model, optimizer=optimizer)

    # A staging directory left behind by a process killed mid-save.
    (tmp_path / "state" / "step-000000020.tmp").mkdir()

    resumed = CheckpointManager(tmp_path).load(model=_tiny_model())
    assert resumed is not None and resumed.global_step == 10


def test_dangling_pointer_falls_back_to_fresh_start(tmp_path, trio):
    """A pointer to a deleted snapshot must not crash the trainer."""
    model, optimizer, _ = trio
    manager = CheckpointManager(tmp_path)
    manager.save(epoch=0, global_step=10, model=model, optimizer=optimizer)

    pointer = tmp_path / "state" / "latest.json"
    body = json.loads(pointer.read_text(encoding="utf-8"))
    body["dir"] = "step-999999999"
    pointer.write_text(json.dumps(body), encoding="utf-8")

    assert CheckpointManager(tmp_path).load(model=_tiny_model()) is None


def test_should_save_fires_on_cadence_and_final_epoch(tmp_path):
    manager = CheckpointManager(tmp_path, save_every_epochs=5)
    assert manager.should_save(4, 200) is True      # (4+1) % 5 == 0
    assert manager.should_save(5, 200) is False
    assert manager.should_save(199, 200) is True    # always the last one
