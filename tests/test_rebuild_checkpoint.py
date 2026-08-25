"""The checkpoint-rebuild path: a pruned run must become samplable again.

Pruning deletes checkpoint-* but leaves state/; the DDPM-1000 check depends
on rebuilding a loadable pipeline from that snapshot. These tests pin the
snapshot layout the trainer writes — latest.json's pointer key, meta.json's
epoch, unet/ in diffusers format — because inference.py and CheckpointManager
must agree on them even though neither imports the other.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from diffusers import DDPMPipeline, UNet2DModel  # noqa: E402

from awwl.methods.finetune.inference import rebuild_checkpoint_from_state  # noqa: E402


def _tiny_unet(tmp_path, snapshot_dir):
    unet = UNet2DModel(
        sample_size=8,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32,),
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
    )
    unet.save_pretrained(snapshot_dir / "unet")


def _fake_run(tmp_path):
    run = tmp_path / "mse_s1"
    snap = run / "state" / "ep49"
    snap.mkdir(parents=True)
    _tiny_unet(tmp_path, snap)
    (run / "config.json").write_text(
        json.dumps({"scheduler": {"num_train_timesteps": 50}}), encoding="utf-8"
    )
    (run / "state" / "latest.json").write_text(json.dumps({"dir": "ep49"}), encoding="utf-8")
    (snap / "meta.json").write_text(json.dumps({"epoch": 49, "global_step": 100}), encoding="utf-8")
    return run


def test_rebuild_restores_the_deleted_path_and_loads_as_a_pipeline(tmp_path):
    run = _fake_run(tmp_path)

    target = rebuild_checkpoint_from_state(run)

    assert target == run / "checkpoint-49"
    # The whole point: what the sampler does with a trained checkpoint must
    # work on a rebuilt one.
    pipeline = DDPMPipeline.from_pretrained(target)
    assert pipeline.unet.config.sample_size == 8


def test_rebuild_honours_an_explicit_target(tmp_path):
    run = _fake_run(tmp_path)

    target = rebuild_checkpoint_from_state(run, tmp_path / "elsewhere" / "ckpt")

    assert target == tmp_path / "elsewhere" / "ckpt"


def test_rebuild_refuses_a_snapshot_without_an_epoch(tmp_path):
    run = _fake_run(tmp_path)
    meta = json.loads((run / "state" / "ep49" / "meta.json").read_text(encoding="utf-8"))
    del meta["epoch"]
    (run / "state" / "ep49" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError):
        rebuild_checkpoint_from_state(run)
