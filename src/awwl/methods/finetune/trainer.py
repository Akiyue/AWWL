"""From-scratch DDPM trainer for the AWWL-Diff recipe.

Replaces ``AWWL-Diff/train_cifar10.py``. The training loop is unchanged, but
hyperparameters come from a config dict and the loss-history JSON logger is
used in every run (the older ``train.py`` did not log it).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from awwl.data.cifar10 import build_cifar10_dataloader
from awwl.losses import get_loss_function
from awwl.models.ddpm_unet import load_or_build_ddpm_unet
from awwl.training.accelerator import build_accelerator
from awwl.training.loss_history import LossHistoryLogger
from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def train_finetune(cfg: dict[str, Any]) -> Path:
    """Train an unconditional DDPM with the configured loss.

    Args:
        cfg: A merged config dict (see ``configs/finetune.yaml``).

    Returns:
        Path to the final saved checkpoint folder.
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    sched_cfg = cfg["scheduler"]
    weights_path = model_cfg.get("model_weights_path")
    out_dir = ensure_dir(_run_dir(cfg))

    accelerator = build_accelerator(
        mixed_precision=cfg["precision"]["mixed_precision"],
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
    )

    dataloader = build_cifar10_dataloader(
        image_size=int(data_cfg["image_size"]),
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 4)),
        split=data_cfg.get("split", "train"),
    )

    builder_kwargs = {
        "image_size": int(model_cfg["image_size"]),
        "in_channels": int(model_cfg["in_channels"]),
        "out_channels": int(model_cfg["out_channels"]),
        "layers_per_block": int(model_cfg["layers_per_block"]),
        "block_out_channels": tuple(model_cfg["block_out_channels"]),
        "down_block_types": tuple(model_cfg["down_block_types"]),
        "up_block_types": tuple(model_cfg["up_block_types"]),
    }
    model = load_or_build_ddpm_unet(weights_path=weights_path, builder_kwargs=builder_kwargs)

    noise_scheduler = DDPMScheduler(num_train_timesteps=int(sched_cfg["num_train_timesteps"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))

    num_epochs = int(train_cfg["num_epochs"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(train_cfg["lr_warmup_steps"]),
        num_training_steps=len(dataloader) * num_epochs,
    )

    loss_fn = get_loss_function(
        loss_cfg["name"],
        noise_scheduler=noise_scheduler,
        alpha=loss_cfg.get("alpha", 0.8),
        power=loss_cfg.get("power", 2.0),
        wavelet_type=loss_cfg.get("wavelet_type", "db1"),
        levels=loss_cfg.get("levels", 1),
        weighting=loss_cfg.get("weighting", "normalized"),
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    history = LossHistoryLogger(
        output_dir=out_dir,
        config={
            "alpha": loss_cfg.get("alpha"),
            "power": loss_cfg.get("power"),
            "wavelet": loss_cfg.get("wavelet_type"),
            "loss_name": loss_cfg["name"],
        },
        resume=weights_path is not None,
    )

    start_epoch = _resume_epoch(weights_path, num_epochs)
    save_every = int(train_cfg.get("save_model_epochs", 20))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))

    last_ckpt: Path | None = None

    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress = tqdm(
            total=len(dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"epoch {epoch}",
        )
        for batch in dataloader:
            clean_images = batch["images"]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = loss_fn(noise_pred, noise, timesteps=timesteps)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            value = float(loss.detach().item())
            history.append(value)
            progress.update(1)
            progress.set_postfix(loss=value)

        if accelerator.is_main_process and (
            (epoch + 1) % save_every == 0 or epoch == num_epochs - 1
        ):
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )
            ckpt = out_dir / f"checkpoint-{epoch}"
            pipeline.save_pretrained(ckpt)
            history.flush()
            last_ckpt = ckpt
            logger.info("saved checkpoint %s", ckpt)

    if last_ckpt is None:
        raise RuntimeError("training finished without saving any checkpoint")
    return last_ckpt


def _run_dir(cfg: dict[str, Any]) -> Path:
    """Mirror the AWWL-Diff convention: ``<output_dir>_a<a>_p<p>_<wavelet>``."""
    base = cfg["output"]["dir"]
    weights_path = cfg["model"].get("model_weights_path")
    if weights_path:
        return Path(weights_path).parent
    loss_cfg = cfg["loss"]
    return Path(
        f"{base}_a{loss_cfg.get('alpha', 0.8)}"
        f"_p{loss_cfg.get('power', 2.0)}"
        f"_{loss_cfg.get('wavelet_type', 'db1')}"
    )


def _resume_epoch(weights_path: str | None, total_epochs: int) -> int:
    """Parse ``checkpoint-XX`` from a resume path, returning ``XX + 1`` (or 0)."""
    if not weights_path:
        return 0
    name = Path(weights_path).name
    if not name.startswith("checkpoint-"):
        return 0
    try:
        epoch = int(name.split("-")[-1]) + 1
    except ValueError:
        return 0
    if epoch >= total_epochs:
        logger.warning(
            "checkpoint already at epoch %d but target is %d — increase num_epochs to continue",
            epoch - 1,
            total_epochs,
        )
    return epoch
