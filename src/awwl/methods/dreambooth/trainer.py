"""DreamBooth full-UNet fine-tuning trainer.

Replaces ``AWWL/dreambooth.py``. The training loop is identical in spirit but
all configuration arrives as a typed dict (loaded from
``configs/dreambooth.yaml``) and every hyperparameter is exposed there.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from awwl.data.dreambooth_dataset import DreamBoothDataset
from awwl.losses import get_loss_function
from awwl.models.sd_components import load_sd_components
from awwl.training.accelerator import build_accelerator, compute_dtype_for
from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def train_dreambooth(cfg: dict[str, Any]) -> Path:
    """Run a DreamBooth fine-tune.

    Args:
        cfg: A merged config dict (see ``configs/dreambooth.yaml``).

    Returns:
        Path to the saved UNet folder.
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    out_dir = ensure_dir(cfg["output"]["dir"])

    accelerator = build_accelerator(
        mixed_precision=cfg["precision"]["mixed_precision"],
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
    )
    compute_dtype = compute_dtype_for(cfg["precision"]["mixed_precision"])

    components = load_sd_components(
        model_cfg["pretrained_model_name_or_path"],
        unet_override_path=model_cfg.get("model_weights_path"),
    )

    dataset = DreamBoothDataset(
        instance_data_root=data_cfg["instance_data_dir"],
        instance_prompt=data_cfg["instance_prompt"],
        tokenizer=components.tokenizer,
        size=int(data_cfg["resolution"]),
    )
    dataloader = DataLoader(dataset, batch_size=int(train_cfg["batch_size"]), shuffle=True)

    optimizer = torch.optim.AdamW(components.unet.parameters(), lr=float(train_cfg["learning_rate"]))

    components.unet, optimizer, dataloader = accelerator.prepare(
        components.unet, optimizer, dataloader
    )
    components.vae.to(accelerator.device, dtype=torch.float32)
    components.text_encoder.to(accelerator.device, dtype=torch.float32)

    loss_fn = get_loss_function(
        loss_cfg["name"],
        noise_scheduler=components.noise_scheduler,
        alpha=loss_cfg.get("alpha", 0.8),
        power=loss_cfg.get("power", 2.0),
        wavelet_type=loss_cfg.get("wavelet_type", "db1"),
        levels=loss_cfg.get("levels", 1),
        weighting=loss_cfg.get("weighting", "normalized"),
    )

    max_steps = int(train_cfg["max_train_steps"])
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    while global_step < max_steps:
        for batch in dataloader:
            with accelerator.accumulate(components.unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                with torch.no_grad():
                    latents = components.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * components.vae.config.scaling_factor
                latents = latents.to(dtype=compute_dtype)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    components.noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = components.noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = components.text_encoder(
                        batch["input_ids"].to(accelerator.device)
                    )[0]

                with accelerator.autocast():
                    model_pred = components.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states.to(compute_dtype),
                    ).sample
                    loss = loss_fn(model_pred, noise, timesteps=timesteps)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(components.unet.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1
                progress.set_postfix({"loss": float(loss.detach().item())})

            if global_step >= max_steps:
                break

    accelerator.wait_for_everyone()
    save_path = out_dir / "unet"
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(components.unet)
        unet_unwrapped.save_pretrained(save_path)
        logger.info("saved DreamBooth UNet to %s", save_path)
    return save_path
