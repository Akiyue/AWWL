"""DreamBooth full-UNet fine-tuning trainer.

Replaces ``AWWL/dreambooth.py``. The training loop is identical in spirit but
all configuration arrives as a typed dict (loaded from
``configs/dreambooth.yaml``) and every hyperparameter is exposed there.

Table 1's differences (~0.01) are far smaller than its reported spread
(~0.035-0.057), so it needs the same seed replication as the CIFAR-10 table.
A run here is 400 steps at batch 1 — minutes, not hours — which makes
multi-seed DreamBooth by far the cheapest rigor available. What that needs
from the trainer is a per-(config, seed) output directory, a config snapshot
and a ledger row, all added below; the optimisation itself is untouched.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from awwl.analysis.results import append_result, result_row
from awwl.data.dreambooth_dataset import DreamBoothDataset
from awwl.losses import get_loss_function, trainable_loss_parameters
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
    seed = int(cfg.get("seed", 42))
    out_dir = ensure_dir(dreambooth_run_dir(cfg))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    started = time.time()

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
    # Seed the shuffle so two seeds of the same config differ in
    # initialisation noise, not also in the order the subject images arrive.
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(components.unet.parameters(), lr=float(train_cfg["learning_rate"]))

    components.unet, optimizer, dataloader = accelerator.prepare(
        components.unet, optimizer, dataloader
    )
    components.vae.to(accelerator.device, dtype=torch.float32)
    components.text_encoder.to(accelerator.device, dtype=torch.float32)

    loss_fn = get_loss_function(
        loss_cfg["name"],
        noise_scheduler=components.noise_scheduler,
        **{k: v for k, v in loss_cfg.items() if k != "name"},
    )
    # Learned objectives (wavelet_learned / _gradnorm / _lifting) carry their
    # own parameters; without this group they would stay frozen at their
    # initialisation while appearing to train normally.
    loss_params = trainable_loss_parameters(loss_fn)
    if loss_params:
        from awwl.losses import loss_module

        module = loss_module(loss_fn)
        if module is not None:
            module.to(accelerator.device)
        optimizer.add_param_group(
            {
                "params": loss_params,
                "lr": float(train_cfg.get("loss_learning_rate", train_cfg["learning_rate"])),
            }
        )
        logger.info("loss '%s' adds %d learnable tensor(s)", loss_cfg["name"], len(loss_params))

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

        output_cfg = cfg.get("output", {})
        ledger = output_cfg.get("ledger")
        append_result(
            Path(ledger) if ledger else out_dir / "results.jsonl",
            result_row(
                cfg,
                exp=str(output_cfg.get("name", out_dir.name)),
                group=str(output_cfg.get("group", loss_cfg["name"])),
                kind="train",
                metrics={
                    "global_step": global_step,
                    "train_seconds": round(time.time() - started, 1),
                    "checkpoint": str(save_path),
                    "instance_prompt": data_cfg.get("instance_prompt"),
                },
            ),
        )
    return save_path


def dreambooth_run_dir(cfg: dict[str, Any]) -> Path:
    """One directory per (config, seed).

    ``output.name`` wins when the pipeline sets it. The fallback appends the
    loss name and seed, because the previous behaviour — writing straight into
    ``output.dir`` — meant two seeds of the same config silently overwrote each
    other's weights.
    """
    output_cfg = cfg.get("output", {})
    base = Path(output_cfg["dir"])
    name = output_cfg.get("name")
    if name:
        return base / str(name)
    loss_name = cfg.get("loss", {}).get("name", "mse")
    return base / f"{loss_name}_s{cfg.get('seed', 42)}"
