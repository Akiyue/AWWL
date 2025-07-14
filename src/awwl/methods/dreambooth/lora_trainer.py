"""DreamBooth + LoRA fine-tuning on a HuggingFace image dataset.

Replaces ``AWWL/finetune.py``. Same SD pipeline as the full DreamBooth
trainer, but only LoRA adapters on the attention layers are updated.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from awwl.data.hf_image_dataset import HuggingFaceImageDataset, load_hf_image_dataset
from awwl.losses import get_loss_function
from awwl.models.lora import add_lora_to_unet
from awwl.models.sd_components import load_sd_components
from awwl.training.accelerator import build_accelerator, compute_dtype_for
from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def train_dreambooth_lora(cfg: dict[str, Any]) -> Path:
    """Run a DreamBooth+LoRA fine-tune.

    Args:
        cfg: A merged config dict (see ``configs/dreambooth_lora.yaml``).

    Returns:
        Path to the saved UNet folder (with LoRA weights merged into
        ``attn_processors``).
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
    try:
        add_lora_to_unet(components.unet, rank=int(model_cfg.get("lora_rank", 4)))
    except Exception as exc:  # diffusers compatibility issue, surfaced loud and continue
        logger.warning("LoRA injection failed: %s — continuing with full UNet", exc)

    components.vae.to(accelerator.device, dtype=torch.float32)
    components.text_encoder.to(accelerator.device, dtype=torch.float32)

    transform = transforms.Compose(
        [
            transforms.Resize(
                int(data_cfg["resolution"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(int(data_cfg["resolution"])),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    hf_ds = load_hf_image_dataset(data_cfg["hf_dataset_name"], split=data_cfg["hf_split"])
    torch_ds = HuggingFaceImageDataset(
        hf_ds, transform=transform, image_column=data_cfg["image_column"]
    )
    dataloader = DataLoader(
        torch_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        components.unet.parameters(), lr=float(train_cfg["learning_rate"])
    )
    components.unet, optimizer, dataloader = accelerator.prepare(
        components.unet, optimizer, dataloader
    )
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
    prompt = data_cfg.get("prompt", "face photo")

    while global_step < max_steps:
        for images in dataloader:
            with accelerator.accumulate(components.unet):
                pixel_values = images.to(accelerator.device, dtype=torch.float32)
                with torch.no_grad():
                    latents = (
                        components.vae.encode(pixel_values).latent_dist.sample()
                        * components.vae.config.scaling_factor
                    )
                latents = latents.to(dtype=compute_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    components.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = components.noise_scheduler.add_noise(latents, noise, timesteps)

                input_ids = components.tokenizer(
                    [prompt] * bsz,
                    padding="max_length",
                    truncation=True,
                    max_length=components.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                with torch.no_grad():
                    encoder_hidden_states = components.text_encoder(input_ids)[0].to(
                        dtype=compute_dtype
                    )

                with accelerator.autocast():
                    model_pred = components.unet(
                        noisy_latents, timesteps, encoder_hidden_states
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
    save_path = out_dir / loss_cfg["name"]
    save_path.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(components.unet)
        unet_unwrapped.save_pretrained(save_path)
        logger.info("saved DreamBooth+LoRA UNet to %s", save_path)

    # Free CUDA cache so a follow-on inference call doesn't OOM.
    del components, optimizer, dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_path
