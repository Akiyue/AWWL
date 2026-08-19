"""Supervised image-restoration trainer for the AWWL reframing.

The original recipe trains a diffusion model, so its loss compares predicted
noise against white noise and the frequency decomposition acts on noise. That
is a weak place to test a frequency-aware objective: the target has no
structure, and the only way the loss can matter is through FID on samples.

Restoration replaces the target with the image itself, where high-frequency
detail is the direct evaluation criterion (PSNR / SSIM / LPIPS). The same
wavelet-weighted loss now decomposes the *prediction residual against a
structured image*, which is exactly the setting its adaptive weighting was
designed for:

* high sigma  -> emphasise low-frequency structure (LL),
* low sigma   -> emphasise high-frequency detail (LH/HL/HH).

Degradation (Gaussian noise of per-sample sigma) is applied on the fly, the
same sigma drives the loss weighting and the model's time embedding, and the
training loop is otherwise the crash-safe recipe shared with
:mod:`awwl.methods.finetune` — same checkpoint manager, EMA, loss-history and
ledger bookkeeping.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from awwl.analysis.results import append_result, result_row
from awwl.data.degradation import add_noise, sample_sigmas, sigma_to_timesteps
from awwl.data.images import build_image_dataloader
from awwl.losses.adaptive_wavelet import AdaptiveWaveletLoss
from awwl.losses import analytic
from awwl.models.ddpm_unet import load_or_build_ddpm_unet
from awwl.training.accelerator import build_accelerator
from awwl.training.checkpointing import CheckpointManager
from awwl.training.ema import build_ema
from awwl.training.loss_history import LossHistoryLogger
from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def restoration_loss(cfg: dict[str, Any]):
    """Build ``(pred, target, sigmas) -> scalar`` for a restoration config.

    Only objectives that make sense without a diffusion schedule are
    available here: the pointwise pixel losses and the wavelet-weighted
    family. ``AdaptiveWaveletLoss`` is called directly with the degradation
    sigma rather than through the factory (whose wrapper derives sigma from a
    ``DDPMScheduler``).
    """
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "mse")
    if name in ("mse", "l1", "huber", "charbonnier"):
        fn = {
            "mse": analytic.mse,
            "l1": analytic.l1,
            "huber": analytic.huber,
            "charbonnier": analytic.charbonnier_loss,
        }[name]

        def _pixel(
            pred: torch.Tensor, target: torch.Tensor, *, sigmas: torch.Tensor
        ) -> torch.Tensor:
            del sigmas
            return fn(pred, target)

        return _pixel

    if name == "adaptive_wavelet":
        wavelet = AdaptiveWaveletLoss(
            levels=int(loss_cfg.get("levels", 1)),
            wavelet_type=str(loss_cfg.get("wavelet_type", "db1")),
            alpha=float(loss_cfg.get("alpha", 0.8)),
            power=float(loss_cfg.get("power", 2.0)),
            weighting=str(loss_cfg.get("weighting", "normalized")),  # type: ignore[arg-type]
            normalize_weights=bool(loss_cfg.get("normalize_weights", False)),
            normalize_scale=float(loss_cfg.get("normalize_scale", 1.0)),
            detail_reduction=str(loss_cfg.get("detail_reduction", "mean")),  # type: ignore[arg-type]
            level_reduction=str(loss_cfg.get("level_reduction", "sum")),  # type: ignore[arg-type]
            dwt_mode=str(loss_cfg.get("dwt_mode", "zero")),
        )

        def _wavelet(
            pred: torch.Tensor, target: torch.Tensor, *, sigmas: torch.Tensor
        ) -> torch.Tensor:
            return wavelet(pred, target, sigmas)

        return _wavelet

    raise ValueError(
        f"loss {name!r} is not available for restoration; use one of "
        "mse, l1, huber, charbonnier, adaptive_wavelet"
    )


def _num_timesteps(cfg: dict[str, Any]) -> int:
    """Timestep resolution of the model's conditioning embedding."""
    return int(cfg.get("model", {}).get("num_train_timesteps", 1000))


def train_restoration(cfg: dict[str, Any]) -> Path:
    """Train a regression UNet that maps noisy images back to clean ones.

    Args:
        cfg: A merged config dict (see ``configs/restoration.yaml``).

    Returns:
        Path to the final checkpoint folder.
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    degrad_cfg = cfg.get("degradation", {})
    seed = int(cfg.get("seed", 42))
    out_dir = ensure_dir(run_dir_for(cfg))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    sigma_min = float(degrad_cfg.get("sigma_min", 0.05))
    sigma_max = float(degrad_cfg.get("sigma_max", 0.5))
    num_timesteps = _num_timesteps(cfg)

    accelerator = build_accelerator(
        mixed_precision=cfg["precision"]["mixed_precision"],
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
    )

    dataloader = build_image_dataloader(
        dataset_name=str(data_cfg.get("dataset_name", "cifar10")),
        image_size=int(data_cfg["image_size"]),
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 4)),
        split=data_cfg.get("split", "train"),
        seed=seed,
        source=str(data_cfg.get("source", "auto")),
        root=str(data_cfg.get("root", "./data")),
    )

    model = load_or_build_ddpm_unet(
        weights_path=model_cfg.get("model_weights_path"),
        builder_kwargs={
            "image_size": int(model_cfg["image_size"]),
            "in_channels": int(model_cfg["in_channels"]),
            "out_channels": int(model_cfg["out_channels"]),
            "layers_per_block": int(model_cfg["layers_per_block"]),
            "block_out_channels": tuple(model_cfg["block_out_channels"]),
            "down_block_types": tuple(model_cfg["down_block_types"]),
            "up_block_types": tuple(model_cfg["up_block_types"]),
        },
    )

    loss_fn = restoration_loss(cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["learning_rate"]),
    )

    num_epochs = int(train_cfg["num_epochs"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(train_cfg["lr_warmup_steps"]),
        num_training_steps=len(dataloader) * num_epochs,
    )

    ema = build_ema(model, train_cfg)

    ckpt = CheckpointManager(
        out_dir,
        keep_last=int(train_cfg.get("keep_last_states", 2)),
        save_every_epochs=int(train_cfg.get("save_state_epochs", 5)),
    )

    resumed = None
    if bool(train_cfg.get("resume", True)):
        resumed = ckpt.load(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, ema=ema)

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    if ema is not None:
        ema.to(accelerator.device)

    start_epoch = resumed.epoch + 1 if resumed else 0
    global_step = resumed.global_step if resumed else 0
    if start_epoch >= num_epochs:
        logger.info("run already complete at epoch %d/%d; nothing to do", start_epoch, num_epochs)
        return _final_checkpoint(out_dir, num_epochs)

    history = LossHistoryLogger(
        output_dir=out_dir,
        config={
            "alpha": loss_cfg.get("alpha"),
            "power": loss_cfg.get("power"),
            "wavelet": loss_cfg.get("wavelet_type"),
            "loss_name": loss_cfg["name"],
            "seed": seed,
        },
        resume=resumed is not None,
    )

    save_every = int(train_cfg.get("save_model_epochs", 20))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    started = time.time()
    last_ckpt: Path | None = None

    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress = tqdm(
            total=len(dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"epoch {epoch}",
        )
        for batch in dataloader:
            with accelerator.accumulate(model):
                clean = batch["images"]
                sigmas = sample_sigmas(
                    clean.shape[0],
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    device=clean.device,
                )
                degraded = add_noise(clean, sigmas)
                timesteps = sigma_to_timesteps(
                    sigmas, num_timesteps=num_timesteps, sigma_min=sigma_min, sigma_max=sigma_max
                )
                restored = model(degraded, timesteps, return_dict=False)[0]
                loss = loss_fn(restored, clean, sigmas=sigmas)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if ema is not None:
                    ema.step(accelerator.unwrap_model(model))

            value = float(loss.detach().item())
            history.append(value)
            progress.update(1)
            progress.set_postfix(loss=value)
        progress.close()

        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
                last_ckpt = _save_model(unwrapped, ema, out_dir / f"checkpoint-{epoch}")
                history.flush()
                logger.info("saved checkpoint %s", last_ckpt)
            if ckpt.should_save(epoch, num_epochs):
                history.flush()
                ckpt.save(
                    epoch=epoch,
                    global_step=global_step,
                    model=unwrapped,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema=ema,
                    meta={"seed": seed, "loss_name": loss_cfg["name"]},
                )

    if last_ckpt is None:
        raise RuntimeError("training finished without saving any checkpoint")

    if accelerator.is_main_process:
        append_result(
            _ledger_path(cfg, out_dir),
            result_row(
                cfg,
                exp=str(cfg.get("output", {}).get("name", out_dir.name)),
                group=str(cfg.get("output", {}).get("group", loss_cfg["name"])),
                kind="train",
                metrics={
                    "final_epoch": num_epochs - 1,
                    "global_step": global_step,
                    "train_seconds": round(time.time() - started, 1),
                    "checkpoint": str(last_ckpt),
                },
            ),
        )
    return last_ckpt


def _save_model(model: torch.nn.Module, ema: Any, target: Path) -> Path:
    """Write a sampling-ready checkpoint, using EMA weights when present."""
    if ema is not None:
        with ema.as_active(model):
            model.save_pretrained(target)
    else:
        model.save_pretrained(target)
    return target


def _final_checkpoint(out_dir: Path, num_epochs: int) -> Path:
    """Path of the last checkpoint of a completed run."""
    final = out_dir / f"checkpoint-{num_epochs - 1}"
    if final.exists():
        return final
    existing = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if existing:
        return existing[-1]
    raise RuntimeError(
        f"{out_dir} reports a completed run but has no checkpoint-* folder; "
        "delete its state/ directory to retrain"
    )


def _ledger_path(cfg: dict[str, Any], out_dir: Path) -> Path:
    """Where to append the results row (shared ledger, or per-run fallback)."""
    configured = cfg.get("output", {}).get("ledger")
    return Path(configured) if configured else out_dir / "results.jsonl"


def run_dir_for(cfg: dict[str, Any]) -> Path:
    """Resolve a run's output directory (``output.name`` wins)."""
    output_cfg = cfg.get("output", {})
    base = Path(output_cfg["dir"])
    name = output_cfg.get("name")
    if name:
        return base / str(name)
    weights_path = cfg.get("model", {}).get("model_weights_path")
    if weights_path:
        return Path(weights_path).parent
    loss_name = cfg.get("loss", {}).get("name", "mse")
    seed = cfg.get("seed", 42)
    return Path(f"{base}_{loss_name}_s{seed}")
