"""From-scratch DDPM trainer for the AWWL-Diff recipe.

Replaces ``AWWL-Diff/train_cifar10.py``. The optimisation recipe is unchanged
from the paper's runs; what has been added is the machinery a multi-seed study
needs:

* **Auto-resume.** Full training state (optimiser moments, LR-scheduler
  position, EMA shadow, RNG) is snapshotted under ``<run>/state`` and picked
  up automatically on restart, so a killed job continues instead of silently
  starting over. See :mod:`awwl.training.checkpointing`.
* **Optional EMA** (``train.use_ema``), off by default — see
  :mod:`awwl.training.ema` for why it matters and why it is not the default.
* **Per-epoch sampling checkpoints** at ``train.save_model_epochs``, which are
  what the convergence-curve experiment evaluates.
* **A results-ledger row** per finished run, so statistics never depend on
  parsing directory names.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from awwl.analysis.results import append_result, result_row
from awwl.data.cifar10 import build_cifar10_dataloader
from awwl.losses import get_loss_function, loss_module, trainable_loss_parameters
from awwl.models.ddpm_unet import load_or_build_ddpm_unet
from awwl.training.accelerator import build_accelerator
from awwl.training.checkpointing import CheckpointManager
from awwl.training.ema import build_ema
from awwl.training.loss_history import LossHistoryLogger
from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def train_finetune(cfg: dict[str, Any]) -> Path:
    """Train an unconditional DDPM with the configured loss.

    Args:
        cfg: A merged config dict (see ``configs/finetune.yaml``).

    Returns:
        Path to the final saved sampling checkpoint folder.
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    sched_cfg = cfg["scheduler"]
    seed = int(cfg.get("seed", 42))
    out_dir = ensure_dir(run_dir_for(cfg))
    # Snapshot the resolved config so downstream evaluation can recover the
    # run's identity without re-deriving it from the directory name.
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    accelerator = build_accelerator(
        mixed_precision=cfg["precision"]["mixed_precision"],
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
    )

    dataloader = build_cifar10_dataloader(
        image_size=int(data_cfg["image_size"]),
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 4)),
        split=data_cfg.get("split", "train"),
        seed=seed,
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
    model = load_or_build_ddpm_unet(
        weights_path=model_cfg.get("model_weights_path"), builder_kwargs=builder_kwargs
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=int(sched_cfg["num_train_timesteps"]))

    loss_fn = get_loss_function(
        loss_cfg["name"],
        noise_scheduler=noise_scheduler,
        **{k: v for k, v in loss_cfg.items() if k != "name"},
    )
    # The learned-weighting and learnable-basis objectives carry parameters of
    # their own. They get their own group: a loss whose weights never update
    # would quietly collapse into a fixed-weight objective frozen at its
    # initialisation, which looks like a working experiment but tests nothing.
    loss_params = trainable_loss_parameters(loss_fn)
    param_groups: list[dict[str, Any]] = [
        {"params": list(model.parameters()), "lr": float(train_cfg["learning_rate"])}
    ]
    if loss_params:
        param_groups.append(
            {
                "params": loss_params,
                "lr": float(train_cfg.get("loss_learning_rate", train_cfg["learning_rate"])),
            }
        )
        logger.info(
            "loss '%s' contributes %d learnable tensor(s) to the optimiser",
            loss_cfg["name"],
            len(loss_params),
        )
    optimizer = torch.optim.AdamW(param_groups)

    num_epochs = int(train_cfg["num_epochs"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(train_cfg["lr_warmup_steps"]),
        num_training_steps=len(dataloader) * num_epochs,
    )

    ema = build_ema(model, train_cfg)

    # Resume before `accelerator.prepare` so the plain module is the one whose
    # state_dict is overwritten; prepare() may wrap it in DDP afterwards.
    ckpt = CheckpointManager(
        out_dir,
        keep_last=int(train_cfg.get("keep_last_states", 2)),
        save_every_epochs=int(train_cfg.get("save_state_epochs", 5)),
    )
    loss_core_pre = loss_module(loss_fn)
    ckpt_extras = {"loss": loss_core_pre} if loss_params else {}

    resumed = None
    if bool(train_cfg.get("resume", True)):
        resumed = ckpt.load(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            extras=ckpt_extras,
        )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    if ema is not None:
        ema.to(accelerator.device)
    if loss_core_pre is not None:
        # Moves parameter .data in place, so the optimiser's references stay valid.
        loss_core_pre.to(accelerator.device)

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

    # GradNorm balances sub-bands using the gradients they impose on the last
    # shared layer. `conv_out` is the natural choice for a UNet: every band's
    # error reaches the network through it.
    loss_core = loss_core_pre
    gradnorm_shared = None
    gradnorm_weights = None
    if loss_core is not None and getattr(loss_core, "gradnorm", None) is not None:
        base = accelerator.unwrap_model(model)
        shared = getattr(base, "conv_out", None)
        gradnorm_shared = list(shared.parameters()) if shared is not None else [
            p for p in base.parameters() if p.requires_grad
        ][-1:]
        gradnorm_weights = [loss_core.gradnorm.weights]
        logger.info("GradNorm active over %d sub-bands", loss_core.gradnorm.n_tasks)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress = tqdm(
            total=len(dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"epoch {epoch}",
        )
        for batch in dataloader:
            with accelerator.accumulate(model):
                clean_images = batch["images"]
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bsz = clean_images.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=clean_images.device,
                ).long()
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = loss_fn(noise_pred, noise, timesteps=timesteps)

                # GradNorm tunes its weights from the gradients the sub-bands
                # impose on the shared trunk, so it needs its own second-order
                # pass *before* the network's backward frees the graph.
                if gradnorm_shared is not None:
                    aux = loss_core.gradnorm_loss(gradnorm_shared)
                    if aux is not None:
                        aux.backward(retain_graph=True, inputs=gradnorm_weights)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if gradnorm_shared is not None:
                    # Rescale the balancer's weights back to a fixed sum, so
                    # their overall magnitude cannot drift into an implicit
                    # learning-rate change.
                    loss_core.after_optimizer_step()

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
                last_ckpt = _export_pipeline(
                    unwrapped, noise_scheduler, ema, out_dir / f"checkpoint-{epoch}"
                )
                history.flush()
                logger.info("saved sampling checkpoint %s", last_ckpt)
            if ckpt.should_save(epoch, num_epochs):
                history.flush()
                ckpt.save(
                    epoch=epoch,
                    global_step=global_step,
                    model=unwrapped,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema=ema,
                    extras=ckpt_extras,
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


def _export_pipeline(
    model: torch.nn.Module,
    noise_scheduler: DDPMScheduler,
    ema: Any | None,
    target: Path,
) -> Path:
    """Write a sampling-ready ``DDPMPipeline``, using EMA weights when present."""
    if ema is not None:
        with ema.as_active(model):
            DDPMPipeline(unet=model, scheduler=noise_scheduler).save_pretrained(target)
    else:
        DDPMPipeline(unet=model, scheduler=noise_scheduler).save_pretrained(target)
    return target


def _final_checkpoint(out_dir: Path, num_epochs: int) -> Path:
    """Path of the last sampling checkpoint of a completed run."""
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
    """Resolve a run's output directory.

    ``output.name`` wins when set (the pipeline runner always sets it, which
    keeps one directory per experiment × seed). Otherwise fall back to the
    AWWL-Diff convention ``<dir>_a<alpha>_p<power>_<wavelet>``, extended with
    the loss name and seed — without those, two seeds or two losses would
    write into the same folder and resume from each other's state.
    """
    output_cfg = cfg.get("output", {})
    base = Path(output_cfg["dir"])
    name = output_cfg.get("name")
    if name:
        return base / str(name)

    weights_path = cfg.get("model", {}).get("model_weights_path")
    if weights_path:
        return Path(weights_path).parent

    loss_cfg = cfg.get("loss", {})
    loss_name = loss_cfg.get("name", "mse")
    seed = cfg.get("seed", 42)
    if loss_name == "adaptive_wavelet":
        tag = (
            f"a{loss_cfg.get('alpha', 0.8)}"
            f"_p{loss_cfg.get('power', 2.0)}"
            f"_{loss_cfg.get('wavelet_type', 'db1')}"
        )
    else:
        tag = loss_name
    return Path(f"{base}_{tag}_s{seed}")
