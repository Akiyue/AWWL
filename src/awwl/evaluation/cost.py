"""Measure the real cost of each loss: time, memory, FLOPs, parameters.

The paper asserts the objective "imposes negligible computational cost
compared to the backbone network" without a single number, while the related
work it is positioned against (WaveDiff and friends) reports parameter,
FLOP and memory tables. This produces the missing evidence — and it is
falsifiable: a DWT on every training step is not free, and a *learned*
weighting adds parameters and a second backward pass.

What is measured, per loss:

* **step time** — median wall-clock of a full training step (forward,
  loss, backward), which is what "negligible" has to be judged against.
  Median, not mean, because a stray scheduler hiccup should not decide it.
* **loss time** — the loss call alone, so its share of the step is visible.
* **peak memory** — CUDA peak allocated during a step; the DWT materialises
  sub-band tensors and that shows up here.
* **FLOPs** — from the PyTorch profiler when it can attribute them.
* **loss parameters** — non-zero only for the learned objectives.

Every figure is reported as an absolute value and as a ratio against MSE, so
the claim under test ("negligible overhead") is read directly off the ratio
column rather than inferred.
"""

from __future__ import annotations

import contextlib
import logging
import statistics
import time
from typing import Any

import torch

from awwl.losses import get_loss_function, trainable_loss_parameters

logger = logging.getLogger(__name__)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def measure_loss_cost(
    loss_name: str,
    *,
    model: torch.nn.Module,
    noise_scheduler: Any,
    batch_size: int = 128,
    image_size: int = 32,
    channels: int = 3,
    device: str = "cuda",
    iters: int = 20,
    warmup: int = 5,
    loss_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Time one loss over ``iters`` full training steps on synthetic data.

    Synthetic inputs are deliberate: the question is the loss's cost, and a
    real dataloader would fold disk and decode time into the measurement.
    """
    loss_fn = get_loss_function(
        loss_name, noise_scheduler=noise_scheduler, **(loss_kwargs or {})
    )
    loss_params = trainable_loss_parameters(loss_fn)
    from awwl.losses import loss_module

    module = loss_module(loss_fn)
    if module is not None:
        module.to(device)

    params = list(model.parameters()) + loss_params
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    clean = torch.randn(batch_size, channels, image_size, image_size, device=device)
    n_timesteps = int(noise_scheduler.config.num_train_timesteps)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    step_times: list[float] = []
    loss_times: list[float] = []

    for it in range(warmup + iters):
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, n_timesteps, (batch_size,), device=device).long()
        noisy = noise_scheduler.add_noise(clean, noise, timesteps)

        _sync(device)
        step_start = time.perf_counter()

        pred = model(noisy, timesteps, return_dict=False)[0]

        _sync(device)
        loss_start = time.perf_counter()
        loss = loss_fn(pred, noise, timesteps=timesteps)
        _sync(device)
        loss_elapsed = time.perf_counter() - loss_start

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        step_elapsed = time.perf_counter() - step_start

        if it >= warmup:  # discard warm-up: first steps pay allocator + cuDNN costs
            step_times.append(step_elapsed)
            loss_times.append(loss_elapsed)

    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if device.startswith("cuda") else float("nan")
    )

    return {
        "loss": loss_name,
        "step_ms": statistics.median(step_times) * 1e3,
        "loss_ms": statistics.median(loss_times) * 1e3,
        "peak_mb": peak_mb,
        "loss_params": sum(p.numel() for p in loss_params),
        "flops": _profile_flops(
            model, loss_fn, clean, noise_scheduler, n_timesteps, device=device
        ),
    }


def _profile_flops(
    model, loss_fn, clean, noise_scheduler, n_timesteps: int, *, device: str
) -> float:
    """FLOPs for one forward+loss, or NaN when the profiler cannot attribute them."""
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:  # pragma: no cover
        return float("nan")

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    try:
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, n_timesteps, (clean.shape[0],), device=device).long()
        noisy = noise_scheduler.add_noise(clean, noise, timesteps)
        with profile(activities=activities, with_flops=True) as prof:
            pred = model(noisy, timesteps, return_dict=False)[0]
            loss_fn(pred, noise, timesteps=timesteps)
        total = sum(evt.flops for evt in prof.key_averages() if evt.flops)
        return float(total) if total else float("nan")
    except Exception as exc:  # pragma: no cover - profiler support varies
        logger.warning("FLOP profiling unavailable: %s", exc)
        return float("nan")


def measure_costs(
    cfg: dict[str, Any],
    loss_names: list[str],
    *,
    iters: int = 20,
    warmup: int = 5,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Measure every loss in ``loss_names`` against the config's backbone.

    A fresh model is built per loss so one measurement cannot inherit
    another's allocator state.
    """
    from diffusers import DDPMScheduler

    from awwl.models.ddpm_unet import build_ddpm_unet

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    sched_cfg = cfg.get("scheduler", {"num_train_timesteps": 1000})

    rows: list[dict[str, Any]] = []
    for name in loss_names:
        logger.info("measuring %s", name)
        model = build_ddpm_unet(
            image_size=int(model_cfg["image_size"]),
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            layers_per_block=int(model_cfg["layers_per_block"]),
            block_out_channels=tuple(model_cfg["block_out_channels"]),
            down_block_types=tuple(model_cfg["down_block_types"]),
            up_block_types=tuple(model_cfg["up_block_types"]),
        ).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=int(sched_cfg["num_train_timesteps"]))

        loss_kwargs = dict(cfg.get("loss", {}))
        loss_kwargs.pop("name", None)
        row = measure_loss_cost(
            name,
            model=model,
            noise_scheduler=scheduler,
            batch_size=int(data_cfg["batch_size"]),
            image_size=int(data_cfg["image_size"]),
            channels=int(model_cfg["in_channels"]),
            device=device,
            iters=iters,
            warmup=warmup,
            loss_kwargs=loss_kwargs,
        )
        row["backbone_params"] = sum(p.numel() for p in model.parameters())
        rows.append(row)

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return rows


def format_cost_table(rows: list[dict[str, Any]], *, baseline: str = "mse") -> str:
    """Render the measurements as markdown, with ratios against ``baseline``."""
    if not rows:
        return "no measurements"
    by_name = {r["loss"]: r for r in rows}
    ref = by_name.get(baseline, rows[0])

    lines = [
        f"| Loss | step (ms) | vs {ref['loss']} | loss (ms) | share of step | peak (MB) | GFLOPs | loss params |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        ratio = row["step_ms"] / ref["step_ms"] if ref["step_ms"] else float("nan")
        share = 100.0 * row["loss_ms"] / row["step_ms"] if row["step_ms"] else float("nan")
        gflops = row["flops"] / 1e9 if row["flops"] == row["flops"] else float("nan")
        peak = "n/a" if row["peak_mb"] != row["peak_mb"] else f"{row['peak_mb']:.0f}"
        gf = "n/a" if gflops != gflops else f"{gflops:.2f}"
        lines.append(
            f"| `{row['loss']}` | {row['step_ms']:.1f} | {ratio:.2f}x | {row['loss_ms']:.2f} "
            f"| {share:.1f}% | {peak} | {gf} | {row['loss_params']:,} |"
        )

    backbone = rows[0].get("backbone_params")
    if backbone:
        lines.append("")
        lines.append(f"Backbone: {backbone:,} parameters.")
    lines.append("")
    lines.append(
        "`vs` is total training-step time relative to the baseline — the number the "
        "'negligible computational cost' claim should be stated against. Peak memory "
        "is CUDA-only and reads `n/a` on CPU."
    )
    return "\n".join(lines)


@contextlib.contextmanager
def _noop():  # pragma: no cover - retained for symmetry with profiler contexts
    yield
