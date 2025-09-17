"""``awwl`` command-line entry point.

Subcommands:

* ``train`` — run a method's trainer with a YAML config.
* ``infer`` — sample images from a fine-tuned checkpoint.
* ``eval`` — run an evaluation suite (CLIP / FID-IS / advanced).
* ``list-checkpoints`` — print logical names from the registry.

Every subcommand accepts ``--override key.path=value`` repeats to tweak the
config without editing the YAML.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from awwl.core.exceptions import AWWLError
from awwl.methods import KNOWN_METHODS, get_trainer
from awwl.utils import apply_overrides, load_yaml, resolve_weights, set_seed, setup_logging

logger = logging.getLogger("awwl.cli")

app = typer.Typer(add_completion=False, no_args_is_help=True, help="AWWL command-line interface.")


def _load_cfg(config: Path, overrides: list[str]) -> dict:
    cfg = load_yaml(config)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    if "seed" in cfg:
        set_seed(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", False)))
    return cfg


@app.command("train")
def train_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to a method YAML config."),
    override: list[str] = typer.Option([], "--override", "-o", help="key.path=value override (repeatable)."),
) -> None:
    """Train a model defined by ``config``."""
    cfg = _load_cfg(config, override)
    method = cfg.get("method")
    if method not in KNOWN_METHODS:
        typer.echo(f"unknown or missing method in config; expected one of {KNOWN_METHODS}", err=True)
        raise typer.Exit(2)
    trainer = get_trainer(method)
    saved = trainer(cfg)
    typer.echo(f"saved: {saved}")


@app.command("infer")
def infer_cmd(
    method: str = typer.Option(..., "--method", help="dreambooth | finetune"),
    weights: Path | None = typer.Option(None, "--weights", help="Direct path to a UNet/checkpoint folder."),
    registry_name: str | None = typer.Option(None, "--registry", help="Logical name in registry.yaml."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o"),
    prompt: str = typer.Option("a photo of sks robot toy on the beach at sunset", "--prompt"),
    num_samples: int = typer.Option(100, "--num-samples", "-n"),
    base_model: str = typer.Option("runwayml/stable-diffusion-v1-5", "--base-model", help="DreamBooth only."),
    project_root: Path | None = typer.Option(None, "--project-root", help="Defaults to the parent of this CLI."),
    device: str = typer.Option("cuda", "--device"),
) -> None:
    """Run inference. Either ``--weights`` or ``--registry`` is required."""
    setup_logging("INFO")
    root = project_root or Path(__file__).resolve().parents[2]
    resolved = resolve_weights(
        method=method,
        explicit_path=str(weights) if weights else None,
        registry_name=registry_name,
        project_root=root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "dreambooth":
        import torch

        from awwl.methods.dreambooth import build_pipeline, generate_images

        pipe = build_pipeline(
            base_model=base_model,
            unet_dir=resolved,
            device=device if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        )
        generate_images(
            pipeline=pipe, prompt=prompt,
            seeds=list(range(1, num_samples + 1)),
            output_dir=output_dir,
        )
    elif method == "finetune":
        from awwl.methods.finetune import generate_samples

        generate_samples(
            checkpoint_path=resolved,
            output_dir=output_dir,
            num_samples=num_samples,
            device=device,
        )
    else:
        typer.echo(f"unknown method {method!r}", err=True)
        raise typer.Exit(2)
    typer.echo(f"images in: {output_dir}")


@app.command("eval")
def eval_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Eval YAML config."),
    override: list[str] = typer.Option([], "--override", "-o", help="key.path=value override."),
) -> None:
    """Run an evaluation suite (``clip``, ``fid_is``, or ``advanced``)."""
    cfg = _load_cfg(config, override)
    suite = cfg.get("eval")

    if suite == "clip":
        from awwl.evaluation import evaluate_clip_over_models

        gen = cfg["generate"]
        prompts = Path(gen["prompts_file"]).read_text(encoding="utf-8").splitlines()
        prompts = [p.strip() for p in prompts if p.strip()]
        evaluate_clip_over_models(
            models_root=gen["models_root"],
            real_images_dir=gen["real_images_dir"],
            prompts=prompts,
            out_root=gen["out_root"],
            base_model=gen["base_model"],
            clip_model_name=cfg["clip"]["model_name"],
            clip_batch_size=int(cfg["clip"]["batch_size"]),
            n_images_per_prompt=int(gen["n_images_per_prompt"]),
            pipeline_batch_size=int(gen["batch_size"]),
            num_inference_steps=int(gen["num_inference_steps"]),
            guidance_scale=float(gen["guidance_scale"]),
            resolution=int(gen["resolution"]),
        )
    elif suite == "fid_is":
        from awwl.evaluation import compute_fid_is

        f = cfg["fid_is"]
        compute_fid_is(
            fake_folder=f["out_folder"],
            real_folder=f["real_folder"],
            log_file=f.get("log_file"),
            exp_name=Path(f["out_folder"]).name,
            is_batch_size=int(f["batch_size"]),
        )
    elif suite == "advanced":
        from awwl.evaluation import compute_advanced_metrics

        a = cfg["advanced"]
        compute_advanced_metrics(
            real_folder=a["real_folder"],
            fake_folder=a["fake_folder"],
            log_file=a.get("log_file"),
            exp_name=Path(a["fake_folder"]).name,
            batch_size=int(a["batch_size"]),
            max_images=int(a["max_images"]),
        )
    else:
        typer.echo(f"unknown eval suite {suite!r}", err=True)
        raise typer.Exit(2)


@app.command("list-checkpoints")
def list_checkpoints(
    project_root: Path | None = typer.Option(None, "--project-root"),
) -> None:
    """Print the registry's logical names per method."""
    setup_logging("INFO")
    root = project_root or Path(__file__).resolve().parents[2]
    registry = load_yaml(root / "configs" / "checkpoints" / "registry.yaml")
    for method, entries in registry.items():
        typer.echo(f"# {method}")
        for name, path in entries.items():
            typer.echo(f"  {name:<24} {path}")


def main() -> None:
    """Module entry-point used both by ``python -m awwl.cli`` and the script alias."""
    try:
        app()
    except AWWLError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
