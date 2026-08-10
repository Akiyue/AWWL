"""``awwl`` command-line entry point.

Subcommands:

* ``train`` — run a method's trainer with a YAML config.
* ``infer`` — sample images from a fine-tuned checkpoint.
* ``eval`` — run an evaluation suite (CLIP / FID-IS / advanced).
* ``eval-samples`` — score one sample folder and append to the results ledger.
* ``pipeline`` — run / inspect / reset a resumable multi-GPU sweep.
* ``stats`` — confidence intervals and significance tests over the ledger.
* ``list-checkpoints`` — print logical names from the registry.

Every config-taking subcommand accepts ``--override key.path=value`` repeats
to tweak the config without editing the YAML.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

from awwl.core.exceptions import AWWLError
from awwl.methods import KNOWN_METHODS, get_trainer
from awwl.utils import (
    apply_overrides,
    load_yaml,
    resolve_weights,
    set_seed,
    setup_logging,
    use_utf8_output,
)

logger = logging.getLogger("awwl.cli")

app = typer.Typer(add_completion=False, no_args_is_help=True, help="AWWL command-line interface.")
pipeline_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Resumable experiment sweeps.")
app.add_typer(pipeline_app, name="pipeline")


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
    sampler: str = typer.Option("ddpm", "--sampler", help="finetune only: ddpm | ddim."),
    steps: int | None = typer.Option(None, "--steps", help="Denoising steps (default 1000 ddpm / 100 ddim)."),
    batch_size: int = typer.Option(128, "--batch-size", help="finetune only: samples per forward pass."),
    sample_seed: int | None = typer.Option(None, "--sample-seed", help="finetune only: seeds the sampling noise."),
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
            batch_size=batch_size,
            num_inference_steps=steps,
            sampler=sampler,  # type: ignore[arg-type]
            seed=sample_seed,
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


@app.command("prepare-data")
def prepare_data_cmd(
    output: Path = typer.Option(Path("./data/cifar10_train_png"), "--output", "-o", help="Reference PNG folder."),
    dataset: str = typer.Option("cifar10", "--dataset", help="'cifar10', a HuggingFace id, or a local folder."),
    image_size: int = typer.Option(32, "--image-size", help="Resolution to write; must match what the model generates."),
    split: str = typer.Option("train", "--split", help="Dataset split to dump."),
    max_images: int | None = typer.Option(None, "--max-images", help="Cap the dump (default: all)."),
) -> None:
    """Dump a dataset to PNGs for use as the FID/KID reference set.

    Run once per (dataset, resolution) before ``awwl pipeline run``. The
    reference must be written at the resolution the model generates — FID
    compares Inception features of both sets, so a mismatch silently produces
    numbers that cannot be compared with published ones.

    Idempotent: re-running skips an existing dump.
    """
    setup_logging("INFO")
    from awwl.data.images import dump_reference_images

    out = dump_reference_images(
        dataset_name=dataset,
        output_dir=output,
        image_size=image_size,
        split=split,
        max_images=max_images,
    )
    typer.echo(f"reference images: {out} ({sum(1 for _ in out.glob('*.png'))} files)")


@app.command("eval-samples")
def eval_samples_cmd(
    run_dir: Path = typer.Option(..., "--run-dir", help="Training run folder (supplies config.json identity)."),
    samples: Path = typer.Option(..., "--samples", exists=True, help="Folder of generated PNGs."),
    real: Path = typer.Option(..., "--real", exists=True, help="Reference image folder."),
    ledger: Path = typer.Option(..., "--ledger", help="results.jsonl to append to."),
    epoch: int | None = typer.Option(None, "--epoch", help="Checkpoint epoch these samples came from."),
    max_images: int = typer.Option(10000, "--max-images", help="Cap for KID / precision-recall."),
    skip_advanced: bool = typer.Option(False, "--skip-advanced", help="FID + IS only (much faster)."),
) -> None:
    """Score one sample folder and append a row to the results ledger.

    Reads ``<run-dir>/config.json`` so the ledger row carries the run's full
    hyperparameter identity, which is what ``awwl stats`` groups on.
    """
    setup_logging("INFO")
    from awwl.analysis.results import append_result, result_row
    from awwl.evaluation import compute_advanced_metrics, compute_fid_is

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        typer.echo(f"no config.json in {run_dir}; was this run produced by `awwl train`?", err=True)
        raise typer.Exit(2)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    metrics = dict(
        compute_fid_is(fake_folder=samples, real_folder=real, exp_name=run_dir.name)
    )
    if not skip_advanced:
        metrics.update(
            compute_advanced_metrics(
                real_folder=real, fake_folder=samples, exp_name=run_dir.name, max_images=max_images
            )
        )

    output_cfg = cfg.get("output", {})
    append_result(
        ledger,
        result_row(
            cfg,
            exp=str(output_cfg.get("name", run_dir.name)),
            group=str(output_cfg.get("group", cfg.get("loss", {}).get("name", "?"))),
            kind="eval",
            metrics=metrics,
            epoch=epoch,
            samples_dir=str(samples),
            num_samples=sum(1 for _ in samples.glob("*.png")),
        ),
    )
    typer.echo(json.dumps(metrics, indent=2, default=str))
    typer.echo(f"appended to {ledger}")


@app.command("eval-dreambooth")
def eval_dreambooth_cmd(
    run_dir: Path = typer.Option(..., "--run-dir", help="DreamBooth run folder (holds unet/ and config.json)."),
    real: Path = typer.Option(..., "--real", exists=True, help="Folder of real subject images."),
    ledger: Path = typer.Option(..., "--ledger", help="results.jsonl to append to."),
    prompts_file: Path = typer.Option(Path("assets/prompts/awwl_dreambooth.txt"), "--prompts", help="One prompt per line."),
    num_images: int = typer.Option(20, "--num-images", help="Images generated per prompt."),
    steps: int = typer.Option(50, "--steps", help="Denoising steps per image."),
    guidance: float = typer.Option(7.5, "--guidance"),
    base_model: str = typer.Option("runwayml/stable-diffusion-v1-5", "--base-model"),
    device: str = typer.Option("cuda", "--device"),
) -> None:
    """Score one DreamBooth run: CLIP alignment + subject similarity.

    Generates images for each prompt, then writes a ledger row carrying both
    metrics — which is what lets ``awwl stats`` put confidence intervals on
    Table 1, whose gaps are currently far smaller than its own error bars.
    """
    setup_logging("INFO")
    import torch
    from transformers import CLIPModel, CLIPProcessor

    from awwl.analysis.results import append_result, result_row
    from awwl.evaluation import image_image_similarity, text_image_similarity
    from awwl.methods.dreambooth import build_pipeline, generate_images

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        typer.echo(f"no config.json in {run_dir}; was this produced by `awwl train`?", err=True)
        raise typer.Exit(2)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    prompts = [p.strip() for p in prompts_file.read_text(encoding="utf-8").splitlines() if p.strip()]
    if not prompts:
        typer.echo(f"no prompts in {prompts_file}", err=True)
        raise typer.Exit(2)

    instance_prompt = str(cfg.get("data", {}).get("instance_prompt", ""))
    mismatched = _prompts_not_matching_subject(instance_prompt, prompts)
    if mismatched:
        typer.echo(
            f"WARNING: {len(mismatched)} of {len(prompts)} prompts do not mention the subject "
            f"this run was trained on ({instance_prompt!r}).\n"
            "  CLIP score then measures agreement with a subject the model never saw, which is "
            "not what subject-driven fidelity means. Use a prompt file for this subject.\n"
            "  Mismatched: " + "; ".join(mismatched),
            err=True,
        )

    unet_dir = run_dir / "unet" if (run_dir / "unet").exists() else run_dir
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    pipe = build_pipeline(
        base_model=base_model,
        unet_dir=unet_dir,
        device=device if use_cuda else "cpu",
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )

    clip_device = "cuda" if use_cuda else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(clip_device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    clip_scores: list[float] = []
    similarities: list[float] = []
    for index, prompt in enumerate(prompts):
        out_dir = run_dir / "samples" / f"prompt{index}"
        generate_images(
            pipeline=pipe,
            prompt=prompt,
            seeds=list(range(1, num_images + 1)),
            output_dir=out_dir,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        images = sorted(p for p in out_dir.glob("*") if p.suffix.lower() in (".png", ".jpg"))
        clip_scores += text_image_similarity(
            clip_model=clip_model, clip_processor=clip_processor,
            prompt=prompt, image_paths=images, device=clip_device,
        )
        similarities += image_image_similarity(
            clip_model=clip_model, clip_processor=clip_processor,
            real_images_dir=real, generated_image_paths=images, device=clip_device,
        )

    import statistics

    metrics = {
        "clip_score": statistics.fmean(clip_scores),
        "clip_score_std": statistics.pstdev(clip_scores) if len(clip_scores) > 1 else 0.0,
        "similarity": statistics.fmean(similarities),
        "similarity_std": statistics.pstdev(similarities) if len(similarities) > 1 else 0.0,
        "n_images": len(clip_scores),
        "n_prompts": len(prompts),
    }
    output_cfg = cfg.get("output", {})
    append_result(
        ledger,
        result_row(
            cfg,
            exp=str(output_cfg.get("name", run_dir.name)),
            group=str(output_cfg.get("group", cfg.get("loss", {}).get("name", "?"))),
            kind="eval",
            metrics=metrics,
            instance_prompt=instance_prompt,
            prompts_file=str(prompts_file),
            prompt_subject_mismatches=len(mismatched),
        ),
    )
    typer.echo(json.dumps(metrics, indent=2))
    typer.echo(f"appended to {ledger}")


def _prompts_not_matching_subject(instance_prompt: str, prompts: list[str]) -> list[str]:
    """Prompts that omit the subject phrase the run was fine-tuned on.

    DreamBooth trains one subject per run, so a prompt naming a different one
    turns CLIP score into a measure of how well the model renders something it
    never learned. The paper's three prompts each belong to a different subject
    (dog / robot toy / vase) and therefore only make sense against three
    separately fine-tuned models — pairing all three with one run silently
    scores two of them against the wrong subject.

    Matching is on the identifier phrase from ``sks`` onward, which is how the
    rare-token convention marks the subject.
    """
    lowered = instance_prompt.lower()
    if "sks" not in lowered:
        return []
    subject = lowered[lowered.index("sks") :].strip().rstrip(".,")
    return [p for p in prompts if subject not in p.lower()]


@app.command("measure-cost")
def measure_cost_cmd(
    config: Path = typer.Option(Path("configs/finetune.yaml"), "--config", "-c", exists=True),
    override: list[str] = typer.Option([], "--override", "-o", help="key.path=value override (repeatable)."),
    losses: str = typer.Option(
        "mse,adaptive_wavelet,wavelet_subband,wavelet_spatial,wavelet_learned,wavelet_lifting",
        "--losses",
        help="Comma-separated loss names to compare.",
    ),
    iters: int = typer.Option(20, "--iters", help="Timed steps per loss (after warm-up)."),
    warmup: int = typer.Option(5, "--warmup"),
    out: Path | None = typer.Option(None, "--out", help="Also write the markdown table here."),
) -> None:
    """Measure step time, memory, FLOPs and parameters per loss.

    Produces the evidence the "negligible computational cost" claim currently
    lacks — and can falsify it: the DWT is not free, and a learned weighting
    adds both parameters and a second backward pass.
    """
    setup_logging("INFO")
    from awwl.evaluation import format_cost_table, measure_costs

    cfg = _load_cfg(config, override)
    names = [n.strip() for n in losses.split(",") if n.strip()]
    table = format_cost_table(measure_costs(cfg, names, iters=iters, warmup=warmup))
    typer.echo(table)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
        typer.echo(f"\nwritten to {out}")


@app.command("plot-curriculum")
def plot_curriculum_cmd(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, help="Run folder with config.json."),
    out: Path | None = typer.Option(None, "--out", help="Output image (default <run>/curriculum.png)."),
    points: int = typer.Option(101, "--points", help="How many sigma values to evaluate."),
) -> None:
    """Plot the sub-band schedule a trained run actually applies.

    For the learned objectives this is the headline figure: it shows whether
    the network rediscovered coarse-to-fine on its own, and whether the total
    weight stays flat across the schedule or drifts the way the published
    equations do.
    """
    setup_logging("INFO")
    from awwl.plotting.curriculum import plot_run_curriculum

    typer.echo(f"wrote {plot_run_curriculum(run_dir, out_path=out, points=points)}")


@pipeline_app.command("run")
def pipeline_run_cmd(
    manifest: Path = typer.Option(..., "--manifest", "-m", exists=True, help="Pipeline manifest YAML."),
    gpus: str | None = typer.Option(None, "--gpus", help="Comma-separated device ids, e.g. 0,1. Defaults to all."),
    max_tier: int | None = typer.Option(None, "--max-tier", help="Only run jobs at or below this tier."),
    max_attempts: int = typer.Option(3, "--max-attempts", help="Retries before a job is parked as failed."),
    stale_after: float = typer.Option(900.0, "--stale-after", help="Seconds without a heartbeat before requeueing."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Expand the manifest and print the plan, run nothing."),
) -> None:
    """Run a sweep, resuming whatever a previous invocation left unfinished.

    Safe to re-run at any time: finished jobs are skipped, jobs interrupted by
    a crash are requeued, and each training job picks up from its own last
    checkpoint. If the server dies mid-sweep, run the identical command again.
    """
    setup_logging("INFO")
    from awwl.pipeline import build_jobs, default_gpus, format_status, load_manifest, run_pipeline
    from awwl.pipeline.store import JobStore

    spec = load_manifest(manifest)
    jobs = build_jobs(spec, manifest_dir=manifest.parent)
    root = Path(spec["output_root"])
    store = JobStore(root / "pipeline" / "state.db", stale_after=stale_after, max_attempts=max_attempts)
    added = store.add_jobs(jobs)
    typer.echo(f"manifest '{spec['name']}': {len(jobs)} job(s), {added} newly queued")

    if dry_run:
        for job in store.list_jobs():
            typer.echo(f"  [{job.status:>7}] tier{job.tier} {job.job_id}")
        raise typer.Exit(0)

    devices = [g.strip() for g in gpus.split(",") if g.strip()] if gpus else default_gpus()
    typer.echo(f"workers: {', '.join('gpu' + d for d in devices)}")

    run_pipeline(
        store,
        gpus=devices,
        log_dir=root / "pipeline" / "logs",
        max_tier=max_tier,
        cwd=Path.cwd(),
    )
    typer.echo("")
    typer.echo(format_status(store))

    # Exit non-zero only for jobs that exhausted their retries — a job that
    # failed once and then succeeded is a successful sweep.
    from awwl.pipeline.store import FAILED

    parked = store.list_jobs(status=FAILED)
    if parked:
        typer.echo("")
        for job in parked:
            log = root / "pipeline" / "logs" / (job.job_id.replace(":", "__") + ".log")
            typer.echo(f"--- {job.job_id}\n    full log: {log}\n", err=True)
        raise typer.Exit(1)


@pipeline_app.command("status")
def pipeline_status_cmd(
    manifest: Path = typer.Option(..., "--manifest", "-m", exists=True, help="Pipeline manifest YAML."),
) -> None:
    """Print queue progress and any failures."""
    setup_logging("WARNING")
    from awwl.pipeline import load_manifest
    from awwl.pipeline.runner import format_status
    from awwl.pipeline.store import JobStore

    spec = load_manifest(manifest)
    store = JobStore(Path(spec["output_root"]) / "pipeline" / "state.db")
    typer.echo(format_status(store))


@pipeline_app.command("reset")
def pipeline_reset_cmd(
    manifest: Path = typer.Option(..., "--manifest", "-m", exists=True, help="Pipeline manifest YAML."),
    running: bool = typer.Option(False, "--running", help="Also requeue jobs stuck in 'running'."),
) -> None:
    """Requeue failed jobs so the next ``pipeline run`` retries them."""
    setup_logging("WARNING")
    from awwl.pipeline import load_manifest
    from awwl.pipeline.store import FAILED, RUNNING, JobStore

    spec = load_manifest(manifest)
    store = JobStore(Path(spec["output_root"]) / "pipeline" / "state.db")
    statuses = (FAILED, RUNNING) if running else (FAILED,)
    typer.echo(f"requeued {store.reset(statuses=statuses)} job(s)")


@app.command("stats")
def stats_cmd(
    ledger: Path = typer.Option(..., "--ledger", "-l", exists=True, help="results.jsonl, or a folder holding some."),
    metric: str = typer.Option("fid", "--metric", help="Ledger field to analyse, e.g. fid, is_mean, kid."),
    baseline: str | None = typer.Option(None, "--baseline", help="Config to test the others against, e.g. mse."),
    kind: str = typer.Option("eval", "--kind", help="Ledger row kind to read."),
    epoch: int | None = typer.Option(None, "--epoch", help="Restrict to one checkpoint epoch."),
    curve: bool = typer.Option(False, "--curve", help="Print metric-vs-epoch instead of a significance test."),
    alpha: float = typer.Option(0.05, "--alpha", help="Family-wise significance level."),
) -> None:
    """Summarise the results ledger with confidence intervals and paired tests.

    Without ``--baseline`` it prints mean ± std and a 95% CI per configuration.
    With one, it adds a seed-paired t-test and Wilcoxon test of every other
    configuration against it, Holm-corrected across the comparisons.
    """
    setup_logging("WARNING")
    from awwl.analysis import compare_to_baseline, load_results, summarize_groups
    from awwl.analysis.stats import convergence_table, format_comparison_table, format_summary_table

    rows = load_results(ledger, kind=kind)
    if epoch is not None:
        rows = [r for r in rows if r.get("epoch") == epoch]
    if not rows:
        typer.echo(f"no '{kind}' rows in {ledger}" + (f" at epoch {epoch}" if epoch else ""), err=True)
        raise typer.Exit(1)

    if curve:
        typer.echo(convergence_table(rows, metric=metric))
        raise typer.Exit(0)

    typer.echo(format_summary_table(summarize_groups(rows, metric=metric), metric=metric))
    if baseline:
        typer.echo("")
        typer.echo(format_comparison_table(compare_to_baseline(rows, metric=metric, baseline=baseline, alpha=alpha)))


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
    use_utf8_output()
    try:
        app()
    except AWWLError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
