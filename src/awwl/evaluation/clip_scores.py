"""CLIP-based evaluation: text-image and image-image similarity.

Replaces ``AWWL/evaluate.py`` (canonical) and ``AWWL/evaluate_image_similarity.py``
(merged into the image-image function below). The older ``AWWL/eval.py`` was
a strict subset of ``evaluate.py`` and was dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from awwl.utils.io import ensure_dir

logger = logging.getLogger(__name__)

_VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in _VALID_EXTS)


def _features(output: object) -> torch.Tensor:
    """The projected embedding, whichever way CLIP hands it back.

    ``get_text_features`` / ``get_image_features`` returned a bare tensor
    through transformers 4.x. In 5.x they return a ``BaseModelOutputWithPooling``
    whose ``pooler_output`` holds the projected embedding. Both are in the wild,
    and reading the wrong one fails loudly rather than quietly -- but only once
    an image has already been generated, which in a sweep means after the
    expensive part.
    """
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    raise TypeError(
        f"CLIP returned {type(output).__name__}, which carries no embedding tensor "
        "on `.pooler_output`. This is a transformers API change; see _features()."
    )


@torch.no_grad()
def _embed_images(
    image_paths: Iterable[Path],
    *,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """L2-normalised image embeddings, shape ``(N, D)``."""
    embs: list[torch.Tensor] = []
    paths = list(image_paths)
    for i in tqdm(range(0, len(paths), batch_size), desc="clip-embed", leave=False):
        batch = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
        inputs = clip_processor(images=batch, return_tensors="pt").to(device)
        emb = _features(clip_model.get_image_features(**inputs))
        embs.append(F.normalize(emb, p=2, dim=-1).cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def text_image_similarity(
    *,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    prompt: str,
    image_paths: list[Path],
    device: str,
    batch_size: int = 32,
) -> list[float]:
    """Cosine similarity between ``prompt`` and each image in ``image_paths``."""
    text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    text_emb = _features(clip_model.get_text_features(**text_inputs))
    text_emb = F.normalize(text_emb, p=2, dim=-1)

    scores: list[float] = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = clip_processor(images=images, return_tensors="pt").to(device)
        image_emb = _features(clip_model.get_image_features(**inputs))
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        sims = (image_emb @ text_emb.T).squeeze(-1).cpu().tolist()
        if isinstance(sims, float):
            sims = [sims]
        scores.extend(sims)
    return scores


@torch.no_grad()
def image_image_similarity(
    *,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    real_images_dir: str | Path,
    generated_image_paths: list[Path],
    device: str,
    batch_size: int = 32,
) -> list[float]:
    """Cosine similarity between each generated image and the *mean* real-image embedding.

    The mean-embedding shortcut comes from the original ``evaluate_image_similarity.py``
    and is the standard way to assess subject fidelity for DreamBooth runs.
    """
    real_paths = _list_images(Path(real_images_dir))
    if not real_paths:
        raise FileNotFoundError(f"no real images in {real_images_dir}")
    real_embs = _embed_images(
        real_paths,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
        batch_size=batch_size,
    )
    avg = F.normalize(real_embs.mean(dim=0, keepdim=True), p=2, dim=-1).to(device)

    scores: list[float] = []
    for i in range(0, len(generated_image_paths), batch_size):
        batch_paths = generated_image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = clip_processor(images=images, return_tensors="pt").to(device)
        gen_emb = _features(clip_model.get_image_features(**inputs))
        gen_emb = F.normalize(gen_emb, p=2, dim=-1)
        sim = (gen_emb @ avg.T).squeeze().cpu().tolist()
        if isinstance(sim, float):
            sim = [sim]
        scores.extend(sim)
    return scores


def evaluate_clip_over_models(
    *,
    models_root: str | Path,
    real_images_dir: str | Path,
    prompts: list[str],
    out_root: str | Path,
    base_model: str,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    n_images_per_prompt: int = 20,
    pipeline_batch_size: int = 4,
    clip_batch_size: int = 32,
    device: str = "cuda",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    resolution: int = 512,
) -> Path:
    """Iterate every model folder under ``models_root``, generate, and score.

    Each ``models_root/<name>`` is expected to contain a ``unet/`` subfolder
    (the ``save_pretrained`` output). Generated images live under
    ``out_root/<name>``; per-prompt and summary CSVs are written too.

    Returns:
        The path to ``summary_all_models.csv``.
    """
    from awwl.methods.dreambooth import (  # local import to avoid cycle
        build_pipeline,
        generate_images,
    )

    models_root = Path(models_root)
    out_root = ensure_dir(out_root)
    device = device if torch.cuda.is_available() else "cpu"

    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    rows: list[dict[str, float | int | str]] = []
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        unet_dir = model_dir / "unet"
        if not unet_dir.exists():
            logger.warning("skipping %s — no unet/ subfolder", model_dir)
            continue
        out_dir = ensure_dir(out_root / model_dir.name)

        pipe = build_pipeline(
            base_model=base_model,
            unet_dir=unet_dir,
            device=device,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        )

        all_clip_scores: list[float] = []
        all_image_sims: list[float] = []
        for prompt in prompts:
            prompt_slug = prompt.replace(" ", "_")[:50]
            prompt_dir = ensure_dir(out_dir / prompt_slug)
            seeds = list(range(n_images_per_prompt))
            paths: list[Path] = []
            for i in tqdm(
                range(0, len(seeds), pipeline_batch_size),
                desc=f"gen {model_dir.name} [{prompt}]",
            ):
                batch_paths = generate_images(
                    pipeline=pipe,
                    prompt=prompt,
                    seeds=seeds[i : i + pipeline_batch_size],
                    output_dir=prompt_dir,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=resolution,
                    width=resolution,
                )
                paths.extend(batch_paths)

            text_scores = text_image_similarity(
                clip_model=clip_model,
                clip_processor=clip_processor,
                prompt=prompt,
                image_paths=paths,
                device=device,
                batch_size=clip_batch_size,
            )
            image_sims = image_image_similarity(
                clip_model=clip_model,
                clip_processor=clip_processor,
                real_images_dir=real_images_dir,
                generated_image_paths=paths,
                device=device,
                batch_size=clip_batch_size,
            )
            all_clip_scores.extend(text_scores)
            all_image_sims.extend(image_sims)

            pd.DataFrame(
                {"image": [str(p) for p in paths], "clip_score": text_scores, "image_sim": image_sims}
            ).to_csv(prompt_dir / "scores.csv", index=False)

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows.append(
            {
                "model": model_dir.name,
                "n_images": len(all_clip_scores),
                "clip_mean": float(np.mean(all_clip_scores)) if all_clip_scores else float("nan"),
                "clip_std": float(np.std(all_clip_scores)) if all_clip_scores else float("nan"),
                "image_sim_mean": float(np.mean(all_image_sims)) if all_image_sims else float("nan"),
                "image_sim_std": float(np.std(all_image_sims)) if all_image_sims else float("nan"),
            }
        )

    summary_path = out_root / "summary_all_models.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    logger.info("summary written to %s", summary_path)
    return summary_path
