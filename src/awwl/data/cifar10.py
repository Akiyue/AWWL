"""CIFAR-10 utilities: streaming loader for training, disk dump for FID ref.

Two sources are supported and the loader prefers whichever works:

* **torchvision** — downloads the original 163 MB archive once into
  ``./data`` and reads it locally forever after. Same copy the FID reference
  dump uses, so a run needs exactly one download.
* **HuggingFace Hub** — the original recipe's source, kept because it is what
  produced the published numbers.

The Hub path is no longer reliable on its own. ``huggingface_hub`` dropped
support for single-segment dataset ids, so the legacy name ``cifar10`` now
raises ``HfUriError`` part-way through resolution even though the redirect to
the canonical ``uoft-cs/cifar10`` succeeds. Rather than pin the whole
dependency chain to work around one dataset name, the loader tries the
canonical id, then the legacy alias, then falls back to torchvision — a fixed
50 000-image dataset does not need a live API at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

Source = Literal["auto", "hf", "torchvision"]

# Canonical id first; the bare alias only works on older hub clients.
_HF_IDS = ("uoft-cs/cifar10", "cifar10")

DEFAULT_ROOT = "./data"


def build_transform(image_size: int, *, horizontal_flip: bool) -> transforms.Compose:
    """The AWWL-Diff preprocessing: resize, optional flip, scale to ``[-1, 1]``."""
    steps: list = [transforms.Resize((image_size, image_size))]
    if horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    return transforms.Compose(steps)


class _TorchvisionCifar(Dataset):
    """torchvision CIFAR-10 in the ``{"images": tensor}`` batch format."""

    def __init__(self, *, root: str | Path, train: bool, transform) -> None:
        self._ds = CIFAR10(root=str(root), train=train, download=True)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, _label = self._ds[int(idx)]
        return {"images": self._transform(img.convert("RGB"))}


def _load_hf(split: str, preprocess):
    """Load from the Hub, trying the canonical id before the legacy alias."""
    from datasets import load_dataset

    last: Exception | None = None
    for repo_id in _HF_IDS:
        try:
            dataset = load_dataset(repo_id, split=split)
        except Exception as exc:  # hub errors are not a single exception type
            logger.debug("could not load %s from the Hub: %s", repo_id, exc)
            last = exc
            continue

        column = "img" if "img" in dataset.column_names else "image"

        def _transform(examples, _column=column):
            return {"images": [preprocess(img.convert("RGB")) for img in examples[_column]]}

        dataset.set_transform(_transform)
        logger.info("CIFAR-10 loaded from the Hub (%s)", repo_id)
        return dataset
    raise RuntimeError(f"no HuggingFace id for CIFAR-10 could be loaded: {last}")


def build_cifar10_dataloader(
    *,
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
    split: str = "train",
    horizontal_flip: bool = True,
    shuffle: bool = True,
    seed: int | None = None,
    source: Source = "auto",
    root: str | Path = DEFAULT_ROOT,
):
    """Return a ``DataLoader`` yielding ``{"images": tensor}`` batches.

    Mirrors the AWWL-Diff training recipe: resize to ``image_size`` (32 by
    default), optional horizontal flip, normalize to ``[-1, 1]``.

    Args:
        seed: When given, drives the shuffling generator so that two runs with
            the same seed see the same batch order. Without it, multi-seed
            comparisons differ in both initialisation *and* data order, which
            confounds the per-seed variance estimate.
        source: ``"auto"`` tries the Hub and falls back to torchvision;
            ``"hf"`` or ``"torchvision"`` force one and fail loudly. Pin this
            when a table must state exactly where its data came from.
        root: Where torchvision caches the archive.
    """
    preprocess = build_transform(image_size, horizontal_flip=horizontal_flip and split == "train")
    train = split == "train"

    if source == "torchvision":
        dataset: Dataset = _TorchvisionCifar(root=root, train=train, transform=preprocess)
    elif source == "hf":
        dataset = _load_hf(split, preprocess)
    elif source == "auto":
        try:
            dataset = _load_hf(split, preprocess)
        except Exception as exc:
            logger.warning(
                "HuggingFace CIFAR-10 unavailable (%s); falling back to the local "
                "torchvision copy under %s — identical images, no Hub required",
                exc,
                root,
            )
            dataset = _TorchvisionCifar(root=root, train=train, transform=preprocess)
    else:
        raise ValueError(f"source must be 'auto', 'hf' or 'torchvision', got {source!r}")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def dump_reference_split(
    *,
    output_dir: str | Path,
    split: str = "train",
    download_root: str | Path = DEFAULT_ROOT,
) -> Path:
    """Write CIFAR-10 ``split`` images to ``output_dir`` as PNGs.

    Used to populate a local FID-reference folder. Idempotent — files are
    skipped if already present and matching the expected count.

    Returns:
        The output directory as a ``Path``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset: Dataset = CIFAR10(root=str(download_root), train=(split == "train"), download=True)

    if len(list(out.glob("*.png"))) >= len(dataset):
        logger.info("reference split already populated in %s; skipping", out)
        return out

    logger.info("dumping %d images to %s", len(dataset), out)
    for idx, (img, _label) in enumerate(tqdm(dataset, desc="cifar10")):
        img.save(out / f"{idx:05d}.png")
    return out
