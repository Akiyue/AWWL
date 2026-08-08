"""One dataloader entry point for every image dataset the trainers use.

The CIFAR-10 loader was wired directly into the trainer, which quietly capped
the whole study at 32x32. That is the binding constraint on two things the
paper needs: the claim that "optimal frequency weighting is tied directly to
dataset resolution" currently rests on **two** data points (CIFAR-10 at 32 and
DreamBooth at 512), and the wavelet-diffusion literature benchmarks on
CelebA-HQ / LSUN, so there is no common ground for comparison.

This dispatches on ``data.dataset_name``:

* ``cifar10`` — the published recipe, unchanged.
* any HuggingFace hub id (``flwrlabs/celeba``, ``huggan/CelebA-HQ`` …) or a
  local directory of images, via :mod:`awwl.data.hf_image_dataset`.

so adding a resolution to the sweep is a config change, not a code change.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from awwl.data.cifar10 import build_cifar10_dataloader
from awwl.data.hf_image_dataset import HuggingFaceImageDataset, load_hf_image_dataset

logger = logging.getLogger(__name__)

CIFAR10 = "cifar10"


def build_transform(image_size: int, *, horizontal_flip: bool) -> transforms.Compose:
    """Resize / centre-crop to ``image_size`` and normalise to ``[-1, 1]``.

    Centre-crop after resizing the short side keeps the aspect ratio, which
    matters for face and scene datasets where a plain resize would squash the
    image and change its frequency content — the very thing being measured.
    """
    steps = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    return transforms.Compose(steps)


class _ImagesOnly(torch.utils.data.Dataset):
    """Wrap a tensor-yielding dataset in the ``{"images": ...}`` batch format."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx):
        return {"images": self._inner[idx]}


def build_image_dataloader(
    *,
    dataset_name: str = CIFAR10,
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
    split: str = "train",
    horizontal_flip: bool = True,
    shuffle: bool = True,
    seed: int | None = None,
    image_column: str = "image",
):
    """Return a ``DataLoader`` yielding ``{"images": tensor}`` batches.

    Args:
        dataset_name: ``"cifar10"``, a HuggingFace hub id, or a local folder.
        seed: Drives the shuffling generator so two runs with the same seed see
            the same batch order — without it a multi-seed comparison varies
            data order as well as initialisation.
    """
    if dataset_name == CIFAR10:
        return build_cifar10_dataloader(
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            split=split,
            horizontal_flip=horizontal_flip,
            shuffle=shuffle,
            seed=seed,
        )

    hf = load_hf_image_dataset(dataset_name, split=split)
    transform = build_transform(image_size, horizontal_flip=horizontal_flip and split == "train")
    dataset = _ImagesOnly(
        HuggingFaceImageDataset(hf, transform=transform, image_column=_pick_column(hf, image_column))
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    logger.info("dataset %s: %d images at %dx%d", dataset_name, len(dataset), image_size, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=True,
    )


def _pick_column(hf_dataset, requested: str) -> str:
    """Fall back to the first image-like column when ``requested`` is absent.

    Hub datasets disagree on the name (``image``, ``img``, ``jpg``); guessing
    here avoids a per-dataset config knob for something unambiguous.
    """
    columns = list(getattr(hf_dataset, "column_names", []) or [])
    if not columns or requested in columns:
        return requested
    for candidate in ("image", "img", "jpg", "png", "picture"):
        if candidate in columns:
            logger.info("image column %r not found; using %r", requested, candidate)
            return candidate
    logger.warning("no obvious image column in %s; using %r", columns, columns[0])
    return columns[0]


def dump_reference_images(
    *,
    dataset_name: str,
    output_dir: str | Path,
    image_size: int,
    split: str = "train",
    max_images: int | None = None,
    image_column: str = "image",
) -> Path:
    """Write a dataset to PNGs for use as the FID/KID reference set.

    The reference images are written at the **same resolution the model
    generates**, because FID compares Inception features of both sets and a
    resolution mismatch between real and fake is a well-known way to produce
    numbers that cannot be compared with anyone else's.

    Idempotent: an existing dump with enough files is left alone.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if dataset_name == CIFAR10 and image_size == 32:
        from awwl.data.cifar10 import dump_reference_split

        return dump_reference_split(output_dir=out, split=split)

    hf = load_hf_image_dataset(dataset_name, split=split)
    column = _pick_column(hf, image_column)
    total = len(hf) if max_images is None else min(len(hf), max_images)

    existing = sum(1 for _ in out.glob("*.png"))
    if existing >= total:
        logger.info("reference set already has %d images in %s; skipping", existing, out)
        return out

    resize = transforms.Compose(
        [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    )
    inner = HuggingFaceImageDataset(hf, transform=None, image_column=column)
    logger.info("dumping %d reference images to %s at %d px", total, out, image_size)
    for idx in tqdm(range(total), desc="reference"):
        resize(inner[idx]).save(out / f"{idx:06d}.png")
    return out
