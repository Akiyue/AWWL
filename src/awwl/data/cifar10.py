"""CIFAR-10 utilities: streaming loader for training, disk dump for FID ref."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def build_cifar10_dataloader(
    *,
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
    split: str = "train",
    horizontal_flip: bool = True,
    shuffle: bool = True,
):
    """Return a ``DataLoader`` yielding ``{"images": tensor}`` batches.

    Mirrors the AWWL-Diff training recipe: resize to ``image_size`` (32 by
    default), optional horizontal flip, normalize to ``[-1, 1]``.
    """
    dataset = load_dataset("cifar10", split=split)
    pipeline = [
        transforms.Resize((image_size, image_size)),
    ]
    if horizontal_flip and split == "train":
        pipeline.append(transforms.RandomHorizontalFlip())
    pipeline += [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    preprocess = transforms.Compose(pipeline)

    def _transform(examples):
        return {"images": [preprocess(img.convert("RGB")) for img in examples["img"]]}

    dataset.set_transform(_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def dump_reference_split(
    *,
    output_dir: str | Path,
    split: str = "train",
    download_root: str | Path = "./data",
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
