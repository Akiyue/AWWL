"""Wrap a HuggingFace ``datasets.Dataset`` of images as a torch ``Dataset``.

Used by the LoRA fine-tuning recipe to ingest e.g. CelebA without copying
files. Handles the variety of image representations HF returns: ``PIL.Image``,
``{"bytes": ...}``, raw ``bytes``, or a numpy array.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HuggingFaceImageDataset(Dataset):
    """Adapt a HF dataset to torch with a torchvision transform.

    Args:
        hf_dataset: The dataset returned by ``datasets.load_dataset``.
        transform: A callable applied to each :class:`PIL.Image` (typically a
            torchvision ``Compose``).
        image_column: Name of the column holding image data.
    """

    def __init__(
        self,
        hf_dataset,
        *,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        image_column: str = "image",
    ) -> None:
        self._ds = hf_dataset
        self._transform = transform
        self._image_column = image_column

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> torch.Tensor | Image.Image:
        item = self._ds[int(idx)]
        img = item.get(self._image_column) if isinstance(item, dict) else item
        img = _to_pil(img)
        if self._transform is not None:
            return self._transform(img)
        return img


def _to_pil(img) -> Image.Image:
    """Coerce HF's image variants into a ``PIL.Image`` in RGB."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, bytes | bytearray):
        return Image.open(io.BytesIO(img)).convert("RGB")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def load_hf_image_dataset(name_or_path: str, *, split: str = "train"):
    """Load a HuggingFace dataset, falling back to ``imagefolder`` for local dirs.

    Args:
        name_or_path: HuggingFace hub id (``flwrlabs/celeba``) or a local
            directory of images.
        split: Dataset split name. Ignored for local folders apart from being
            forwarded to ``load_dataset``.
    """
    from pathlib import Path as _Path  # local import keeps top imports tidy
    if _Path(name_or_path).is_dir():
        logger.info("loading local imagefolder from %s", name_or_path)
        return load_dataset("imagefolder", data_dir=name_or_path, split=split)
    logger.info("loading HF dataset %s split=%s", name_or_path, split)
    return load_dataset(name_or_path, split=split)
