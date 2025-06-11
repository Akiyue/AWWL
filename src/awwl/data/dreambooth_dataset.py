"""Subject-image dataset for DreamBooth fine-tuning."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

_VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class DreamBoothDataset(Dataset):
    """Loads a small folder of subject images and pairs each with one prompt.

    Each item is a dict::

        {"pixel_values": Tensor(C, H, W), "input_ids": Tensor(L,)}

    Args:
        instance_data_root: Folder containing subject images.
        instance_prompt: The text prompt shared across every image (e.g.
            ``"a photo of sks robot toy"``).
        tokenizer: A HuggingFace ``CLIPTokenizer`` already loaded for the
            target Stable-Diffusion model.
        size: Square edge length the images are resized + center-cropped to.
    """

    def __init__(
        self,
        *,
        instance_data_root: str | Path,
        instance_prompt: str,
        tokenizer,
        size: int = 512,
    ) -> None:
        root = Path(instance_data_root)
        if not root.is_dir():
            raise FileNotFoundError(f"instance_data_root does not exist: {root}")
        paths = sorted(p for p in root.iterdir() if p.suffix.lower() in _VALID_EXTS)
        if not paths:
            raise FileNotFoundError(f"no images found in {root}")
        self._paths = paths
        self._size = size
        self._image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self._prompt_input_ids = tokenizer(
            instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        logger.info("loaded %d instance images from %s", len(paths), root)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self._paths[idx % len(self._paths)]
        image = Image.open(path).convert("RGB")
        return {
            "pixel_values": self._image_transforms(image),
            "input_ids": self._prompt_input_ids.squeeze(0),
        }
