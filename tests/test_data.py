"""Smoke tests for the data pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image


class _StubTokenizer:
    """Just enough of CLIPTokenizer for the dataset constructor."""

    model_max_length = 16

    def __call__(self, *_args, **_kwargs):
        class _Out:
            input_ids = torch.zeros(1, 16, dtype=torch.long)

        return _Out()


@pytest.fixture
def tiny_image_dir(tmp_path: Path) -> Path:
    """Two 16×16 PNGs in a temp folder."""
    for i in range(2):
        img = Image.new("RGB", (16, 16), color=(i * 80, i * 40, 0))
        img.save(tmp_path / f"{i:02d}.png")
    return tmp_path


def test_dreambooth_dataset_yields_expected_shape(tiny_image_dir):
    from awwl.data import DreamBoothDataset

    ds = DreamBoothDataset(
        instance_data_root=tiny_image_dir,
        instance_prompt="a photo of sks robot",
        tokenizer=_StubTokenizer(),
        size=16,
    )
    assert len(ds) == 2
    item = ds[0]
    assert item["pixel_values"].shape == (3, 16, 16)
    assert item["pixel_values"].min() >= -1.0 and item["pixel_values"].max() <= 1.0
    assert item["input_ids"].shape == (16,)


def test_dreambooth_dataset_rejects_missing_dir(tmp_path):
    from awwl.data import DreamBoothDataset

    with pytest.raises(FileNotFoundError):
        DreamBoothDataset(
            instance_data_root=tmp_path / "does_not_exist",
            instance_prompt="x",
            tokenizer=_StubTokenizer(),
            size=16,
        )


def test_hf_image_to_pil_handles_array():
    """Internal helper should accept a HxWx3 uint8 numpy array."""
    import numpy as np

    from awwl.data.hf_image_dataset import _to_pil

    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    img = _to_pil(arr)
    assert img.size == (8, 8)
    assert img.mode == "RGB"
