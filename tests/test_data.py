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


def test_radial_gain_plants_the_requested_spectral_change():
    """RadialGain shifts the luminance radial profile by exactly its gain.

    It is the data-side twin of the pricing transplant: the planted-mechanism
    arm trains MSE on images carrying AWWL's measured spectral change, and the
    comparison is only meaningful if the change planted is the change asked for.
    """
    import numpy as np
    import torch

    from awwl.data.cifar10 import RadialGain
    from awwl.evaluation.price_fid import radial_profile

    rng = np.random.default_rng(0)
    imgs = rng.uniform(0.2, 0.8, size=(64, 3, 32, 32)).astype(np.float32)  # far from the clip range
    x = torch.from_numpy(imgs)
    assert torch.allclose(RadialGain(np.zeros(23))(x[0]), x[0], atol=1e-6)

    gain = np.linspace(0.0, 0.5, 23)
    y = torch.stack([RadialGain(gain)(im) for im in x]).numpy()
    shift = radial_profile(y.transpose(0, 2, 3, 1) * 255.0) - radial_profile(imgs.transpose(0, 2, 3, 1) * 255.0)
    assert np.max(np.abs(shift - gain)) < 1e-6
