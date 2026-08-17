"""CLIP scoring against both shapes the transformers API returns.

`get_text_features` / `get_image_features` handed back a bare tensor through
transformers 4.x and a `BaseModelOutputWithPooling` from 5.x. The 5.x shape
crashed every DreamBooth eval -- but only after the images had been generated,
so the sweep paid the whole cost of each job before failing it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from awwl.evaluation.clip_scores import _features, image_image_similarity, text_image_similarity

DIM = 8


class _Pooled:
    """Stands in for ``BaseModelOutputWithPooling``: embedding on pooler_output."""

    def __init__(self, tensor: torch.Tensor):
        self.pooler_output = tensor
        self.last_hidden_state = torch.zeros(tensor.shape[0], 3, DIM)


class _StubCLIP:
    """A CLIP whose feature calls return whichever shape the test asks for."""

    def __init__(self, *, wrap: bool):
        self.wrap = wrap

    def _out(self, n: int) -> object:
        torch.manual_seed(n)
        t = torch.randn(n, DIM)
        return _Pooled(t) if self.wrap else t

    def get_text_features(self, **kwargs):
        return self._out(1)

    def get_image_features(self, **kwargs):
        return self._out(kwargs["n"])


class _Movable(dict):
    """Processor output: a mapping that also answers ``.to(device)``."""

    def to(self, _device):
        return self


class _StubProcessor:
    """Records the batch size so the stub model can return a matching shape."""

    def __call__(self, *, images=None, text=None, **kwargs):
        return _Movable(n=len(images) if images is not None else len(text))


@pytest.fixture
def processor():
    return _StubProcessor()


@pytest.fixture
def images(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


def test_features_accepts_a_bare_tensor():
    tensor = torch.randn(2, DIM)
    assert _features(tensor) is tensor


def test_features_unwraps_the_pooled_output():
    tensor = torch.randn(2, DIM)
    assert _features(_Pooled(tensor)) is tensor


def test_features_rejects_something_carrying_no_embedding():
    class Empty:
        pass

    with pytest.raises(TypeError, match="carries no embedding tensor"):
        _features(Empty())


@pytest.mark.parametrize("wrap", [False, True], ids=["transformers-4", "transformers-5"])
def test_text_image_similarity_works_for_both_return_shapes(wrap, images, processor):
    scores = text_image_similarity(
        clip_model=_StubCLIP(wrap=wrap), clip_processor=processor,
        prompt="a photo of sks robot toy", image_paths=images, device="cpu",
    )

    assert len(scores) == len(images)
    assert all(-1.0 <= s <= 1.0 for s in scores), "cosine similarity must be in [-1, 1]"


@pytest.mark.parametrize("wrap", [False, True], ids=["transformers-4", "transformers-5"])
def test_image_image_similarity_works_for_both_return_shapes(wrap, images, processor, tmp_path):
    scores = image_image_similarity(
        clip_model=_StubCLIP(wrap=wrap), clip_processor=processor,
        real_images_dir=tmp_path, generated_image_paths=images, device="cpu",
    )

    assert len(scores) == len(images)
    assert all(-1.0 <= s <= 1.0 for s in scores)
