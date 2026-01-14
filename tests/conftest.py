"""Shared fixtures for smoke tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Force tests onto CPU even when a GPU is present.

    Smoke tests must run in <30 s on CPU; a flaky GPU should never affect them.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def synthetic_images() -> torch.Tensor:
    """A tiny ``(N=2, C=3, H=32, W=32)`` batch in ``[-1, 1]``."""
    torch.manual_seed(0)
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def synthetic_latents() -> torch.Tensor:
    """A tiny ``(N=2, C=4, H=8, W=8)`` latent tensor (matches SD VAE output)."""
    torch.manual_seed(0)
    return torch.randn(2, 4, 8, 8)
