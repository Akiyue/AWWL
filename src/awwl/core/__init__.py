"""Shared abstractions: registries and domain exceptions."""

from __future__ import annotations

from awwl.core.exceptions import (
    AWWLError,
    CheckpointNotFoundError,
    ConfigError,
    UnknownLossError,
    UnknownMethodError,
)
from awwl.core.registry import Registry

__all__ = [
    "AWWLError",
    "CheckpointNotFoundError",
    "ConfigError",
    "Registry",
    "UnknownLossError",
    "UnknownMethodError",
]
