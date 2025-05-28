"""Resolve external-checkpoint paths via the registry or direct override."""

from __future__ import annotations

import logging
from pathlib import Path

from awwl.core.exceptions import CheckpointNotFoundError
from awwl.utils.io import load_yaml

logger = logging.getLogger(__name__)


def resolve_weights(
    *,
    method: str,
    explicit_path: str | None,
    registry_name: str | None = None,
    registry_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve the weights path for a method, preferring an explicit override.

    Args:
        method: Top-level key in ``registry.yaml`` (``dreambooth`` or
            ``finetune``).
        explicit_path: If given, this absolute or relative path is returned
            directly after existence check — no registry lookup.
        registry_name: Logical name to look up under ``method:`` in
            ``registry.yaml``.
        registry_path: Path to the registry YAML. Defaults to
            ``configs/checkpoints/registry.yaml`` relative to ``project_root``.
        project_root: Base for resolving relative paths inside the registry.
            Defaults to the registry file's parent.

    Raises:
        CheckpointNotFoundError: The resolved path does not exist on disk.
        ValueError: Neither ``explicit_path`` nor ``registry_name`` was given.
    """
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.exists():
            raise CheckpointNotFoundError(f"weights path does not exist: {candidate}")
        return candidate

    if not registry_name:
        raise ValueError("either explicit_path or registry_name is required")

    if registry_path is None:
        if project_root is None:
            raise ValueError("project_root is required when registry_path is unset")
        registry_path = Path(project_root) / "configs" / "checkpoints" / "registry.yaml"

    registry_path = Path(registry_path).resolve()
    registry = load_yaml(registry_path)
    if method not in registry:
        raise CheckpointNotFoundError(
            f"method {method!r} not in registry {registry_path}"
        )
    if registry_name not in registry[method]:
        available = ", ".join(sorted(registry[method])) or "<none>"
        raise CheckpointNotFoundError(
            f"checkpoint {registry_name!r} not registered for {method}; available: {available}"
        )

    raw = registry[method][registry_name]
    base = Path(project_root) if project_root else registry_path.parent
    candidate = (base / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if not candidate.exists():
        raise CheckpointNotFoundError(
            f"registry entry {method}.{registry_name} points at missing path: {candidate}"
        )
    return candidate
