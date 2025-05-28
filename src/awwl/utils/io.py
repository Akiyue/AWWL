"""Filesystem and YAML helpers used across the package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from awwl.core.exceptions import ConfigError


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing, return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, applying a single ``_base_`` include if present.

    The include is resolved relative to the file containing it, so configs can
    sit in subdirectories. Keys in the child override the base, recursively
    for nested dicts. Lists and scalars are replaced wholesale.

    Raises ``ConfigError`` if the YAML is malformed or the base file is
    missing.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}: {p}")

    base_ref = data.pop("_base_", None)
    if base_ref is None:
        return data
    base_path = (p.parent / base_ref).resolve()
    base = load_yaml(base_path)
    return _deep_merge(base, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key.path=value`` CLI overrides to a config dict in place.

    Values are parsed as YAML scalars, so ``true``, ``42``, ``1e-4``, and
    ``[1, 2]`` all do the right thing. Keys that don't exist yet are created.
    """
    for raw in overrides:
        if "=" not in raw:
            raise ConfigError(f"override {raw!r} must be of the form key=value")
        key, _, value = raw.partition("=")
        parsed = _parse_scalar(value)
        _set_nested(cfg, key.strip().split("."), parsed)
    return cfg


def _parse_scalar(value: str):
    """Parse ``value`` as YAML, then try int/float fallbacks for forms YAML 1.1 misses.

    YAML 1.1 doesn't recognise ``1e-4`` as a float (it requires a dot). Falling
    through to ``float`` keeps CLI overrides ergonomic while preserving full
    YAML semantics for everything else (lists, bools, null, etc.).
    """
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse override value {value!r}: {exc}") from exc
    if isinstance(parsed, str):
        try:
            return int(parsed)
        except ValueError:
            pass
        try:
            return float(parsed)
        except ValueError:
            pass
    return parsed


def _set_nested(cfg: dict[str, Any], parts: list[str], value: Any) -> None:
    cur = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value
