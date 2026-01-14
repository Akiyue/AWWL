"""Smoke tests for the utility layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from awwl.core.exceptions import ConfigError
from awwl.utils import apply_overrides, ensure_dir, load_yaml, set_seed


def test_load_yaml_with_base(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("a: 1\nnested:\n  x: 1\n  y: 2\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("_base_: base.yaml\nb: 2\nnested:\n  y: 99\n", encoding="utf-8")

    cfg = load_yaml(child)
    assert cfg["a"] == 1
    assert cfg["b"] == 2
    assert cfg["nested"] == {"x": 1, "y": 99}


def test_apply_overrides_creates_nested_keys():
    cfg = {"loss": {"alpha": 0.8}}
    apply_overrides(cfg, ["loss.power=3.0", "train.lr=1e-4"])
    assert cfg["loss"]["alpha"] == 0.8
    assert cfg["loss"]["power"] == 3.0
    assert cfg["train"]["lr"] == 1e-4


def test_apply_overrides_rejects_malformed():
    with pytest.raises(ConfigError):
        apply_overrides({}, ["not_an_assignment"])


def test_ensure_dir_idempotent(tmp_path):
    d = tmp_path / "a" / "b"
    ensure_dir(d)
    ensure_dir(d)
    assert d.is_dir()


def test_set_seed_does_not_raise():
    set_seed(0)
    set_seed(42, deterministic=False)
