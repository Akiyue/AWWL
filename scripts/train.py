#!/usr/bin/env python3
"""Thin shim that calls ``awwl train``. Convenient for ``python scripts/train.py``."""

from __future__ import annotations

import sys

from awwl.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "train")
    app()
