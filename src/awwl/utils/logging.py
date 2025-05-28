"""Project-wide logging setup.

The library uses ``logging`` exclusively — never ``print`` — so callers can
silence or reroute output without code changes.
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: str | int = "INFO", fmt: str = _DEFAULT_FMT) -> None:
    """Configure the root logger.

    Idempotent: calling twice does not add duplicate handlers.

    Args:
        level: Standard ``logging`` level name or integer.
        fmt: ``logging.Formatter`` format string.
    """
    root = logging.getLogger()
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setFormatter(logging.Formatter(fmt))
