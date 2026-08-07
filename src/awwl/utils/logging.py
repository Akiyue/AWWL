"""Project-wide logging setup.

The library uses ``logging`` exclusively — never ``print`` — so callers can
silence or reroute output without code changes.
"""

from __future__ import annotations

import contextlib
import logging
import sys

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def use_utf8_output() -> None:
    """Make stdout/stderr tolerate non-ASCII on a legacy Windows console.

    Report tables carry ``σ``, ``α`` and ``↑``/``↓``; a cp1252 console raises
    ``UnicodeEncodeError`` on the first one and takes the whole command down
    with it. Falling back to replacement characters loses a glyph instead of
    the run. No-op where the stream is already UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


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
