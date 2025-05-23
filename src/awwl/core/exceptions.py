"""Domain-specific exceptions raised by the awwl package.

Catching ``AWWLError`` lets a CLI distinguish expected misconfiguration from
unexpected programming bugs (which propagate naturally).
"""

from __future__ import annotations


class AWWLError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(AWWLError):
    """A YAML config or CLI override is malformed or contradictory."""


class CheckpointNotFoundError(AWWLError):
    """A path expected to contain saved weights does not exist."""


class UnknownLossError(AWWLError):
    """``loss.name`` does not match any registered loss factory."""


class UnknownMethodError(AWWLError):
    """``method`` does not match any registered training method."""
