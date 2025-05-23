"""Lightweight string-keyed registries for losses, methods, and datasets.

The registry pattern lets new components be added by users without editing
``__init__`` import lists, and gives a single place to enumerate what is
available (useful in CLI ``--help`` output).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A string → factory mapping with friendly error messages.

    Args:
        kind: Human-readable name of what is registered (used in error
            messages — e.g. ``"loss"``, ``"method"``).
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator to add a factory under ``name``.

        Returns the original object unchanged so it can still be used
        directly. Raises ``ValueError`` on duplicate names — registries are
        append-only by design.
        """

        def _decorator(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self._kind} {name!r} is already registered")
            self._items[name] = obj
            return obj

        return _decorator

    def get(self, name: str) -> T:
        """Look up a registered factory or raise ``KeyError``."""
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"unknown {self._kind} {name!r}. Registered: {available}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        """Return all registered names, sorted."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items
