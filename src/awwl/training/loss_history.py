"""Per-step loss-history JSON logging.

Mirrors the AWWL-Diff format so existing analysis tools (``plot_losses.py``)
keep working without changes::

    {"config": {...}, "losses": [float, float, ...]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LossHistoryLogger:
    """Append per-step losses to an in-memory list and dump JSON each epoch.

    Args:
        output_dir: Folder where ``loss_history.json`` will live.
        config: Hyperparameter snapshot to embed alongside the loss list.
        resume: When ``True`` and a JSON file already exists at the target,
            load its losses and continue appending.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path,
        config: dict[str, Any],
        resume: bool = False,
    ) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "loss_history.json"
        self._config = dict(config)
        self._losses: list[float] = []
        if resume and self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self._losses = list(payload.get("losses", []))
            logger.info("resumed loss history with %d entries", len(self._losses))

    def append(self, value: float) -> None:
        """Record one step's loss value."""
        self._losses.append(float(value))

    def flush(self) -> None:
        """Write the current state to ``loss_history.json``."""
        with self._path.open("w", encoding="utf-8") as f:
            json.dump({"config": self._config, "losses": self._losses}, f)

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self._losses)
