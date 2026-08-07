"""Append-only ledger of experiment results.

Every training / evaluation job appends one JSON object per line to a shared
``results.jsonl``. Line-oriented and append-only so that N worker processes
on N GPUs can write concurrently without a lock, and so a crash truncates at
most the final line instead of corrupting the file (the reader skips
unparseable lines).

A row carries the full experiment identity alongside the metrics, so the
statistics layer never has to parse directory names::

    {"exp": "cifar_awwl", "group": "awwl", "seed": 1, "epoch": 199,
     "loss_name": "adaptive_wavelet", "alpha": 0.2, "power": 1.0,
     "use_ema": false, "fid": 16.62, "is_mean": 7.95, ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LEDGER = "results.jsonl"


def result_row(
    cfg: dict[str, Any],
    *,
    exp: str,
    group: str,
    kind: str,
    metrics: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a ledger row from a merged config plus measured metrics.

    Pulls the identifying hyperparameters out of ``cfg`` so that two rows for
    the same configuration are always comparable, whatever produced them.
    """
    loss_cfg = cfg.get("loss", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    row: dict[str, Any] = {
        "exp": exp,
        "group": group,
        "kind": kind,
        "seed": cfg.get("seed"),
        "method": cfg.get("method"),
        "dataset": data_cfg.get("dataset_name"),
        "image_size": data_cfg.get("image_size"),
        "loss_name": loss_cfg.get("name"),
        "alpha": loss_cfg.get("alpha"),
        "power": loss_cfg.get("power"),
        "wavelet_type": loss_cfg.get("wavelet_type"),
        "levels": loss_cfg.get("levels"),
        "weighting": loss_cfg.get("weighting"),
        "normalize_weights": loss_cfg.get("normalize_weights", False),
        "detail_reduction": loss_cfg.get("detail_reduction", "mean"),
        "use_ema": train_cfg.get("use_ema", False),
        "num_epochs": train_cfg.get("num_epochs"),
    }
    row.update(metrics or {})
    row.update(extra)
    return row


def append_result(ledger_path: str | Path, row: dict[str, Any]) -> Path:
    """Append one row to ``ledger_path``, creating parent dirs as needed.

    A single ``write`` of a short line in append mode is atomic enough for the
    concurrent-worker case; no locking is used deliberately.
    """
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, default=str)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def load_results(
    ledger_path: str | Path,
    *,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    """Read a ledger, skipping blank and unparseable lines.

    Args:
        ledger_path: File written by :func:`append_result`. A directory is
            also accepted, in which case every ``results.jsonl`` beneath it is
            concatenated.
        kind: Keep only rows with this ``kind`` (e.g. ``"eval"``).
    """
    path = Path(ledger_path)
    files = sorted(path.rglob(DEFAULT_LEDGER)) if path.is_dir() else [path]

    rows: list[dict[str, Any]] = []
    for f in files:
        if not f.exists():
            continue
        for lineno, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning("skipping malformed ledger line %s:%d", f, lineno)
    if kind is not None:
        rows = [r for r in rows if r.get("kind") == kind]
    return rows
