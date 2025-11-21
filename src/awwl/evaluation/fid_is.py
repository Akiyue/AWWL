"""FID + Inception Score evaluation for the Finetune method.

Replaces ``AWWL-Diff/eval_fid.py``. Uses ``clean-fid`` for FID and
``torch-fidelity`` for IS (the same combo as the original script — both are
referee-friendly, well-cited implementations).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_fid_is(
    *,
    fake_folder: str | Path,
    real_folder: str | Path,
    log_file: str | Path | None = None,
    exp_name: str = "unknown",
    is_batch_size: int = 128,
) -> dict[str, float]:
    """Compute FID and Inception Score; optionally append to ``log_file``.

    Both libraries are imported lazily — they pull large dependency trees that
    aren't needed for training or non-FID evals.

    Returns:
        Dict with keys ``fid``, ``is_mean``, ``is_std``. Values default to
        ``-1.0`` if the corresponding computation raised.
    """
    from cleanfid import fid as cleanfid
    from torch_fidelity import calculate_metrics

    fake_folder = Path(fake_folder)
    real_folder = Path(real_folder)

    fid_score = -1.0
    try:
        logger.info("computing FID via clean-fid: %s vs %s", fake_folder, real_folder)
        fid_score = cleanfid.compute_fid(fdir1=str(fake_folder), fdir2=str(real_folder))
    except Exception as exc:
        logger.error("FID failed: %s", exc)

    is_mean = -1.0
    is_std = -1.0
    try:
        logger.info("computing IS via torch-fidelity: %s", fake_folder)
        metrics = calculate_metrics(
            input1=str(fake_folder),
            isc=True,
            fid=False,
            cuda=True,
            batch_size=is_batch_size,
            verbose=False,
        )
        is_mean = float(metrics["inception_score_mean"])
        is_std = float(metrics["inception_score_std"])
    except Exception as exc:
        logger.error("IS failed: %s", exc)

    if log_file is not None:
        _append_log(Path(log_file), exp_name=exp_name, fid=fid_score, is_mean=is_mean, is_std=is_std)

    return {"fid": float(fid_score), "is_mean": is_mean, "is_std": is_std}


def _append_log(path: Path, *, exp_name: str, fid: float, is_mean: float, is_std: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            f.write(f"{'Experiment Name':<30} | {'FID':<10} | Inception Score\n")
            f.write("-" * 70 + "\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{exp_name:<30} | {fid:.4f}    | {is_mean:.4f} ± {is_std:.4f}\n")
