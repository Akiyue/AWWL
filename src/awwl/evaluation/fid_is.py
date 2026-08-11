"""FID + Inception Score + a second KID estimate, for the Finetune method.

Replaces ``AWWL-Diff/eval_fid.py``. Uses ``clean-fid`` for FID and
``torch-fidelity`` for IS (the same combo as the original script — both are
referee-friendly, well-cited implementations).

``kid_tf`` is a **cross-check**, not a duplicate. The project's other KID
(:func:`awwl.evaluation.advanced_metrics.compute_advanced_metrics`) is a
hand-written polynomial-kernel MMD, and on the first multi-seed replication it
was the *only* metric on which the proposed loss beat the baseline. A headline
claim resting on a non-standard implementation is a claim waiting to be
overturned in review, so the same quantity is computed here by an established
library as well. The two use different feature extractors and subset schemes,
so they will not agree to the last digit — what matters is that they agree on
sign and rough magnitude. If they disagree, neither should be reported until
the disagreement is understood.
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
    kid_subset_size: int = 1000,
) -> dict[str, float]:
    """Compute FID, Inception Score and an independent KID estimate.

    Both libraries are imported lazily — they pull large dependency trees that
    aren't needed for training or non-FID evals.

    Args:
        kid_subset_size: torch-fidelity's MMD subset size. Must not exceed the
            number of generated images, so it is clamped rather than left to
            raise mid-evaluation.

    Returns:
        Dict with keys ``fid``, ``is_mean``, ``is_std``, ``kid_tf``,
        ``kid_tf_std``. Values default to ``-1.0`` if that computation raised.
    """
    _patch_scipy_sqrtm_disp()

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

    kid_tf = -1.0
    kid_tf_std = -1.0
    try:
        n_fake = sum(1 for p in fake_folder.iterdir() if p.suffix.lower() in (".png", ".jpg"))
        subset = max(2, min(kid_subset_size, n_fake))
        if subset < kid_subset_size:
            logger.info("KID subset clamped to %d (only %d generated images)", subset, n_fake)
        logger.info("computing KID via torch-fidelity (cross-check)")
        metrics = calculate_metrics(
            input1=str(fake_folder),
            input2=str(real_folder),
            kid=True,
            isc=False,
            fid=False,
            cuda=True,
            batch_size=is_batch_size,
            kid_subset_size=subset,
            verbose=False,
        )
        kid_tf = float(metrics["kernel_inception_distance_mean"])
        kid_tf_std = float(metrics["kernel_inception_distance_std"])
    except Exception as exc:
        logger.error("KID (torch-fidelity) failed: %s", exc)

    if log_file is not None:
        _append_log(Path(log_file), exp_name=exp_name, fid=fid_score, is_mean=is_mean, is_std=is_std)

    return {
        "fid": float(fid_score),
        "is_mean": is_mean,
        "is_std": is_std,
        "kid_tf": kid_tf,
        "kid_tf_std": kid_tf_std,
    }


def _patch_scipy_sqrtm_disp() -> bool:
    """Restore ``scipy.linalg.sqrtm``'s ``disp`` argument for ``clean-fid``.

    SciPy removed ``disp`` in 1.17; ``clean-fid`` still calls
    ``sqrtm(sigma1 @ sigma2, disp=False)``, so FID dies with ``sqrtm() got an
    unexpected keyword argument 'disp'`` *after* both feature-extraction passes
    have run — several minutes of GPU time thrown away per evaluation, and on a
    sweep, once per configuration.

    The old contract was: ``disp=False`` returns ``(X, errest)`` instead of
    ``X``. This shim reproduces it, including ``errest``, so FID values are
    unchanged — it restores a calling convention, it does not alter the
    computation.

    One deliberate difference: SciPy also *silenced* the ill-conditioning
    warning under ``disp=False``, and this does not. clean-fid discards the
    error estimate, so suppressing the warning too would leave a singular
    covariance completely unreported. It fires when the sample count is small
    relative to the 2048-dimensional features — expected on a few-hundred-image
    smoke test, a real signal on a full evaluation.

    Returns ``True`` when a patch was applied. No-op on SciPy versions that
    still accept the argument.
    """
    try:
        import numpy as np
        from scipy import linalg
    except ImportError:  # pragma: no cover - scipy is an eval extra
        return False

    if getattr(linalg.sqrtm, "_awwl_disp_shim", False):
        return True

    import inspect

    try:
        if "disp" in inspect.signature(linalg.sqrtm).parameters:
            return False
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callable
        pass

    original = linalg.sqrtm

    def sqrtm(A, disp=True, **legacy):
        # SciPy 1.17 narrowed the signature to sqrtm(A) alone; `blocksize` went
        # the same way as `disp`, so accept and drop any leftover keywords
        # rather than trading one TypeError for another.
        legacy.pop("blocksize", None)
        result = original(A, **legacy)
        if disp:
            return result
        # SciPy's historical error estimate for the returned root.
        with np.errstate(invalid="ignore", divide="ignore"):
            arg_norm = np.linalg.norm(A, "fro")
            errest = (
                np.linalg.norm(result @ result - A, "fro") / arg_norm
                if arg_norm
                else 0.0
            )
        return result, errest

    sqrtm._awwl_disp_shim = True
    linalg.sqrtm = sqrtm
    logger.info("patched scipy.linalg.sqrtm to accept 'disp' (removed in SciPy 1.17)")
    return True


def _append_log(path: Path, *, exp_name: str, fid: float, is_mean: float, is_std: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            f.write(f"{'Experiment Name':<30} | {'FID':<10} | Inception Score\n")
            f.write("-" * 70 + "\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{exp_name:<30} | {fid:.4f}    | {is_mean:.4f} ± {is_std:.4f}\n")
