"""Run the FID pricing measurements and append them to a ledger.

Every number the paper quotes about what a spectral intervention is worth in
FID comes from the ledger this writes (``runs/pricing/results.jsonl``), one row
per (dataset, operator, dose or target, seed). Nothing is printed for a human
to copy.

Operators, all applied to the baseline's own samples (``--baseline``, MSE by
default) at the evaluated epoch:

``identity``   no change; reproduces the run's ledger FID (a check on this path)
``boost``      raised-cosine high-band gain of ``--dose`` dB
``transplant`` per-radius gain moving the baseline's profile onto ``--target``'s
               profile for the same seed: the objective's measured spectral
               change, applied as a filter
``match``      per-radius gain moving the baseline's profile onto the reference
               images' profile: the whole mean radial deficit removed
``noise``      uniform white noise of +/- ``--dose`` LSB per channel, unrounded:
               how much FID responds to noise of the size 8-bit quantisation adds

``--bands low,mid,high`` restricts a ``transplant`` or ``match`` gain to the
listed index thirds of the radial profile (gain 1 elsewhere), which splits an
operator's value by band.

``--quantize float`` (default) scores the filtered images without rounding;
``dither`` reproduces the original 8-bit measurement so its quantisation share
can be isolated.

``--source`` prices an arm other than the baseline (e.g. boosting the objective
itself, for the no-interaction check).

Usage::

    python scripts/price_fid.py --dataset cifar10 --op boost --dose 0.41
    python scripts/price_fid.py --dataset cifar10 --op transplant --target awwl
    python scripts/price_fid.py --dataset celeba64 --op match
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from awwl.evaluation import price_fid as pf  # noqa: E402

DATASETS = {
    "cifar10": {"root": "runs/phase0", "real": "data/cifar10_train_png", "epoch": 199},
    "celeba64": {"root": "runs/celeba64", "real": "data/celeba64_train_png", "epoch": 199},
    "cifar10_long": {"root": "runs/converged", "real": "data/cifar10_train_png", "epoch": 799},
}


def _cached(path: Path, build):
    """Load ``path`` or build and save it.

    The write goes through a temporary name and ``os.replace``, so a second
    process reading the cache (the table generators share it) never sees a
    half-written array.
    """
    if path.exists():
        return np.load(path)
    value = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, value)
    os.replace(tmp, path)
    return value


def _restrict(gain_db_per_bin: np.ndarray, bands: str | None) -> np.ndarray:
    """Zero the per-bin dB gain outside the requested index thirds."""
    if not bands:
        return gain_db_per_bin
    keep = {b.strip() for b in bands.split(",")}
    unknown = keep - {"low", "mid", "high"}
    if unknown:
        raise SystemExit(f"unknown band(s) {sorted(unknown)}")
    m = len(gain_db_per_bin)
    edges = np.linspace(0, m, 4).astype(int)
    out = np.zeros_like(gain_db_per_bin)
    for name, a, b in zip(("low", "mid", "high"), edges[:-1], edges[1:], strict=True):
        if name in keep:
            out[a:b] = gain_db_per_bin[a:b]
    return out


def _profile_gain(target: np.ndarray, source: np.ndarray, h: int, w: int, bands: str | None) -> np.ndarray:
    m = min(len(target), len(source))
    db = _restrict(np.asarray(target[:m]) - np.asarray(source[:m]), bands)
    return pf.profile_gain(db, np.zeros(m), h, w)


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _samples(root: str, group: str, seed: int, epoch: int) -> Path:
    return Path(root) / f"{group}_s{seed}" / "samples" / f"ep{epoch}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="cifar10")
    ap.add_argument("--op", choices=["identity", "boost", "transplant", "match", "noise"], required=True)
    ap.add_argument("--dose", type=float, default=0.0, help="boost in dB, or noise amplitude in LSB")
    ap.add_argument("--bands", default=None, help="comma list of low,mid,high to restrict a profile gain")
    ap.add_argument("--target", default=None, help="arm whose profile a transplant copies")
    ap.add_argument("--baseline", default="mse")
    ap.add_argument("--source", default=None, help="arm to transform (default: the baseline)")
    ap.add_argument("--source-epoch", type=int, default=None)
    ap.add_argument("--target-epoch", type=int, default=None, help="epoch of the transplant target (default: the dataset's)")
    ap.add_argument("--source-folder", default=None,
                    help="score a fixed folder instead of a run's samples (e.g. held-out real images); "
                         "recorded under --source, one row with seed 0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--quantize", choices=["float", "round", "dither"], default="float")
    ap.add_argument("--ledger", default="runs/pricing/results.jsonl")
    ap.add_argument("--cache", default="runs/pricing/cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", choices=["inception", "dinov2"], default="inception",
                    help="feature space of the Frechet distance (and KID)")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    epoch = cfg["epoch"]
    source = args.source or args.baseline
    source_epoch = args.source_epoch or epoch
    cache = Path(args.cache) / args.dataset
    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)

    scorer = pf.DinoScorer(device=args.device) if args.features == "dinov2" else pf.Scorer(device=args.device)
    t0 = time.time()
    ref_name = "ref_features.npy" if args.features == "inception" else f"ref_features_{args.features}.npy"
    ref_feats = _cached(cache / ref_name, lambda: scorer.folder_features(cfg["real"]))
    ref_profile = _cached(cache / "ref_profile.npy", lambda: pf.radial_profile(pf.load_images(cfg["real"])))
    print(f"reference ready ({len(ref_feats)} images) in {time.time() - t0:.0f}s", flush=True)

    target_epoch = args.target_epoch or epoch
    seeds = [0] if args.source_folder else args.seeds
    for seed in seeds:
        folder = Path(args.source_folder) if args.source_folder else _samples(cfg["root"], source, seed, source_epoch)
        if not folder.is_dir():
            print(f"skip seed {seed}: no samples at {folder}", flush=True)
            continue
        t1 = time.time()
        images = pf.load_images(folder)
        n, h, w, _ = images.shape
        before = pf.radial_profile(images)

        if args.op == "identity":
            out = images.astype(np.float32)
        elif args.op == "boost":
            out = pf.apply_gain(images, pf.raised_cosine_gain(h, w, db=args.dose))
        elif args.op == "match":
            out = pf.apply_gain(images, _profile_gain(ref_profile, before, h, w, args.bands))
        elif args.op == "noise":
            rng = np.random.default_rng(1000 + seed)
            out = images.astype(np.float32) + rng.uniform(-args.dose, args.dose, images.shape).astype(np.float32)
        else:  # transplant
            if not args.target:
                ap.error("--op transplant needs --target")
            tfolder = _samples(cfg["root"], args.target, seed, target_epoch)
            if not tfolder.is_dir():
                print(f"skip seed {seed}: no target samples at {tfolder}", flush=True)
                continue
            target_profile = _cached(
                cache / "profiles" / f"{args.target}_s{seed}_ep{target_epoch}.npy",
                lambda tf=tfolder: pf.radial_profile(pf.load_images(tf)),
            )
            out = pf.apply_gain(images, _profile_gain(target_profile, before, h, w, args.bands))

        if args.op != "identity" or args.quantize != "float":
            out = pf.finalize(out, quantize=args.quantize, seed=seed)
        after = pf.radial_profile(out)
        feats = scorer.features(out)
        np.random.seed(0)  # KID draws random subsets
        row = {
            "dataset": args.dataset,
            "op": args.op,
            "dose_db": args.dose if args.op in ("boost", "noise") else None,
            "bands": args.bands,
            "target": args.target if args.op == "transplant" else None,
            "target_epoch": target_epoch if args.op == "transplant" else None,
            "source_folder": args.source_folder,
            "baseline": args.baseline,
            "source": source,
            "source_epoch": source_epoch,
            "seed": seed,
            "quantize": args.quantize,
            "features": None if args.features == "inception" else args.features,
            "n_images": int(n),
            "n_reference": int(len(ref_feats)),
            "fid": pf.fid_from_features(feats, ref_feats),
            "kid": pf.kid_from_features(feats, ref_feats),
            "dev_before": pf.band_deviations(before, ref_profile),
            "dev_after": pf.band_deviations(after, ref_profile),
            "profile_before": [float(v) for v in before],
            "profile_after": [float(v) for v in after],
            "seconds": round(time.time() - t1, 1),
            "git": _git_rev(),
        }
        with ledger.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"{args.dataset} {args.op} {args.dose if args.op == 'boost' else args.target or ''} "
            f"s{seed} [{args.quantize}]  FID {row['fid']:.3f}  KID {row['kid']:.5f}  "
            f"high {row['dev_before'][2]:+.3f} -> {row['dev_after'][2]:+.3f} dB  ({row['seconds']}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
