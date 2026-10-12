"""A second intervention operator for the pricing test: full spectrum matching.

The pricing test of `\\S pricing-protocol` prices a mechanism by applying it to
the baseline's own samples. Everything in the paper so far applies it one way:
a raised-cosine boost of the high band by a fixed number of decibels. That
operator was chosen to mirror what the audited objective does, but a result
that depends on the shape of the intervention is a result about the
intervention.

This is the other end of the family of operators the test admits. Rather than
lifting one band by a chosen amount, it rescales *every* radial frequency so
the sample set's mean radial spectrum matches the reference set's exactly.
That is the strongest possible version of the correction: it removes the
spectral discrepancy entirely rather than by a measured amount, so the FID it
buys is an upper bound on what any spectral intervention can be worth.

Two operators bracket the answer. If the band boost and the full match agree,
the price is a property of the spectrum rather than of how it was touched. If
the full match buys much more, the audited objective was leaving value on the
table; if it buys no more, the objective's correction was already most of
what spectral correction can do.

Usage::

    python scripts/spectral_match.py --source runs/phase0/mse_s1/samples/ep199 \\
        --real data/cifar10_train_png --out runs/match/mse_s1

Reports the deviation before and after, so the correction can be checked
rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from awwl.evaluation.spectrum import radial_profile  # noqa: E402


def _radii(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ys, xs = np.indices((h, w))
    cy, cx = h // 2, w // 2
    return np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)


def match_folder(
    source: Path, real: Path, out: Path, *, max_images: int = 2000, eps: float = 1e-8
) -> dict[str, float]:
    """Rescale each image's spectrum so the set's mean radial profile matches.

    The gain is computed once from the two *set-level* profiles and applied to
    every image, rather than per image. Matching each image to the mean would
    also erase the variation between images, which is part of what the metric
    measures and nothing the objective claims to change.
    """
    from PIL import Image

    src_profile = radial_profile(source, max_images=max_images)
    ref_profile = radial_profile(real, max_images=max_images)
    if src_profile is None or ref_profile is None:
        raise SystemExit("could not profile one of the folders")

    n = min(len(src_profile), len(ref_profile))
    # Profiles are in dB; the multiplicative gain per radial bin is the
    # difference brought back to linear.
    gain_db = np.asarray(ref_profile[:n]) - np.asarray(src_profile[:n])
    gain = np.power(10.0, gain_db / 20.0)

    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    written = 0
    for path in files:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        h, w, _ = arr.shape
        bins = np.clip(_radii((h, w)).astype(int), 0, n - 1)
        mask = gain[bins]
        chans = []
        for c in range(3):
            f = np.fft.fftshift(np.fft.fft2(arr[..., c]))
            chans.append(np.real(np.fft.ifft2(np.fft.ifftshift(f * mask))))
        matched = np.clip(np.stack(chans, axis=-1), 0.0, 1.0)
        Image.fromarray((matched * 255.0 + 0.5).astype(np.uint8)).save(out / path.name)
        written += 1

    after = radial_profile(out, max_images=max_images)
    m = min(len(after), len(ref_profile))
    band = slice(2 * m // 3, m)
    return {
        "images": written,
        "high_before": float(np.mean(np.asarray(src_profile[:m])[band] - np.asarray(ref_profile[:m])[band])),
        "high_after": float(np.mean(np.asarray(after[:m])[band] - np.asarray(ref_profile[:m])[band])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--real", type=Path, default=Path("data/cifar10_train_png"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=2000)
    args = parser.parse_args()

    stats = match_folder(args.source, args.real, args.out, max_images=args.max_images)
    print(f"matched {stats['images']} images -> {args.out}")
    print(f"high-band deviation  before {stats['high_before']:+.3f} dB"
          f"  after {stats['high_after']:+.3f} dB")
    print("\nScore the output folder against the same reference to price it:")
    print(f"  awwl eval --config configs/eval/fid_is.yaml "
          f"-o fid_is.out_folder={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
