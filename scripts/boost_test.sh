#!/usr/bin/env bash
# Does a spectral correction improve FID at the operating point?
#
# The calibration on real images measures FID's sensitivity at its *minimum*,
# where two identical distributions sit. There the first derivative is zero by
# construction, so perturbations cost quadratically and everything looks
# insensitive. Models are far from that minimum, and a correction there pays
# at first order. The two answers differ by more than an order of magnitude,
# and the second one is the one that bears on the question.
#
# So: take a model's own samples, boost the high band by the amount a
# frequency-aware loss achieves, and score against real images. Repeat per
# seed so the difference gets an error bar rather than an anecdote.
#
# Boosting AWWL as well is the control. If AWWL has genuinely corrected its
# spectrum, the same boost should help it *less* than it helps MSE. If it
# helps both equally, AWWL never moved the spectrum in a way that mattered.
#
# Usage:
#   bash scripts/boost_test.sh
#   BOOST=1.0 CONFIGS="mse awwl static_wavelet" bash scripts/boost_test.sh
#
# Environment:
#   ROOT     sweep directory              (default runs/phase0)
#   CONFIGS  configs to test              (default "mse awwl")
#   SEEDS    seeds                        (default 1 2 3 4 5)
#   EPOCH    checkpoint epoch             (default 199)
#   REAL     real reference folder        (default ./data/cifar10_train_png)
#   BOOST    high-band boost in dB        (default 0.41, AWWL's measured gain)
#   COUNT    images per comparison        (default 10000)
#   WORK     scratch directory            (default runs/boost)

set -uo pipefail

ROOT=${ROOT:-runs/phase0}
CONFIGS=${CONFIGS:-"mse awwl"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
EPOCH=${EPOCH:-199}
REAL=${REAL:-./data/cifar10_train_png}
BOOST=${BOOST:-0.41}
COUNT=${COUNT:-10000}
WORK=${WORK:-runs/boost}

if [ ! -d "${REAL}" ]; then
  echo "reference folder not found: ${REAL}" >&2
  exit 1
fi

mkdir -p "${WORK}"
echo "boost ${BOOST} dB, ${COUNT} images, reference ${REAL}"
echo

for c in ${CONFIGS}; do
  for s in ${SEEDS}; do
    samples="${ROOT}/${c}_s${s}/samples/ep${EPOCH}"
    if [ ! -d "${samples}" ]; then
      echo "skip ${c}_s${s}: no samples"
      continue
    fi
    echo "=== ${c}_s${s}"
    # 0 dB and the boost are scored in one call so both sides get identical
    # dithering and quantisation; only the spectral change differs.
    awwl sensitivity --real "${REAL}" --source "${samples}" \
      --deltas "0,-${BOOST}" --count "${COUNT}" --skip-advanced \
      --work "${WORK}/${c}_s${s}" \
      | tee "${WORK}/${c}_s${s}.txt" | grep -E "^ *(0d|-)" || true
    echo
  done
done

echo "per-run tables under ${WORK}/"
echo
echo "Read the pair: the 0 dB row is the model as trained, the boosted row is"
echo "the same samples with the spectral correction applied as post-processing."
echo "If the boost improves FID, the correction is worth something FID can see —"
echo "and a loss that achieves it while scoring worse overall is losing more"
echo "elsewhere than it gains here."
