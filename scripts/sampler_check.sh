#!/usr/bin/env bash
# Does the paper's conclusion survive a different sampler?
#
# Every number in the study comes from DDIM-100. That is a defensible operating
# point and it is applied identically to every arm, but it is one point. A
# frequency-aware objective could plausibly lose there and win under the full
# DDPM chain: few-step samplers have their own spectral signature, and an
# objective that shifts high-frequency content might interact with it.
#
# If the ordering reverses under DDPM-1000, the paper's claim has to narrow to
# "at few-step sampling" -- which is a much smaller claim, and one a reviewer
# will ask about. This is the cheapest way to find out, and it uses checkpoints
# that already exist: no training.
#
# Sampling is ~10x DDIM-100 per image, so this is hours, not days. Keep the
# image count at the main table's value: the point is to compare against those
# numbers, and FID is biased by sample count.
#
# Usage:
#   bash scripts/sampler_check.sh
#   CONFIGS="mse awwl static_wavelet" SEEDS="1 2 3" bash scripts/sampler_check.sh
#
# Environment:
#   ROOT     sweep directory        (default runs/phase0)
#   CONFIGS  arms to re-sample      (default "mse awwl")
#   SEEDS    seeds                  (default 1 2 3 4 5)
#   EPOCH    checkpoint epoch       (default 199)
#   REAL     reference images       (default ./data/cifar10_train_png)
#   COUNT    images per run         (default 10000, matching the main table)
#   STEPS    denoising steps        (default 1000)
#   LEDGER   where rows land        (default runs/phase0/sampler_check.jsonl)

set -uo pipefail

ROOT=${ROOT:-runs/phase0}
CONFIGS=${CONFIGS:-"mse awwl"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
EPOCH=${EPOCH:-199}
REAL=${REAL:-./data/cifar10_train_png}
COUNT=${COUNT:-10000}
STEPS=${STEPS:-1000}
LEDGER=${LEDGER:-runs/phase0/sampler_check.jsonl}

if [ ! -d "${REAL}" ]; then
  echo "reference folder not found: ${REAL}" >&2
  exit 1
fi

echo "DDPM-${STEPS}, ${COUNT} images per run, reference ${REAL}"
echo "ledger: ${LEDGER}"
echo

for c in ${CONFIGS}; do
  for s in ${SEEDS}; do
    run="${ROOT}/${c}_s${s}"
    ckpt="${run}/checkpoint-${EPOCH}"
    out="${run}/samples/ddpm${STEPS}_ep${EPOCH}"

    if [ ! -d "${ckpt}" ]; then
      echo "skip ${c}_s${s}: no checkpoint at ${ckpt}"
      continue
    fi

    echo "=== ${c}_s${s}"
    # The sampling seed is fixed across arms so the comparison differs by the
    # trained model alone, exactly as the DDIM table does.
    awwl infer --method finetune --weights "${ckpt}" \
      --output-dir "${out}" --num-samples "${COUNT}" \
      --sampler ddpm --steps "${STEPS}" --sample-seed 12345 || continue

    awwl eval-samples --run-dir "${run}" --samples "${out}" \
      --real "${REAL}" --epoch "${EPOCH}" --ledger "${LEDGER}" || true
    echo
  done
done

echo "compare the ordering against the DDIM-100 table:"
echo "  awwl stats -l ${LEDGER} --metric fid --baseline mse"
echo
echo "The claim to check is not whether FID improves -- it will, DDPM-1000 is a"
echo "better sampler -- but whether the arms keep their order and their spacing."
echo "If they do, the conclusion is not an artefact of few-step sampling."
