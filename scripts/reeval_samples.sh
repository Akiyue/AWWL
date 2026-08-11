#!/usr/bin/env bash
# Re-score sample folders that already exist on disk.
#
# Generation is the expensive half of an evaluation; the metrics are cheap. So
# when a metric is added or fixed after a sweep has run, there is no reason to
# retrain or resample — point this at the finished runs and only the scoring
# repeats.
#
# Rows are appended, not replaced. The statistics layer keeps the last value
# per (config, seed) *for each metric independently*, so a run with
# --skip-advanced adds the new numbers without discarding precision/recall
# from the original pass.
#
# Usage:
#   bash scripts/reeval_samples.sh
#   GROUPS="awwl_normalized awwl_eq7" bash scripts/reeval_samples.sh
#   EPOCH=119 FULL=1 bash scripts/reeval_samples.sh
#
# Environment overrides:
#   ROOT    sweep output directory        (default runs/phase0)
#   GROUPS  space-separated config names  (default the tier-1 three)
#   SEEDS   space-separated seeds         (default 1..5)
#   EPOCH   checkpoint epoch to score     (default 199)
#   REAL    reference image folder        (default ./data/cifar10_train_png)
#   LEDGER  results.jsonl to append to    (default $ROOT/results.jsonl)
#   FULL    set to 1 to also recompute KID / precision / recall (slower)

set -uo pipefail

ROOT=${ROOT:-runs/phase0}
GROUPS=${GROUPS:-"mse static_wavelet awwl"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
EPOCH=${EPOCH:-199}
REAL=${REAL:-./data/cifar10_train_png}
LEDGER=${LEDGER:-${ROOT}/results.jsonl}
FULL=${FULL:-0}

EXTRA="--skip-advanced"
if [ "${FULL}" = "1" ]; then
  EXTRA=""
fi

if [ ! -d "${REAL}" ]; then
  echo "reference folder not found: ${REAL}" >&2
  exit 1
fi

total=0
ok=0
skipped=0
failed=0

for g in ${GROUPS}; do
  for s in ${SEEDS}; do
    run_dir="${ROOT}/${g}_s${s}"
    samples="${run_dir}/samples/ep${EPOCH}"
    total=$((total + 1))

    if [ ! -d "${samples}" ]; then
      echo "skip ${g}_s${s}: no samples at ${samples}"
      skipped=$((skipped + 1))
      continue
    fi

    n=$(find "${samples}" -maxdepth 1 -name '*.png' | wc -l)
    echo "=== ${g}_s${s}  (${n} images)"

    if awwl eval-samples --run-dir "${run_dir}" --samples "${samples}" \
        --real "${REAL}" --ledger "${LEDGER}" --epoch "${EPOCH}" ${EXTRA}; then
      ok=$((ok + 1))
    else
      echo "FAILED: ${g}_s${s}" >&2
      failed=$((failed + 1))
    fi
  done
done

echo
echo "re-scored ${ok}/${total}  (skipped ${skipped}, failed ${failed})"
echo "ledger: ${LEDGER}"
echo
echo "next:"
echo "  awwl stats -l ${LEDGER} --metric kid_tf --epoch ${EPOCH} --baseline mse"
