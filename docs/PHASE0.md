# Phase 0 — running the survival test

Everything here runs from a clone of this repo on the training server. Nothing
in this document needs more than the two RTX 5000 Ada cards.

---

## Why this comes before any extension

The CIFAR-10 table in the CITA paper separates its three best rows by less
than a tenth of a FID point:

| | FID |
|---|---|
| MSE | 16.68 |
| Static Wavelet (p=0) | **16.55** |
| AWWL | 16.62 |

Those are single-seed numbers. Seed-to-seed spread for this recipe is normally
several times that gap, which means the published ordering may be noise. The
same holds for the DreamBooth table, where the differences (~0.01) sit well
inside the reported standard deviations (~0.035-0.057).

So the first question is not "which journal?" — it is **is there an effect at
all?** Phase 0 answers that, plus two questions about the objective itself
that a reviewer reading the released code would raise. Committing weeks of
compute to CelebA-HQ or flow matching before knowing the answer is the main
way this project can waste a semester.

---

## 0. Setup (once)

```bash
git clone https://github.com/Akiyue/AWWL.git awwl && cd awwl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[eval,plot,dev]"

# CIFAR-10 as PNGs, used as the FID/KID reference set
awwl prepare-data --output ./data/cifar10_train_png
```

Check the GPUs are visible:

```bash
python -c "import torch; print(torch.cuda.device_count())"   # expect 2
```

---

## 1. The free check: loss math (CPU, ~5 seconds)

```bash
python scripts/verify_loss_math.py --out runs/loss_math_report.md
```

This needs no GPU and settles three things:

1. **Parseval.** Confirms that for `db1` the four sub-band errors carry exactly
   the pixel MSE, so AWWL is a *reallocation* of a fixed error budget over
   orthogonal bands. This is the paper's real theoretical footing, and it is
   stronger stated plainly than the current "Theoretical Alignment" bullet.
   It also makes clear the loss decomposes the **prediction residual**, not
   the image — worth fixing in the motivation, since the ε-prediction target
   is white noise and has no structure to decompose.

2. **Eq. (7) vs the code.** The paper sums the three detail bands; the
   implementation averages them. That is a factor of 3 on the detail term, so
   the published `α` does not mean what the equation says. Either correct the
   equation to an average, or set `loss.detail_reduction: sum` and re-tune.
   Leaving the paper and the public code disagreeing is the kind of thing
   reviewers now check.

3. **The α confound.** Eqs. (4)-(5) do **not** sum to a constant — the total
   runs from `α` at high noise to `1-α` at low noise. At α=0.8 early timesteps
   get 4× the gradient magnitude of late ones; at α=0.2 the reverse. So α is
   simultaneously a frequency balance *and* a global timestep reweighting.

   The size of that second effect is what makes it a problem. Measured against
   the Min-SNR (γ=5) curve, the script reports **r = +0.92 at α=0.8 and
   r = −0.92 at α=0.2**: high α tracks Min-SNR (down-weighting low-noise,
   high-SNR steps), low α is its mirror image (up-weighting them). The two
   published optima are therefore not just different frequency balances —
   they are *opposite timestep schedules*.

   That undercuts the conclusion "optimal frequency weighting is tied directly
   to dataset resolution", which rests entirely on the α=0.2 (CIFAR) vs α=0.8
   (DreamBooth) split. A from-scratch DDPM and a 400-step fine-tune plausibly
   want opposite timestep emphasis for reasons unrelated to resolution — and
   with two data points, resolution and training regime are perfectly
   confounded anyway. Tier 2 separates them.

---

## 2. The sweep

```bash
awwl pipeline run -m configs/pipeline/phase0.yaml --gpus 0,1
```

That is the whole command. It is safe to re-run at any time — see
[Crash recovery](#4-crash-recovery).

Preview the plan without running anything:

```bash
awwl pipeline run -m configs/pipeline/phase0.yaml --dry-run
```

Run one tier at a time (recommended — stop and read the tier-1 result before
paying for tier 3):

```bash
awwl pipeline run -m configs/pipeline/phase0.yaml --gpus 0,1 --max-tier 1
```

### What each tier buys

| Tier | Runs | Question |
|---|---|---|
| 1 | 15 | Is the AWWL−MSE gap larger than seed noise? Does AWWL converge *faster*? |
| 2 | 16 | Is the effect frequency-driven or a timestep-reweighting artefact? Does EMA preserve the ordering? Does eq. (7)'s literal form behave like the code's? |
| 3 | 25 | Error bars on the rest of the published Table 2. |

### Cost

One training run is ~3 h on one card; the two cards run two different
experiments concurrently rather than splitting one (these cards have no
NVLink, and at 35M parameters a sweep is throughput-bound, so independent runs
beat DDP). Sampling 10 000 images with DDIM-100 is ~6 min per checkpoint.

| Tier | Wall-clock on 2 GPUs |
|---|---|
| 1 | ~1 day |
| 1+2 | ~2.5 days |
| 1+2+3 | ~4 days |

Sampling uses **DDIM-100, not DDPM-1000** — a tenth of the cost, and what
makes a 56-run matrix affordable at all. FID depends on the sampler, so every
row in a table must use the same one; never mix DDIM rows with DDPM rows.

---

## 3. Reading the results

```bash
# Mean ± std and 95% CI per configuration, at the final checkpoint
awwl stats -l runs/phase0/results.jsonl --metric fid --epoch 199

# The actual test: paired-by-seed t-test + Wilcoxon against MSE, Holm-corrected
awwl stats -l runs/phase0/results.jsonl --metric fid --epoch 199 --baseline mse
awwl stats -l runs/phase0/results.jsonl --metric is_mean --epoch 199 --baseline mse

# Convergence curves (tier 1) — the "cheaper training" claim
awwl stats -l runs/phase0/results.jsonl --metric fid --curve
```

Comparisons are **paired by seed**: seed 1 of AWWL against seed 1 of MSE, and
so on. Pairing removes the variance the seed itself contributes and is much
more sensitive than an unpaired test at N=5.

Both a t-test and a Wilcoxon signed-rank test are reported. Read the Wilcoxon
column as a robustness check on direction, not as a second verdict — its
smallest possible two-sided p is `2/2^N`, i.e. **0.0625 at five seeds, so it
can never clear α=0.05 here**. A 0.0625 next to a small t-test p means "as
significant as this test can get". If the budget allows a sixth and seventh
seed for the tier-1 groups, the floor drops to 0.031 and 0.016 and the two
tests become directly comparable — cheap insurance for a headline claim.

### Deciding what to do next

**If AWWL beats MSE with p < 0.05 after correction** — the core claim holds.
Proceed to the extension work, and report CIs in every future table.

**If `awwl_normalized` keeps the advantage but plain `awwl` does not, or vice
versa** — this is the most informative outcome. It means α's two roles can be
separated, and *that separation is a paper*: an analysis of frequency versus
timestep reweighting in diffusion losses, with the current manuscript's
"Alpha Paradox" as the motivating observation rather than an embarrassment.

**If nothing is significant on FID** — do not add datasets; adding breadth to
a null result just costs more to reject. Two honest routes remain, both
affordable here:

- *Convergence speed.* If AWWL reaches MSE's final FID at epoch 120 instead of
  200, that is a compute-saving claim measured cleanly by the tier-1 curves,
  independent of whether final quality differs.
- *Image restoration.* On SR / deblur / denoising, high-frequency detail is
  the direct evaluation criterion (LPIPS, NIQE) rather than something inferred
  through FID, effect sizes are typically much larger, and the models fit
  these GPUs comfortably. This is the strongest reframing available on this
  hardware.

Either way, report the null honestly — a negative result with five seeds and
confidence intervals is publishable in a way that an unreplicated 0.06 FID win
is not.

---

## 4. Crash recovery

Re-running the sweep is the recovery procedure:

```bash
awwl pipeline run -m configs/pipeline/phase0.yaml --gpus 0,1
```

What happens on restart:

- **Finished jobs are skipped.** Job state lives in
  `runs/phase0/pipeline/state.db` (SQLite, WAL) and survives a hard power cut.
- **Jobs that were running when the machine died are requeued.** A running job
  records a heartbeat; a stale one is reclaimed automatically at startup.
- **A requeued training job resumes from its own last checkpoint**, not from
  scratch. Optimiser moments, LR-scheduler position, EMA shadow weights and
  RNG state are all snapshotted every 5 epochs under `<run>/state/`, written
  to a staging directory and renamed into place, so a crash mid-write leaves
  either the old snapshot or the new one — never a corrupt half.
- **Interrupted sampling resumes too**: images already on disk are counted and
  only the remainder is generated.

Worst case, a crash costs about five minutes of training per in-flight job.

Other useful commands:

```bash
awwl pipeline status -m configs/pipeline/phase0.yaml   # progress + failures
awwl pipeline reset  -m configs/pipeline/phase0.yaml   # requeue failed jobs
```

A job that fails 3 times is parked as `failed` and the sweep continues past
it. Its full log is at `runs/phase0/pipeline/logs/<job_id>.log`. Fix the
cause, then `pipeline reset` and run again.

`Ctrl-C` stops cleanly: in-flight subprocesses are terminated, their jobs
return to the queue, and the next run picks them up.

### Running detached

```bash
nohup awwl pipeline run -m configs/pipeline/phase0.yaml --gpus 0,1 \
      > runs/phase0/pipeline.out 2>&1 &
```

If the SSH session drops, the sweep keeps going; if the machine reboots, run
the same command again.

---

## 5. Layout produced

```
runs/phase0/
├── results.jsonl                 # one row per finished train/eval job
├── pipeline/
│   ├── state.db                  # job queue — the crash-recovery record
│   └── logs/<job_id>.log
└── awwl_s1/                      # one directory per (config, seed)
    ├── config.json               # resolved config, read back by eval jobs
    ├── loss_history.json
    ├── checkpoint-39 … -199/     # sampling checkpoints (EMA weights if on)
    ├── samples/ep199/            # generated PNGs
    └── state/                    # resume snapshots (pruned to last 2)
```

`results.jsonl` is append-only and carries the full hyperparameter identity on
every row, so `awwl stats` never has to parse a directory name, and two
workers can write concurrently without a lock.

---

## 6. What Phase 0 does *not* cover

Deliberately out of scope, in rough order of when to consider them:

- **DreamBooth multi-seed.** Cheap (400 steps at batch 1 — minutes per run),
  and Table 1's differences are even smaller relative to their spread than
  CIFAR-10's. Worth doing right after tier 1; it needs a small addition to the
  pipeline manifest for the `dreambooth` method.
- **Learned weighting (GradNorm / uncertainty).** The fix for the α problem
  rather than a measurement of it. Highest research value, but only worth
  building once tier 2 says what α is actually doing.
- **Multi-level DWT.** The loss already takes `levels`; note that
  `level_reduction` controls whether the detail term's magnitude grows with
  the number of levels, which needs to be held fixed for the comparison to
  mean anything.
- **Prior-preservation DreamBooth, WaveDM/WaveDiff baselines, higher
  resolutions.** All expensive, all premature before Phase 0 reports.
