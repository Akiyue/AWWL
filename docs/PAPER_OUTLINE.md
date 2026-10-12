# Paper draft — working document

**Local only.** Gitignored, deliberately: the repo is public and this contains
unpublished results and candid notes on what went wrong.

Every number is measured. Sections marked ⏳ are waiting on runs; nothing
pending can overturn §2–§6, which are closed.

Numbers in the manuscript come from the ledger via `scripts/make_tables.py`;
the ones in *this* file are transcribed and can go stale. Trust `paper/tables/`.

---

## Title candidates

Lead with the phenomenon, not the method — nobody outside this project cares
about one conference paper's loss.

1. **Frequency-Aware Diffusion Losses Pay More Than They Buy**
2. What a Spectral Correction Is Worth: Auditing Frequency-Aware Losses for Diffusion Models
3. The Spectrum Is Not the Bottleneck: On Frequency-Weighted Objectives in Diffusion Training

(1) is the sharpest. (2) is safest for a journal.

---

## Abstract (draft)

Diffusion models under-produce high spatial frequencies, and a growing family
of objectives sets out to correct this by weighting wavelet or Fourier
sub-bands of the training loss. We audit one such objective across five seeds
and show that although it does exactly what it claims — closing 0.42 dB of a
2.2 dB high-frequency deficit on CIFAR-10 — it is no better than plain MSE on
FID, Inception Score or KID, and slightly worse on all three. We then quantify
what that correction should have been worth. Applying the identical correction
to the baseline's own samples as a post-processing step improves FID by
0.346 ± 0.015, so the trained objective ends 0.79 FID short of what its own
mechanism predicts: it buys a real spectral improvement and pays for it
several times over elsewhere. A twenty-line Fourier post-process on the
baseline outperforms the trained objective by the same 0.79 FID. We further
show that the published weighting conflates frequency balance with a
Min-SNR-style timestep reweighting (|r| = 0.82); isolating the frequency
component while holding the gradient budget fixed makes results significantly
worse (+1.34 FID, p = 0.003). Finally we report two measurement hazards that
reversed conclusions during this study: a hand-written KID estimator that
inverted the sign of the only positive result, and the fact that calibrating
metric sensitivity on real images measures it at a minimum where it is
quadratically insensitive, understating the true operating-point sensitivity
by roughly thirty-fold.

---

## §1 Introduction

- Spectral deficiency of diffusion samples is real and measurable: on
  CIFAR-10 the generated radial spectrum sits 0.36 dB below real at low
  frequency and **2.20 dB below at high frequency** — a 6× tilt.
- A family of objectives proposes to fix this by weighting sub-bands of the
  loss by noise level. The premise is sound; this paper asks whether the
  remedy works and, when it does not, why.
- Contributions:
  1. An analytic characterisation: on an orthonormal basis, sub-band
     weighting reallocates a fixed error budget and adds no information.
  2. A confound: the standard weighting is simultaneously a timestep
     reweighting; a protocol that separates them.
  3. A five-seed audit showing the mechanism works while the metrics do not
     move.
  4. A direct measurement of what the correction is worth, and the finding
     that the objective falls 0.79 FID short of it.
  5. Two measurement hazards that reversed conclusions mid-study.

## §2 The objective family — CLOSED

**Parseval.** For an orthonormal wavelet basis, the unweighted sub-band loss
*is* the pixel MSE:

    pixel_MSE = ¼ · Σ_b mean(b²),   relative error 7.4e-8 (db1, verified)

So the entire family reallocates one fixed error budget. Any gain comes from
*how* the budget is split, never from added information. Worth stating
plainly; the literature is vague about it.

Note also that the DWT is applied to the **prediction residual**, not to an
image: the ε-prediction target is white noise and has no structure to
decompose. Motivations phrased as "separating image structure from texture"
describe something the objective does not do.

→ `python scripts/verify_loss_math.py`

## §3 The α confound, and a protocol — CLOSED

The published weights share a denominator but do not sum to a constant: the
total runs from α at high noise to 1−α at low noise. So α sets the
frequency balance **and** rescales gradient magnitude across the schedule.

Correlation with the Min-SNR (γ=5) curve, over the σ of all 1000 training
timesteps: **r = +0.82 at α = 0.8, −0.82 at
α = 0.2.** The two published optima are opposite timestep schedules, not
merely different frequency balances — which undercuts the original paper's
attribution of the α split to image resolution.

**Protocol.** Hold Σw constant *at the original mean* (0.5), so only the shape
varies. Normalising naively to 1.0 doubles the mean gradient and confounds the
ablation with a 2× learning rate — a mistake this study made first and caught
by measuring.

| CIFAR-10, 5 seeds | Δ FID vs MSE | p (Holm) |
|---|---|---|
| `awwl` (published) | +0.44 | 0.43 |
| `awwl_normalized` (naive, 2× LR confound) | +1.15 | 0.045 |
| **`awwl_norm_matched`** (budget held fixed) | **+1.34** | **0.0029** |

Isolated from the timestep effect, the frequency weighting is **significantly
harmful**. The residual appeal of the published form comes from the timestep
reweighting riding along with it.

## §4 Protocol

- 5 seeds; mean ± std with 95% CI; paired-by-seed t-test with
  Holm–Bonferroni across the family.
- Wilcoxon reported as a direction check only: its floor is 2/2^N = 0.0625 at
  N = 5, so it cannot clear α = 0.05 and is not a verdict.
- DDIM-100, 10 000 samples, identical across arms. DDPM-1000 check **done**
  (§sampler): the gap widens rather than closing, so the sampler was
  concealing the effect, not creating it. That threat is closed.
- EMA reported as its own configuration.
- Spectra in dB (20·log10); an earlier revision used a natural log, 2.3×
  larger and not dB.

## §5 The metrics do not move — CLOSED

CIFAR-10, 200 epochs, epoch 199, 5 seeds.

| | Δ vs MSE | p (Holm) |
|---|---|---|
| FID | +0.44 (worse) | 0.43 |
| IS | −0.059 (worse) | 0.44 |
| KID (torch-fidelity) | +0.0003 (worse) | 0.77 |

- The published IS gain of **+0.17 lies outside our 95% CI of
  [−0.171, +0.054]**. The data actively excludes it.
- Convergence is *slower* at every checkpoint (ep39: 59.9 vs 52.4), closing
  the "same quality, less compute" fallback.
- `awwl_eq7`, the published equation implemented as literally written
  (detail bands summed rather than averaged): **+2.63 FID, p = 0.0004.** The
  paper's equation and its released code are different objectives, and the
  equation is much worse. A reader reimplementing from the paper gets
  something worse than the baseline.

**Measurement hazard 1 — main text, not a footnote.** A hand-written
polynomial-kernel KID showed AWWL *better* (p = 0.028 raw); `torch-fidelity`'s
KID shows it *worse*. The sign flips with the implementation. The only
positive result in the study was an artefact of non-standard code.

## §6 The mechanism works — CLOSED

Signed deviation from the real spectrum, dB (negative = too little energy):

| | low | mid | high |
|---|---|---|---|
| mse | −0.36 | −0.80 | **−2.20** |
| static_wavelet | −0.35 | −0.79 | **−2.23** |
| awwl | −0.31 | −0.56 | **−1.79** |

- The premise holds: the deficit is ~6× worse at high frequency.
- AWWL's improvement is largest at high frequency in absolute terms
  (0.41 dB vs 0.05 dB low). Be precise in the text: **broadband with a
  high-frequency tilt**, not an exclusively high-frequency correction — the
  proportional gain is largest in the mid band, and a reviewer will recompute.
- **The wavelet decomposition contributes nothing.** `static_wavelet` overlaps
  MSE across the entire frequency axis (Fig. spectrum). The temporal schedule
  is the whole effect — which validates the "adaptive" component at the
  mechanism level even as the metrics stay flat.
- Survives the §3 protocol (p = 0.0056), so it is a genuine frequency effect.
- **Perceptual:** matched samples from identical initial noise are
  indistinguishable across losses (Fig. compare). A 0.4 dB high-frequency
  change on 32×32 is below the visible threshold.

## §7 What the correction is worth — CLOSED, the paper's core

Apply the identical correction to a model's **own samples** as a Fourier
post-process and score it. Five seeds per arm; the 0 dB rung reproduces the
sweep's FID exactly (18.460 / 18.901), validating the apparatus.

| | spectral deficit | FID | FID after +0.41 dB | Δ |
|---|---|---|---|---|
| MSE | 2.220 dB | 18.460 ± 0.210 | **18.114 ± 0.202** | **−0.346 ± 0.015** |
| AWWL | 1.800 dB | 18.901 ± 0.344 | 18.568 ± 0.335 | −0.334 ± 0.011 |

Two independent measurements agree that **AWWL corrects 0.42 ± 0.18 dB**
(2.220 − 1.800), matching the 0.41 dB from §6 by a different route.

**The accounting:**

| | FID |
|---|---|
| MSE baseline | 18.460 |
| Predicted for AWWL from its own spectral correction | **18.114** |
| AWWL, actual | **18.901** |
| **Unexplained penalty** | **+0.787** |

The objective buys 0.35 FID of spectral improvement and pays roughly 1.1 FID
for it somewhere else.

Two framings of the same fact, both worth stating:

- **MSE + post-process (18.114) beats AWWL trained (18.901) by 0.79 FID.** A
  twenty-line Fourier operation captures the entire spectral benefit at zero
  training cost.
- **AWWL + post-process (18.568) is still worse than plain MSE (18.460).** The
  correction cannot recover what the objective spends to obtain it.

**Measurement hazard 2.** Calibrating FID's sensitivity on *real* images —
attenuating one half and scoring against the other — measures it at the
minimum of the distance, where the first derivative vanishes and perturbations
cost quadratically. That calibration implied a 0.41 dB correction was worth
0.010 FID and would need ~17 500 seeds to detect. Measured at the operating
point, where the derivative is non-zero, the same correction is worth 0.346 —
**thirty-fold larger.** Sensitivity calibrations must be performed away from
the minimum. This study drew, and then retracted, the wrong conclusion from
the wrong calibration.

## §8 Does a learned weighting escape? — CLOSED

Tier 4 complete. No route escapes: every added axis of adaptivity is
significantly worse than MSE and worse than the budget-held-fixed weighting it
was meant to improve on (nearest: +1.3 FID). Two arms diverged rather than
lost — GradNorm (348.8 ± 28.0) and, separately, raw SNR. The decisive row is
the σ-conditioned learned schedule at +6.66 FID (p = 0.0001): gradient descent
finds something worse than grid search published.

## §9 Generality — mostly closed

- DreamBooth, 5 seeds × 9 objectives — **done**, §generality written.
- Restoration (CIFAR-10 denoising, 3 seeds × 6 objectives) — **done**, and a
  complete null on PSNR/SSIM/LPIPS/NIQE. Worth more than it looks: this task
  was chosen *because* larger effects were expected, and there are none.
  LPIPS was recorded as 0.0 for the whole sweep by a swallowed device error;
  re-evaluated on CPU, now real. Table generated from the ledger.
- DDPM-1000 sampler check — **done**, and it strengthens the paper: the gap
  widens from +0.44 (n.s.) to +0.982 (p = 0.0002), and the static ablation
  becomes significant too. Written up as §sampler.
- Second resolution (CelebA-64) — **running, 1 worker only.** GPU 0 is held by
  unrelated processes, so the sweep gets one card: 15.5 days remaining for the
  10 outstanding runs, not the ~4 the manifest assumed. Only MSE (5/5) is done.
- Second architecture (LDM or DiT) — not run. **Needed for top tier.**

## §11 A second family member — ANALYTIC HALF CLOSED, ARM NOT RUN

Focal Frequency Loss implemented (`losses/focal_frequency.py`, 10 tests).
Two properties settled without a GPU, both now in the paper:

- **Parseval holds on the Fourier basis too**, exact to machine precision.
  So §family's claim generalises from "orthonormal wavelet" to "any unitary
  transform", which is what makes it a family claim rather than a claim about
  one basis. Written into §family, §intro contributions, §discussion.

- **The confound recurs in a weighting with no σ in it.** FFL's total is
  *exactly* invariant to the residual's scale but not to its spectral shape —
  and spectral shape is what the schedule moves. So it imposes a timestep
  reweighting as an emergent consequence, and **the §confound protocol does
  not transfer**: holding Σw fixed needs a parameter separating shape from
  total, which an error-keyed weighting has none of. Only its unweighted
  limit is available as a control, so a positive result from such a member
  could not be credited to the frequency axis. New subsection
  §confound-family, plus §extensions "A second published member".

Both computed by `verify_loss_math.py` checks 4-5, not transcribed. The
empirical arm is queued (14 GPU-h); until it runs the paper has **no
family-level empirical claim**, and §extensions says so explicitly.

Watch the wording: it is *characterised*, not *trained*. Three places said
"we run" and had to be corrected.

## §10 The SNR row is mislabelled — FIXED IN CODE, ARM NOT YET RUN

`snr_weighted` is the *raw* ratio, not Min-SNR-γ, and §generality cited Hang
et al. for it — on the paper's only positive result. Measured: the raw weight
peaks at 9997, spans 2.5e8, and puts 95% of the gradient mass in the lowest 5%
of timesteps, which is exactly why it craters on from-scratch training and
survives a 400-step fine-tune. The prose now says raw SNR and explains the
divergence; a real `min_snr` loss (γ=5) is implemented, tested and queued in
both manifests. 14 GPU-hours on CIFAR-10, minutes on DreamBooth.

## §10 Limitations — write these; do not wait to be asked

- Single architecture and dataset until §9 lands.
- DDIM-100 sampling; DDPM-1000 check outstanding.
- `spec_dist` is a hand-written radial FFT. Two independent implementations
  here agree, which is reassuring and not proof — and after §5 we are in no
  position to treat our own implementations as authoritative.
- The Fourier post-process leaves a spatial signature unlike a model's
  under-production, so §7's figure is an estimate of what the correction is
  worth, not an exact accounting.
- Absence of an effect at 32×32 does not establish absence at resolutions
  where high-frequency content carries more of the signal.

---

## Figures

1. Spectral deviation vs frequency, all arms — `awwl spectrum`. The one that
   shows mse and static_wavelet overlapping exactly.
2. Matched samples, same noise, one row per loss — `awwl compare-samples`.
3. FID vs applied boost, both arms — from §7.
4. Weight profile vs σ with the total-weight panel — `awwl plot-curriculum`.
5. Convergence curves — `awwl stats --curve`.

## Still missing before submission

Everything experimental is done: 176 training runs, 525 GPU-hours, 324
evaluated checkpoints, 40 arms, three tasks, two family members, four
operating points, a seven-point convergence curve. The manuscript has no
`\pending` or `\todo` markers left.

| | Blocking | Who |
|---|---|---|
| Authorship + CRediT | **yes** | the group — longest pole, nothing else needs it |
| Build the PDF and read it end to end | **yes** | whoever has TeX |
| Presubmission inquiry to Neurocomputing on revising a prior conference paper | **yes** | corresponding author |
| Fill the 13 placeholders in `paper/submission/` | yes | submitter |
| Zenodo DOI from a tagged release | yes | whoever owns the repo |
| CelebA-64 second arm | no — stated as a limitation, kept as revision ammunition | — |

## What the final sweeps changed

Three claims had to be **scoped to the 200-epoch recipe** because each
attenuates by roughly half at four times the budget:

| | 200 ep | 800 ep |
|---|---|---|
| Pricing shortfall | 0.787 | 0.335 ± 0.237 |
| Ancestral-sampler gap | +0.982 (p = 0.0002) | +0.344 (p = 0.12) |
| DDIM gap | +0.441 | +0.146 |

What survives every operating point is the null and its direction: behind at
all seven checkpoints, both samplers, both budgets, never ahead. The paper now
says this explicitly rather than leading with a single number.

Two results arrived that were not planned and are worth more than what they
replaced:

* **Min-SNR-γ dissolves the study's only positive finding.** The +0.0411 CLIP
  gain belongs to the *unclamped* ratio, which craters on CIFAR-10 at FID 325.
  The clamped method moves nothing (−0.0019, p = 0.45) and on CIFAR-10 trains
  normally while still losing to MSE by +2.09. Neither task gives the timestep
  axis a result once the weighting is bounded.
* **The spectral deficit widens with training.** Within one sweep, paired by
  seed, the baseline's high-band deficit grows by 0.43 dB by the epoch where
  FID is *best* and 0.67 dB by the end, while FID improves. That refutes the
  under-convergence explanation directly, and it says the spectrum and the
  metric come apart during ordinary training — which is the paper's thesis,
  measured without any frequency-aware objective involved.

## Framing notes

- Declare the CITA 2026 conference version; check the venue's extension policy.
- **Do not oversell §6.** "Corrects the spectrum" is supported; "improves
  high-frequency detail" is not — nothing visible changed.
- The two measurement hazards are a genuine contribution, not an apology.
  Both reversed a conclusion; both generalise past this paper.
- Strongest one-line summary: **a loss can do exactly what it was designed to
  do, and still lose.**
