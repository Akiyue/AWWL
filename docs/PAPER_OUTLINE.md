# Paper draft — working document

**Local only.** Gitignored, deliberately: the repo is public and this contains
unpublished results and candid notes on what went wrong.

Every number is measured. Sections marked ⏳ are waiting on runs; nothing
pending can overturn §2–§6, which are closed.

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
- DDIM-100, 10 000 samples, identical across arms. ⏳ DDPM-1000 sensitivity
  check outstanding — the one open threat to the null.
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

## §8 Does a learned weighting escape? ⏳

Tier 4, running. `awwl_learned` (σ-conditioned uncertainty weighting, removes
α and p entirely), `awwl_gradnorm` (cited in the original related work, never
implemented there), `awwl_subband`, `awwl_spatial`, `awwl_lifting`.

Either outcome is reportable. A learned weighting that improves the spectrum
further while FID stays flat confirms the §7 accounting on an independent
mechanism. One that succeeds becomes the paper's positive result.

## §9 Generality ⏳

- DreamBooth, 5 seeds — 25/25 trained, evaluation rerunning.
- Second resolution (CelebA-64) — supported, not run. **Needed for top tier.**
- Second architecture (LDM or DiT) — not run. **Needed for top tier.**

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

| | Blocking | Cost |
|---|---|---|
| Tier 4 (§8) | yes | running, ~2.5 days |
| DreamBooth eval (§9) | yes | ~1 h |
| Second resolution (§9) | for top tier | ~2 days |
| Second architecture (§9) | for top tier | ~4 days |
| DDPM-1000 sampler check (§4) | yes | ~6 h |
| Cost table | no | minutes |

## Framing notes

- Declare the CITA 2026 conference version; check the venue's extension policy.
- **Do not oversell §6.** "Corrects the spectrum" is supported; "improves
  high-frequency detail" is not — nothing visible changed.
- The two measurement hazards are a genuine contribution, not an apology.
  Both reversed a conclusion; both generalise past this paper.
- Strongest one-line summary: **a loss can do exactly what it was designed to
  do, and still lose.**
