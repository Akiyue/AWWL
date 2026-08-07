# Adaptive Weight Wavelet Loss (AWWL)
### A Dynamic Frequency-Aware Loss Function for Diffusion Model Training

> **Do Thi Phuong Thao**¹·², **Tran Ngoc Duc Anh**¹, **Nguyen Van An**¹, **Le Quang Ngoc**¹, **Tran Huy Hoang Son**¹
> ¹ Faculty of Information Technology, Hanoi University, Vietnam · ² School of Computing, Phenikaa University, Vietnam
>
> *The Conference on Information Technology and its Applications* (CITA 2026), Hanoi University.

---

This repository is the official implementation of **AWWL**, a parameter-free,
plug-and-play loss for diffusion-model training that aligns frequency
prioritisation with the progressive denoising schedule. AWWL decomposes the
noise-prediction residual with a 2-D Discrete Wavelet Transform and weights
the low-frequency (LL) and high-frequency (LH/HL/HH) sub-bands by a function
of the noise magnitude σₜ:

* **High-noise phase** (σₜ → 1): the loss prioritises **global structure** (LL).
* **Low-noise phase** (σₜ → 0): the loss focuses on **fine detail** (HF bands).

AWWL adds no parameters, requires no architectural changes, and outperforms
pixel-domain (MSE / L₁ / Huber / Charbonnier) and fixed-weight wavelet
baselines on both subject-driven fine-tuning (DreamBooth) and unconditional
image generation (CIFAR-10).

## Method

The loss is a weighted sum of L₂ distances on the 2-D DWT of the predicted
and ground-truth noise (paper eqs. 4–7):

```
                  α · σₜᵖ                          (1-α) · (1-σₜ)ᵖ
  w_LL(σₜ) = ──────────────────       w_det(σₜ) = ──────────────────
              σₜᵖ + (1-σₜ)ᵖ                       σₜᵖ + (1-σₜ)ᵖ

  L_AWWL = w_LL(σₜ) · ‖ε̂_LL − ε_LL‖²₂
         + w_det(σₜ) · (‖ε̂_LH − ε_LH‖²₂ + ‖ε̂_HL − ε_HL‖²₂ + ‖ε̂_HH − ε_HH‖²₂)
```

* **σₜ** = √(1 − ᾱₜ): noise magnitude from the diffusion schedule.
* **α ∈ (0, 1)**: global bias between structure and detail.
* **p > 0**: sharpness of the temporal transition (linear vs quadratic).

The reference implementation lives in
[`src/awwl/losses/adaptive_wavelet.py`](src/awwl/losses/adaptive_wavelet.py).

## Reported results

### Table 1 — DreamBooth subject-driven fine-tuning (Stable Diffusion 1.5)

| Loss                  | CLIP Score ↑   | Image Similarity ↑ |
| --------------------- | -------------- | ------------------ |
| MSE (L₂)              | 0.298 ± 0.035  | 0.864 ± 0.057      |
| L₁                    | 0.283 ± 0.037  | 0.890 ± 0.055      |
| Charbonnier           | 0.274 ± 0.036  | **0.907 ± 0.049**  |
| Perceptual (VGG)      | 0.284 ± 0.037  | 0.876 ± 0.049      |
| SNR-Weighted          | **0.315 ± 0.032** | 0.821 ± 0.062   |
| Static Wavelet (p=0)  | 0.298 ± 0.035  | 0.878 ± 0.057      |
| **AWWL (Ours)** *(α=0.8, p=2.0)* | 0.306 ± 0.036 | 0.889 ± 0.057 |

AWWL is among the strongest methods on both metrics at once — 2nd on CLIP,
and within noise of 2nd on Similarity (L₁'s 0.890 leads by 0.001, far inside
the ±0.055 spread) — avoiding the sharp polarisation of the single-objective
baselines.

> ⚠️ **These are single-run numbers.** The gaps between methods (~0.01-0.02)
> are much smaller than the reported standard deviations (~0.035-0.057), so
> the ordering above is not yet established. See
> [`docs/PHASE0.md`](docs/PHASE0.md) for the multi-seed replication that
> tests it.

### Table 2 — CIFAR-10 unconditional generation (200 epochs)

| Method                        | FID ↓     | IS ↑      | KID ↓     | Precision ↑ | Recall ↑  |
| ----------------------------- | --------- | --------- | --------- | ----------- | --------- |
| MSE                           | 16.68     | 7.78      | 0.00746   | 0.7855      | 0.6386    |
| L₁                            | 19.70     | 7.69      | 0.00897   | 0.7891      | 0.6213    |
| Huber                         | 20.13     | 7.62      | 0.00888   | 0.7840      | 0.6133    |
| Perceptual                    | 20.02     | 7.70      | 0.00874   | **0.7905**  | 0.6096    |
| SNR-Weighted                  | 19.83     | 7.58      | 0.00914   | 0.7866      | 0.6153    |
| Static Wavelet (p=0)          | **16.55** | 7.81      | 0.00728   | 0.7792      | **0.6446**|
| **AWWL (Ours)** *(α=0.2, p=1.0)* | _16.62_ | **7.95**  | **0.00722** | 0.7805    | _0.6440_  |

AWWL achieves the best Inception Score and KID, with FID and Recall on par
with the strongest baselines.

> ⚠️ **Single seed per row.** The top three rows are separated by less than
> 0.15 FID, which is smaller than the usual seed-to-seed spread for this
> recipe — and the best FID in the table belongs to the static (p=0) ablation,
> not to AWWL. Sampling used ancestral DDPM without EMA weights, which is why
> the absolute values sit well above published DDPM CIFAR-10 FIDs. The
> replication in [`docs/PHASE0.md`](docs/PHASE0.md) addresses all three.

### Key ablation findings (Table 3)

* **α — frequency balance.** CIFAR-10 is best with low α (α=0.2 → FID 16.62);
  high α degrades sharply (α=0.95 → FID 27.95). DreamBooth, by contrast,
  prefers α=0.8 — high-resolution photos rely on global structure for
  identity, low-resolution images on edges.
* **p — schedule sharpness.** Linear (p=1.0) outperforms quadratic (p=2.0)
  on CIFAR-10 (IS 7.95 vs 7.71). The static (p=0) variant matches AWWL on
  FID but loses 0.14 IS points, validating the temporal-adaptivity claim.
* **Wavelet basis.** Haar (`db1`) marginally beats `db4` on CIFAR-10
  (FID 16.62 vs 16.76); recommended default for low-resolution data.

## Installation

```bash
git clone https://github.com/Akiyue/AWWL.git awwl && cd awwl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[eval,plot,dev]"
```

The `eval` extra adds `clean-fid`, `torch-fidelity`, `prdc`, and `scipy`;
`plot` adds `matplotlib` and `seaborn`; `dev` adds `pytest`, `ruff`, `black`.

## Reproducing the paper

### Task 1 — DreamBooth (Table 1)

```bash
# Train AWWL on the 'sks robot toy' subject (paper config).
# The default points at ../AWWL/dataset/; override --data.instance_data_dir
# to point at your own folder of subject images.
awwl train --config configs/dreambooth.yaml \
    --override data.instance_prompt="a photo of sks robot toy"

# Sample with the three paper prompts
for prompt in "a photo of sks dog in a cyberpunk neon city" \
              "a photo of sks robot toy on the beach at sunset" \
              "a photo of sks vase as a watercolor painting"; do
  awwl infer --method dreambooth \
      --weights ./runs/dreambooth/unet \
      --output-dir "./runs/dreambooth/samples/$(echo "$prompt" | tr ' ' _ | head -c 32)" \
      --prompt "$prompt" --num-samples 50
done

# Score against the real subject images
awwl eval --config configs/eval/clip.yaml \
    --override generate.models_root=./runs/dreambooth_all \
    --override generate.real_images_dir=../AWWL/dataset
```

The full benchmark sweep (8 baselines + AWWL, three subjects) is captured
in `configs/checkpoints/registry.yaml`; see *Pre-trained checkpoints* below.

### Task 2 — CIFAR-10 unconditional (Table 2)

```bash
# Train AWWL with the paper-optimal config (alpha=0.2, p=1.0, db1)
awwl train --config configs/finetune.yaml

# Sample 10 000 images for FID/IS
awwl infer --method finetune \
    --weights ./runs/finetune_a0.2_p1.0_db1/checkpoint-199 \
    --output-dir ./runs/finetune_samples \
    --num-samples 10000

# FID + IS against the CIFAR-10 reference split
awwl eval --config configs/eval/fid_is.yaml \
    --override fid_is.out_folder=./runs/finetune_samples

# KID + Precision/Recall + spectral distance
awwl eval --config configs/eval/advanced.yaml \
    --override advanced.fake_folder=./runs/finetune_samples
```

### Reproducing ablations (Table 3)

Override the relevant hyperparameter on the CLI:

```bash
# Impact of alpha at p=2.0 (Table 3, top half)
for a in 0.5 0.8 0.95; do
  awwl train --config configs/finetune.yaml \
      --override loss.alpha=$a --override loss.power=2.0
done

# Impact of p at alpha=0.2 (Table 3, middle)
for p in 0 0.5 1.0 2.0; do
  awwl train --config configs/finetune.yaml \
      --override loss.alpha=0.2 --override loss.power=$p
done

# Wavelet basis (Table 3, bottom)
awwl train --config configs/finetune.yaml \
    --override loss.alpha=0.2 --override loss.power=1.0 \
    --override loss.wavelet_type=db4
```

## Pre-trained checkpoints

`configs/checkpoints/registry.yaml` maps every named experiment in the paper
to its on-disk checkpoint. Inspect with:

```bash
awwl list-checkpoints
```

Use a logical name with `awwl infer`:

```bash
# DreamBooth, paper's "Proposed Best" run (Table 1, AWWL row).
awwl infer --method dreambooth --registry proposed_best \
    --output-dir ./runs/eval/dreambooth_paper \
    --prompt "a photo of sks robot toy on the beach at sunset" --num-samples 50

# Finetune, paper's α=0.2 / p=1.0 / db1 run (Table 2, AWWL row).
awwl infer --method finetune --registry proposed_best \
    --output-dir ./runs/eval/finetune_paper --num-samples 10000
```

Pointing directly at any checkpoint folder works too:

```bash
awwl infer --method dreambooth \
    --weights /path/to/AWWL/output/full_benchmark_final_v3/17_proposed_best/unet \
    --output-dir ./runs/eval/custom \
    --prompt "a photo of sks robot toy on the beach at sunset"
```

Weights are **never copied or modified** — the registry resolves to the
original `AWWL/` and `AWWL-Diff/` folders that produced the paper's numbers.

## Replication and sweeps

Single runs cannot separate the methods above, so the repo ships a resumable
multi-GPU pipeline for multi-seed replication. Full rationale, cost estimates
and how to read the outcome: [`docs/PHASE0.md`](docs/PHASE0.md).

```bash
# Free, CPU-only: check three mathematical properties of the objective
python scripts/verify_loss_math.py

# One-off: CIFAR-10 as PNGs for the FID/KID reference set
awwl prepare-data --output ./data/cifar10_train_png

# The sweep. Re-run this exact command after any crash — it resumes.
awwl pipeline run -m configs/pipeline/phase0.yaml --gpus 0,1

awwl pipeline status -m configs/pipeline/phase0.yaml
awwl pipeline reset  -m configs/pipeline/phase0.yaml   # requeue failures
```

Jobs are tracked in SQLite (`runs/phase0/pipeline/state.db`), each runs as a
subprocess pinned to one GPU, and a job whose worker dies is requeued from its
heartbeat. Training resumes from a full state snapshot — optimiser moments,
LR-scheduler position, EMA shadow and RNG — written every few epochs, so a
crash costs minutes rather than hours. Interrupted sampling resumes too.

Results land in an append-only `results.jsonl`, analysed with:

```bash
awwl stats -l runs/phase0/results.jsonl --metric fid --epoch 199 --baseline mse
awwl stats -l runs/phase0/results.jsonl --metric fid --curve
```

which reports mean ± std with 95% CIs per configuration, then a seed-paired
t-test and Wilcoxon signed-rank test against the baseline, Holm-Bonferroni
corrected across the comparisons.

## Loss options beyond the paper's configuration

`AdaptiveWaveletLoss` defaults reproduce the published runs exactly. Three
options exist to resolve discrepancies between the manuscript and this code
(all verified numerically by `scripts/verify_loss_math.py`):

| Option | Default | What it changes |
| --- | --- | --- |
| `normalize_weights` | `false` | Eqs. (4)-(5) do **not** sum to a constant: the total runs from `α` at high noise to `1-α` at low noise, so `α` is also a Min-SNR-style timestep reweighting. `true` pins the total at 1 for every σ, isolating the frequency balance. |
| `detail_reduction` | `mean` | Eq. (7) *sums* the three detail bands; the code averages them — a factor of 3 on the detail term. `sum` implements the equation as written. |
| `level_reduction` | `sum` | With `levels > 1`, whether the detail term's magnitude grows with the number of levels. |

## Repository layout

```
awwl/
├── configs/
│   ├── base.yaml
│   ├── dreambooth.yaml      # Table 1, paper-optimal alpha=0.8, p=2.0
│   ├── dreambooth_lora.yaml
│   ├── finetune.yaml        # Table 2, paper-optimal alpha=0.2, p=1.0, db1
│   ├── eval/{clip,fid_is,advanced}.yaml
│   ├── pipeline/phase0.yaml # Multi-seed replication sweep
│   └── checkpoints/registry.yaml
├── src/awwl/
│   ├── losses/              # AWWL + 8 baseline losses, one factory
│   │   ├── adaptive_wavelet.py   # paper eqs. (4)-(7); also Static (p=0)
│   │   ├── analytic.py           # mse, l1, huber, charbonnier, vlb, snr, kl
│   │   ├── perceptual.py         # VGG perceptual loss
│   │   └── factory.py            # get_loss_function(name, **cfg)
│   ├── methods/
│   │   ├── dreambooth/      # SD 1.5 trainer + LoRA variant + inference
│   │   └── finetune/        # CIFAR-10 DDPM trainer + inference (DDPM/DDIM)
│   ├── pipeline/            # Resumable sweeps
│   │   ├── store.py              # SQLite job queue, crash recovery
│   │   ├── manifest.py           # experiment matrix -> job DAG
│   │   └── runner.py             # one subprocess worker per GPU
│   ├── analysis/            # Results ledger + significance testing
│   │   ├── results.py            # append-only results.jsonl
│   │   └── stats.py              # CIs, paired t-test, Wilcoxon, Holm
│   ├── data/                # DreamBooth / HF-image / CIFAR-10 datasets
│   ├── models/              # SD components, DDPM UNet, LoRA injection
│   ├── evaluation/          # CLIP, FID/IS, KID/PR/spectral, timestep
│   ├── plotting/            # Bar/scatter, radar, ablation, grids, etc.
│   ├── training/            # Accelerator, EMA, crash-safe checkpointing
│   ├── utils/               # YAML loader, seeding, logging, paths
│   ├── core/                # Registries, exceptions
│   └── cli.py               # train | infer | eval | eval-samples |
│                            # prepare-data | pipeline | stats
├── scripts/
│   └── verify_loss_math.py  # CPU checks of Parseval / eq.(7) / the alpha confound
├── docs/PHASE0.md           # How to run the replication, and how to read it
├── tests/                   # CPU tests (~30 s)
├── assets/prompts/          # The three paper prompts
└── MIGRATION.md             # Per-file map AWWL/, AWWL-Diff/ → src/awwl/
```

## Loss zoo

Every baseline in Tables 1–2 ships in [`src/awwl/losses/`](src/awwl/losses/):

| Name in paper          | `loss.name`         | Module                                       |
| ---------------------- | ------------------- | -------------------------------------------- |
| MSE (L₂)               | `mse`               | `analytic.py::mse`                           |
| L₁                     | `l1`                | `analytic.py::l1`                            |
| Huber                  | `huber`             | `analytic.py::huber`                         |
| Charbonnier            | `charbonnier`       | `analytic.py::charbonnier_loss`              |
| Perceptual (VGG)       | `perceptual`        | `perceptual.py::PerceptualLoss`              |
| Variational Lower Bound| `vlb`               | `analytic.py::vlb_loss`                      |
| SNR-Weighted (Min-SNR) | `snr_weighted`      | `analytic.py::snr_weighted_loss`             |
| KL on x₀               | `kl_x0`             | `analytic.py::kl_loss_x0`                    |
| Static Wavelet (p=0)   | `adaptive_wavelet`<br>+ `power=0.0` | `adaptive_wavelet.py::AdaptiveWaveletLoss` |
| **AWWL (Ours)**        | `adaptive_wavelet`  | `adaptive_wavelet.py::AdaptiveWaveletLoss`   |

## Tests + lint

```bash
pytest                    # CPU only, no GPU or network needed
ruff check src tests      # static checks
black --check src tests
```

The suite covers the loss zoo, the Parseval / eq.(7) / α-confound properties
asserted in `docs/PHASE0.md`, checkpoint save-resume (including a torn
snapshot and a dangling pointer), and the job store's exclusivity,
dependency ordering and stale-heartbeat recovery.

## Reproducibility

Every config seeds Python, NumPy, and PyTorch (default seed 42). For
bit-identical training across machines, set `deterministic: true` in the
config — this also engages cuDNN-deterministic mode. Off by default since
mixed-precision relies on non-deterministic kernels.

## Citation

```bibtex
@inproceedings{thao2026awwl,
  title     = {Adaptive Weight Wavelet Loss: A Dynamic Frequency-Aware
               Loss Function for Diffusion Model Training},
  author    = {Do, Thi Phuong Thao and Tran, Ngoc Duc Anh and
               Nguyen, Van An and Le, Quang Ngoc and Tran, Huy Hoang Son},
  booktitle = {Proceedings of The Conference on Information Technology and its Applications (CITA)},
  year      = {2026},
  address   = {Hanoi, Vietnam},
}
```

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

For questions about the paper or implementation, please open a GitHub issue
or contact the corresponding authors:
* `thaodtp_fit@hanu.edu.vn`
* `25900004@st.phenikaa-uni.edu.vn`
