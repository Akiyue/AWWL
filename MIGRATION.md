# Migration mapping

Every file from `AWWL/` and `AWWL-Diff/` is listed below. The two source
folders are **not modified** — they remain on disk as-is. Existing model
weights (`*.safetensors` under `AWWL/output/...` and
`AWWL-Diff/{ablat_,baseline_,rescue_,test_,mse}*/checkpoint-*`) are referenced
by the registry at `configs/checkpoints/registry.yaml`; none are copied.

## Where the AWWL-Diff version was preferred over AWWL

* `train.py` (AWWL-Diff) was a strict subset of `train_cifar10.py` (no JSON
  loss history); the latter is canonical and powers
  `methods/finetune/trainer.py`.

## Canonical loss formula

The default `loss.weighting` is **normalized**, matching the published
paper (eqs. 4–5):

```
w_LL  = α · σ^p / (σ^p + (1-σ)^p)
w_det = (1-α) · (1-σ)^p / (σ^p + (1-σ)^p)
```

Every number in the paper's Tables 1–3 was produced under this formula.
The **boosted** variant (`w = 0.5 + σ^p`, an AWWL-Diff experimental form)
is still available via `loss.weighting: boosted` for ablations.

## AWWL/ → awwl/

| Old path                                   | New path                                                                | Notes |
| ------------------------------------------ | ----------------------------------------------------------------------- | ----- |
| `AWWL/dreambooth.py`                       | `src/awwl/methods/dreambooth/trainer.py`                                | Argparse → typed config dict |
| `AWWL/finetune.py`                         | `src/awwl/methods/dreambooth/lora_trainer.py`, `src/awwl/models/lora.py`| LoRA helper extracted into `models/` |
| `AWWL/losses.py`                           | `src/awwl/losses/{adaptive_wavelet,perceptual,analytic,factory}.py`     | Split by concern; weighting flag added |
| `AWWL/inference.py`                        | `src/awwl/methods/dreambooth/inference.py` (SD-pipeline path)           | Canonical inference path |
| `AWWL/inference_lora.py`                   | EXCLUDE                                                                 | Manual DDPM loop superseded by SD pipeline (Q4 resolution) |
| `AWWL/eval.py`                             | EXCLUDE                                                                 | Strict subset of `evaluate.py` |
| `AWWL/evaluate.py`                         | `src/awwl/evaluation/clip_scores.py`                                    | Multi-prompt + image-image |
| `AWWL/evaluate_image_similarity.py`        | merged into `src/awwl/evaluation/clip_scores.py` (`image_image_similarity`) | |
| `AWWL/plot.py`                             | `src/awwl/plotting/bar_scatter.py`                                      | Same figure, parameterised |
| `AWWL/train.py`                            | EXCLUDE                                                                 | 0-byte file |
| `AWWL/run.sh`, `AWWL/run2.sh`              | EXCLUDE (replaced by configs)                                           | Hyperparameters live in `configs/dreambooth.yaml` |
| `AWWL/prompts.txt`                         | `assets/prompts/awwl_dreambooth.txt`                                    | |
| `AWWL/dataset/`, `AWWL/dataset2/`          | reference-only                                                          | `configs/dreambooth.yaml::data.instance_data_dir` |
| `AWWL/datasets/`                           | EXCLUDE                                                                 | Empty folder |
| `AWWL/output/**`                           | reference-only via `configs/checkpoints/registry.yaml`                  | |
| `AWWL/result*/`, `AWWL/results*/`, `AWWL/results_news/`, `AWWL/generated/` | EXCLUDE | Regeneratable artifacts |
| `AWWL/*.png`, `AWWL/*.csv`                 | EXCLUDE                                                                 | Regeneratable artifacts |
| `AWWL/__pycache__/`                        | EXCLUDE                                                                 | |

## AWWL-Diff/ → awwl/

| Old path                                  | New path                                                  | Notes |
| ----------------------------------------- | --------------------------------------------------------- | ----- |
| `AWWL-Diff/train_cifar10.py`              | `src/awwl/methods/finetune/trainer.py`                    | Canonical; loss-history logger preserved |
| `AWWL-Diff/train.py`                      | EXCLUDE                                                   | Subset of `train_cifar10.py` |
| `AWWL-Diff/losses.py`                     | merged into `src/awwl/losses/*` (boosted = canonical)     | |
| `AWWL-Diff/data.py`                       | `src/awwl/data/cifar10.py::dump_reference_split`          | |
| `AWWL-Diff/eval_fid.py`                   | `src/awwl/evaluation/fid_is.py` + `src/awwl/methods/finetune/inference.py::generate_samples` | Generation and metrics split |
| `AWWL-Diff/advanced.py`                   | `src/awwl/evaluation/advanced_metrics.py`                 | |
| `AWWL-Diff/timestep.py`                   | `src/awwl/evaluation/timestep_analysis.py`                | Hardcoded paths removed |
| `AWWL-Diff/spectrum_plot.py`              | `src/awwl/evaluation/spectrum.py` + `src/awwl/plotting/spectrum.py` | Compute / plot split |
| `AWWL-Diff/radar_plot.py`                 | `src/awwl/plotting/radar.py`                              | Hardcoded data → `RadarPlotSpec` |
| `AWWL-Diff/plot.py`                       | `src/awwl/plotting/ablation.py`                           | Hardcoded data → `AlphaSeries`/`PowerSeries` |
| `AWWL-Diff/plot_losses.py`                | `src/awwl/plotting/loss_curve.py`                         | Hardcoded experiments → `LossCurveSpec` |
| `AWWL-Diff/create_grid.py`                | `src/awwl/plotting/grids.py`                              | |
| `AWWL-Diff/run.sh`, `AWWL-Diff/run_ablation.sh` | EXCLUDE                                             | Replaced by configs |
| `AWWL-Diff/requirements.txt`              | merged into `pyproject.toml` + `requirements.txt`         | |
| `AWWL-Diff/data/`, `AWWL-Diff/cifar10_train_ref/` | reference-only                                    | `configs/eval/fid_is.yaml::fid_is.real_folder` |
| `AWWL-Diff/{ablat_,baseline_,rescue_,test_,mse}*/` | reference-only via `configs/checkpoints/registry.yaml` | |
| `AWWL-Diff/*.png`, `AWWL-Diff/*.txt` (logs/results) | EXCLUDE                                       | Regeneratable artifacts |
| `AWWL-Diff/__pycache__/`                  | EXCLUDE                                                   | |

## Functional changes vs. the originals

* **No hardcoded paths.** `timestep.py`, `spectrum_plot.py`, `radar_plot.py`,
  `plot.py`, `plot_losses.py`, and `create_grid.py` all carried hardcoded
  paths or hardcoded data; these now arrive via configs or function arguments.
* **No `print` in library code.** Every module logs through `logging`. Only
  the CLI ``awwl`` writes directly to stdout (typer `echo`).
* **One loss factory.** Trainers and tests all go through
  `awwl.losses.get_loss_function(name, ...)`; adding a new loss is a
  single edit in `losses/factory.py`.
* **Config-driven `output_dir` for Finetune.** The auto-suffix scheme
  (`<output>_a<a>_p<p>_<wavelet>`) from `train_cifar10.py` is preserved.
* **Loss history is always logged** for Finetune runs — the older
  `train.py` skipped this; we picked the newer behaviour as canonical.
* **`AdaptiveWaveletLoss` weighting is configurable.** Default `boosted`
  (AWWL-Diff). Set `loss.weighting: normalized` to reproduce AWWL/output
  checkpoints exactly.
