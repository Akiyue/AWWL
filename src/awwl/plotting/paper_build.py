"""Every figure in ``paper/figures``, rebuilt from the ledgers by one call.

:data:`FIGURES` maps each file name the manuscript includes to the function
that writes it; :func:`build` runs them. ``awwl figures --out paper/figures``
is the command-line front end. A figure whose inputs are absent is skipped
with a message, so the command is safe to run part-way through a sweep, and
``tests/test_paper_figures.py`` checks that every figure the manuscript
includes has an entry here.

Spectral profiles are expensive (one FFT pass over 10,000 samples per run),
so they are looked up before they are computed: first in the pricing cache
(``runs/pricing/cache/<dataset>/profiles/<arm>_s<seed>_ep<epoch>.npy``, the
float pipeline of :mod:`awwl.evaluation.price_fid`, all images), then in the
pricing ledger's ``identity`` rows, and only then computed from the samples
with the same function. ``profile_cache_out`` saves what was computed.

Each figure is authored at the width the manuscript includes it at, read from
its ``\\guardedfigure`` in ``paper/sections`` (:func:`include_fraction`), so
its type prints at the size it was set in.

The three pricing figures (``teaser``, ``price_curve``, ``pricing_arms``)
take every number through the helpers of ``scripts/make_tables.py``
(:func:`tables_module`), the code that writes Tables accounting, operators,
price-curve and pricing-arms, so a figure cannot disagree with its table.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Sources:
    """Where each figure's inputs live. Defaults are the repository layout."""

    ledger: Path = Path("runs/phase0/results.jsonl")
    epoch: int = 199
    baseline: str = "mse"
    converged: Path = Path("runs/converged/results.jsonl")
    converged_epoch: int = 799
    #: The positive control's ``A``: the 800-epoch MSE runs (with EMA) at their FID
    #: optimum (make_numbers.py ``longBestEpochMSE``, ``posOpt*``).
    positive_epoch: int = 399
    ddpm1000: Path = Path("runs/phase0_ddpm1000/results.jsonl")
    converged_ddpm1000: Path = Path("runs/converged_ddpm1000/results.jsonl")
    dreambooth: Path = Path("runs/dreambooth_seeds/results.jsonl")
    restoration: Path = Path("runs/restoration_cifar/results.jsonl")
    pricing: Path = Path("runs/pricing/results.jsonl")
    # Read by no figure since the boost-only correction figure was retired;
    # kept because ``awwl figures`` still accepts --boost and --boost-dose.
    boost: Path = Path("runs/boost")
    boost_dose: float | None = None
    phase0_root: Path = Path("runs/phase0")
    converged_root: Path = Path("runs/converged")
    real: Path = Path("data/cifar10_train_png")
    profile_cache: Path = Path("runs/pricing/cache")
    profile_cache_out: Path | None = None
    alpha: float = 0.2
    power: float = 1.0
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    compare_seed: int = 1
    _memo: dict = field(default_factory=dict, repr=False)


class SkipFigure(Exception):  # noqa: N818 -- a signal, not an error
    """A figure's inputs are not there; report and carry on."""


# --------------------------------------------------------------------------
# Ledger access


def _eval_rows(path: Path) -> list[dict]:
    from awwl.analysis.results import load_results

    if not Path(path).exists():
        raise SkipFigure(f"no ledger at {path}")
    rows = load_results(path, kind="eval")
    if not rows:
        raise SkipFigure(f"no eval rows in {path}")
    return rows


def _pricing_rows(src: Sources) -> list[dict]:
    import json

    key = ("pricing", str(src.pricing))
    if key not in src._memo:
        rows = []
        if Path(src.pricing).exists():
            for line in Path(src.pricing).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        src._memo[key] = rows
    return src._memo[key]


# --------------------------------------------------------------------------
# Spectral profiles


def _dataset_of(root: Path) -> str:
    return "cifar10_long" if Path(root).name == "converged" else "cifar10"


def reference_profile(src: Sources) -> np.ndarray:
    """The real images' radial profile (all images), cached or computed."""
    if "ref" in src._memo:
        return src._memo["ref"]
    for dataset in ("cifar10", "cifar10_long"):
        path = Path(src.profile_cache) / dataset / "ref_profile.npy"
        if path.exists():
            src._memo["ref"] = np.load(path)
            return src._memo["ref"]
    from awwl.evaluation import price_fid as pf

    if not Path(src.real).is_dir():
        raise SkipFigure(f"no reference images at {src.real}")
    logger.info("computing the reference profile from %s", src.real)
    ref = pf.radial_profile(pf.load_images(src.real))
    if src.profile_cache_out is not None:
        out = Path(src.profile_cache_out) / "cifar10" / "ref_profile.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, ref)
    src._memo["ref"] = ref
    return ref


def run_profile(src: Sources, root: Path, group: str, seed: int, epoch: int) -> np.ndarray | None:
    """One run's radial profile (all samples, float pipeline), or ``None``."""
    dataset = _dataset_of(root)
    name = f"{group}_s{seed}_ep{epoch}.npy"
    for base in [src.profile_cache_out, src.profile_cache]:
        if base is None:
            continue
        path = Path(base) / dataset / "profiles" / name
        if path.exists():
            return np.load(path)
    for row in _pricing_rows(src):
        if (row.get("op") == "identity" and row.get("source") == group
                and row.get("seed") == seed and row.get("source_epoch") == epoch
                and row.get("quantize", "float") == "float"
                and row.get("dataset") == dataset and row.get("profile_before")):
            return np.asarray(row["profile_before"], dtype=float)
    folder = Path(root) / f"{group}_s{seed}" / "samples" / f"ep{epoch}"
    if not folder.is_dir():
        return None
    from awwl.evaluation import price_fid as pf

    logger.info("computing the radial profile of %s", folder)
    profile = pf.radial_profile(pf.load_images(folder))
    if src.profile_cache_out is not None:
        out = Path(src.profile_cache_out) / dataset / "profiles" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, profile)
    return profile


def _high_band_deficit(profile: np.ndarray, ref: np.ndarray) -> float:
    from awwl.evaluation.price_fid import band_deviations

    return -band_deviations(profile, ref)[2]


# --------------------------------------------------------------------------
# Printed width


def include_fraction(name: str, default: float) -> float:
    """The fraction of the line width the manuscript includes ``name`` at.

    Read from ``\\guardedfigure{<fraction>}{<name>}...`` in the section
    files, so a figure is authored at the width it prints at (and its type at
    the size it is set in) even after an author changes the include width;
    ``default`` when no section includes it. The sections are looked up beside
    the source tree and then under the working directory.
    """
    for base in (Path(__file__).resolve().parents[3], Path.cwd()):
        sections = base / "paper" / "sections"
        if sections.is_dir():
            break
    else:
        return default
    pattern = re.compile(r"\\guardedfigure\{([0-9.]+)\}\{" + re.escape(name) + r"\}")
    for tex in sorted(sections.glob("*.tex")):
        body = re.sub(r"(?<!\\)%.*", "", tex.read_text(encoding="utf-8"))
        found = pattern.search(body)
        if found:
            return float(found.group(1))
    return default


def _width(out: Path, default: float) -> float:
    return include_fraction(Path(out).name, default)


# --------------------------------------------------------------------------
# Builders: one per figure file.


def build_weights(src: Sources, out: Path) -> Path:
    from awwl.plotting.paper_figures import plot_weight_schedule

    return plot_weight_schedule(alpha=src.alpha, power=src.power, out_path=out,
                                fraction=_width(out, 0.95))


def build_gradient_scale(src: Sources, out: Path) -> Path:
    from awwl.plotting.paper_figures import plot_gradient_scale

    return plot_gradient_scale(alpha=src.alpha, power=src.power, out_path=out,
                               fraction=_width(out, 1.0))


def build_effect_sizes(src: Sources, out: Path) -> Path:
    from awwl.plotting.paper_figures import plot_effect_sizes

    rows = [r for r in _eval_rows(src.ledger) if r.get("epoch") == src.epoch]
    return plot_effect_sizes(rows, metric="fid", baseline=src.baseline, out_path=out,
                             fraction=_width(out, 0.92))


def build_convergence(src: Sources, out: Path) -> Path:
    from awwl.plotting.paper_figures import plot_convergence

    rows = _eval_rows(src.ledger)
    return plot_convergence(rows, metric="fid", baseline=src.baseline, out_path=out,
                            configs=[src.baseline, "static_wavelet", "awwl"], interval="bars",
                            fraction=_width(out, 1.0))


def build_budget_curve(src: Sources, out: Path) -> Path:
    from awwl.plotting.paper_figures import plot_convergence

    rows = _eval_rows(src.converged)
    return plot_convergence(rows, metric="fid", baseline=src.baseline, out_path=out,
                            configs=[src.baseline, "awwl"], interval="band",
                            fraction=_width(out, 1.0))


def build_budget_sampler(src: Sources, out: Path) -> Path:
    """Four of Table budget's operating points, corrected as the table corrects them.

    The points come from make_tables.py ``budget_points``, the code behind
    Table budget: the two 800-epoch points carry the Holm p of the Budget
    family (those two contrasts), and each 200-epoch point the corrected p of
    the family its contrast belongs to (the main family under DDIM-100, the
    sampler family under DDPM-1000). A marker's fill therefore agrees with the
    table's bold. The 200-epoch EMA point is in the table but not drawn.
    """
    from awwl.plotting.paper_figures import plot_budget_sampler

    mt = tables_module()
    ledgers = [mt.load_eval(src.ledger, src.epoch), mt.load_eval(src.converged, src.converged_epoch),
               mt.load_eval(src.ddpm1000, src.epoch),
               mt.load_eval(src.converged_ddpm1000, src.converged_epoch)]
    if not all(ledgers):
        raise SkipFigure("a budget or sampler ledger is missing")
    short, long_, anc, long_anc = ledgers
    found = {(b, s): res for b, s, _mb, _ma, res in mt.budget_points(short, short, long_, anc, long_anc)}
    n_short = src.epoch + 1
    panels = [
        ("DDIM-100", "200 epochs, no EMA", f"{n_short} epochs,\nno EMA"),
        ("DDIM-100", "800 epochs, EMA", f"{4 * n_short} epochs,\nEMA"),
        ("DDPM-1000", "200 epochs, no EMA", f"{n_short} epochs,\nno EMA"),
        ("DDPM-1000", "800 epochs, EMA", f"{4 * n_short} epochs,\nEMA"),
    ]
    points: dict[str, list] = {}
    for sampler, point, label in panels:
        res = found.get((point, sampler))
        if res is None:
            raise SkipFigure(f"no AWWL comparison for {point}, {sampler}")
        points.setdefault(sampler, []).append(
            (label, res.mean, res.half, (res.p_holm or 1.0) < 0.05, res.n))
    return plot_budget_sampler(points, out_path=out, fraction=_width(out, 0.72))


def build_deficit_vs_metric(src: Sources, out: Path) -> Path:
    from awwl.analysis.stats import summarize_groups
    from awwl.plotting.paper_figures import plot_deficit_versus_metric

    rows = _eval_rows(src.converged)
    epochs = [199, 399, 799]
    ref = reference_profile(src)
    deficit: dict[str, list[list[float]]] = {}
    fid: dict[str, list[list[float]]] = {}
    for group in (src.baseline, "awwl"):
        for e in epochs:
            vals = []
            for seed in (1, 2, 3):
                prof = run_profile(src, src.converged_root, group, seed, e)
                if prof is not None:
                    vals.append(_high_band_deficit(prof, ref))
            if not vals:
                raise SkipFigure(f"no samples for {group} at epoch {e} under {src.converged_root}")
            deficit.setdefault(group, []).append(vals)
            s = summarize_groups([r for r in rows if r.get("group") == group
                                  and r.get("epoch") == e], metric="fid")
            if not s:
                raise SkipFigure(f"no FID for {group} at epoch {e} in {src.converged}")
            fid.setdefault(group, []).append(list(s[0].by_seed.values()))
    return plot_deficit_versus_metric(epochs=epochs, deficit=deficit, fid=fid, out_path=out,
                                      fraction=_width(out, 0.82))


def build_spectrum(src: Sources, out: Path) -> Path:
    from awwl.plotting.spectrum import plot_spectrum_deviation

    ref = reference_profile(src)
    profiles = {}
    for group in (src.baseline, "static_wavelet", "awwl"):
        per = [run_profile(src, src.phase0_root, group, s, src.epoch) for s in src.seeds]
        per = [p for p in per if p is not None]
        if len(per) < 2:
            raise SkipFigure(f"fewer than two seeds of samples for {group}")
        profiles[group] = np.stack(per)
    return plot_spectrum_deviation(
        real=ref,
        baselines={k: v for k, v in profiles.items() if k != "awwl"},
        highlight=("awwl", profiles["awwl"]),
        output_path=out,
        fraction=_width(out, 0.93),
    )


def build_compare(src: Sources, out: Path) -> Path:
    from awwl.plotting.paired_grid import plot_paired_samples

    folders = {g: src.phase0_root / f"{g}_s{src.compare_seed}" / "samples" / f"ep{src.epoch}"
               for g in (src.baseline, "static_wavelet", "awwl")}
    missing = [str(p) for p in folders.values() if not p.is_dir()]
    if missing:
        raise SkipFigure(f"no samples at {missing}")
    return plot_paired_samples(folders, out_path=out, count=8, difference=("awwl", src.baseline),
                               fraction=_width(out, 1.0))


def build_dreambooth_tradeoff(src: Sources, out: Path) -> Path:
    from awwl.plotting.paper_figures import plot_tradeoff

    rows = _eval_rows(src.dreambooth)
    return plot_tradeoff(
        rows, x_metric="clip_score", y_metric="similarity", baseline=src.baseline,
        x_label="$\\Delta$ CLIP text–image score vs. MSE ($\\rightarrow$ better)",
        y_label="$\\Delta$ subject similarity vs. MSE ($\\uparrow$ better)",
        out_path=out, fraction=_width(out, 0.80),
    )


def build_restoration_null(src: Sources, out: Path) -> Path:
    from awwl.plotting.paper_figures import plot_null_panel

    rows = _eval_rows(src.restoration)
    epoch = max(r["epoch"] for r in rows if r.get("epoch") is not None)
    rows = [r for r in rows if r.get("epoch") == epoch]
    return plot_null_panel(
        rows, baseline=src.baseline, out_path=out, fraction=_width(out, 1.0),
        # Rows in the presentation order of Table restoration (make_tables.ROWS).
        metrics=[("psnr", "PSNR (dB)", False), ("ssim", "SSIM", False),
                 ("lpips", "LPIPS", True), ("niqe", "NIQE-feat. dist.", True)],
    )


# --------------------------------------------------------------------------
# The pricing figures read the ledgers through the generated tables' own
# helpers (scripts/make_tables.py), so a figure and its table cannot disagree.

_TABLES = None


def tables_module():
    """``scripts/make_tables.py`` as a module, imported once.

    Looked up beside the source tree first and then under the working
    directory. A figure that needs it is skipped when neither has it.
    """
    global _TABLES
    import sys

    if _TABLES is not None:
        return _TABLES
    if "make_tables" in sys.modules:
        _TABLES = sys.modules["make_tables"]
        return _TABLES
    import importlib.util

    for base in (Path(__file__).resolve().parents[3], Path.cwd()):
        path = base / "scripts" / "make_tables.py"
        if path.exists():
            break
    else:
        raise SkipFigure("scripts/make_tables.py not found (run from the repository root)")
    spec = importlib.util.spec_from_file_location("make_tables", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: its dataclasses look their module up by name.
    sys.modules["make_tables"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules["make_tables"]
        raise
    _TABLES = module
    return module


def _pricing_ledger(src: Sources) -> list[dict]:
    mt = tables_module()
    pr = mt.load_pricing(src.pricing)
    if not pr:
        raise SkipFigure(f"no pricing rows in {src.pricing}")
    return pr


def _phase0_final(src: Sources) -> list[dict]:
    mt = tables_module()
    rows = mt.load_eval(src.ledger, src.epoch)
    if not rows:
        raise SkipFigure(f"no eval rows at epoch {src.epoch} in {src.ledger}")
    return rows


def _need(res, what: str):
    if res is None:
        raise SkipFigure(f"pricing ledger incomplete: {what}")
    return res


def _mean(values: dict) -> float:
    return float(np.mean(list(values.values())))


def teaser_verdicts(src: Sources):
    """The two verdicts of the teaser: AWWL, and the positive control.

    AWWL: ``B`` = MSE and ``A`` = AWWL at epoch 199 (five seeds), ``T_A`` the
    transplant of each AWWL run's radial profile onto the MSE run of the same
    seed; exactly the transplant column of Table accounting. Positive control:
    ``B`` = MSE at epoch 199 of the 800-epoch runs and ``A`` = the same runs at
    the schedule's FID optimum, ``src.positive_epoch`` (three seeds), ``T_A``
    the transplant of that checkpoint's profile onto the epoch-199 samples
    (make_numbers.py ``posOpt*``). The end of the schedule (``posCtrl*``) is
    not drawn here; Fig. pricing-arms carries it as a secondary row.
    """
    from awwl.plotting.paper_figures import Interval, Verdict

    mt = tables_module()
    pr = _pricing_ledger(src)
    rows = _phase0_final(src)
    b = mt.price(pr, "cifar10", "identity")
    tb = mt.price(pr, "cifar10", "transplant", target="awwl")
    a = mt.by_seed(rows, "awwl", "fid")
    out = [Verdict(
        title=f"AWWL ($n = {len(set(b) & set(a))}$):\n$B$ = MSE, $A$ = AWWL",
        n=len(set(b) & set(a)), fid_b=_mean(b), fid_tb=_mean(tb), fid_a=_mean(a),
        value=Interval.of(_need(mt.value_of(b, tb), "transplant of awwl")),
        gap=Interval.of(_need(mt.paired(a, b), "awwl vs mse")),
        mrg=Interval.of(_need(mt.mrg(a, tb), "MRG of awwl")), arm="awwl",
    )]
    epoch = src.positive_epoch
    pos = positive_control(src, epoch)
    if pos is None:
        logger.warning("positive control incomplete in %s / %s; teaser drawn without it",
                       src.pricing, src.converged)
        return out
    value, gap, mrg, n = pos
    means = src._memo[("positive_control_means", epoch)]
    out.append(Verdict(
        title=(f"Positive control, MSE + EMA\n($n = {n}$): $B$ at epoch {POSITIVE_BASE_EPOCH}, "
               f"$A$ at {epoch}"),
        n=n, fid_b=means[0], fid_tb=means[1], fid_a=means[2],
        value=Interval.of(value), gap=Interval.of(gap), mrg=Interval.of(mrg), arm="mse",
    ))
    return out


#: The positive control's ``B``: the 800-epoch MSE runs at epoch 199.
POSITIVE_BASE_EPOCH = 199


def positive_control(src: Sources, epoch: int | None = None):
    """``(value, gap, mrg, n)`` of the positive control, or ``None`` while incomplete.

    ``B`` is MSE at epoch 199 of the four-times-budget runs and ``A`` the same
    runs at ``epoch`` (default ``src.positive_epoch``, the schedule's FID
    optimum); ``T_A`` transplants each run's radial profile at that epoch onto
    its own epoch-199 samples. ``gap`` = M(A) - M(B). The numbers are those of
    make_numbers.py: at the optimum ``posOpt*`` (``A`` scored on the pricing
    path, ``gap`` the negative of ``posOptTrainGain``), at the last epoch
    ``posCtrl*`` (``A`` from the training ledger, ``gap`` the negative of
    ``posCtrlTrainGain``).
    """
    mt = tables_module()
    pr = _pricing_ledger(src)
    epoch = src.positive_epoch if epoch is None else epoch
    b199 = mt.price(pr, "cifar10_long", "identity", source_epoch=POSITIVE_BASE_EPOCH)
    if epoch == src.converged_epoch:
        t = mt.price(pr, "cifar10_long", "transplant", target="mse", source_epoch=POSITIVE_BASE_EPOCH)
        a = mt.by_seed(mt.load_eval(src.converged, epoch), "mse", "fid")
    else:
        t = mt.price(pr, "cifar10_long", "transplant", target="mse", source_epoch=POSITIVE_BASE_EPOCH,
                     target_epoch=epoch)
        a = mt.price(pr, "cifar10_long", "identity", source_epoch=epoch)
    value, gap, mrg = mt.value_of(b199, t), mt.paired(a, b199), mt.mrg(a, t)
    if value is None or gap is None or mrg is None:
        return None
    src._memo[("positive_control_means", epoch)] = (_mean(b199), _mean(t), _mean(a))
    return value, gap, mrg, len(set(b199) & set(a))


def build_teaser(src: Sources, out: Path) -> Path:
    from awwl.plotting.paper_figures import plot_teaser

    return plot_teaser(teaser_verdicts(src), out_path=out, fraction=_width(out, 1.0))


#: Band-restricted operators of Fig. price-curve (b): ``(header, op, target,
#: [(bands, label), ...])``; ``bands=None`` is the operator on all bands.
PRICE_BANDS = [
    ("Transplant of AWWL’s change", "transplant", "awwl",
     [("low", "low third"), ("mid", "mid third"), ("high", "high third"),
      (None, "all bands")]),
    ("Match to the reference spectrum", "match", None,
     [("low", "low third"), ("mid", "mid third"), ("high", "high third"),
      (None, "all bands")]),
]


def price_curve_points(src: Sources):
    """``(curve, controls, bands)`` for :func:`plot_price_curve`.

    CIFAR-10, epoch 199, MSE's samples, five seeds; every quantity through
    ``make_tables.price`` exactly as Tables price-curve and operators compute
    it. Holm runs over the points of each panel: the boost doses and the
    controls in (a), the band operators present in (b).
    """
    from awwl.plotting.paper_figures import Interval, PricePoint

    mt = tables_module()
    pr = _pricing_ledger(src)
    base = mt.price(pr, "cifar10", "identity")
    if not base:
        raise SkipFigure("no complete identity rows for MSE in the pricing ledger")

    def measure(op, **kw):
        value = mt.value_of(base, mt.price(pr, "cifar10", op, **kw))
        shift = mt.paired(mt.price(pr, "cifar10", op, field=mt.high_after, **kw),
                          mt.price(pr, "cifar10", op, field=mt.high_before, **kw))
        return (value, shift) if value is not None and shift is not None else None

    def points(specs):
        """A PricePoint per spec (``None`` while incomplete); Holm over those present."""
        found = [(label, kind, measure(op, **kw)) for label, kind, op, kw in specs]
        adjusted = iter(mt.holm([m[0].p for _l, _k, m in found if m is not None]))
        return [None if m is None else PricePoint(
            label, Interval.of(m[0], significant=next(adjusted) < 0.05), Interval.of(m[1]), kind)
            for label, kind, m in found]

    dose = mt.BOOST_DOSE
    noise = sorted({float(r["dose_db"]) for r in pr if r.get("dataset") == "cifar10"
                    and r.get("op") == "noise" and r.get("source", "mse") == "mse"
                    and r.get("dose_db") is not None})
    specs_a = [(f"+{d:g} dB", "boost", "boost", dict(dose=d)) for d in mt.boost_doses(pr)]
    specs_a.append((f"+{dose:g} dB boost, dithered to 8 bits", "dither", "boost",
                    dict(dose=dose, quantize="dither")))
    specs_a += [(f"$\\pm${a:g} LSB", "noise", "noise", dict(dose=a)) for a in noise]
    measured = [p for p in points(specs_a) if p is not None]
    curve = [p for p in measured if p.kind == "boost"]
    controls = [p for p in measured if p.kind != "boost"]
    if len(curve) < 2:
        raise SkipFigure("fewer than two complete boost doses in the pricing ledger")
    specs_b = [(label, op, op, dict(target=target, bands=bands) if target else dict(bands=bands))
               for _h, op, target, rows in PRICE_BANDS for bands, label in rows]
    found = iter(points(specs_b))
    blocks = [(header, [p for p in (next(found) for _ in rows) if p is not None])
              for header, _op, _target, rows in PRICE_BANDS]
    return curve, controls, blocks


def pricing_arm_rows(src: Sources):
    """The blocks of :func:`plot_pricing_arms`, from ``make_tables.pricing_arms``.

    V and the MRG of every arm of FAMILY_PRICING (transplant onto MSE's
    samples, CIFAR-10, epoch 199), exactly as Table pricing-arms computes
    them. Fills: the MRG's Holm correction is the table's (over
    FAMILY_PRICING); V is corrected over the same family on its own. The
    positive control is a one-test family for each quantity, at the long
    schedule's optimum (``src.positive_epoch``) and, as a secondary row, at
    its end (``src.converged_epoch``).
    """
    from awwl.plotting.paper_figures import POSITIVE_ROWS, PRICING_BLOCKS, Interval, PricedArm

    mt = tables_module()
    pr = _pricing_ledger(src)
    data = mt.pricing_arms(pr, _phase0_final(src))
    family = list(mt.FAMILY_PRICING)
    listed = [k for _h, arms in PRICING_BLOCKS for k in arms if k not in POSITIVE_ROWS]
    if sorted(listed) != sorted(family):
        raise ValueError(f"PRICING_BLOCKS and FAMILY_PRICING differ: listed only "
                         f"{sorted(set(listed) - set(family))}, family only "
                         f"{sorted(set(family) - set(listed))}")
    v_holm = dict(zip(family, mt.holm([data[a]["value"].p if data[a]["value"] else 1.0
                                       for a in family]), strict=True))
    arms: dict[str, PricedArm] = {}
    for a in family:
        d = data[a]
        if d["value"] is None or d["mrg"] is None:
            logger.warning("no pricing rows yet for %s; omitted from the figure", a)
            continue
        arms[a] = PricedArm(a, d["n"], Interval.of(d["value"], significant=v_holm[a] < 0.05),
                            Interval.of(d["mrg"], significant=(d["mrg"].p_holm or 1.0) < 0.05))
    for key, epoch, secondary in ((POSITIVE_ROWS[0], src.positive_epoch, False),
                                  (POSITIVE_ROWS[1], src.converged_epoch, True)):
        pos = positive_control(src, epoch)
        if pos is None:
            logger.warning("no positive control at epoch %d yet; omitted from the figure", epoch)
            continue
        value, _gap, mrg, n = pos
        arms[key] = PricedArm(key, n, Interval.of(value, significant=value.p < 0.05),
                              Interval.of(mrg, significant=mrg.p < 0.05), style="mse",
                              secondary=secondary)
    return [(h, [arms[k] for k in keys if k in arms]) for h, keys in PRICING_BLOCKS]


def build_pricing_arms(src: Sources, out: Path) -> Path:
    from awwl.plotting.paper_figures import POSITIVE_ROWS, plot_pricing_arms

    names = {POSITIVE_ROWS[0]: f"epoch {src.positive_epoch} vs. {POSITIVE_BASE_EPOCH}",
             POSITIVE_ROWS[1]: f"epoch {src.converged_epoch} vs. {POSITIVE_BASE_EPOCH} (end)"}
    return plot_pricing_arms(pricing_arm_rows(src), out_path=out, names=names,
                             fraction=_width(out, 0.9))


def build_price_curve(src: Sources, out: Path) -> Path:
    from awwl.plotting.paper_figures import plot_price_curve

    curve, controls, bands = price_curve_points(src)
    return plot_price_curve(curve, controls, bands, out_path=out, fraction=_width(out, 0.95))


#: Every figure the manuscript includes, and what writes it.
FIGURES: dict[str, Callable[[Sources, Path], Path]] = {
    "weights.png": build_weights,
    "gradient_scale.png": build_gradient_scale,
    "effect_sizes.png": build_effect_sizes,
    "convergence.png": build_convergence,
    "budget_curve.png": build_budget_curve,
    "budget_sampler.png": build_budget_sampler,
    "deficit_vs_metric.png": build_deficit_vs_metric,
    "spectrum.png": build_spectrum,
    "compare.png": build_compare,
    "teaser.png": build_teaser,
    "price_curve.png": build_price_curve,
    "pricing_arms.png": build_pricing_arms,
    "dreambooth_tradeoff.png": build_dreambooth_tradeoff,
    "restoration_null.png": build_restoration_null,
}


def build(out_dir: str | Path, src: Sources | None = None, *, only=None, echo=print):
    """Write every figure (or those named in ``only``) into ``out_dir``.

    Returns ``(written, skipped)``: paths written, and ``(name, reason)`` for
    figures whose inputs were missing.
    """
    src = src or Sources()
    out_dir = Path(out_dir)
    names = list(FIGURES)
    if only:
        wanted = {n if n.endswith(".png") else f"{n}.png" for n in only}
        unknown = wanted - set(FIGURES)
        if unknown:
            raise ValueError(f"unknown figure(s): {sorted(unknown)}; known: {names}")
        names = [n for n in names if n in wanted]
    written, skipped = [], []
    for name in names:
        try:
            path = FIGURES[name](src, out_dir / name)
            written.append(path)
            echo(f"wrote {path}")
        except SkipFigure as exc:
            skipped.append((name, str(exc)))
            echo(f"skipped {name}: {exc}")
    return written, skipped
