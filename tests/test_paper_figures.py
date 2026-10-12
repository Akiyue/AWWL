"""Smoke tests for the paper figures.

These check that each figure renders, that the numbers reaching it are the
numbers measured, and that the paper's figure set is reproducible from one
entry point -- not that a figure looks right, which only the eye settles. The
parsing tests guard the path from the boost experiment's text output to the
figure, where a silent regex failure would produce a plausible-looking plot of
nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from awwl.plotting.paper_figures import (  # noqa: E402
    BLUE,
    DISPLAY_NAMES,
    EFFECT_BLOCKS,
    SERIES,
    _mean_ci,
    display_name,
    parse_boost_tables,
    plot_budget_sampler,
    plot_convergence,
    plot_correction_value,
    plot_deficit_versus_metric,
    plot_effect_sizes,
    plot_gradient_scale,
    plot_null_panel,
    plot_pricing_protocol,
    plot_tradeoff,
    plot_weight_schedule,
    style_of,
)

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"

# A small comparison family for the forest-plot tests, standing in for the
# paper's fixed list (which the synthetic ledger does not cover).
SMALL_BLOCKS = [
    ("Published objective", [("awwl", False), ("static_wavelet", False)]),
    ("Controls", [("awwl_norm_matched", False), ("awwl_grad_matched_mean", True)]),
]


def boost_table(*, fid: float, boosted_fid: float) -> str:
    """A run's output, produced by the same function that writes the real ones.

    Hand-typing the format here would test the fixture rather than the code:
    the parser's job is to read what ``awwl sensitivity`` actually emits, so the
    emitter is what generates the input.
    """
    from awwl.evaluation.sensitivity import SensitivityPoint, format_sensitivity_table

    return format_sensitivity_table(
        [
            SensitivityPoint(0.0, 2.220, {"fid": fid, "is_mean": 7.9, "kid_tf": 0.0123}),
            SensitivityPoint(-0.44, 1.782, {"fid": boosted_fid, "is_mean": 7.9, "kid_tf": 0.0118}),
        ]
    )


@pytest.fixture
def ledger() -> list[dict]:
    """Five seeds x five epochs, with the measured CIFAR-10 effect sizes."""
    rng = np.random.default_rng(0)
    deltas = {"mse": 0.0, "static_wavelet": 0.06, "awwl": 0.44, "awwl_norm_matched": 1.34}
    seeds = range(5)
    base = {s: 18.46 + rng.normal(0, 0.55) for s in seeds}
    rows = []
    for group, delta in deltas.items():
        for seed in seeds:
            final = base[seed] + delta + rng.normal(0, 0.18)
            for epoch in (39, 79, 119, 159, 199):
                decay = 1.0 + 2.6 * float(np.exp(-(epoch - 39) / 55.0))
                rows.append(
                    {"group": group, "seed": seed, "epoch": epoch, "fid": final * decay}
                )
    return rows


def _final(rows):
    return [r for r in rows if r["epoch"] == 199]


# --------------------------------------------------------------------------
# Style and identity


def test_style_is_one_face_at_print_size():
    """Text and mathematics share STIX, at 9 pt, with no bold chrome."""
    from awwl.plotting._style import apply_style

    apply_style()
    rc = matplotlib.rcParams
    assert rc["mathtext.fontset"] == "stix"
    assert rc["font.serif"][0] == "STIXGeneral"
    assert rc["font.size"] == 9
    assert min(rc["xtick.labelsize"], rc["ytick.labelsize"], rc["legend.fontsize"]) >= 7
    assert rc["axes.labelweight"] == "normal" and rc["axes.titleweight"] == "normal"


def test_series_colours_are_distinct():
    """Two configurations sharing a colour would be unreadable in every figure."""
    colours = [spec[1] for spec in SERIES.values()]
    assert len(set(colours)) == len(colours)


def test_series_markers_are_unique():
    """Identity is never carried by colour alone, so markers must differ too."""
    from awwl.plotting.paper_figures import EXTRA_MARKERS

    markers = [spec[3] for spec in SERIES.values()] + list(EXTRA_MARKERS.values())
    assert len(set(markers)) == len(markers)


def test_only_the_audited_objective_family_is_ever_blue():
    """An arm without a fixed hue falls back to grey, never to AWWL's blue."""
    for group in list(DISPLAY_NAMES) + ["some_future_arm"]:
        if not group.startswith("awwl"):
            assert style_of(group)[0] != BLUE, group


def test_every_arm_has_a_printed_name():
    for _title, arms in EFFECT_BLOCKS:
        for group, _optional in arms:
            assert display_name(group) != group, f"{group} has no display name"
    # The detail-summed shape is named, since it is not AWWL's shape.
    assert "detail bands summed" in display_name("awwl_grad_matched")
    assert display_name("awwl_grad_matched_mean") != display_name("awwl_grad_matched")
    assert len(set(DISPLAY_NAMES.values())) == len(DISPLAY_NAMES), "two arms share a name"


def _tables():
    from awwl.plotting.paper_build import tables_module

    return tables_module()


def test_figure_names_are_the_tables_names():
    """One display name per arm: the figures' copy of the map is the tables' map."""
    from awwl.plotting.paper_figures import for_mathtext

    mt = _tables()
    assert DISPLAY_NAMES == mt.NAMES
    for key, (name, *_style) in SERIES.items():
        assert name == for_mathtext(mt.NAMES[key]), key


def test_figure_blocks_and_families_are_the_tables():
    """Same groups, headers, order and Holm families in the figures and the tables."""
    from awwl.plotting.paper_figures import (
        BLOCKS,
        EFFECT_FAMILY,
        EFFECT_OWN_FAMILIES,
        EFFECT_PAIRS,
        POSITIVE_ROWS,
        PRICING_BLOCKS,
        PRICING_FAMILY,
    )

    mt = _tables()
    assert BLOCKS == mt.BLOCKS
    assert [(t, [k for k, _o in arms]) for t, arms in EFFECT_BLOCKS] == [
        (t, list(keys)) for t, keys in mt.BLOCKS]
    assert tuple(EFFECT_FAMILY) == tuple(mt.FAMILY_MAIN)
    assert tuple(PRICING_FAMILY) == tuple(mt.FAMILY_PRICING)
    assert {mt.EMA_PAIR[0]: mt.EMA_PAIR[1]} == EFFECT_PAIRS
    # The timestep-only arm has its own family and is not in the main one.
    assert EFFECT_OWN_FAMILIES[mt.TIMESTEP_ARM] == tuple(mt.FAMILY_TIMESTEP)
    assert mt.TIMESTEP_ARM not in EFFECT_FAMILY
    # The pricing figure's arm blocks are the shared blocks, in their order.
    arm_blocks = [(t, keys) for t, keys in PRICING_BLOCKS if not set(keys) & set(POSITIVE_ROWS)]
    assert arm_blocks == [(t, [k for k in keys if k in mt.FAMILY_PRICING]) for t, keys in mt.BLOCKS]


def test_every_display_name_renders_in_mathtext(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for i, name in enumerate(DISPLAY_NAMES.values()):
        ax.text(0, i, name)
    fig.savefig(tmp_path / "names.png")
    plt.close(fig)


def test_saved_figures_are_print_width_rgb_at_300_dpi(tmp_path):
    from PIL import Image

    from awwl.plotting._style import DPI, TEXT_WIDTH_IN

    out = plot_gradient_scale(alpha=0.2, power=1.0, out_path=tmp_path / "g.png")
    with Image.open(out) as im:
        assert im.mode == "RGB"
        assert abs(im.size[0] - TEXT_WIDTH_IN * DPI) <= 1
        assert round(im.info["dpi"][0]) == DPI


# --------------------------------------------------------------------------
# Reproducibility: one entry point for every figure the manuscript includes


def _included_figures() -> set[str]:
    names = set()
    for tex in (PAPER / "sections").glob("*.tex"):
        body = tex.read_text(encoding="utf-8")
        body = re.sub(r"(?<!\\)%.*", "", body)
        names.update(re.findall(r"\\guardedfigure\{[^}]*\}\{([^}]+)\}", body))
        names.update(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body))
    return {n if Path(n).suffix else f"{n}.png" for n in names}


#: The paper's figure set: what ``awwl figures`` builds by default.
PAPER_FIGURES = {
    "weights.png", "gradient_scale.png", "effect_sizes.png", "convergence.png",
    "budget_curve.png", "budget_sampler.png", "deficit_vs_metric.png", "spectrum.png",
    "compare.png", "teaser.png", "price_curve.png", "pricing_arms.png",
    "dreambooth_tradeoff.png", "restoration_null.png",
}
#: Figures the paper no longer carries. Their builders are gone from the
#: default run and their files from paper/figures, so a stale copy cannot be
#: picked up by \guardedfigure.
RETIRED_FIGURES = {"correction_value.png", "pricing_protocol.png"}


def test_every_included_figure_has_a_producer():
    """``awwl figures`` must be able to rebuild every figure the paper includes."""
    from awwl.plotting.paper_build import FIGURES

    included = _included_figures()
    if not included:
        pytest.skip("no manuscript sections found")
    missing = sorted(included - set(FIGURES))
    assert not missing, f"figures with no producer in paper_build.FIGURES: {missing}"


def test_the_producers_are_the_paper_figure_set():
    """The default run builds the new pricing figures and none of the retired ones."""
    from awwl.plotting.paper_build import FIGURES

    assert set(FIGURES) == PAPER_FIGURES
    assert not RETIRED_FIGURES & set(FIGURES)


def test_retired_figures_are_neither_included_nor_on_disk():
    included = _included_figures()
    assert not RETIRED_FIGURES & included, sorted(RETIRED_FIGURES & included)
    stale = sorted(n for n in RETIRED_FIGURES if (PAPER / "figures" / n).exists())
    assert not stale, f"retired figures still in paper/figures: {stale}"


def _include_fractions() -> dict[str, float]:
    fractions = {}
    for tex in (PAPER / "sections").glob("*.tex"):
        body = re.sub(r"(?<!\\)%.*", "", tex.read_text(encoding="utf-8"))
        for frac, name in re.findall(r"\\guardedfigure\{([^}]*)\}\{([^}]+)\}", body):
            fractions[name] = float(frac)
        for frac, name in re.findall(
                r"\\includegraphics\[width=([0-9.]*)\\linewidth\]\{([^}]+)\}", body):
            fractions[name] = float(frac or 1.0)
    return fractions


def test_committed_figures_are_authored_at_their_include_width():
    """A figure authored at one width and included at another prints its type
    at the wrong size; this is how a 2700-px spectrum once slipped in."""
    from PIL import Image

    from awwl.plotting._style import DPI, TEXT_WIDTH_IN

    checked = 0
    for name, frac in _include_fractions().items():
        path = PAPER / "figures" / name
        if not path.exists():
            continue
        with Image.open(path) as im:
            expected = frac * TEXT_WIDTH_IN * DPI
            assert abs(im.size[0] - expected) <= 2, (name, im.size[0], round(expected))
        checked += 1
    if not checked:
        pytest.skip("no built figures to check")


def test_figures_are_built_at_the_width_the_sections_include_them():
    """The builders read the include width, so an author's change of width
    reaches the artwork on the next build instead of rescaling its type."""
    from awwl.plotting.paper_build import include_fraction

    fractions = _include_fractions()
    if not fractions:
        pytest.skip("no manuscript sections found")
    for name, frac in fractions.items():
        assert include_fraction(name, -1.0) == pytest.approx(frac), name
    assert include_fraction("no_such_figure.png", 0.5) == 0.5


def test_build_rejects_unknown_figure_names(tmp_path):
    from awwl.plotting.paper_build import build

    with pytest.raises(ValueError):
        build(tmp_path, only=["no_such_figure"], echo=lambda *_: None)


def test_build_skips_figures_whose_inputs_are_missing(tmp_path):
    from awwl.plotting.paper_build import Sources, build

    src = Sources(ledger=tmp_path / "absent.jsonl")
    written, skipped = build(tmp_path, src, only=["effect_sizes"], echo=lambda *_: None)
    assert not written and skipped and skipped[0][0] == "effect_sizes.png"


# --------------------------------------------------------------------------
# Forest plot


def test_effect_sizes_renders(ledger, tmp_path):
    out = plot_effect_sizes(
        _final(ledger), metric="fid", baseline="mse", out_path=tmp_path / "effects.png",
        blocks=SMALL_BLOCKS,
    )
    assert out.exists() and out.stat().st_size > 0


def test_effect_sizes_adds_the_optional_control_when_it_has_rows(ledger, tmp_path):
    rows = _final(ledger) + [
        {"group": "awwl_grad_matched_mean", "seed": r["seed"], "epoch": 199,
         "fid": r["fid"] + 0.9}
        for r in _final(ledger) if r["group"] == "mse"
    ]
    out = plot_effect_sizes(rows, metric="fid", baseline="mse",
                            out_path=tmp_path / "effects.png", blocks=SMALL_BLOCKS)
    assert out.exists()


def test_effect_sizes_rejects_a_missing_baseline(ledger, tmp_path):
    with pytest.raises(ValueError):
        plot_effect_sizes(
            _final(ledger), metric="fid", baseline="nonexistent",
            out_path=tmp_path / "x.png", blocks=SMALL_BLOCKS,
        )


def test_effect_sizes_rejects_a_missing_required_arm(ledger, tmp_path):
    """The row list is fixed: an arm silently dropping out must fail loudly."""
    blocks = SMALL_BLOCKS + [("Other", [("min_snr", False)])]
    with pytest.raises(ValueError, match="min_snr"):
        plot_effect_sizes(_final(ledger), metric="fid", baseline="mse",
                          out_path=tmp_path / "x.png", blocks=blocks)


def _timestep_rows(ledger, seeds):
    """The synthetic ledger plus a timestep-only arm at ``seeds`` seeds."""
    base = {r["seed"]: r["fid"] for r in _final(ledger) if r["group"] == "mse"}
    return _final(ledger) + [
        {"group": "awwl_timestep", "seed": s, "epoch": 199, "fid": base[s] + 0.3 + 0.02 * s}
        for s in sorted(base)[:seeds]]


def test_timestep_arm_is_drawn_only_with_every_seed(ledger):
    from awwl.plotting.paper_figures import effect_results

    blocks = SMALL_BLOCKS + [("Timestep", [("awwl_timestep", True)])]
    results, _pairs, own = effect_results(_timestep_rows(ledger, 4), metric="fid", baseline="mse",
                                          blocks=blocks)
    assert "awwl_timestep" not in results and "awwl_timestep" in own
    results, _pairs, _own = effect_results(_timestep_rows(ledger, 5), metric="fid",
                                           baseline="mse", blocks=blocks)
    assert "awwl_timestep" in results


def test_timestep_arm_is_corrected_in_its_own_family(ledger):
    """Its fill uses Holm over its two contrasts, not the main family."""
    from awwl.analysis.stats import _holm, compare_to_baseline
    from awwl.plotting.paper_figures import effect_results

    rows = _timestep_rows(ledger, 5)
    blocks = SMALL_BLOCKS + [("Timestep", [("awwl_timestep", True)])]
    results, _pairs, _own = effect_results(rows, metric="fid", baseline="mse", blocks=blocks)

    def p_of(arm, ref):
        sub = [r for r in rows if r["group"] in (arm, ref)]
        return next(c.p_value for c in compare_to_baseline(sub, metric="fid", baseline=ref)
                    if c.group == arm)

    expected = _holm([p_of("awwl_timestep", "mse"), p_of("awwl", "awwl_timestep")])[0]
    assert results["awwl_timestep"].p_holm == pytest.approx(expected)


def test_effect_sizes_renders_the_timestep_arm(ledger, tmp_path):
    blocks = SMALL_BLOCKS + [("Timestep", [("awwl_timestep", True)])]
    out = plot_effect_sizes(_timestep_rows(ledger, 5), metric="fid", baseline="mse",
                            out_path=tmp_path / "effects_t.png", blocks=blocks)
    assert out.exists()


def test_effect_blocks_cover_the_cifar_comparison_family():
    """Every arm of the phase-0 ledger's epoch-199 family is in the fixed list."""
    path = REPO / "runs" / "phase0" / "results.jsonl"
    if not path.exists():
        pytest.skip("phase-0 ledger not present")
    from awwl.analysis.results import load_results

    groups = {r["group"] for r in load_results(path, kind="eval") if r.get("epoch") == 199}
    listed = {g for _t, arms in EFFECT_BLOCKS for g, _o in arms}
    assert groups - {"mse"} <= listed
    optional = {g for _t, arms in EFFECT_BLOCKS for g, o in arms if o}
    assert "awwl_grad_matched_mean" in optional


# --------------------------------------------------------------------------
# Training dynamics


def test_convergence_renders_both_panels(ledger, tmp_path):
    for interval in ("bars", "band"):
        out = plot_convergence(
            ledger, metric="fid", baseline="mse", out_path=tmp_path / f"conv_{interval}.png",
            interval=interval,
        )
        assert out.exists() and out.stat().st_size > 0


def test_convergence_without_a_baseline_is_single_panel(ledger, tmp_path):
    out = plot_convergence(ledger, metric="fid", out_path=tmp_path / "conv1.png")
    assert out.exists() and out.stat().st_size > 0


def test_convergence_needs_more_than_one_epoch(tmp_path):
    rows = [{"group": "mse", "seed": s, "epoch": 199, "fid": 18.0 + s} for s in range(3)]
    with pytest.raises(ValueError):
        plot_convergence(rows, metric="fid", out_path=tmp_path / "conv2.png")


def test_paired_interval_matches_the_t_interval():
    # The epoch-159 AWWL - MSE differences of the phase-0 ledger.
    diffs = [0.74, -5.24, -1.02, -0.72, -0.09]
    mean, half = _mean_ci(diffs)
    assert mean == pytest.approx(-1.266, abs=1e-3)
    assert (mean - half, mean + half) == pytest.approx((-4.15, 1.62), abs=0.01)


def test_budget_sampler_renders(tmp_path):
    points = {
        "DDIM-100": [("200 epochs", 0.44, 0.59, False, 5), ("800 epochs", 0.15, 0.59, False, 3)],
        "DDPM-1000": [("200 epochs", 0.98, 0.18, True, 5), ("800 epochs", 0.34, 0.56, False, 3)],
    }
    out = plot_budget_sampler(points, out_path=tmp_path / "budget.png")
    assert out.exists()


def test_deficit_versus_metric_renders(tmp_path):
    out = plot_deficit_versus_metric(
        epochs=[199, 399, 799],
        deficit={"mse": [[1.8, 1.9], [2.3, 2.3], [2.6, 2.6]],
                 "awwl": [[1.6, 1.7], [1.9, 1.9], [2.3, 2.4]]},
        fid={"mse": [[14.8, 14.7], [12.0, 12.1], [13.7, 13.6]],
             "awwl": [[15.2, 15.1], [12.3, 12.2], [13.8, 13.9]]},
        out_path=tmp_path / "deficit.png",
    )
    assert out.exists()


# --------------------------------------------------------------------------
# The weighting


def test_weight_schedule_renders(tmp_path):
    out = plot_weight_schedule(out_path=tmp_path / "weights.png", alpha=0.2, power=1.0)
    assert out.exists() and out.stat().st_size > 0


def test_published_weighting_averages_over_training_timesteps():
    """The numbers the gradient-scale figure normalises by (verified values)."""
    from awwl.losses import AdaptiveWaveletLoss
    from awwl.plotting.paper_figures import _training_sigmas

    sig = _training_sigmas("linear")
    w = AdaptiveWaveletLoss(alpha=0.2, power=1.0).weights_at(sig)
    assert float(w["total"].mean()) == pytest.approx(0.3165, abs=5e-4)
    sq = w["w_ll"] ** 2 + w["w_det"] ** 2 / 3
    assert float(sq.mean()) == pytest.approx(0.0530, abs=5e-4)
    assert float((sig > 0.9).float().mean()) == pytest.approx(0.60, abs=0.01)


# --------------------------------------------------------------------------
# Pricing figures


def test_correction_value_renders(tmp_path):
    arms = {"mse": (2.220, 18.460, 1.782, 18.114), "awwl": (1.800, 18.901, 1.398, 18.568)}
    out = plot_correction_value(arms, out_path=tmp_path / "correction.png",
                                mrg_ci=(0.20, 1.37))
    assert out.exists() and out.stat().st_size > 0


def test_correction_value_takes_the_mrg_interval_from_seeds(tmp_path):
    per_seed = {
        "mse": {1: (2.2, 18.2, 1.8, 17.9), 2: (2.2, 18.7, 1.8, 18.3), 3: (2.3, 18.4, 1.8, 18.1)},
        "awwl": {1: (1.8, 19.3, np.nan, np.nan), 2: (1.8, 19.0, np.nan, np.nan),
                 3: (1.9, 18.5, np.nan, np.nan)},
    }
    means = {g: tuple(float(np.mean([v[i] for v in by.values()])) for i in range(4))
             for g, by in per_seed.items()}
    out = plot_correction_value(means, per_seed=per_seed, out_path=tmp_path / "c.png")
    assert out.exists()


def test_pricing_protocol_takes_every_number_as_an_argument(tmp_path):
    out = plot_pricing_protocol(
        baseline_fid=18.460, corrected_fid=18.114, trained_fid=18.901,
        mrg=0.787, mrg_ci=(0.20, 1.37), delta_db=0.42, delta_ci=(0.24, 0.60),
        out_path=tmp_path / "protocol.png",
    )
    assert out.exists()
    # Without the optional numbers it still renders (MRG from the two means).
    out = plot_pricing_protocol(baseline_fid=None, corrected_fid=18.1, trained_fid=18.9,
                                out_path=tmp_path / "protocol2.png")
    assert out.exists()


def _iv(mean, half, sig=True):
    from awwl.plotting.paper_figures import Interval

    return Interval(mean, mean - half, mean + half, sig)


def test_teaser_renders_with_and_without_the_positive_control(tmp_path):
    from awwl.plotting.paper_figures import Verdict, plot_teaser

    awwl = Verdict(title="AWWL ($n$ = 5):\n$B$ = MSE, $A$ = AWWL", n=5, fid_b=18.46,
                   fid_tb=18.12, fid_a=18.90, value=_iv(0.34, 0.10), gap=_iv(0.44, 0.59),
                   mrg=_iv(0.78, 0.50))
    pos = Verdict(title="Positive control ($n$ = 3):\n$A$ = $B$ trained 4$\\times$ longer",
                  n=3, fid_b=14.78, fid_tb=14.92, fid_a=13.70, value=_iv(-0.14, 0.16),
                  gap=_iv(-1.08, 0.22), mrg=_iv(-1.22, 0.21), arm="mse")
    for verdicts, name in (([awwl, pos], "t2.png"), ([awwl], "t1.png")):
        out = plot_teaser(verdicts, out_path=tmp_path / name)
        assert out.exists() and out.stat().st_size > 0


def test_interval_of_a_paired_result_can_be_negated():
    from types import SimpleNamespace

    from awwl.plotting.paper_figures import Interval

    res = SimpleNamespace(mean=0.34, lo=0.24, hi=0.44)
    iv = Interval.of(res, sign=-1.0)
    assert (iv.mean, iv.lo, iv.hi) == pytest.approx((-0.34, -0.44, -0.24))
    assert Interval.of(None) is None


def test_price_curve_renders(tmp_path):
    from awwl.plotting.paper_figures import PricePoint, plot_price_curve

    curve = [PricePoint(f"+{d:g} dB", _iv(0.15 * d, 0.005), _iv(0.92 * d, 0.001))
             for d in (0.1, 0.2, 0.41, 0.8, 1.2, 1.6, 2.4)]
    controls = [PricePoint("+0.41 dB boost, dithered to 8 bits", _iv(0.34, 0.02),
                           _iv(0.43, 0.01), "dither"),
                PricePoint("$\\pm$0.5 LSB", _iv(0.32, 0.02), _iv(0.09, 0.01), "noise"),
                PricePoint("$\\pm$1 LSB", _iv(0.93, 0.06), _iv(0.31, 0.01), "noise")]
    bands = [("Transplant of AWWL's change",
              [PricePoint("low third only", _iv(-0.03, 0.08, False), _iv(0.0, 0.004),
                          "transplant"),
               PricePoint("all bands", _iv(0.34, 0.10), _iv(0.37, 0.15), "transplant")]),
             ("Match to the reference spectrum",
              [PricePoint("mid third only", _iv(1.26, 0.09), _iv(0.01, 0.005), "match")])]
    out = plot_price_curve(curve, controls, bands, out_path=tmp_path / "price.png")
    assert out.exists() and out.stat().st_size > 0


def test_pricing_arms_renders_and_marks_off_scale_rows(tmp_path):
    from awwl.plotting.paper_figures import PricedArm, plot_pricing_arms

    blocks = [
        ("AWWL", [PricedArm("awwl", 5, _iv(0.34, 0.10), _iv(0.78, 0.50, False))]),
        ("Controls", [PricedArm("positive_control", 3, _iv(-0.14, 0.16, False),
                                _iv(-1.22, 0.21), style="mse"),
                      PricedArm("static_wavelet", 5, _iv(-0.02, 0.02, False),
                                _iv(0.16, 0.32, False))]),
        ("Other losses", [PricedArm("l1", 5, _iv(-2.68, 0.10), _iv(4.46, 0.37)),
                          PricedArm("awwl_full", 3, _iv(-0.28, 0.10), _iv(12.25, 1.0))]),
    ]
    out = plot_pricing_arms(blocks, out_path=tmp_path / "arms.png",
                            names={"positive_control": "MSE trained 4$\\times$ longer"})
    assert out.exists() and out.stat().st_size > 0


def test_pricing_arms_draws_a_secondary_row(tmp_path):
    from awwl.plotting.paper_figures import POSITIVE_ROWS, PricedArm, plot_pricing_arms

    blocks = [("AWWL", [PricedArm("awwl", 5, _iv(0.34, 0.10), _iv(0.78, 0.50, False))]),
              ("Positive control", [
                  PricedArm(POSITIVE_ROWS[0], 3, _iv(-0.10, 0.14, False), _iv(-2.87, 0.62),
                            style="mse"),
                  PricedArm(POSITIVE_ROWS[1], 3, _iv(-0.14, 0.16, False), _iv(-1.22, 0.21),
                            style="mse", secondary=True)])]
    out = plot_pricing_arms(blocks, out_path=tmp_path / "arms2.png",
                            names={POSITIVE_ROWS[0]: "epoch 399 vs. 199",
                                   POSITIVE_ROWS[1]: "epoch 799 vs. 199 (end)"})
    assert out.exists() and out.stat().st_size > 0


def test_column_labels_leaders_do_not_cross(tmp_path):
    """The crowded DreamBooth quadrant: leaders leave each label at its left
    middle, and no two of them cross."""
    import itertools

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation
    from matplotlib.transforms import Bbox

    from awwl.plotting.paper_figures import _column_labels

    pts = {"AWWL": (-0.011, 0.029), "$L_1$": (-0.0145, 0.0205), "AWWL, x": (-0.0105, 0.0205),
           "Charbonnier": (-0.0125, 0.0135), "Static": (-0.0063, 0.011), "Min-SNR": (-0.002, 0.006)}
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.set_xlim(-0.03, 0.06)
    ax.set_ylim(-0.08, 0.065)
    fig.canvas.draw()
    whiskers = [(x, y, 0.01, 0.025) for x, y in pts.values()]
    to_px = ax.transData.transform
    hard = [Bbox.from_extents(*(to_px(p) - 9), *(to_px(p) + 9)) for p in pts.values()]
    _column_labels(ax, pts, whiskers, hard)
    notes = [t for t in ax.texts if isinstance(t, Annotation)]
    assert sorted(t.get_text() for t in notes) == sorted(pts)
    for t in notes:
        assert t.arrow_patch is not None and t.arrowprops.get("relpos") == (0.0, 0.5)
    segs = [(tuple(to_px(t.xy)), tuple(to_px(t.xyann))) for t in notes]

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    for (a, b), (c, d) in itertools.combinations(segs, 2):
        assert not (orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0)
    plt.close(fig)


def test_pricing_blocks_list_the_pricing_family_exactly():
    """Every arm the pricing table corrects over is a row of the forest plot."""
    from awwl.plotting.paper_build import tables_module
    from awwl.plotting.paper_figures import POSITIVE_ROWS, PRICING_BLOCKS, PRICING_HIGHLIGHT

    mt = tables_module()
    listed = [k for _h, arms in PRICING_BLOCKS for k in arms]
    assert len(listed) == len(set(listed))
    assert sorted(set(listed) - set(POSITIVE_ROWS)) == sorted(mt.FAMILY_PRICING)
    assert set(POSITIVE_ROWS) <= set(listed)
    assert set(PRICING_HIGHLIGHT) <= set(listed)


def _macros() -> dict[str, str]:
    path = PAPER / "generated" / "numbers.tex"
    if not path.exists():
        return {}
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                           path.read_text(encoding="utf-8")))


def _agrees(value: float, macro: str) -> bool:
    digits = len(macro.split(".")[1]) if "." in macro else 0
    return f"{value:.{digits}f}" == macro or (float(macro) == 0 and abs(value) < 10**-digits)


def test_pricing_figures_draw_the_numbers_the_text_quotes():
    """The figures and numbers.tex go through the same helpers, so they agree.

    A failure here means one of them is stale: regenerate numbers.tex
    (scripts/make_numbers.py) and the figures (awwl figures) together.
    """
    import os

    from awwl.plotting import paper_build as pb

    macros = _macros()
    if not macros or not (REPO / "runs" / "pricing" / "results.jsonl").exists():
        pytest.skip("numbers.tex or the pricing ledger not present")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        src = pb.Sources()
        verdicts = pb.teaser_verdicts(src)
        curve, controls, bands = pb.price_curve_points(src)
        blocks = pb.pricing_arm_rows(src)
    finally:
        os.chdir(cwd)
    arms = {a.key: a for _h, rows in blocks for a in rows}
    by_label = {(h, p.label): p for h, pts in bands for p in pts}
    header = pb.PRICE_BANDS[0][0]
    checks = {
        "priceTransplantValue": verdicts[0].value.mean,
        "mrgMain": verdicts[0].mrg.mean,
        "mrgMainLo": verdicts[0].mrg.lo,
        "posOptMRG": verdicts[1].mrg.mean,
        "posOptMRGLo": verdicts[1].mrg.lo,
        "posOptValue": verdicts[1].value.mean,
        "posOptTrainGain": -verdicts[1].gap.mean,
        "posCtrlMRG": arms["positive_control_end"].mrg.mean,
        "posCtrlValue": arms["positive_control_end"].value.mean,
        "priceTransplantMidValue": by_label[(header, "mid third")].value.mean,
        "priceTransplantHighValue": by_label[(header, "high third")].value.mean,
        "priceBoostDitherValue": next(p for p in controls if p.kind == "dither").value.mean,
        "priceNoiseOneValue": controls[-1].value.mean,
        "mrgArmFFL": arms["focal_frequency"].mrg.mean,
        "valueArmLOne": arms["l1"].value.mean,
        "mrgArmStatic": arms["static_wavelet"].mrg.mean,
    }
    boost = next(p for p in curve if p.label == "+0.41 dB")
    checks["priceBoostValue"] = boost.value.mean
    bad = {k: (round(v, 4), macros.get(k)) for k, v in checks.items()
           if k in macros and "?" not in macros[k] and not _agrees(v, macros[k])}
    assert not bad, bad


def test_parse_boost_tables_recovers_the_measured_numbers(tmp_path):
    (tmp_path / "mse_s0.txt").write_text(
        boost_table(fid=18.460, boosted_fid=18.114), encoding="utf-8"
    )

    parsed = parse_boost_tables(tmp_path)

    assert set(parsed) == {"mse"}
    deficit, fid, boosted_deficit, boosted_fid = parsed["mse"]
    # FID round-trips exactly (%.3f); the deficit carries three significant
    # figures because that is all %.3g writes. Enough for a dB axis, and the
    # reason no paper number is quoted from the deficit past two decimals.
    assert deficit == pytest.approx(2.22, abs=5e-3)
    assert boosted_deficit == pytest.approx(1.78, abs=5e-3)
    assert fid == pytest.approx(18.460)
    assert boosted_fid == pytest.approx(18.114)


def test_parse_boost_tables_averages_across_seeds(tmp_path):
    for seed, (orig, boosted) in enumerate([(18.40, 18.10), (18.52, 18.13)]):
        (tmp_path / f"mse_s{seed}.txt").write_text(
            boost_table(fid=orig, boosted_fid=boosted), encoding="utf-8"
        )

    _, fid, _, boosted_fid = parse_boost_tables(tmp_path)["mse"]

    assert fid == pytest.approx(18.46, abs=1e-6)
    assert boosted_fid == pytest.approx(18.115, abs=1e-6)

    per_seed = parse_boost_tables(tmp_path, per_seed=True)["mse"]
    assert sorted(per_seed) == [0, 1]
    assert per_seed[1][1] == pytest.approx(18.52)


def test_parse_boost_tables_ignores_unparseable_files(tmp_path):
    (tmp_path / "empty_s0.txt").write_text("skip empty_s0: no samples", encoding="utf-8")

    assert parse_boost_tables(tmp_path) == {}


# --------------------------------------------------------------------------
# Second tasks


def _second_task_rows():
    rng = np.random.default_rng(1)
    rows = []
    for group, (dx, dy) in {"mse": (0, 0), "awwl": (-0.011, 0.029), "min_snr": (-0.002, 0.006),
                            "snr_weighted": (0.041, -0.044), "l1": (-0.014, 0.020)}.items():
        for seed in range(1, 6):
            rows.append({"group": group, "seed": seed,
                         "clip_score": 0.31 + dx + rng.normal(0, 0.005),
                         "similarity": 0.84 + dy + rng.normal(0, 0.01),
                         "psnr": 30 + dx * 10 + rng.normal(0, 0.02),
                         "ssim": 0.9 + rng.normal(0, 1e-4)})
    return rows


def test_tradeoff_renders(tmp_path):
    out = plot_tradeoff(_second_task_rows(), x_metric="clip_score", y_metric="similarity",
                        baseline="mse", out_path=tmp_path / "tradeoff.png")
    assert out.exists()
    direct = plot_tradeoff(_second_task_rows(), x_metric="clip_score", y_metric="similarity",
                           baseline="mse", out_path=tmp_path / "direct.png", labels="direct")
    assert direct.exists()
    with pytest.raises(ValueError):
        plot_tradeoff(_second_task_rows(), x_metric="clip_score", y_metric="similarity",
                      baseline="mse", out_path=tmp_path / "x.png", labels="leaders")


def test_tradeoff_legend_names_every_arm_with_no_leader_lines(tmp_path, monkeypatch):
    """PE-11: identity by marker and a legend, so no leader line crosses a whisker."""
    import matplotlib.pyplot as plt

    from awwl.plotting import paper_figures as pf

    seen = {}

    def keep(fig, out_path, **_kw):
        seen["fig"] = fig
        return Path(out_path)

    monkeypatch.setattr(pf, "save_figure", keep)
    rows = _second_task_rows()
    plot_tradeoff(rows, x_metric="clip_score", y_metric="similarity", baseline="mse",
                  out_path=tmp_path / "t.png")
    ax = seen["fig"].axes[0]
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    arms = {r["group"] for r in rows} - {"mse"}
    assert {display_name(a) for a in arms} <= set(texts)
    assert any("p_{\\mathrm{Holm}}" in t for t in texts)
    assert not [c for c in ax.texts if getattr(c, "arrow_patch", None) is not None]
    plt.close(seen["fig"])


def test_every_scatter_arm_has_its_own_marker():
    """Arms that share the grey fallback colour are told apart by marker alone."""
    from awwl.plotting.paper_figures import EXTRA_MARKERS

    dreambooth = ("l1", "huber", "charbonnier", "snr_weighted", "min_snr", "static_wavelet",
                  "awwl", "awwl_normalized", "awwl_learned")
    styles = {style_of(a) for a in dreambooth}
    assert len(styles) == len(dreambooth)
    assert set(EXTRA_MARKERS) >= {"l1", "huber", "charbonnier", "awwl_learned"}


def test_training_timestep_axes_run_from_one_to_the_last_timestep():
    """F18: Section 3.1 samples t from {1, ..., T}; the figures label the same range."""
    from awwl.plotting.paper_figures import TIMESTEP_TICKS, _training_timesteps

    t = _training_timesteps(1000)
    assert t[0] == 1 and t[-1] == 1000 and len(t) == 1000
    assert TIMESTEP_TICKS[0] == 1 and TIMESTEP_TICKS[-1] == 1000


def test_null_panel_renders_in_the_given_order(tmp_path):
    out = plot_null_panel(
        _second_task_rows(), baseline="mse", out_path=tmp_path / "null.png",
        metrics=[("psnr", "PSNR (dB)", False), ("ssim", "SSIM", False)],
        order=["l1", "awwl", "min_snr"],
    )
    assert out.exists()


def test_paired_samples_grid_renders(tmp_path):
    from PIL import Image

    from awwl.plotting.paired_grid import plot_paired_samples

    rng = np.random.default_rng(0)
    folders = {}
    for group in ("mse", "awwl"):
        folder = tmp_path / group
        folder.mkdir()
        for i in range(4):
            Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(
                folder / f"{i:05d}.png")
        folders[group] = folder
    out = plot_paired_samples(folders, out_path=tmp_path / "compare.png", count=4)
    with Image.open(out) as im:
        assert im.mode == "RGB"


def test_spectrum_labels_terminate_and_separate(tmp_path):
    """The end-label nudge must finish, and must actually separate the labels.

    Both halves of this were broken. The nudge was a ``while`` loop whose body
    did not depend on the value it tested, so when ``prev + gap`` rounded to
    something less than ``gap`` above ``prev`` -- which it does for the real
    CIFAR-10 profiles -- it spun forever at full CPU. And the position it
    computed was then discarded, the label being drawn at the raw endpoint
    regardless. The profiles here reproduce the rounding case: three series
    whose endpoints sit close enough to trigger the nudge.
    """
    from awwl.plotting.spectrum import LABEL_LEADING, LABEL_PT, plot_spectrum_deviation

    freqs = np.linspace(0.0, 1.0, 16)
    real = np.zeros_like(freqs)
    # Endpoints -2.5645.., -2.4928.., -2.0383.. are the measured ones; their
    # separation is below the label height, so all three must be moved apart.
    curves = {
        "mse": real - 2.492822941353797 * freqs,
        "static_wavelet": real - 2.5645072557403665 * freqs,
        "awwl": real - 2.038341920863079 * freqs,
    }

    out = plot_spectrum_deviation(
        real=real,
        baselines={k: v for k, v in curves.items() if k != "awwl"},
        highlight=("awwl", curves["awwl"]),
        output_path=tmp_path / "spectrum.png",
    )

    assert out.exists() and out.stat().st_size > 0
    assert LABEL_LEADING > 1.0, "a leading of 1 leaves ascenders touching descenders"
    assert LABEL_PT > 0


def test_spectrum_accepts_per_seed_profiles(tmp_path):
    from awwl.plotting.spectrum import _axial_nyquist, plot_spectrum_deviation

    assert _axial_nyquist(23) == 16  # 32 x 32 images
    assert _axial_nyquist(46) == 32  # 64 x 64 images
    rng = np.random.default_rng(0)
    real = np.linspace(100, 30, 23)
    per_seed = {g: real - np.linspace(0, d, 23) + rng.normal(0, 0.05, (5, 23))
                for g, d in {"mse": 2.8, "static_wavelet": 2.9, "awwl": 2.4}.items()}
    out = plot_spectrum_deviation(
        real=real, baselines={k: v for k, v in per_seed.items() if k != "awwl"},
        highlight=("awwl", per_seed["awwl"]), output_path=tmp_path / "s.png",
    )
    assert out.exists()
