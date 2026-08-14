"""Smoke tests for the paper figures.

These check that each figure renders and that the numbers reaching it are the
numbers measured -- not that it looks right, which only the eye settles. The
parsing test is the one that matters: it guards the path from the boost
experiment's text output to the figure, where a silent regex failure would
produce a plausible-looking plot of nothing.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from awwl.plotting.paper_figures import (  # noqa: E402
    SERIES,
    parse_boost_tables,
    plot_convergence,
    plot_correction_value,
    plot_effect_sizes,
    plot_weight_schedule,
)


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


def test_effect_sizes_renders(ledger, tmp_path):
    out = plot_effect_sizes(
        ledger, metric="fid", baseline="mse", out_path=tmp_path / "effects.png"
    )
    assert out.exists() and out.stat().st_size > 0


def test_effect_sizes_rejects_a_missing_baseline(ledger, tmp_path):
    with pytest.raises(ValueError):
        plot_effect_sizes(
            ledger, metric="fid", baseline="nonexistent", out_path=tmp_path / "x.png"
        )


def test_convergence_renders_both_panels(ledger, tmp_path):
    out = plot_convergence(
        ledger, metric="fid", baseline="mse", out_path=tmp_path / "conv.png"
    )
    assert out.exists() and out.stat().st_size > 0


def test_convergence_without_a_baseline_is_single_panel(ledger, tmp_path):
    out = plot_convergence(ledger, metric="fid", out_path=tmp_path / "conv1.png")
    assert out.exists() and out.stat().st_size > 0


def test_convergence_needs_more_than_one_epoch(tmp_path):
    rows = [{"group": "mse", "seed": s, "epoch": 199, "fid": 18.0 + s} for s in range(3)]
    with pytest.raises(ValueError):
        plot_convergence(rows, metric="fid", out_path=tmp_path / "conv2.png")


def test_weight_schedule_renders(tmp_path):
    out = plot_weight_schedule(out_path=tmp_path / "weights.png", alpha=0.2, power=1.0)
    assert out.exists() and out.stat().st_size > 0


def test_correction_value_renders(tmp_path):
    arms = {"mse": (2.220, 18.460, 1.782, 18.114), "awwl": (1.800, 18.901, 1.398, 18.568)}
    out = plot_correction_value(arms, out_path=tmp_path / "correction.png")
    assert out.exists() and out.stat().st_size > 0


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


def test_parse_boost_tables_ignores_unparseable_files(tmp_path):
    (tmp_path / "empty_s0.txt").write_text("skip empty_s0: no samples", encoding="utf-8")

    assert parse_boost_tables(tmp_path) == {}


def test_series_colours_are_distinct():
    """Two configurations sharing a colour would be unreadable in every figure."""
    colours = [spec[1] for spec in SERIES.values()]
    assert len(set(colours)) == len(colours)
