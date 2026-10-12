"""Checks on the table and number generators (scripts/make_*.py).

The manuscript takes every table and every quoted number from these scripts,
so a statistical slip in them reaches the paper unseen. These tests pin the
conventions the captions state (sample SD, paired-by-seed intervals, Holm over
a fixed family), check that the builders run on the real ledgers and emit
LaTeX that cannot break the build, and check that a missing pricing row yields
a visible placeholder rather than a number.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

mt = pytest.importorskip("make_tables")
scipy_stats = pytest.importorskip("scipy.stats")

PHASE0 = ROOT / "runs" / "phase0" / "results.jsonl"
needs_ledger = pytest.mark.skipif(not PHASE0.exists(), reason="no CIFAR-10 ledger in this checkout")


# ----------------------------------------------------------------- helpers


def _holm_reference(p: list[float]) -> list[float]:
    """Holm's adjusted p-values straight from the definition.

    With p sorted ascending, p_adj(i) = max_{j <= i} min(1, (m - j + 1) p(j)).
    """
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    out = [0.0] * m
    for rank, idx in enumerate(order):
        out[idx] = max(min(1.0, (m - j) * p[order[j]]) for j in range(rank + 1))
    return out


def _strip_escaped(text: str) -> str:
    return text.replace(r"\{", "").replace(r"\}", "").replace(r"\$", "").replace(r"\%", "")


def _latex_safe(name: str, text: str) -> None:
    """The failures a generated file can inflict on the build, without a compiler."""
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())
    body = _strip_escaped(body)
    assert body.count("{") == body.count("}"), f"{name}: unbalanced braces"
    for lineno, line in enumerate(body.splitlines(), 1):
        assert line.count("$") % 2 == 0, f"{name}:{lineno}: odd number of $ in {line!r}"
    assert not re.search(r"\b(nan|NaN|inf|None)\b", body), f"{name}: NaN/inf/None leaked into the table"
    assert body.count(r"\begin{table}") == body.count(r"\end{table}")
    assert body.count(r"\begin{tabular}") == body.count(r"\end{tabular}")


# --------------------------------------------------------------- statistics


def test_holm_matches_the_definition():
    rng = random.Random(0)
    for m in (1, 2, 3, 5, 8, 21):
        for _ in range(50):
            p = [rng.random() ** rng.choice((1, 3, 6)) for _ in range(m)]
            got = mt.holm(p)
            ref = _holm_reference(p)
            assert got == pytest.approx(ref, abs=1e-12)


def test_holm_with_ties_and_ones():
    p = [0.01, 0.01, 1.0, 0.04, 0.5]
    assert mt.holm(p) == pytest.approx(_holm_reference(p))


def test_holm_is_monotone_and_bounded():
    rng = random.Random(1)
    for _ in range(200):
        p = [rng.random() for _ in range(rng.randint(1, 25))]
        adj = mt.holm(p)
        assert all(a >= r - 1e-15 for a, r in zip(adj, p, strict=True))
        assert all(a <= 1.0 for a in adj)
        for (pi, ai), (pj, aj) in itertools.combinations(zip(p, adj, strict=True), 2):
            if pi < pj:
                assert ai <= aj + 1e-15


def test_family_counts_absent_arms_as_p_one():
    """The correction depends on the declared family, not on what has been run."""
    a = mt.Paired(5, 1.0, 0.5, 0.4, 1.6, 4.5, 0.01, None, (), ())
    b = mt.Paired(5, 1.0, 0.5, 0.4, 1.6, 4.5, 0.03, None, (), ())
    results = {"a": a, "b": b, "c": None}
    mt.apply_family(results, ["a", "b", "c"])
    expect = mt.holm([0.01, 0.03, 1.0])
    assert a.p_holm == pytest.approx(expect[0])
    assert b.p_holm == pytest.approx(expect[1])
    # A smaller family would have corrected less.
    assert b.p_holm > mt.holm([0.01, 0.03])[1]


def test_sample_sd_uses_ddof_one():
    rng = np.random.default_rng(2)
    for n in (2, 3, 5, 10):
        x = rng.normal(size=n)
        assert mt.sample_sd(x.tolist()) == pytest.approx(float(np.std(x, ddof=1)))
    assert mt.sample_sd([3.0]) == 0.0


def test_paired_interval_and_test_match_scipy():
    a = {1: 18.9, 2: 19.4, 3: 18.8, 4: 18.4, 5: 19.1}
    b = {1: 18.2, 2: 18.5, 3: 18.8, 4: 18.4, 5: 18.4}
    res = mt.paired(a, b)
    d = np.array([a[s] - b[s] for s in sorted(a)])
    assert res.n == 5
    assert res.mean == pytest.approx(d.mean())
    assert res.sd == pytest.approx(d.std(ddof=1))
    lo, hi = scipy_stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / math.sqrt(len(d)))
    assert (res.lo, res.hi) == pytest.approx((lo, hi))
    assert res.p == pytest.approx(scipy_stats.ttest_rel([a[s] for s in sorted(a)], [b[s] for s in sorted(b)]).pvalue)


def test_pairing_is_by_seed_id_not_position():
    a = {3: 3.0, 1: 1.0, 2: 2.0, 5: 50.0}
    b = {1: 0.5, 2: 1.5, 3: 2.5, 4: 99.0}
    res = mt.paired(a, b)
    assert res.n == 3
    assert res.deltas == pytest.approx((0.5, 0.5, 0.5))
    assert mt.paired({1: 1.0}, {1: 2.0}) is None


def test_one_sided_against_published():
    res = mt.one_sample([0.9, 1.0, 1.1, 0.95, 1.05])
    # Replicated +1.0 against a published -0.06: overwhelmingly worse.
    assert mt.one_sided_p(res, -0.06, greater=True) < 1e-3
    assert mt.one_sided_p(res, -0.06, greater=False) > 0.99


def test_p_value_formatting():
    assert mt.fp(0.4312) == "0.43"
    assert mt.fp(0.0539) == "0.054"
    assert mt.fp(0.00049) == "0.0005"
    assert mt.fp(3e-6) == r"<10^{-4}"
    assert mt.fp_wilcoxon(0.0625) == "$0.0625$"
    assert mt.fsigned(-0.0001, 2) == "0.00"
    assert mt.fsigned(0.44, 2) == "+0.44"


# ----------------------------------------------------------------- pricing


def _pricing_rows(ops: dict[tuple, list[float]], dataset: str = "cifar10") -> list[dict]:
    rows = []
    for (op, extra), fids in ops.items():
        for seed, fid in enumerate(fids, start=1):
            row = {"dataset": dataset, "op": op, "dose_db": None, "target": None, "source": "mse",
                   "source_epoch": mt.PRICING_EPOCH[dataset], "seed": seed, "quantize": "float",
                   "fid": fid, "kid": 0.01, "dev_before": [-0.3, -0.8, -2.3],
                   "dev_after": [-0.3, -0.8, -1.9], "n_images": 10000}
            row.update(dict(extra))
            rows.append(row)
    return rows


def test_pricing_deduplicates_keeping_the_last_row(tmp_path):
    rows = _pricing_rows({("identity", ()): [18.0] * 5})
    rows.append(dict(rows[0], fid=99.0))
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    loaded = mt.load_pricing(path)
    assert len(loaded) == 5
    assert mt.price(loaded, "cifar10", "identity")[1] == 99.0


def test_partial_pricing_measurement_is_withheld():
    rows = _pricing_rows({("identity", ()): [18.0, 18.2]})
    assert mt.price(rows, "cifar10", "identity") == {}
    table = mt.build_worth_table(rows)
    assert mt.PLACEHOLDER in table


def test_pricing_tables_are_placeholders_without_a_ledger():
    for build in (lambda: mt.build_worth_table([]),
                  lambda: mt.build_operators_table([], []),
                  lambda: mt.build_accounting_table([], [], []),
                  lambda: mt.build_pricing_arms_table([], []),
                  lambda: mt.build_price_curve_table([])):
        text = build()
        assert mt.PLACEHOLDER in text
        _latex_safe("pricing", mt.PREAMBLE + text)


def test_mrg_is_gap_plus_value():
    b = [18.2, 18.5, 18.8, 18.4, 18.4]
    tb = [17.9, 18.1, 18.4, 18.1, 18.0]
    a = {s: v for s, v in enumerate([19.4, 18.8, 18.8, 18.4, 19.1], start=1)}
    rows = _pricing_rows({("identity", ()): b,
                          ("boost", (("dose_db", mt.BOOST_DOSE),)): tb,
                          ("transplant", (("target", "awwl"),)): tb})
    B = mt.price(rows, "cifar10", "identity")
    TB = mt.price(rows, "cifar10", "boost", dose=mt.BOOST_DOSE)
    gap, value, m = mt.paired(a, B), mt.value_of(B, TB), mt.mrg(a, TB)
    assert m.mean == pytest.approx(gap.mean + value.mean)
    phase0 = [{"group": "awwl", "seed": s, "fid": v, "epoch": 199} for s, v in a.items()]
    table = mt.build_accounting_table(rows, phase0, [])
    assert "$+" + mt.fnum(m.mean, 2) + "$" in table
    # Bold means p_Holm < 0.05 in every table; the MRG row is not bold.
    mrg_line = next(line for line in table.splitlines() if line.startswith("MRG"))
    assert r"\mathbf" not in mrg_line
    _latex_safe("accounting", table)


# ------------------------------------------------ names, blocks, families


def test_every_arm_of_every_family_has_one_name_and_one_place():
    listed = [k for _title, keys in mt.BLOCKS for k in keys]
    assert len(listed) == len(set(listed)), "an arm sits in two blocks"
    assert set(listed) | {"mse"} == set(mt.NAMES)
    assert len(set(mt.NAMES.values())) == len(mt.NAMES), "two arms share a display name"
    for family in (mt.FAMILY_MAIN, mt.FAMILY_PRICING, mt.FAMILY_EXTENSIONS, mt.FAMILY_DDPM,
                   mt.FAMILY_DREAMBOOTH, mt.FAMILY_RESTORATION):
        assert set(family) <= set(listed), set(family) - set(listed)
    assert mt.TIMESTEP_ARM in listed and mt.TIMESTEP_ARM not in mt.FAMILY_MAIN


def _eval_rows(means: dict[str, float], seeds=(1, 2, 3, 4, 5), epoch: int = 199, spread: float = 0.2):
    rng = random.Random(7)
    base = {s: 18.0 + rng.gauss(0, 0.5) for s in seeds}
    return [{"group": g, "seed": s, "epoch": epoch, "fid": base[s] + d + rng.gauss(0, spread)}
            for g, d in means.items() for s in seeds]


def test_budget_family_is_the_two_long_contrasts():
    """E14: each contrast is corrected once; the 200-epoch rows keep their home family's p."""
    short = _eval_rows({"mse": 0, "awwl": 0.4, "static_wavelet": 0.1, "l1": 7, "mse_ema": 0.0,
                        "awwl_ema": 0.5})
    anc = _eval_rows({"mse": 0, "awwl": 1.0, "static_wavelet": 0.3})
    long_ = _eval_rows({"mse": -4.5, "awwl": -4.3}, seeds=(1, 2, 3), epoch=799)
    long_anc = _eval_rows({"mse": -6.5, "awwl": -6.1}, seeds=(1, 2, 3), epoch=799)
    points = {(b, s): res for b, s, _mb, _ma, res in
              mt.budget_points(short, short, long_, anc, long_anc)}
    assert len(points) == len(mt.BUDGET_POINTS)
    long_pts = [points[mt.BUDGET_POINTS[i]] for i in mt.FAMILY_BUDGET]
    assert [r.p_holm for r in long_pts] == pytest.approx(mt.holm([r.p for r in long_pts]))
    main = mt.compare_family(short, "fid", mt.FAMILY_MAIN)["awwl"]
    assert points[("200 epochs, no EMA", "DDIM-100")].p_holm == pytest.approx(main.p_holm)
    ema = points[("200 epochs, EMA", "DDIM-100")]
    assert ema.p_holm == pytest.approx(ema.p)
    ddpm = mt.compare_family(anc, "fid", mt.FAMILY_DDPM)["awwl"]
    assert points[("200 epochs, no EMA", "DDPM-1000")].p_holm == pytest.approx(ddpm.p_holm)
    table = mt.build_budget_table(mt.budget_points(short, short, long_, anc, long_anc))
    assert table.count(r"$^{\dagger}$") == 1 + 3   # the note and the three repeated rows
    assert r"\textsuperscript" not in table
    _latex_safe("budget", table)


def test_timestep_arm_enters_the_tables_only_with_every_seed():
    """It has its own two-test family; until all seeds are in, no row is shown."""
    means = {"mse": 0, "awwl": 0.4, "static_wavelet": 0.1}
    four = _eval_rows(means) + [r for r in _eval_rows({mt.TIMESTEP_ARM: 0.6}) if r["seed"] < 5]
    five = _eval_rows(means) + _eval_rows({mt.TIMESTEP_ARM: 0.6})
    name = mt.NAMES[mt.TIMESTEP_ARM]
    for build in (lambda r: mt.build_comparison_table(r, "fid", "FID", digits=2),
                  lambda r: mt.build_main_table(r, 199)):
        assert name not in build(four)
        assert name in build(five)
    table = mt.build_comparison_table(five, "fid", "FID", digits=2)
    row = next(line for line in table.splitlines() if line.startswith(name))
    assert r"$^{\dagger}$" in row
    tt = mt.contrasts(five, "fid", mt.FAMILY_TIMESTEP)[mt.FAMILY_TIMESTEP[0]]
    assert mt.fp(tt.p_holm) in row
    _latex_safe("compare", table)


def _multi_metric_rows(shifts: dict[str, float], seeds=(1, 2, 3, 4, 5)) -> list[dict]:
    """Rows with every metric of Table compare-other, each arm shifted by ``shifts``."""
    rng = random.Random(11)
    out = []
    for g, d in shifts.items():
        for s in seeds:
            out.append({"group": g, "seed": s, "epoch": 199, "fid": 18.5 + d + rng.gauss(0, 0.2),
                        "is_mean": 7.7 - 0.1 * d + rng.gauss(0, 0.05),
                        "kid_tf": 0.012 + 0.001 * d + rng.gauss(0, 2e-4),
                        "precision": 0.66 + rng.gauss(0, 0.004), "recall": 0.55 + rng.gauss(0, 0.004)})
    return out


def test_compare_other_merges_the_four_metrics():
    rows = _multi_metric_rows({"mse": 0, "awwl": 0.4, "static_wavelet": 0.1, "l1": 7.0,
                               mt.TIMESTEP_ARM: 0.8})
    text = mt.build_compare_other_table(rows)
    assert r"\label{tab:compare-other}" in text
    for header in (r"IS $\uparrow$", r"KID$\times 10^{3}$", r"Precision $\uparrow$", r"Recall $\uparrow$"):
        assert header in text
    lines = text.splitlines()
    body = [line for line in lines[lines.index(r"\midrule") + 1:]
            if line.endswith(r"\\") and not line.startswith(r"\multicolumn")]
    ncols = 1 + 3 * len(mt.OTHER_METRICS)
    assert all(line.count(" & ") == ncols - 1 for line in body)
    # The timestep-only arm carries its own family, marked on each corrected
    # p (as in Table compare-fid), not on its name.
    row = next(line for line in body if line.startswith(mt.NAMES[mt.TIMESTEP_ARM]))
    assert not row.startswith(mt.NAMES[mt.TIMESTEP_ARM] + r"$^{\dagger}$")
    cells = row.removesuffix(r" \\").split(" & ")
    assert all(cells[3 + 3 * i].endswith(r"$^{\dagger}$") for i in range(len(mt.OTHER_METRICS)))
    assert "carries no dagger" in text and r"$p_{\mathrm{Holm}}$ carries no dagger" in text
    # Bold where, and only where, the corrected p is below 0.05, per metric.
    for i, (metric, _h, _d, scale) in enumerate(mt.OTHER_METRICS):
        for arm, r in mt._metric_results(rows, metric, scale).items():
            line = next(line for line in body if line.startswith(mt.NAMES[arm]))
            cell = line.removesuffix(r" \\").split(" & ")[3 + 3 * i]
            assert (r"\mathbf" in cell) == (r.p_holm < 0.05), (arm, metric, cell)
    _latex_safe("compare_other", text)


def test_confound_table_orders_its_rows_and_draws_the_grid():
    rows = _eval_rows({"mse": 0, mt.TIMESTEP_ARM: 0.8, "awwl": 0.4, "awwl_normalized": 1.1,
                       "awwl_norm_matched": 1.3, "awwl_grad_matched_mean": 0.9})
    text = mt.build_confound_table(rows, Path("/nonexistent"))
    names = [mt.NAMES[a] for _t, arms in mt.CONFOUND_BLOCKS for a in arms]
    starts = [next(i for i, line in enumerate(text.splitlines()) if line.startswith(n + " & "))
              for n in names]
    assert starts == sorted(starts), "rows out of order"
    # The timestep-only arm: both contrasts corrected in its own two-test family.
    tt = mt.contrasts(rows, "fid", mt.FAMILY_TIMESTEP)
    row = next(line for line in text.splitlines() if line.startswith(mt.NAMES[mt.TIMESTEP_ARM] + " & "))
    for contrast in mt.FAMILY_TIMESTEP:
        assert mt.fp(tt[contrast].p_holm) in row
    assert row.count(r"$^{\dagger}$") == 2
    # Arm minus AWWL, the negative of AWWL minus the arm.
    b = tt[mt.FAMILY_TIMESTEP[1]]
    assert mt.fsigned(-b.mean, 2) in row
    # The 2x2 summary: one cell per (schedule, shape).
    grid = text[text.index(r"\par\medskip"):]
    for arm in mt.CONFOUND_GRID.values():
        assert mt.NAMES[arm] in grid
    assert r"\label{tab:confound}" in text
    _latex_safe("confound", text)
    # Without every seed of the timestep-only arm, neither its row nor the grid.
    four = [r for r in rows if not (r["group"] == mt.TIMESTEP_ARM and r["seed"] == 5)]
    text4 = mt.build_confound_table(four, Path("/nonexistent"))
    assert mt.NAMES[mt.TIMESTEP_ARM] not in text4 and r"\medskip" not in text4


def test_accounting_bolds_an_mrg_only_when_holm_significant():
    b = [18.2, 18.5, 18.8, 18.4, 18.4]
    a = {s: v + 1.0 + 0.001 * s for s, v in enumerate(b, start=1)}
    tb = [v - 0.3 for v in b]
    rows = _pricing_rows({("identity", ()): b,
                          ("boost", (("dose_db", mt.BOOST_DOSE),)): tb,
                          ("transplant", (("target", "awwl"),)): tb})
    phase0 = [{"group": "awwl", "seed": s, "fid": v, "epoch": 199} for s, v in a.items()]
    table = mt.build_accounting_table(rows, phase0, [])
    mrg_line = next(line for line in table.splitlines() if line.startswith("MRG"))
    holm = mt.pricing_arms(rows, phase0)["awwl"]["mrg"].p_holm
    assert holm < 0.05
    cells = mrg_line.split(" & ")
    # Boost column: no family, never bold; transplant column: bold.
    assert r"\mathbf" not in cells[1] and r"\mathbf" in cells[2]
    _latex_safe("accounting", table)


def test_replication_details_carry_the_one_sided_tests_and_intervals():
    ddim = _multi_metric_rows({"mse": 0, "awwl": 0.4, "static_wavelet": 0.1})
    ddpm = _multi_metric_rows({"mse": -2, "awwl": -1.0, "static_wavelet": -1.7})
    text = mt.build_replication_details_table(ddim, ddpm)
    assert r"\label{tab:replication-details}" in text
    res = mt.compare_family(ddpm, "fid", mt.FAMILY_DDPM)["awwl"]
    pub = mt.CONFERENCE_CIFAR10["awwl"]["fid"] - mt.CONFERENCE_CIFAR10["mse"]["fid"]
    assert mt.fp(mt.one_sided_p(res, pub, greater=True)) in text
    lo, hi = mt.prediction_interval(res)
    assert mt.fci(lo, hi, 2) in text
    main = mt.build_replication_table(ddim, ddpm)
    assert r"\ref{tab:replication-details}" in main and "Inception Score}" not in main
    _latex_safe("replication_details", text)


def test_replication_details_list_every_row_of_the_conference_table():
    """Critic C1: the replication is not limited to three of the seven rows."""
    ddim = _multi_metric_rows({"mse": 0, "awwl": 0.4, "static_wavelet": 0.1, "l1": 7.0,
                               "huber": 0.3, "snr_weighted": 300.0})
    ddpm = _multi_metric_rows({"mse": -2, "awwl": -1.0, "static_wavelet": -1.7})
    text = mt.build_replication_details_table(ddim, ddpm)
    assert len(mt.CONFERENCE_ROWS) == 7 and mt.CONFERENCE_ROWS[0] == "mse"
    lines = text.splitlines()
    for arm in mt.CONFERENCE_ROWS:
        name = mt.NAMES.get(arm) or mt.CONFERENCE_ONLY_NAMES[arm]
        row = next(line for line in lines if line.startswith(name + " & "))
        for metric in ("fid", "is_mean"):
            assert "$" + mt.fnum(mt.CONFERENCE_CIFAR10[arm][metric], 2) + "$" in row
    # An arm sampled only with DDIM-100 says so; one not retrained says so twice per metric.
    assert next(line for line in lines if line.startswith(mt.NAMES["l1"] + " & ")).count("not run") == 2
    assert next(line for line in lines if line.startswith("Perceptual (VGG) & ")).count("not run") == 4
    # A mean in the hundreds prints without decimals.
    snr = next(line for line in lines if line.startswith(mt.NAMES["snr_weighted"] + " & "))
    assert "." not in snr.split(" & ")[3].split(r"\pm")[0]
    _latex_safe("replication_details_seven", text)


def test_pricing_caption_names_the_family_and_the_fid_symbol():
    """The family's membership is stated, and FID is M, never F (the DFT)."""
    text = mt.build_pricing_arms_table([], [])
    stable = [a for a in mt.FAMILY_MAIN if a in mt.FAMILY_PRICING]
    assert f"the {len(stable)} arms of the main family" in text
    assert mt.NAMES["awwl_ema"] in text
    for builder in (lambda: mt.build_pricing_arms_table([], []), lambda: mt.build_worth_table([]),
                    lambda: mt.build_operators_table([], []), lambda: mt.build_price_curve_table([]),
                    lambda: mt.build_accounting_table([], [], [])):
        assert "F(" not in builder()


def test_band_table_shows_the_timestep_arm_only_with_every_seed():
    mb = pytest.importorskip("make_band_table")
    assert mt.TIMESTEP_ARM in [a for a, _ in mb.ARMS]
    rng = np.random.default_rng(4)
    devs = {arm: {s: (rng.normal(size=3) - [0.3, 0.8, 2.3]).tolist() for s in (1, 2, 3, 4, 5)}
            for arm in ("mse", "awwl", mt.TIMESTEP_ARM)}
    arms = [(a, lab) for a, lab in mb.ARMS if a in devs]
    assert mt.NAMES[mt.TIMESTEP_ARM] in mb.build_band_table(devs, np.zeros(23), arms=arms)
    del devs[mt.TIMESTEP_ARM][5]
    assert mt.NAMES[mt.TIMESTEP_ARM] not in mb.build_band_table(devs, np.zeros(23), arms=arms)


# --------------------------------------------------------- on the real data


@needs_ledger
def test_builders_run_on_the_real_ledgers():
    written = mt.build_all(**{k: v for k, v in mt.DEFAULTS.items()})
    for name in ("full.tex", "compare_fid.tex", "compare_other.tex", "replication.tex",
                 "replication_details.tex", "confound.tex", "extensions.tex", "worth.tex",
                 "accounting.tex", "operators.tex", "pricing_arms.tex"):
        assert name in written, f"{name} not built"
    for name, text in written.items():
        assert text.startswith("% Generated by scripts/"), name
        assert r"\label{tab:" in text, name
        assert r"\caption{" in text, name
        _latex_safe(name, text)
        # Equation numbers change when the paper is reordered; tables must use \ref.
        assert not re.search(r"Eq\.~\(\d", text), f"{name}: hard-coded equation number"


@needs_ledger
def test_family_is_fixed_not_read_from_the_ledger():
    rows = mt.load_eval(PHASE0, 199)
    res = mt.compare_family(rows, "fid", mt.FAMILY_MAIN)
    assert set(res) == set(mt.FAMILY_MAIN)
    present = [r.p for r in res.values() if r is not None]
    absent = sum(1 for r in res.values() if r is None)
    ref = mt.holm(present + [1.0] * absent)
    got = [r.p_holm for r in res.values() if r is not None]
    assert got == pytest.approx(ref[: len(present)])


@needs_ledger
def test_dreambooth_caption_counts_its_objectives():
    path = ROOT / "runs" / "dreambooth_seeds" / "results.jsonl"
    if not path.exists():
        pytest.skip("no DreamBooth ledger")
    rows = mt.load_eval(path)
    text = mt.build_dreambooth_table(rows)
    n = len(mt.groups_of(rows))
    assert f"Rank is over the {mt.cw(n)} objectives" in text
    if n != 9:
        assert "Rank is over the nine" not in text
    assert r"$\Delta$" not in text


@needs_ledger
def test_numbers_file_is_well_formed():
    mn = pytest.importorskip("make_numbers")
    macros = mn.build(compute=False)
    text = macros.text()
    assert text.startswith("% Generated by scripts/make_numbers.py")
    _latex_safe("numbers.tex", text)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(r"\newcommand"):
            assert lines[i - 1].startswith("% "), f"macro without a source comment: {line}"
    names = re.findall(r"\\newcommand\{\\([^}]*)\}", text)
    assert names and all(re.fullmatch(r"[A-Za-z]+", n) for n in names)
    assert len(names) == len(set(names))
    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    clash = set(names) & set(re.findall(r"\\newcommand\{\\(\w+)\}", main))
    assert not clash, f"generated macros collide with main.tex: {sorted(clash)}"
    for key in ("fidMSEmean", "fidAWWLmean", "dAWWL", "dAWWLLo", "dAWWLHi", "dAWWLP", "dAWWLHolm",
                "convDOneFiftyNine", "nRunsTotal", "gpuHoursTotal", "celebaFIDMSEmean", "mrgMain",
                "mrgMainLo", "mrgMainHi", "priceBoostDiffAWWL", "priceBoostDiffAWWLLo",
                "priceBoostDiffAWWLHi", "priceBoostDiffAWWLP"):
        assert key in macros.names, key
    # The published weighting's averages over the training timesteps (verified
    # values; the two sum-of-squares averages are what the gradient-scale arms pin).
    assert macros.names["wTotalMean"] == "0.3165"
    assert macros.names["wSqMean"] == "0.0530"
    assert macros.names["wSqMeanSummed"] == "0.2454"
    assert float(macros.names["levelSqFixedSum"]) == pytest.approx(float(macros.names["wSqMeanSummed"]))
    assert float(macros.names["levelSqFixed"]) == pytest.approx(float(macros.names["wSqMean"]))
    # The headline comparison against the ledger, recomputed independently.
    rows = mt.load_eval(PHASE0, 199)
    m = mt.by_seed(rows, "mse", "fid")
    assert macros.names["fidMSEmean"] == f"{np.mean(list(m.values())):.3f}"
    assert macros.names["fidMSEsd"] == f"{np.std(list(m.values()), ddof=1):.3f}"


def test_band_edges_match_band_deviations():
    mb = pytest.importorskip("make_band_table")
    from awwl.evaluation.price_fid import band_deviations

    for n in (17, 23, 46):
        profile = np.arange(n, dtype=float)
        devs = band_deviations(profile, np.zeros(n))
        for (lo, hi), dev in zip(mb.band_edges(n), devs, strict=True):
            assert dev == pytest.approx(np.mean(profile[lo : hi + 1]))
    assert mb.band_edges(23) == [(0, 6), (7, 14), (15, 22)]


def test_atomic_save_leaves_no_temporary(tmp_path):
    mb = pytest.importorskip("make_band_table")
    target = tmp_path / "sub" / "profile.npy"
    mb.atomic_save(target, np.arange(5.0))
    assert np.load(target).tolist() == [0, 1, 2, 3, 4]
    assert [p.name for p in target.parent.iterdir()] == ["profile.npy"]


def test_incomplete_sample_folder_is_not_profiled(tmp_path, monkeypatch):
    """A folder a sampling job is still filling must not reach the shared cache."""
    mb = pytest.importorskip("make_band_table")
    from PIL import Image

    folder = tmp_path / "samples" / "ep199"
    folder.mkdir(parents=True)
    for i in range(3):
        Image.fromarray(np.full((8, 8, 3), 40 * i, dtype=np.uint8)).save(folder / f"{i:05d}.png")
    cache = tmp_path / "cache" / "arm_s1_ep199.npy"
    monkeypatch.setattr(mb, "EXPECTED_SAMPLES", 4)
    mb.ensure_profiles([(folder, cache)], workers=1)
    assert not cache.exists()
    monkeypatch.setattr(mb, "EXPECTED_SAMPLES", 3)
    mb.ensure_profiles([(folder, cache)], workers=1)
    assert cache.exists() and np.load(cache).shape == (len(np.bincount(mb.pf.radial_bins(8, 8).ravel())),)


def test_band_table_builder_is_latex_safe():
    mb = pytest.importorskip("make_band_table")
    rng = np.random.default_rng(3)
    devs = {arm: {s: (rng.normal(size=3) - [0.3, 0.8, 2.3]).tolist() for s in (1, 2, 3, 4, 5)}
            for arm, _ in mb.ARMS[:4]}
    ref = np.zeros(23)
    text = mb.build_band_table(devs, ref, arms=mb.ARMS[:4])
    _latex_safe("bands.tex", text)
    assert "radii 0--6 (low), 7--14 (mid) and 15--22 (high)" in text
