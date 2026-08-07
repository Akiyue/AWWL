"""Results-ledger and significance-testing tests."""

from __future__ import annotations

import math

from awwl.analysis.results import append_result, load_results, result_row
from awwl.analysis.stats import (
    _holm,
    compare_to_baseline,
    convergence_table,
    format_comparison_table,
    format_summary_table,
    summarize_groups,
)


def _rows(group: str, values: dict[int, float], *, metric: str = "fid", epoch: int = 199):
    return [
        {"group": group, "seed": seed, "kind": "eval", "epoch": epoch, metric: value}
        for seed, value in values.items()
    ]


# ------------------------------------------------------------------- ledger


def test_ledger_roundtrip(tmp_path):
    path = tmp_path / "results.jsonl"
    append_result(path, {"group": "mse", "seed": 1, "fid": 16.7})
    append_result(path, {"group": "awwl", "seed": 1, "fid": 16.6})
    assert len(load_results(path)) == 2


def test_ledger_skips_corrupt_trailing_line(tmp_path):
    """A crash mid-append must cost one row, not the whole file."""
    path = tmp_path / "results.jsonl"
    append_result(path, {"group": "mse", "seed": 1, "fid": 16.7})
    with path.open("a", encoding="utf-8") as f:
        f.write('{"group": "awwl", "seed": 1, "fi')  # truncated by a crash
    rows = load_results(path)
    assert len(rows) == 1
    assert rows[0]["group"] == "mse"


def test_ledger_filters_by_kind(tmp_path):
    path = tmp_path / "results.jsonl"
    append_result(path, {"kind": "train", "group": "mse"})
    append_result(path, {"kind": "eval", "group": "mse", "fid": 1.0})
    assert len(load_results(path, kind="eval")) == 1


def test_result_row_captures_experiment_identity():
    cfg = {
        "seed": 3,
        "method": "finetune",
        "data": {"dataset_name": "cifar10", "image_size": 32},
        "loss": {"name": "adaptive_wavelet", "alpha": 0.2, "power": 1.0, "normalize_weights": True},
        "train": {"use_ema": True, "num_epochs": 200},
    }
    row = result_row(cfg, exp="awwl_s3", group="awwl", kind="eval", metrics={"fid": 16.6})
    assert row["seed"] == 3
    assert row["alpha"] == 0.2
    assert row["normalize_weights"] is True
    assert row["use_ema"] is True
    assert row["fid"] == 16.6


# ---------------------------------------------------------------- summaries


def test_summarize_computes_ci():
    rows = _rows("mse", {1: 16.0, 2: 17.0, 3: 18.0})
    (summary,) = summarize_groups(rows, metric="fid")
    assert summary.n == 3
    assert math.isclose(summary.mean, 17.0)
    assert math.isclose(summary.std, 1.0)
    assert summary.ci_low < summary.mean < summary.ci_high


def test_summarize_ignores_failed_metric_sentinel():
    """The eval helpers report a failed metric as -1.0; it must not be averaged in."""
    rows = _rows("mse", {1: 16.0, 2: -1.0, 3: 18.0})
    (summary,) = summarize_groups(rows, metric="fid")
    assert summary.n == 2
    assert math.isclose(summary.mean, 17.0)


def test_summarize_deduplicates_reruns():
    rows = _rows("mse", {1: 16.0}) + _rows("mse", {1: 15.0})
    (summary,) = summarize_groups(rows, metric="fid")
    assert summary.n == 1 and summary.mean == 15.0


# -------------------------------------------------------------- comparisons


def test_holm_is_monotone_and_bounded():
    corrected = _holm([0.01, 0.02, 0.04])
    assert corrected == sorted(corrected)
    assert all(c <= 1.0 for c in corrected)
    assert math.isclose(corrected[0], 0.03)  # 3 x 0.01


def test_holm_preserves_input_order():
    corrected = _holm([0.5, 0.001])
    assert corrected[1] < corrected[0]


def test_paired_test_detects_a_consistent_shift():
    """Every seed improves by the same amount: unambiguously significant."""
    rows = _rows("mse", {1: 17.0, 2: 18.0, 3: 19.0, 4: 20.0, 5: 21.0})
    rows += _rows("awwl", {1: 16.0, 2: 17.0, 3: 18.0, 4: 19.0, 5: 20.0})
    (result,) = compare_to_baseline(rows, metric="fid", baseline="mse")
    assert result.n_pairs == 5
    assert math.isclose(result.mean_delta, -1.0)
    assert result.better, "lower FID must count as better"
    assert result.significant


def test_paired_test_rejects_a_noise_sized_gap():
    """The published pattern: a tiny mean gap swamped by seed variance."""
    rows = _rows("mse", {1: 16.2, 2: 17.4, 3: 16.0, 4: 17.9, 5: 16.5})
    rows += _rows("awwl", {1: 17.1, 2: 16.3, 3: 16.9, 4: 16.6, 5: 17.3})
    (result,) = compare_to_baseline(rows, metric="fid", baseline="mse")
    assert not result.significant


def test_higher_is_better_metric_flips_direction():
    rows = _rows("mse", {1: 7.8, 2: 7.7}, metric="is_mean")
    rows += _rows("awwl", {1: 7.9, 2: 7.8}, metric="is_mean")
    (result,) = compare_to_baseline(rows, metric="is_mean", baseline="mse")
    assert result.mean_delta > 0
    assert result.better


def test_comparison_uses_only_shared_seeds():
    rows = _rows("mse", {1: 17.0, 2: 18.0, 3: 19.0})
    rows += _rows("awwl", {1: 16.0, 2: 17.0})
    (result,) = compare_to_baseline(rows, metric="fid", baseline="mse")
    assert result.n_pairs == 2


def test_comparison_skips_untestable_group(caplog):
    rows = _rows("mse", {1: 17.0, 2: 18.0})
    rows += _rows("awwl", {1: 16.0})
    assert compare_to_baseline(rows, metric="fid", baseline="mse") == []


def test_holm_applied_across_multiple_baselines():
    rows = _rows("mse", {1: 17.0, 2: 18.0, 3: 19.0})
    rows += _rows("a", {1: 16.9, 2: 17.9, 3: 18.9})
    rows += _rows("b", {1: 16.8, 2: 17.8, 3: 18.8})
    results = compare_to_baseline(rows, metric="fid", baseline="mse")
    assert len(results) == 2
    assert all(r.p_holm >= r.p_value for r in results)


# ----------------------------------------------------------------- renderers


def test_renderers_produce_text():
    rows = _rows("mse", {1: 17.0, 2: 18.0}) + _rows("awwl", {1: 16.0, 2: 17.0})
    assert "mse" in format_summary_table(summarize_groups(rows, metric="fid"), metric="fid")
    assert "Holm" in format_comparison_table(compare_to_baseline(rows, metric="fid", baseline="mse"))


def test_convergence_table_spans_epochs():
    rows = _rows("awwl", {1: 30.0}, epoch=39) + _rows("awwl", {1: 16.0}, epoch=199)
    table = convergence_table(rows, metric="fid")
    assert "ep39" in table and "ep199" in table
    assert "30.0000" in table and "16.0000" in table
