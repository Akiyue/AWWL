"""Pricing a spectral correction on DreamBooth samples.

The CIFAR-10 analysis asks what the objective's spectral correction is worth by
applying it to the baseline's own samples. This is the same question where the
metrics are CLIP-based; the point of the test is that the correction is
*measured per run* rather than assumed, so each arm is priced at what it
actually achieves.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from awwl.evaluation.pricing import format_pricing_table, high_band_deficit, price_run


def write_images(folder, n, *, smooth):
    """`smooth=True` starves the high band, which is the deficit under test."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0 if smooth else 1)
    for i in range(n):
        a = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(a)
        if smooth:
            from PIL import ImageFilter

            img = img.filter(ImageFilter.GaussianBlur(radius=1.4))
        img.save(folder / f"{i}.png")
    return folder


@pytest.fixture
def run(tmp_path):
    real = write_images(tmp_path / "real", 6, smooth=False)
    run_dir = tmp_path / "mse_s1"
    for i in range(2):
        write_images(run_dir / "samples" / f"prompt{i}", 4, smooth=True)
    (run_dir / "config.json").write_text(
        json.dumps({"data": {"instance_data_dir": str(real)}}), encoding="utf-8"
    )
    return run_dir, real


def constant_scorer(_folder, _prompt):
    return [0.3, 0.3], [0.8, 0.8]


def test_smoothed_samples_show_a_positive_high_band_deficit(run):
    run_dir, real = run
    dirs = sorted((run_dir / "samples").glob("prompt*"))

    deficit = high_band_deficit(dirs, real, max_images=100)

    assert deficit > 0, "blurred samples must read as missing high-frequency energy"


def test_price_run_reports_both_metrics_before_and_after(run):
    run_dir, _ = run

    priced = price_run(run_dir, prompts=["a", "b"], score_folder=constant_scorer,
                       work=run_dir.parent / "work")

    assert priced.exp == "mse_s1"
    assert priced.n_images == 4, "two prompts scored, two images each"
    assert priced.clip_score == pytest.approx(0.3)
    assert priced.similarity == pytest.approx(0.8)


def test_the_boost_defaults_to_the_run_s_own_measured_deficit(run):
    """Each arm is priced at the correction it achieves, not a shared constant."""
    run_dir, real = run
    dirs = sorted((run_dir / "samples").glob("prompt*"))
    expected = high_band_deficit(dirs, real, max_images=100)

    priced = price_run(run_dir, prompts=["a", "b"], score_folder=constant_scorer,
                       work=run_dir.parent / "work")

    assert priced.boost_db == pytest.approx(expected)


def test_boosting_reduces_the_deficit(run):
    run_dir, _ = run

    priced = price_run(run_dir, prompts=["a", "b"], score_folder=constant_scorer,
                       work=run_dir.parent / "work")

    assert priced.boosted_deficit_db < priced.deficit_db, (
        "the post-process must move the spectrum in the direction it claims"
    )


def test_an_explicit_boost_overrides_the_measurement(run):
    run_dir, _ = run

    priced = price_run(run_dir, prompts=["a", "b"], score_folder=constant_scorer,
                       boost_db=0.5, work=run_dir.parent / "work")

    assert priced.boost_db == pytest.approx(0.5)


def test_a_prompt_count_mismatch_is_refused(run):
    """Three prompts against two folders would silently mis-pair prompt and image."""
    run_dir, _ = run

    with pytest.raises(ValueError, match="each prompt directory"):
        price_run(run_dir, prompts=["a", "b", "c"], score_folder=constant_scorer,
                  work=run_dir.parent / "work")


def test_a_missing_reference_folder_says_so(run, tmp_path):
    run_dir, _ = run

    with pytest.raises(ValueError, match="reference images not found"):
        price_run(run_dir, prompts=["a", "b"], score_folder=constant_scorer,
                  real_dir=tmp_path / "nope", work=run_dir.parent / "work")


def test_the_table_shows_the_change_in_both_metrics(run):
    run_dir, _ = run
    priced = price_run(run_dir, prompts=["a", "b"], score_folder=constant_scorer,
                       work=run_dir.parent / "work")

    table = format_pricing_table([priced])

    assert "dCLIP" in table and "dsim" in table
    assert "mse_s1" in table
