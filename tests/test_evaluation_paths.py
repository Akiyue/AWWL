"""Regression tests for the two failures that only appear on a real eval run.

Both cost minutes of GPU time before surfacing, and both were invisible to the
existing suite: one needs a *relative* folder path, the other needs a modern
SciPy. They are cheap to pin down once known, so they are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from awwl.evaluation.advanced_metrics import _ImagePathDataset, _list_images
from awwl.evaluation.fid_is import _patch_scipy_sqrtm_disp


@pytest.fixture
def image_folder(tmp_path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in range(3):
        Image.new("RGB", (8, 8), color=(i * 40, 0, 0)).save(folder / f"{i:05d}.png")
    (folder / "notes.txt").write_text("ignored", encoding="utf-8")
    return folder


def test_listing_skips_non_images(image_folder):
    assert [p.name for p in _list_images(image_folder)] == ["00000.png", "00001.png", "00002.png"]


def test_dataset_works_from_a_relative_path(image_folder, monkeypatch):
    """The exact failure: 'data/x/data/x/00000.png', No such file or directory.

    `iterdir()` already returns folder-prefixed paths, so re-joining them onto
    the folder doubled the prefix. It went unnoticed because pathlib drops the
    left operand when the right side is absolute — so absolute inputs worked
    and relative ones did not.
    """
    monkeypatch.chdir(image_folder.parent)

    dataset = _ImagePathDataset("imgs")
    assert len(dataset) == 3
    assert dataset[0].shape == (3, 299, 299)  # loads, rather than raising


def test_dataset_still_works_from_an_absolute_path(image_folder):
    assert len(_ImagePathDataset(str(image_folder))) == 3


def test_max_images_caps_the_listing(image_folder):
    assert len(_ImagePathDataset(image_folder, max_imgs=2)) == 2


# ------------------------------------------------------------- sqrtm shim


def test_sqrtm_shim_restores_the_disp_contract():
    """clean-fid calls sqrtm(..., disp=False); SciPy 1.17 removed the argument.

    The old contract returned ``(X, errest)`` when ``disp=False``. FID dies
    without it — after both feature passes have already run.
    """
    scipy_linalg = pytest.importorskip("scipy.linalg")
    _patch_scipy_sqrtm_disp()

    matrix = np.array([[4.0, 0.0], [0.0, 9.0]])
    result = scipy_linalg.sqrtm(matrix, disp=False)

    assert isinstance(result, tuple) and len(result) == 2
    root, errest = result
    assert np.allclose(root, np.array([[2.0, 0.0], [0.0, 3.0]]))
    assert errest == pytest.approx(0.0, abs=1e-10)


def test_sqrtm_shim_leaves_the_default_call_untouched():
    """Without `disp` the return value must stay the bare matrix."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    _patch_scipy_sqrtm_disp()

    root = scipy_linalg.sqrtm(np.array([[4.0, 0.0], [0.0, 9.0]]))
    assert not isinstance(root, tuple)
    assert np.allclose(root, np.array([[2.0, 0.0], [0.0, 3.0]]))


def test_sqrtm_shim_is_idempotent():
    """It runs before every evaluation; re-patching must not nest wrappers."""
    pytest.importorskip("scipy.linalg")
    _patch_scipy_sqrtm_disp()
    _patch_scipy_sqrtm_disp()
    assert _patch_scipy_sqrtm_disp() in (True, False)

    from scipy import linalg

    root, _ = linalg.sqrtm(np.eye(2), disp=False)
    assert np.allclose(root, np.eye(2))


# ------------------------------------------------------- spectral banding


def test_band_deviations_locates_the_difference():
    """A scalar distance cannot say *where* a loss changed the spectrum.

    The whole premise of a frequency-aware objective is a correction at high
    frequencies, so the band split is what separates 'the mechanism works' from
    'something changed, elsewhere'.
    """
    from awwl.evaluation.spectrum import band_deviations

    real = np.zeros(30)
    model = np.zeros(30)
    model[20:] = -2.0  # deficit confined to the top third

    low, mid, high = band_deviations(model, real)
    assert low == pytest.approx(0.0)
    assert mid == pytest.approx(0.0)
    assert high == pytest.approx(-2.0)


def test_band_deviations_are_signed():
    """Sign distinguishes too-little energy (over-smoothing) from too much."""
    from awwl.evaluation.spectrum import band_deviations

    real = np.zeros(30)
    assert band_deviations(np.full(30, 1.5), real)[0] > 0
    assert band_deviations(np.full(30, -1.5), real)[0] < 0


def test_band_deviations_tolerate_length_mismatch():
    from awwl.evaluation.spectrum import band_deviations

    assert len(band_deviations(np.zeros(24), np.zeros(30))) == 3


def test_band_table_renders(tmp_path):
    from awwl.evaluation.spectrum import format_band_table

    real = np.zeros(30)
    table = format_band_table(real, {"mse": np.full(30, -3.0), "awwl": np.full(30, -1.0)})
    assert "mse" in table and "awwl" in table and "high" in table


def test_profiles_by_config_skips_missing_runs(tmp_path, caplog):
    from awwl.evaluation.spectrum import profiles_by_config

    folder = tmp_path / "awwl_s1" / "samples" / "ep199"
    folder.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (32, 32), color=(i * 60, 10, 10)).save(folder / f"{i}.png")

    profiles = profiles_by_config(tmp_path, configs=["awwl", "absent"], seeds=[1, 2], epoch=199)
    assert set(profiles) == {"awwl"}, "a config with no samples must be skipped, not crash"
    assert profiles["awwl"].ndim == 1


# ------------------------------------------------------ paired sample grid


def _fake_samples(root, config, seed, n=6):
    folder = root / f"{config}_s{seed}" / "samples" / "ep199"
    folder.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (32, 32), color=(i * 30, 0, 0)).save(folder / f"{i:05d}.png")
    return folder


def test_paired_grid_puts_one_row_per_config(tmp_path):
    from awwl.plotting.paired_grid import paired_sample_grid

    folders = {c: _fake_samples(tmp_path, c, 1) for c in ("mse", "awwl")}
    out = paired_sample_grid(folders, output_path=tmp_path / "cmp.png", count=4, scale=2)

    assert out.exists()
    image = Image.open(out)
    assert image.width > image.height, "4 columns of 2 rows should be wider than tall"


def test_paired_grid_reports_a_missing_folder(tmp_path):
    from awwl.plotting.paired_grid import paired_sample_grid

    folders = {"mse": _fake_samples(tmp_path, "mse", 1), "absent": tmp_path / "nope"}
    with pytest.raises(FileNotFoundError, match="absent"):
        paired_sample_grid(folders, output_path=tmp_path / "cmp.png")


def test_paired_grid_start_offset_selects_later_samples(tmp_path):
    from awwl.plotting.paired_grid import paired_sample_grid

    folders = {"mse": _fake_samples(tmp_path, "mse", 1, n=10)}
    assert paired_sample_grid(
        folders, output_path=tmp_path / "cmp.png", count=3, start=5
    ).exists()
