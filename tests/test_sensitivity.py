"""Tests for the FID-vs-spectral-deficit calibration.

The experiment's entire value rests on one property: the filter must remove
the amount of high-frequency energy it says it does, measured in the same band
the study reports. A filter that asks for 0.5 dB and delivers 0.9 dB would
move the threshold and quietly invert the conclusion.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from awwl.evaluation.sensitivity import (
    SensitivityPoint,
    _radial_gain,
    attenuate_folder,
    format_sensitivity_table,
    split_reference,
)
from awwl.evaluation.spectrum import band_deviations, radial_profile


@pytest.fixture
def noise_folder(tmp_path):
    """White noise: flat spectrum, so attenuation is unambiguous to measure."""
    rng = np.random.default_rng(0)
    folder = tmp_path / "real"
    folder.mkdir()
    for i in range(24):
        arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(folder / f"{i:04d}.png")
    return folder


# ------------------------------------------------------------------- gain


def test_gain_leaves_low_frequencies_untouched():
    gain = _radial_gain((64, 64), db=6.0, cutoff=0.667, width=0.15)
    assert gain[32, 32] == pytest.approx(1.0), "DC must not be attenuated"


def test_gain_reaches_the_requested_attenuation_at_the_corner():
    gain = _radial_gain((64, 64), db=6.0, cutoff=0.667, width=0.15)
    assert gain[0, 0] == pytest.approx(10 ** (-6.0 / 20.0), rel=1e-6)


def _overshoot(image: np.ndarray, gain: np.ndarray) -> float:
    """How far a filtered image escapes the original value range (Gibbs ringing)."""
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(image)) * gain)))
    return float(max(filtered.max() - image.max(), image.min() - filtered.min()))


def test_smooth_edge_rings_less_than_a_hard_cutoff():
    """The reason for the raised cosine, tested on the artefact it prevents.

    A hard cutoff produces Gibbs ringing around edges, and FID would charge
    that ringing to the "spectral deficit" being measured — contaminating the
    calibration with a penalty the filter itself created.
    """
    edge = np.zeros((128, 128))
    edge[:, 64:] = 255.0

    smooth = _overshoot(edge, _radial_gain((128, 128), db=8.0, cutoff=0.667, width=0.5))
    hard = _overshoot(edge, _radial_gain((128, 128), db=8.0, cutoff=0.667, width=1e-6))

    assert hard > 0, "the hard cutoff should ring, or this test proves nothing"
    assert smooth < 0.6 * hard, f"smooth overshoot {smooth:.2f} vs hard {hard:.2f}"


def test_ringing_stays_small_at_the_deltas_actually_used():
    """Bounds the artefact the calibration cannot separate from the deficit."""
    edge = np.zeros((128, 128))
    edge[:, 64:] = 255.0

    overshoot = _overshoot(edge, _radial_gain((128, 128), db=1.0, cutoff=0.667, width=0.5))
    assert overshoot / 255.0 < 0.01, "over 1% overshoot at 1 dB contaminates the curve"


def test_zero_db_is_a_no_op_gain():
    assert np.allclose(_radial_gain((32, 32), db=0.0, cutoff=0.667, width=0.15), 1.0)


# --------------------------------------------------------------- filtering


@pytest.mark.parametrize("db", [1.0, 3.0, 6.0])
def test_narrow_transition_delivers_the_requested_attenuation(tmp_path, noise_folder, db):
    """Validates the mask maths itself, with the band-spreading turned off.

    At the default (wide) transition the attenuation deliberately bleeds into
    the mid band and the achieved value falls short of the request — which is
    why the curve is plotted against the measured deviation. Narrowing the
    transition isolates the top third and lets the arithmetic be checked.
    """
    reference = attenuate_folder(noise_folder, tmp_path / "ref", db=0.0, width=0.02)
    filtered = attenuate_folder(noise_folder, tmp_path / f"f{db:g}", db=db, width=0.02)

    measured = -band_deviations(radial_profile(filtered), radial_profile(reference))[-1]
    assert measured == pytest.approx(db, rel=0.35), (
        f"requested {db} dB, measured {measured:.2f} dB in the high band"
    )


def test_default_transition_attenuates_less_than_requested(tmp_path, noise_folder):
    """The documented consequence of the wide edge — report measured, not asked."""
    reference = radial_profile(attenuate_folder(noise_folder, tmp_path / "ref", db=0.0))
    filtered = radial_profile(attenuate_folder(noise_folder, tmp_path / "f", db=6.0))

    measured = -band_deviations(filtered, reference)[-1]
    assert 0 < measured < 6.0


def test_attenuation_is_monotone_in_db(tmp_path, noise_folder):
    reference = radial_profile(attenuate_folder(noise_folder, tmp_path / "ref", db=0.0))
    measured = []
    for db in (0.0, 2.0, 6.0):
        folder = attenuate_folder(noise_folder, tmp_path / f"f{db:g}", db=db)
        measured.append(-band_deviations(radial_profile(folder), reference)[-1])
    assert measured == sorted(measured)


def test_low_band_is_left_alone(tmp_path, noise_folder):
    """The deficit must sit at the top, or the calibration measures the wrong thing."""
    reference = radial_profile(attenuate_folder(noise_folder, tmp_path / "ref", db=0.0))
    filtered = radial_profile(attenuate_folder(noise_folder, tmp_path / "f", db=6.0))

    low, mid, high = band_deviations(filtered, reference)
    assert abs(low) < 0.15, f"low band moved by {low:.3f} dB"
    assert abs(high) > abs(mid) > abs(low), "attenuation should increase with frequency"


def test_zero_db_still_round_trips_through_the_fft(tmp_path, noise_folder):
    """The 0 dB rung must carry the same processing noise as the others."""
    out = attenuate_folder(noise_folder, tmp_path / "z", db=0.0)
    assert len(list(out.glob("*.png"))) == 24


# ------------------------------------------------------------------ split


def test_split_returns_disjoint_halves(noise_folder):
    left, right = split_reference(noise_folder, count=8)
    assert len(left) == len(right) == 8
    assert not (set(left) & set(right)), "halves must not share images"


def test_split_interleaves_rather_than_slicing(noise_folder):
    """Contiguous files can share a class; a block split would confound the floor."""
    left, right = split_reference(noise_folder, count=4)
    assert left[0].name < right[0].name < left[1].name


def test_split_refuses_when_there_are_too_few_images(noise_folder):
    with pytest.raises(ValueError, match="need 200"):
        split_reference(noise_folder, count=100)


# ----------------------------------------------------------------- report


def test_table_locates_an_effect_on_the_curve():
    points = [
        SensitivityPoint(0.0, 0.0, {"fid": 2.0, "is_mean": 10.0, "kid_tf": 0.001}),
        SensitivityPoint(0.5, 0.48, {"fid": 2.1, "is_mean": 10.0, "kid_tf": 0.001}),
        SensitivityPoint(4.0, 3.9, {"fid": 22.0, "is_mean": 9.0, "kid_tf": 0.02}),
    ]
    table = format_sensitivity_table(points, reference_effect_db=0.5)
    assert "floor" in table
    assert "+0.100" in table, "delta against the floor should be shown"
    assert "0.48 dB rung" in table or "0.48" in table


def test_table_without_an_effect_marker_still_renders():
    points = [SensitivityPoint(0.0, 0.0, {"fid": 2.0})]
    assert "floor" in format_sensitivity_table(points)
