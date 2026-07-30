"""Tests for the radar micro-Doppler feature helpers and bird discriminant."""

from __future__ import annotations

import pytest

from frontline_drones import microdoppler as md


def test_wavelength_x_band():
    # ~9.5 GHz => ~3.16 cm.
    assert md.wavelength_m(9.5) == pytest.approx(0.03156, rel=1e-2)


def test_wavelength_rejects_nonpositive():
    with pytest.raises(ValueError):
        md.wavelength_m(0.0)


def test_wavelength_inverse_with_frequency():
    # Doubling frequency halves wavelength.
    assert md.wavelength_m(20.0) == pytest.approx(md.wavelength_m(10.0) / 2.0)


def test_band_wavelength_known():
    assert md.band_wavelength_m("X") == pytest.approx(md.wavelength_m(9.5))


def test_band_wavelength_case_insensitive():
    assert md.band_wavelength_m("ku") == md.band_wavelength_m("Ku")


def test_band_wavelength_unknown_raises():
    with pytest.raises(KeyError):
        md.band_wavelength_m("Z")


def test_doppler_shift_sign_and_magnitude():
    # Positive radial velocity (closing) gives positive shift.
    shift = md.doppler_shift_hz(10.0, 9.5)
    assert shift > 0


def test_doppler_shift_scales_with_velocity():
    assert md.doppler_shift_hz(20.0, 9.5) == pytest.approx(2 * md.doppler_shift_hz(10.0, 9.5))


def test_doppler_zero_velocity():
    assert md.doppler_shift_hz(0.0, 9.5) == 0.0


def test_blade_tip_velocity():
    # 2*pi*r*rpm/60. r=0.12, rpm=6000 => ~75.4 m/s.
    v = md.blade_tip_velocity_mps(6000, 0.12)
    assert v == pytest.approx(75.398, rel=1e-3)


def test_blade_tip_velocity_rejects_bad_radius():
    with pytest.raises(ValueError):
        md.blade_tip_velocity_mps(6000, 0.0)


def test_blade_flash_rate():
    assert md.blade_flash_rate_hz(6000, 2) == pytest.approx(200.0)


def test_blade_flash_rate_rejects_zero_blades():
    with pytest.raises(ValueError):
        md.blade_flash_rate_hz(6000, 0)


def test_micro_doppler_bandwidth_positive():
    v = md.blade_tip_velocity_mps(6000, 0.12)
    bw = md.micro_doppler_bandwidth_hz(v, 9.5)
    assert bw > 0


def test_micro_doppler_bandwidth_is_double_shift():
    assert md.micro_doppler_bandwidth_hz(50.0, 9.5) == pytest.approx(
        2 * md.doppler_shift_hz(50.0, 9.5)
    )


def test_discriminate_drone_high_rate_symmetric():
    a = md.discriminate_bird(flash_rate_hz=200.0, symmetric_sidebands=True)
    assert a.classification == "drone_like"
    assert a.drone_likelihood >= 0.6
    assert a.symmetric is True


def test_discriminate_bird_low_rate_asymmetric():
    a = md.discriminate_bird(flash_rate_hz=5.0, symmetric_sidebands=False)
    assert a.classification == "bird_like"
    assert a.drone_likelihood <= 0.35


def test_discriminate_likelihood_in_unit_interval():
    a = md.discriminate_bird(flash_rate_hz=50.0, symmetric_sidebands=True)
    assert 0.0 <= a.drone_likelihood <= 1.0


def test_discriminate_symmetry_clamped():
    a = md.discriminate_bird(
        flash_rate_hz=100.0, symmetric_sidebands=True, spectrum_symmetry=5.0
    )
    assert 0.0 <= a.drone_likelihood <= 1.0


def test_discriminate_rejects_negative_rate():
    with pytest.raises(ValueError):
        md.discriminate_bird(flash_rate_hz=-1.0, symmetric_sidebands=True)


def test_discriminate_has_cues():
    a = md.discriminate_bird(flash_rate_hz=200.0, symmetric_sidebands=True)
    assert len(a.cues) >= 2


def test_discriminate_higher_rate_more_dronelike():
    low = md.discriminate_bird(flash_rate_hz=15.0, symmetric_sidebands=True)
    high = md.discriminate_bird(flash_rate_hz=300.0, symmetric_sidebands=True)
    assert high.drone_likelihood >= low.drone_likelihood


def test_discriminate_to_dict():
    d = md.discriminate_bird(flash_rate_hz=200.0, symmetric_sidebands=True).to_dict()
    assert set(d) == {
        "drone_likelihood", "classification", "symmetric", "flash_rate_hz", "cues"
    }


def test_discriminate_symmetric_beats_asymmetric():
    sym = md.discriminate_bird(flash_rate_hz=100.0, symmetric_sidebands=True)
    asym = md.discriminate_bird(flash_rate_hz=100.0, symmetric_sidebands=False)
    assert sym.drone_likelihood >= asym.drone_likelihood


def test_recommended_bands_covers_library():
    advice = md.recommended_bands()
    assert advice
    platforms = {a.platform for a in advice}
    assert "dji-mavic-3" in platforms


def test_recommended_bands_have_band_and_discriminant():
    advice = md.recommended_bands()
    for a in advice:
        assert a.recommended_band != ""
        assert a.bird_discriminant != ""


def test_recommended_bands_to_dict():
    d = md.recommended_bands()[0].to_dict()
    assert set(d) == {
        "platform", "name", "recommended_band", "bird_discriminant", "source_url"
    }


def test_bird_ceiling_constant_reasonable():
    assert 5.0 <= md.BIRD_MAX_FLAP_HZ <= 20.0


def test_all_bands_have_positive_wavelength():
    for band in md.RADAR_BANDS_GHZ:
        assert md.band_wavelength_m(band) > 0


def test_discriminate_deterministic():
    a = md.discriminate_bird(flash_rate_hz=123.0, symmetric_sidebands=True).to_dict()
    b = md.discriminate_bird(flash_rate_hz=123.0, symmetric_sidebands=True).to_dict()
    assert a == b
