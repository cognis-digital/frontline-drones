"""Acoustic beamforming: DoA estimation + self-noise rejection."""
import math

import pytest

from frontline_drones.beamforming import (
    BearingEstimate,
    MicArray,
    angular_spectrum,
    estimate_bearing,
    simulate_samples,
    uniform_circular_array,
)

FREQ = 2000.0


def _ang_diff(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def uca():
    return uniform_circular_array(8, 0.05)


# ---- array ----

def test_array_needs_two_mics():
    with pytest.raises(ValueError):
        MicArray(((0.0, 0.0),))


def test_uca_element_count():
    assert uca().n == 8


def test_uca_bad_params():
    with pytest.raises(ValueError):
        uniform_circular_array(1, 0.05)
    with pytest.raises(ValueError):
        uniform_circular_array(8, 0)


def test_steering_vector_length_and_unit_magnitude():
    sv = uca().steering_vector(45.0, FREQ)
    assert len(sv) == 8
    for v in sv:
        assert abs(v) == pytest.approx(1.0, abs=1e-9)


# ---- direction finding ----

@pytest.mark.parametrize("bearing", [0, 30, 45, 90, 135, 180, 225, 270, 315])
def test_single_source_bearing_recovered(bearing):
    arr = uca()
    samples = simulate_samples(arr, FREQ, [(bearing, 1.0)])
    est = estimate_bearing(arr, samples, FREQ)
    assert _ang_diff(est.bearing_deg, bearing) <= 8.0


def test_spectrum_peak_matches_estimate():
    arr = uca()
    samples = simulate_samples(arr, FREQ, [(120.0, 1.0)])
    spec = angular_spectrum(arr, samples, FREQ)
    peak = max(spec, key=lambda t: t[1])[0]
    est = estimate_bearing(arr, samples, FREQ)
    assert _ang_diff(peak, est.bearing_deg) <= 1.0


def test_bad_scan_step():
    arr = uca()
    samples = simulate_samples(arr, FREQ, [(90.0, 1.0)])
    with pytest.raises(ValueError):
        angular_spectrum(arr, samples, FREQ, scan_step_deg=0)


# ---- self-noise rejection (the SECTR problem) ----

def test_self_noise_rejected_recovers_target():
    arr = uca()
    # Weak target at 90, LOUD self-noise (own motors) at 250.
    samples = simulate_samples(arr, FREQ, [(90.0, 1.0), (250.0, 5.0)])
    # Without rejection, the loud self-noise wins.
    naive = estimate_bearing(arr, samples, FREQ)
    assert _ang_diff(naive.bearing_deg, 250.0) <= 12.0
    # With self-bearing nulled, the true target is recovered.
    est = estimate_bearing(arr, samples, FREQ, self_bearing_deg=250.0, guard_deg=25.0)
    assert est.rejected_self
    assert _ang_diff(est.bearing_deg, 90.0) <= 12.0


def test_guard_band_excludes_self_bearing():
    arr = uca()
    samples = simulate_samples(arr, FREQ, [(100.0, 1.0), (100.0, 3.0)])  # colocated
    est = estimate_bearing(arr, samples, FREQ, self_bearing_deg=100.0, guard_deg=20.0)
    assert _ang_diff(est.bearing_deg, 100.0) > 20.0 or est.rejected_self


def test_estimate_to_dict():
    arr = uca()
    samples = simulate_samples(arr, FREQ, [(45.0, 1.0)])
    d = estimate_bearing(arr, samples, FREQ).to_dict()
    for k in ("bearing_deg", "power", "rejected_self"):
        assert k in d


# ---- property sweeps ----

@pytest.mark.parametrize("n", [4, 6, 8, 12])
@pytest.mark.parametrize("bearing", [20, 90, 160, 240, 300])
def test_recovery_across_array_sizes(n, bearing):
    arr = uniform_circular_array(n, 0.05)
    samples = simulate_samples(arr, FREQ, [(bearing, 1.0)])
    est = estimate_bearing(arr, samples, FREQ)
    assert _ang_diff(est.bearing_deg, bearing) <= 12.0


@pytest.mark.parametrize("target,self_b", [
    (30, 200), (90, 300), (150, 20), (270, 100), (350, 170),
])
def test_self_rejection_various_geometries(target, self_b):
    arr = uca()
    samples = simulate_samples(arr, FREQ, [(target, 1.0), (self_b, 6.0)])
    est = estimate_bearing(arr, samples, FREQ, self_bearing_deg=self_b, guard_deg=25.0)
    assert _ang_diff(est.bearing_deg, target) <= 15.0


@pytest.mark.parametrize("power_ratio", [2, 4, 8, 16])
def test_louder_self_still_rejected(power_ratio):
    arr = uca()
    samples = simulate_samples(arr, FREQ, [(80.0, 1.0), (260.0, float(power_ratio))])
    est = estimate_bearing(arr, samples, FREQ, self_bearing_deg=260.0, guard_deg=25.0)
    assert _ang_diff(est.bearing_deg, 80.0) <= 15.0


@pytest.mark.parametrize("step", [0.5, 1.0, 2.0, 5.0])
def test_scan_step_still_recovers(step):
    arr = uca()
    samples = simulate_samples(arr, FREQ, [(135.0, 1.0)])
    est = estimate_bearing(arr, samples, FREQ, scan_step_deg=step)
    assert _ang_diff(est.bearing_deg, 135.0) <= 8.0 + step
