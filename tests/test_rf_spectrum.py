"""Tests for passive RF spectrum occupancy and FHSS burst analysis."""

from __future__ import annotations

import pytest

from frontline_drones.rf_spectrum import (
    Burst,
    SpectrumSummary,
    analyze_spectrum,
    band_of,
    channel_histogram,
)

# -- band_of ------------------------------------------------------------------

def test_band_of_24():
    assert band_of(2440.0) == "2.4 GHz"


def test_band_of_58():
    assert band_of(5800.0) == "5.8 GHz"


def test_band_of_900():
    assert band_of(915.0) == "900 MHz"


def test_band_of_433():
    assert band_of(433.0) == "433 MHz"


def test_band_of_12ghz():
    assert band_of(1280.0) == "1.2 GHz"


def test_band_of_52ghz():
    assert band_of(5200.0) == "5.2 GHz"


def test_band_of_out_of_plan():
    assert band_of(100.0) is None
    assert band_of(3500.0) is None


def test_burst_band_property():
    assert Burst(2450.0, 0.0, 5.0).band == "2.4 GHz"


# -- empty / degenerate -------------------------------------------------------

def test_empty_summary():
    s = analyze_spectrum([])
    assert isinstance(s, SpectrumSummary)
    assert s.num_bursts == 0
    assert s.hopping == "unknown"
    assert s.bands == ()


def test_bad_channel_khz_raises():
    with pytest.raises(ValueError):
        analyze_spectrum([(2440, 0, 5)], channel_khz=0)


# -- fixed link ---------------------------------------------------------------

def test_single_channel_is_fixed():
    bursts = [(2440.0, i * 0.1, 50.0) for i in range(10)]
    s = analyze_spectrum(bursts)
    assert s.distinct_channels == 1
    assert s.hopping == "fixed"
    assert s.hop_rate_hz == 0.0


def test_fixed_link_band():
    bursts = [(5800.0, i * 0.1, 20.0) for i in range(5)]
    s = analyze_spectrum(bursts)
    assert s.bands == ("5.8 GHz",)
    assert s.classify_bands == ("5.8 GHz",)


# -- FHSS ---------------------------------------------------------------------

def test_frequency_hopping_detected():
    # Many channels across the 2.4 GHz band, short dwell, fast hops.
    bursts = [(2400.0 + (i * 7 % 80), i * 0.002, 1.0) for i in range(40)]
    s = analyze_spectrum(bursts)
    assert s.hopping == "fhss"
    assert s.distinct_channels > 5
    assert s.hop_rate_hz > 0


def test_fhss_confidence_high():
    bursts = [(2400.0 + (i * 11 % 80), i * 0.002, 0.5) for i in range(50)]
    s = analyze_spectrum(bursts)
    assert s.hopping == "fhss"
    assert s.hopping_confidence > 0.5


def test_wide_span_reported():
    bursts = [(2405.0, 0.0, 1.0), (2480.0, 0.01, 1.0)]
    s = analyze_spectrum(bursts)
    assert s.freq_span_mhz == pytest.approx(75.0)


# -- dwell / duty / hop-rate --------------------------------------------------

def test_mean_dwell():
    bursts = [(2440.0, 0.0, 10.0), (2440.0, 1.0, 20.0), (2440.0, 2.0, 30.0)]
    s = analyze_spectrum(bursts)
    assert s.mean_dwell_ms == pytest.approx(20.0)
    assert s.min_dwell_ms == 10.0
    assert s.max_dwell_ms == 30.0


def test_hop_rate_counts_changes():
    # Alternate two channels every 0.1 s over ~0.5 s => several changes.
    bursts = []
    for i in range(6):
        freq = 2410.0 if i % 2 == 0 else 2450.0
        bursts.append((freq, i * 0.1, 5.0))
    s = analyze_spectrum(bursts)
    assert s.hop_rate_hz > 0


def test_duty_cycle_bounded():
    bursts = [(2440.0, 0.0, 100.0), (2440.0, 0.05, 100.0)]  # overlapping dwell
    s = analyze_spectrum(bursts)
    assert 0.0 <= s.duty_cycle <= 1.0


def test_duty_cycle_low_for_sparse():
    bursts = [(2440.0, 0.0, 1.0), (2440.0, 10.0, 1.0)]
    s = analyze_spectrum(bursts)
    assert s.duty_cycle < 0.01


def test_window_span():
    bursts = [(2440.0, 0.0, 100.0), (2440.0, 1.0, 500.0)]
    s = analyze_spectrum(bursts)
    # ends at 1.0 + 0.5 = 1.5 s, starts at 0 -> window 1.5 s.
    assert s.window_s == pytest.approx(1.5)


# -- channelisation -----------------------------------------------------------

def test_channel_granularity_merges():
    # Two freqs 0.5 MHz apart collapse to one 1 MHz channel.
    bursts = [(2440.0, 0.0, 5.0), (2440.4, 0.1, 5.0)]
    s = analyze_spectrum(bursts, channel_khz=1000.0)
    assert s.distinct_channels == 1


def test_channel_granularity_separates():
    bursts = [(2440.0, 0.0, 5.0), (2445.0, 0.1, 5.0)]
    s = analyze_spectrum(bursts, channel_khz=1000.0)
    assert s.distinct_channels == 2


# -- channel_histogram --------------------------------------------------------

def test_histogram_counts():
    bursts = [(2440.0, 0.0, 5.0), (2440.0, 0.1, 5.0), (2450.0, 0.2, 5.0)]
    hist = channel_histogram(bursts)
    assert hist["2440 MHz"] == 2
    assert hist["2450 MHz"] == 1


def test_histogram_sorted_by_frequency():
    bursts = [(2470.0, 0.0, 5.0), (2410.0, 0.1, 5.0), (2440.0, 0.2, 5.0)]
    hist = channel_histogram(bursts)
    keys = list(hist)
    assert keys == sorted(keys, key=lambda k: float(k.split()[0]))


def test_histogram_empty():
    assert channel_histogram([]) == {}


# -- input coercion -----------------------------------------------------------

def test_accepts_burst_objects():
    s = analyze_spectrum([Burst(2440.0, 0.0, 5.0), Burst(2440.0, 0.1, 5.0)])
    assert s.num_bursts == 2


def test_accepts_dicts():
    bursts = [{"center_freq_mhz": 2440.0, "start_s": 0.0, "duration_ms": 5.0},
              {"freq_mhz": 2440.0, "start_s": 0.1, "duration_ms": 5.0}]
    s = analyze_spectrum(bursts)
    assert s.num_bursts == 2
    assert s.bands == ("2.4 GHz",)


def test_accepts_tuples_with_bandwidth_power():
    s = analyze_spectrum([(2440.0, 0.0, 5.0, 20.0, -50.0)])
    assert s.num_bursts == 1


# -- multi-band ---------------------------------------------------------------

def test_dual_band_operation():
    bursts = [(2440.0, 0.0, 5.0), (5800.0, 0.1, 5.0)]
    s = analyze_spectrum(bursts)
    assert set(s.bands) == {"2.4 GHz", "5.8 GHz"}


def test_out_of_plan_freq_no_band():
    s = analyze_spectrum([(100.0, 0.0, 5.0)])
    assert s.bands == ()


# -- determinism / serialisation ----------------------------------------------

def test_deterministic():
    bursts = [(2400.0 + (i * 7 % 80), i * 0.002, 1.0) for i in range(30)]
    a = analyze_spectrum(bursts).to_dict()
    b = analyze_spectrum(bursts).to_dict()
    assert a == b


def test_to_dict_keys():
    s = analyze_spectrum([(2440.0, 0.0, 5.0)])
    d = s.to_dict()
    for k in ("num_bursts", "window_s", "bands", "distinct_channels", "freq_span_mhz",
              "mean_dwell_ms", "min_dwell_ms", "max_dwell_ms", "hop_rate_hz",
              "duty_cycle", "hopping", "hopping_confidence", "classify_bands"):
        assert k in d


def test_feeds_rf_classify_band_tokens():
    # classify_bands should be usable directly as rf_classify observation bands.
    from frontline_drones.rf_classify import RFObservation, classify
    bursts = [(2440.0 + (i * 5 % 60), i * 0.002, 1.0) for i in range(30)]
    s = analyze_spectrum(bursts)
    obs = RFObservation(bands=s.classify_bands, hopping=s.hopping)
    result = classify(obs)
    assert result.caveat  # returns a valid classification object
