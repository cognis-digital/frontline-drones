"""Tests for the acoustic BPF detector / library matcher."""

from __future__ import annotations

import pytest

from frontline_drones import acoustic


def test_bpf_basic():
    # 6000 RPM, 2 blades => 100 rev/s * 2 = 200 Hz.
    assert acoustic.blade_passing_frequency(6000, 2) == pytest.approx(200.0)


def test_bpf_four_blades():
    assert acoustic.blade_passing_frequency(3000, 4) == pytest.approx(200.0)


def test_bpf_zero_rpm():
    assert acoustic.blade_passing_frequency(0, 3) == 0.0


def test_bpf_rejects_bad_blade_count():
    with pytest.raises(ValueError):
        acoustic.blade_passing_frequency(3000, 0)


def test_bpf_rejects_negative_rpm():
    with pytest.raises(ValueError):
        acoustic.blade_passing_frequency(-100, 2)


def test_rpm_from_bpf_inverts():
    rpm = acoustic.rpm_from_bpf(200.0, 2)
    assert rpm == pytest.approx(6000.0)


def test_rpm_bpf_roundtrip():
    for rpm in (1200, 4800, 9000):
        bpf = acoustic.blade_passing_frequency(rpm, 3)
        assert acoustic.rpm_from_bpf(bpf, 3) == pytest.approx(rpm)


def test_harmonics_default_count():
    h = acoustic.harmonics(100.0)
    assert h == [200.0, 300.0, 400.0, 500.0, 600.0]


def test_harmonics_custom_count():
    assert acoustic.harmonics(50.0, 2) == [100.0, 150.0]


def test_harmonics_zero_count():
    assert acoustic.harmonics(100.0, 0) == []


def test_harmonics_negative_raises():
    with pytest.raises(ValueError):
        acoustic.harmonics(100.0, -1)


def test_wavelength():
    assert acoustic.wavelength_m(343.0) == pytest.approx(1.0)


def test_wavelength_rejects_nonpositive():
    with pytest.raises(ValueError):
        acoustic.wavelength_m(0.0)


def test_parse_range_two_numbers():
    assert acoustic.parse_hz_range("~110-170") == (110.0, 170.0)


def test_parse_range_single_number():
    assert acoustic.parse_hz_range("90") == (90.0, 90.0)


def test_parse_range_none_for_empty():
    assert acoustic.parse_hz_range("") is None
    assert acoustic.parse_hz_range("n/a") is None


def test_parse_range_reorders():
    assert acoustic.parse_hz_range("200-100") == (100.0, 200.0)


def test_parse_range_en_dash():
    assert acoustic.parse_hz_range("110–170") == (110.0, 170.0)


def test_match_in_band_scores_one():
    cands = acoustic.match_bpf(130.0)
    assert cands
    assert cands[0].score == pytest.approx(1.0)
    assert cands[0].in_band is True


def test_match_sorted_descending():
    cands = acoustic.match_bpf(130.0)
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)


def test_match_limit_respected():
    cands = acoustic.match_bpf(130.0, limit=2)
    assert len(cands) <= 2


def test_match_rejects_negative():
    with pytest.raises(ValueError):
        acoustic.match_bpf(-5.0)


def test_match_far_value_low_or_empty():
    # Way above any library band -> either empty or low scores, never in-band.
    cands = acoustic.match_bpf(5000.0)
    assert all(not c.in_band for c in cands)


def test_match_candidate_to_dict():
    c = acoustic.match_bpf(130.0)[0]
    d = c.to_dict()
    assert set(d) == {"platform", "name", "score", "in_band", "library_range_hz", "source_url"}
    assert isinstance(d["library_range_hz"], list)


def test_detect_estimates_rpm():
    det = acoustic.detect(200.0, blade_count=2)
    assert det.estimated_rpm == pytest.approx(6000.0)


def test_detect_no_blade_count_no_rpm():
    det = acoustic.detect(130.0)
    assert det.estimated_rpm is None


def test_detect_has_best_and_advisory():
    det = acoustic.detect(130.0)
    assert det.best is not None
    assert "300-500" in det.advisory or "short-range" in det.advisory


def test_detect_to_dict():
    d = acoustic.detect(130.0, blade_count=2).to_dict()
    assert d["best"] is not None
    assert isinstance(d["candidates"], list)
    assert d["estimated_rpm"] == pytest.approx(3900.0)


def test_detect_matches_shahed_low_bpf():
    # Shahed-136 single pusher prop sits at a low fundamental (~90-110 Hz).
    det = acoustic.detect(100.0)
    platforms = [c.platform for c in det.candidates]
    assert "shahed-136" in platforms


def test_detect_deterministic():
    assert acoustic.detect(130.0).to_dict() == acoustic.detect(130.0).to_dict()


def test_custom_data_dir(tmp_path):
    csv = tmp_path / "acoustic-signatures.csv"
    csv.write_text(
        "id,name,platform,bpf_fundamental_hz,source_url\n"
        "x1,Test One,x1,~100-120,http://example.test\n",
        encoding="utf-8",
    )
    cands = acoustic.match_bpf(110.0, data_dir=str(tmp_path))
    assert len(cands) == 1 and cands[0].platform == "x1"
