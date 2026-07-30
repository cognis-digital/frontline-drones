"""Tests for the Remote ID plausibility / spoof-flag checks."""

from __future__ import annotations

import pytest

from frontline_drones import remoteid_plausibility as rp
from frontline_drones.remoteid import decode_message, pack_location


def test_clean_location_is_plausible():
    fields = {
        "latitude": 37.4419, "longitude": -122.1430,
        "geodetic_altitude_m": 80.0, "direction_deg": 90.0,
        "speed_mps": 12.0, "ua_type": "multirotor",
    }
    rep = rp.check_location_fields(fields)
    assert rep.plausible is True
    assert rep.score == pytest.approx(1.0)


def test_null_island_is_suspect():
    rep = rp.check_location_fields({"latitude": 0.0, "longitude": 0.0})
    assert rep.plausible is False
    assert any(f.code == "null_island" for f in rep.flags)


def test_coords_out_of_range_suspect():
    rep = rp.check_location_fields({"latitude": 200.0, "longitude": 0.0})
    assert rep.plausible is False
    assert any(f.code == "coords_out_of_range" for f in rep.flags)


def test_altitude_implausible_warns():
    rep = rp.check_location_fields(
        {"latitude": 37.0, "longitude": -122.0, "geodetic_altitude_m": 50000.0}
    )
    assert any(f.code == "altitude_implausible" for f in rep.flags)


def test_altitude_within_window_ok():
    rep = rp.check_location_fields(
        {"latitude": 37.0, "longitude": -122.0, "geodetic_altitude_m": 100.0}
    )
    assert all(f.code != "altitude_implausible" for f in rep.flags)


def test_direction_out_of_range():
    rep = rp.check_location_fields(
        {"latitude": 37.0, "longitude": -122.0, "direction_deg": 400.0}
    )
    assert any(f.code == "direction_out_of_range" for f in rep.flags)


def test_negative_speed_suspect():
    rep = rp.check_location_fields(
        {"latitude": 37.0, "longitude": -122.0, "speed_mps": -5.0}
    )
    assert rep.plausible is False
    assert any(f.code == "negative_speed" for f in rep.flags)


def test_speed_exceeds_multirotor_envelope():
    rep = rp.check_location_fields(
        {"latitude": 37.0, "longitude": -122.0, "speed_mps": 80.0, "ua_type": "multirotor"}
    )
    assert any(f.code == "speed_exceeds_envelope" for f in rep.flags)


def test_speed_ok_for_aeroplane():
    rep = rp.check_location_fields(
        {"latitude": 37.0, "longitude": -122.0, "speed_mps": 80.0, "ua_type": "aeroplane"}
    )
    assert all(f.code != "speed_exceeds_envelope" for f in rep.flags)


def test_unknown_ua_type_uses_default_envelope():
    # 200 m/s exceeds default 120 envelope.
    rep = rp.check_location_fields(
        {"latitude": 37.0, "longitude": -122.0, "speed_mps": 200.0, "ua_type": "mystery"}
    )
    assert any(f.code == "speed_exceeds_envelope" for f in rep.flags)


def test_airborne_but_static_info():
    rep = rp.check_location_fields(
        {"latitude": 37.0, "longitude": -122.0, "speed_mps": 0.0,
         "operational_status": "airborne", "height_m": 0.0}
    )
    assert any(f.code == "airborne_but_static" for f in rep.flags)


def test_info_flag_stays_plausible():
    rep = rp.check_location_fields(
        {"latitude": 37.0, "longitude": -122.0, "speed_mps": 0.0,
         "operational_status": "airborne", "height_m": 0.0}
    )
    # Only an info flag -> still "plausible" (no suspect).
    assert rep.plausible is True


def test_report_always_carries_caveat():
    rep = rp.check_location_fields({"latitude": 37.0, "longitude": -122.0})
    assert "spoofable" in rep.caveat
    assert "NOT authenticate" in rep.caveat


def test_report_to_dict():
    d = rp.check_location_fields({"latitude": 0.0, "longitude": 0.0}).to_dict()
    assert set(d) == {"plausible", "score", "flags", "caveat"}
    assert isinstance(d["flags"], list)


def test_flag_to_dict():
    rep = rp.check_location_fields({"latitude": 0.0, "longitude": 0.0})
    fd = rep.flags[0].to_dict()
    assert set(fd) == {"code", "severity", "detail"}


def test_score_bounded():
    rep = rp.check_location_fields(
        {"latitude": 200.0, "longitude": 400.0, "speed_mps": -5.0}
    )
    assert 0.0 <= rep.score <= 1.0


def test_decoded_null_island_message():
    msg = decode_message(pack_location(0.0, 0.0))
    rep = rp.check_location_fields(msg.fields)
    assert any(f.code == "null_island" for f in rep.flags)


def test_decoded_valid_message_plausible():
    msg = decode_message(pack_location(37.4, -122.1, speed_mps=10.0, direction_deg=90))
    rep = rp.check_location_fields(msg.fields)
    assert rep.plausible is True


def test_track_teleport_flagged():
    fixes = [
        {"t": 0.0, "latitude": 37.0, "longitude": -122.0},
        {"t": 1.0, "latitude": 38.0, "longitude": -122.0},  # ~111 km in 1 s
    ]
    rep = rp.check_track(fixes, ua_type="multirotor")
    assert rep.plausible is False
    assert any(f.code == "teleport" for f in rep.flags)


def test_track_normal_movement_ok():
    fixes = [
        {"t": 0.0, "latitude": 37.0000, "longitude": -122.0000},
        {"t": 1.0, "latitude": 37.0001, "longitude": -122.0000},  # ~11 m in 1 s
        {"t": 2.0, "latitude": 37.0002, "longitude": -122.0000},
    ]
    rep = rp.check_track(fixes, ua_type="multirotor")
    assert rep.plausible is True
    assert all(f.code != "teleport" for f in rep.flags)


def test_track_nonmonotonic_time():
    # Two fixes at the same timestamp -> zero dt is flagged non-monotonic.
    dup = [
        {"t": 1.0, "latitude": 37.0, "longitude": -122.0},
        {"t": 1.0, "latitude": 37.0, "longitude": -122.0},
    ]
    rep = rp.check_track(dup, ua_type="multirotor")
    assert any(f.code == "nonmonotonic_time" for f in rep.flags)


def test_track_propagates_perfix_flags():
    fixes = [
        {"t": 0.0, "latitude": 0.0, "longitude": 0.0},  # null island
        {"t": 1.0, "latitude": 0.0, "longitude": 0.0},
    ]
    rep = rp.check_track(fixes, ua_type="multirotor")
    assert any(f.code == "null_island" for f in rep.flags)


def test_track_missing_coords_skips_teleport():
    fixes = [
        {"t": 0.0, "latitude": None, "longitude": None},
        {"t": 1.0, "latitude": 37.0, "longitude": -122.0},
    ]
    rep = rp.check_track(fixes, ua_type="multirotor")
    assert all(f.code != "teleport" for f in rep.flags)


def test_track_empty():
    rep = rp.check_track([], ua_type="multirotor")
    assert rep.plausible is True
    assert rep.score == pytest.approx(1.0)


def test_speed_envelope_lookup():
    assert rp._speed_envelope("multirotor") == rp.UA_TYPE_MAX_SPEED_MPS["multirotor"]
    assert rp._speed_envelope(None) == rp.DEFAULT_MAX_SPEED_MPS
    assert rp._speed_envelope(123) == rp.DEFAULT_MAX_SPEED_MPS


def test_as_float_helpers():
    assert rp._as_float(None) is None
    assert rp._as_float(True) is None
    assert rp._as_float("3.5") == 3.5
    assert rp._as_float("bad") is None
    assert rp._as_float(7) == 7.0
