"""Tests for the geospatial-awareness primitives."""

from __future__ import annotations

import pytest

from frontline_drones import geo


def test_haversine_zero_distance():
    assert geo.haversine_m(37.0, -122.0, 37.0, -122.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_one_degree_latitude():
    # ~111.2 km per degree of latitude.
    d = geo.haversine_m(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111_195, rel=0.001)


def test_haversine_symmetric():
    a = geo.haversine_m(51.5, -0.1, 48.85, 2.35)
    b = geo.haversine_m(48.85, 2.35, 51.5, -0.1)
    assert a == pytest.approx(b)


def test_haversine_london_paris_known():
    # London to Paris is ~343 km.
    d = geo.haversine_m(51.5074, -0.1278, 48.8566, 2.3522)
    assert d == pytest.approx(343_000, rel=0.02)


def test_bearing_due_north():
    assert geo.initial_bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_bearing_due_east():
    assert geo.initial_bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0, abs=1e-3)


def test_bearing_due_south():
    assert geo.initial_bearing_deg(1.0, 0.0, 0.0, 0.0) == pytest.approx(180.0, abs=1e-6)


def test_bearing_in_range():
    b = geo.initial_bearing_deg(10.0, 20.0, -5.0, -15.0)
    assert 0.0 <= b < 360.0


def test_destination_point_roundtrips_with_haversine():
    lat, lon = geo.destination_point(37.0, -122.0, 90.0, 1000.0)
    assert geo.haversine_m(37.0, -122.0, lat, lon) == pytest.approx(1000.0, rel=1e-3)


def test_destination_point_north_increases_latitude():
    lat, lon = geo.destination_point(0.0, 0.0, 0.0, 111_000.0)
    assert lat > 0.99
    assert lon == pytest.approx(0.0, abs=1e-6)


def test_enu_offset_east_positive_for_greater_lon():
    east, north = geo.enu_offset_m(37.0, -121.99, 37.0, -122.0)
    assert east > 0
    assert north == pytest.approx(0.0, abs=1e-6)


def test_enu_offset_north_positive_for_greater_lat():
    east, north = geo.enu_offset_m(37.01, -122.0, 37.0, -122.0)
    assert north > 0
    assert east == pytest.approx(0.0, abs=1e-6)


def test_enu_offset_zero_at_origin():
    assert geo.enu_offset_m(37.0, -122.0, 37.0, -122.0) == (0.0, 0.0)


def test_slant_range():
    assert geo.slant_range_m(300.0, 400.0) == pytest.approx(500.0)


def test_elevation_angle_45():
    assert geo.elevation_angle_deg(100.0, 100.0) == pytest.approx(45.0)


def test_elevation_angle_overhead():
    assert geo.elevation_angle_deg(0.0, 100.0) == pytest.approx(90.0)


def test_elevation_angle_ground_zero():
    assert geo.elevation_angle_deg(0.0, 0.0) == pytest.approx(0.0)


def test_protected_zone_to_dict():
    z = geo.ProtectedZone("base", 37.0, -122.0, 500.0, "test base")
    d = z.to_dict()
    assert d["name"] == "base" and d["radius_m"] == 500.0 and d["label"] == "test base"


def test_zone_status_inside():
    z = geo.ProtectedZone("base", 37.0, -122.0, 500.0)
    st = geo.zone_status(z, 37.001, -122.0)  # ~111 m north
    assert st.inside is True
    assert st.range_to_edge_m < 0


def test_zone_status_outside():
    z = geo.ProtectedZone("base", 37.0, -122.0, 50.0)
    st = geo.zone_status(z, 37.001, -122.0)  # ~111 m north
    assert st.inside is False
    assert st.range_to_edge_m > 0


def test_zone_status_bearing_north():
    z = geo.ProtectedZone("base", 37.0, -122.0, 500.0)
    st = geo.zone_status(z, 37.001, -122.0)
    assert st.bearing_from_center_deg == pytest.approx(0.0, abs=1e-3)


def test_zone_status_to_dict_keys():
    z = geo.ProtectedZone("base", 0.0, 0.0, 100.0)
    d = geo.zone_status(z, 0.0, 0.0).to_dict()
    assert set(d) == {"zone", "range_m", "inside", "range_to_edge_m", "bearing_from_center_deg"}


def test_cpa_head_on_reaches_zero():
    # Moving straight toward the asset from 100 m east at 10 m/s west.
    cpa = geo.closest_point_of_approach((100.0, 0.0, 0.0), (-10.0, 0.0, 0.0))
    assert cpa.cpa_distance_m == pytest.approx(0.0, abs=1e-9)
    assert cpa.time_to_cpa_s == pytest.approx(10.0)
    assert cpa.approaching is True


def test_cpa_offset_pass():
    # Passing 50 m north of the asset, moving west.
    cpa = geo.closest_point_of_approach((100.0, 50.0, 0.0), (-10.0, 0.0, 0.0))
    assert cpa.cpa_distance_m == pytest.approx(50.0, abs=1e-6)
    assert cpa.approaching is True


def test_cpa_receding_clamps_to_now():
    # Already moving away.
    cpa = geo.closest_point_of_approach((100.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert cpa.approaching is False
    assert cpa.time_to_cpa_s == 0.0
    assert cpa.current_distance_m == pytest.approx(100.0)


def test_cpa_stationary():
    cpa = geo.closest_point_of_approach((30.0, 40.0, 0.0), (0.0, 0.0, 0.0))
    assert cpa.cpa_distance_m == pytest.approx(50.0)
    assert cpa.time_to_cpa_s == 0.0
    assert cpa.approaching is False


def test_cpa_with_asset_offset():
    cpa = geo.closest_point_of_approach((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert cpa.cpa_distance_m == pytest.approx(0.0, abs=1e-9)
    assert cpa.time_to_cpa_s == pytest.approx(10.0)


def test_cpa_to_dict():
    d = geo.closest_point_of_approach((100.0, 0.0, 0.0), (-10.0, 0.0, 0.0)).to_dict()
    assert set(d) == {"cpa_distance_m", "time_to_cpa_s", "approaching", "current_distance_m"}


def test_approach_advisory_inside():
    z = geo.ProtectedZone("base", 0.0, 0.0, 500.0)
    adv = geo.approach_advisory(z, (100.0, 0.0, 10.0), (0.0, 0.0, 0.0))
    assert adv.inside is True
    assert adv.breaches_zone is True
    assert "INSIDE" in adv.advisories[0]


def test_approach_advisory_projected_breach():
    z = geo.ProtectedZone("base", 0.0, 0.0, 500.0)
    adv = geo.approach_advisory(z, (2000.0, 0.0, 50.0), (-100.0, 0.0, 0.0))
    assert adv.inside is False
    assert adv.breaches_zone is True
    assert any("breach" in a for a in adv.advisories)


def test_approach_advisory_stays_clear():
    z = geo.ProtectedZone("base", 0.0, 0.0, 100.0)
    adv = geo.approach_advisory(z, (2000.0, 1000.0, 50.0), (-100.0, 0.0, 0.0))
    assert adv.breaches_zone is False


def test_approach_advisory_opening():
    z = geo.ProtectedZone("base", 0.0, 0.0, 100.0)
    adv = geo.approach_advisory(z, (2000.0, 0.0, 50.0), (100.0, 0.0, 0.0))
    assert adv.breaches_zone is False
    assert any("opening" in a for a in adv.advisories)


def test_approach_advisory_to_dict():
    z = geo.ProtectedZone("base", 0.0, 0.0, 100.0)
    d = geo.approach_advisory(z, (200.0, 0.0, 0.0), (-10.0, 0.0, 0.0)).to_dict()
    assert d["zone"] == "base"
    assert "advisories" in d and isinstance(d["advisories"], list)
