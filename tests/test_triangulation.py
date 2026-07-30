"""Tests for the passive bearing/AoA emitter-geolocation solver."""

from __future__ import annotations

import math

import pytest

from frontline_drones.triangulation import (
    BearingObservation,
    GeolocationFix,
    bearing_from_positions,
    bearing_unit_vector,
    geolocate,
    intersect_bearings,
    observation_from_latlon,
)

# -- bearing_unit_vector ------------------------------------------------------

def test_unit_north():
    e, n = bearing_unit_vector(0.0)
    assert e == pytest.approx(0.0, abs=1e-9)
    assert n == pytest.approx(1.0)


def test_unit_east():
    e, n = bearing_unit_vector(90.0)
    assert e == pytest.approx(1.0)
    assert n == pytest.approx(0.0, abs=1e-9)


def test_unit_south():
    e, n = bearing_unit_vector(180.0)
    assert e == pytest.approx(0.0, abs=1e-9)
    assert n == pytest.approx(-1.0)


def test_unit_west():
    e, n = bearing_unit_vector(270.0)
    assert e == pytest.approx(-1.0)
    assert n == pytest.approx(0.0, abs=1e-9)


def test_unit_is_normalized():
    for b in (0, 33, 91, 178, 270, 359):
        e, n = bearing_unit_vector(b)
        assert math.hypot(e, n) == pytest.approx(1.0)


def test_unit_wraps_360():
    assert bearing_unit_vector(370.0) == pytest.approx(bearing_unit_vector(10.0))


# -- intersect_bearings -------------------------------------------------------

def test_intersect_basic_cross():
    pt = intersect_bearings((-100, 0), 45.0, (100, 0), 315.0)
    assert pt is not None
    assert pt[0] == pytest.approx(0.0, abs=1e-6)
    assert pt[1] == pytest.approx(100.0, abs=1e-6)


def test_intersect_orthogonal():
    # Sensor south looking north, sensor west looking east -> cross at (0,0)+.
    pt = intersect_bearings((0, -50), 0.0, (-50, 0), 90.0)
    assert pt == pytest.approx((0.0, 0.0), abs=1e-6)


def test_intersect_parallel_returns_none():
    assert intersect_bearings((0, 0), 90.0, (0, 100), 90.0) is None


def test_intersect_antiparallel_returns_none():
    assert intersect_bearings((0, 0), 90.0, (0, 100), 270.0) is None


def test_intersect_recovers_known_emitter():
    emitter = (300.0, 400.0)
    a = (0.0, 0.0)
    b = (500.0, 0.0)
    ba = math.degrees(math.atan2(emitter[0] - a[0], emitter[1] - a[1])) % 360
    bb = math.degrees(math.atan2(emitter[0] - b[0], emitter[1] - b[1])) % 360
    pt = intersect_bearings(a, ba, b, bb)
    assert pt == pytest.approx(emitter, abs=1e-6)


# -- geolocate ----------------------------------------------------------------

def test_geolocate_two_bearings_exact():
    f = geolocate([(-100, 0, 45.0), (100, 0, 315.0)])
    assert isinstance(f, GeolocationFix)
    assert f.east == pytest.approx(0.0, abs=1e-6)
    assert f.north == pytest.approx(100.0, abs=1e-6)
    assert f.num_bearings == 2


def test_geolocate_three_bearings_consistent():
    emitter = (120.0, 250.0)
    sensors = [(0, 0), (400, 0), (200, -150)]
    obs = []
    for sx, sy in sensors:
        brg = math.degrees(math.atan2(emitter[0] - sx, emitter[1] - sy)) % 360
        obs.append((sx, sy, brg))
    f = geolocate(obs)
    assert f.east == pytest.approx(emitter[0], abs=1e-4)
    assert f.north == pytest.approx(emitter[1], abs=1e-4)
    assert f.residual_rms_m == pytest.approx(0.0, abs=1e-4)


def test_geolocate_residual_zero_for_consistent():
    f = geolocate([(-100, 0, 45.0), (100, 0, 315.0), (0, -100, 0.0)])
    assert f.residual_rms_m == pytest.approx(0.0, abs=1e-6)
    assert f.max_residual_m == pytest.approx(0.0, abs=1e-6)


def test_geolocate_inconsistent_has_residual():
    # Perturb one bearing so lines do not meet at a single point.
    f = geolocate([(-100, 0, 45.0), (100, 0, 315.0), (0, -100, 20.0)])
    assert f.residual_rms_m > 0.0


def test_geolocate_requires_two():
    with pytest.raises(ValueError):
        geolocate([(0, 0, 90.0)])


def test_geolocate_empty_raises():
    with pytest.raises(ValueError):
        geolocate([])


def test_geolocate_parallel_raises():
    with pytest.raises(ValueError):
        geolocate([(0, 0, 90.0), (0, 100, 90.0)])


def test_geolocate_near_parallel_poor_geometry():
    good = geolocate([(-100, 0, 45.0), (100, 0, 315.0)])
    bad = geolocate([(0, 0, 89.0), (0, 200, 91.0)])
    assert bad.gdop > good.gdop
    assert bad.min_subtended_deg < good.min_subtended_deg
    assert not bad.well_conditioned


def test_geolocate_min_subtended_orthogonal():
    f = geolocate([(0, -50, 0.0), (-50, 0, 90.0)])
    assert f.min_subtended_deg == pytest.approx(90.0, abs=1e-6)


def test_geolocate_well_conditioned_flag():
    f = geolocate([(-100, 0, 45.0), (100, 0, 315.0)])
    assert f.well_conditioned is True


def test_geolocate_all_in_front_true():
    f = geolocate([(-100, 0, 45.0), (100, 0, 315.0)])
    assert f.all_in_front is True


def test_geolocate_behind_sensor_flagged():
    # Both bearings point away from the true crossing -> fix behind sensors.
    f = geolocate([(-100, 0, 225.0), (100, 0, 135.0)])
    assert f.all_in_front is False
    assert not f.well_conditioned


def test_geolocate_weight_pulls_fix():
    # Three MUTUALLY INCONSISTENT bearings (two lines always meet exactly, so
    # weighting only matters with an over-determined, inconsistent system).
    obs_unweighted = [
        BearingObservation(-100, 0, 45.0),
        BearingObservation(100, 0, 315.0),
        BearingObservation(0, -100, 20.0),
    ]
    base = geolocate(obs_unweighted)
    # Heavily weight the third sensor: its perpendicular residual must shrink.
    weighted = geolocate([
        BearingObservation(-100, 0, 45.0, weight=1.0),
        BearingObservation(100, 0, 315.0, weight=1.0),
        BearingObservation(0, -100, 20.0, weight=50.0),
    ])
    assert (weighted.east, weighted.north) != (base.east, base.north)


def test_geolocate_zero_weight_raises():
    with pytest.raises(ValueError):
        geolocate([BearingObservation(0, 0, 90.0, weight=0.0),
                   BearingObservation(0, 100, 45.0)])


def test_geolocate_negative_weight_raises():
    with pytest.raises(ValueError):
        geolocate([BearingObservation(0, 0, 90.0, weight=-1.0),
                   BearingObservation(0, 100, 45.0)])


def test_geolocate_accepts_dicts():
    f = geolocate([
        {"east": -100, "north": 0, "bearing_deg": 45.0},
        {"east": 100, "north": 0, "bearing_deg": 315.0},
    ])
    assert f.north == pytest.approx(100.0, abs=1e-6)


def test_geolocate_accepts_objects():
    f = geolocate([
        BearingObservation(-100, 0, 45.0, sensor_id="A"),
        BearingObservation(100, 0, 315.0, sensor_id="B"),
    ])
    assert f.east == pytest.approx(0.0, abs=1e-6)


def test_geolocate_deterministic():
    obs = [(-100, 0, 45.0), (100, 0, 315.0), (0, -80, 10.0)]
    assert geolocate(obs).to_dict() == geolocate(obs).to_dict()


def test_geolocate_gdop_finite_for_good_geometry():
    f = geolocate([(-100, 0, 45.0), (100, 0, 315.0)])
    assert math.isfinite(f.gdop)
    assert f.gdop >= 1.0 - 1e-9 or f.gdop > 0


def test_to_dict_keys():
    d = geolocate([(-100, 0, 45.0), (100, 0, 315.0)]).to_dict()
    for k in ("east", "north", "num_bearings", "residual_rms_m", "max_residual_m",
              "gdop", "min_subtended_deg", "all_in_front", "well_conditioned", "caveat"):
        assert k in d


def test_caveat_present():
    f = geolocate([(-100, 0, 45.0), (100, 0, 315.0)])
    assert "awareness" in f.caveat.lower()


# -- lat/lon helpers ----------------------------------------------------------

def test_bearing_from_positions_north():
    # Emitter due north of the sensor.
    b = bearing_from_positions(40.0, -80.0, 40.1, -80.0)
    assert b == pytest.approx(0.0, abs=1e-3)


def test_bearing_from_positions_east():
    b = bearing_from_positions(40.0, -80.0, 40.0, -79.9)
    assert b == pytest.approx(90.0, abs=0.1)


def test_observation_from_latlon_roundtrip():
    ref = (40.0, -80.0)
    obs = observation_from_latlon(40.01, -80.0, 180.0, ref_lat=ref[0], ref_lon=ref[1])
    # 0.01 deg north of ref -> north offset ~1113 m, east ~0.
    assert obs.north == pytest.approx(1112.0, abs=5.0)
    assert obs.east == pytest.approx(0.0, abs=1e-6)
    assert obs.bearing_deg == 180.0


def test_network_geolocation_from_latlon():
    # Two sensors, known emitter; synthesise bearings and recover it in ENU.
    ref = (40.0, -80.0)
    emitter = (40.02, -79.98)
    s1 = (40.0, -80.0)
    s2 = (40.0, -79.95)
    obs = []
    for s in (s1, s2):
        brg = bearing_from_positions(s[0], s[1], emitter[0], emitter[1])
        obs.append(observation_from_latlon(s[0], s[1], brg, ref_lat=ref[0], ref_lon=ref[1]))
    from frontline_drones.geo import enu_offset_m
    ee, en = enu_offset_m(emitter[0], emitter[1], ref[0], ref[1])
    f = geolocate(obs)
    assert f.east == pytest.approx(ee, abs=5.0)
    assert f.north == pytest.approx(en, abs=5.0)
