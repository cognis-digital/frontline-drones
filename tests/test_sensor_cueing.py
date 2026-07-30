"""Tests for the sensor cueing / cross-sensor track-handoff module."""

from __future__ import annotations

import json
import math

import pytest

from frontline_drones import sensor_cueing as sc
from frontline_drones.sensor_cueing import (
    CueCommand,
    CustodySchedule,
    HandoffPlan,
    Sensor,
    Target,
    cue_sensor,
    plan_handoff,
    schedule_custody,
)

# -- angle helpers ------------------------------------------------------------

def test_bearing_north():
    assert sc.bearing_deg(0.0, 10.0) == pytest.approx(0.0)


def test_bearing_east():
    assert sc.bearing_deg(10.0, 0.0) == pytest.approx(90.0)


def test_bearing_south():
    assert sc.bearing_deg(0.0, -10.0) == pytest.approx(180.0)


def test_bearing_west():
    assert sc.bearing_deg(-10.0, 0.0) == pytest.approx(270.0)


def test_bearing_zero_offset_is_zero():
    assert sc.bearing_deg(0.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (0.0, 0.0, 0.0),
        (10.0, 20.0, 10.0),
        (350.0, 10.0, 20.0),
        (0.0, 180.0, 180.0),
        (90.0, 270.0, 180.0),
        (270.0, 90.0, 180.0),
        (5.0, 355.0, 10.0),
    ],
)
def test_angular_sep(a, b, expected):
    assert sc.angular_sep_deg(a, b) == pytest.approx(expected)


@pytest.mark.parametrize("a", [0.0, 45.0, 123.4, 200.0, 359.9])
@pytest.mark.parametrize("b", [0.0, 45.0, 123.4, 200.0, 359.9])
def test_angular_sep_symmetric_and_bounded(a, b):
    s1 = sc.angular_sep_deg(a, b)
    s2 = sc.angular_sep_deg(b, a)
    assert s1 == pytest.approx(s2)
    assert 0.0 <= s1 <= 180.0 + 1e-9


# -- Sensor validation --------------------------------------------------------

def test_sensor_constructs():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    assert s.max_range_m == 1000.0
    assert s.role == "tracker"


def test_sensor_bad_max_range():
    with pytest.raises(ValueError):
        Sensor("s", 0, 0, max_range_m=0)


def test_sensor_negative_min_range():
    with pytest.raises(ValueError):
        Sensor("s", 0, 0, max_range_m=100, min_range_m=-1)


def test_sensor_min_ge_max():
    with pytest.raises(ValueError):
        Sensor("s", 0, 0, max_range_m=100, min_range_m=100)


def test_sensor_bad_slew_rate():
    with pytest.raises(ValueError):
        Sensor("s", 0, 0, max_range_m=100, slew_rate_dps=0)


def test_sensor_negative_settle():
    with pytest.raises(ValueError):
        Sensor("s", 0, 0, max_range_m=100, settle_s=-1)


def test_sensor_bad_role():
    with pytest.raises(ValueError):
        Sensor("s", 0, 0, max_range_m=100, role="shooter")


def test_sensor_roles_constant():
    assert sc.SENSOR_ROLES == ("detector", "tracker")


def test_sensor_to_dict_roundtrips_json():
    s = Sensor("s", 1.0, 2.0, max_range_m=500.0, min_range_m=50.0)
    json.dumps(s.to_dict())


# -- Target / coercion --------------------------------------------------------

def test_target_position_at():
    t = Target("t", east=0.0, north=0.0, vel_east=2.0, vel_north=1.0)
    assert t.position_at(3.0) == (6.0, 3.0)


def test_target_speed():
    t = Target("t", vel_east=3.0, vel_north=4.0)
    assert t.speed_mps == pytest.approx(5.0)


def test_as_target_passthrough():
    t = Target("t", east=1.0, north=2.0)
    assert sc.as_target(t) is t


def test_as_target_from_dict():
    t = sc.as_target({"east": 3.0, "north": 4.0, "vel_east": 1.0})
    assert (t.east, t.north, t.vel_east) == (3.0, 4.0, 1.0)


def test_as_target_dict_xy_aliases():
    t = sc.as_target({"x": 5.0, "y": 6.0})
    assert (t.east, t.north) == (5.0, 6.0)


def test_as_target_from_tuple():
    t = sc.as_target((1.0, 2.0, 3.0, 4.0))
    assert (t.east, t.north, t.vel_east, t.vel_north) == (1.0, 2.0, 3.0, 4.0)


def test_as_target_from_pair():
    t = sc.as_target((7.0, 8.0))
    assert (t.east, t.north, t.vel_east, t.vel_north) == (7.0, 8.0, 0.0, 0.0)


def test_as_target_from_duck_typed_track():
    # Mimic a frontline_drones.track_association.Track by attributes.
    class FakeTrack:
        east = 10.0
        north = 20.0
        vel_east = 1.0
        vel_north = 2.0
        track_id = 99

    t = sc.as_target(FakeTrack())
    assert (t.east, t.north, t.track_id) == (10.0, 20.0, 99)


def test_track_association_track_is_accepted():
    from frontline_drones.track_association import Track

    tr = Track(track_id=7, east=100.0, north=0.0, vel_east=5.0, vel_north=0.0)
    t = sc.as_target(tr)
    assert t.track_id == 7
    assert t.east == 100.0
    assert t.vel_east == 5.0


# -- instantaneous geometry ---------------------------------------------------

def test_ground_range():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    t = Target("t", east=3.0, north=4.0)
    assert sc.ground_range_m(s, t) == pytest.approx(5.0)


def test_slant_range_uses_altitude():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, height_m=0.0)
    t = Target("t", east=0.0, north=0.0, up=10.0)
    assert sc.slant_range_m(s, t) == pytest.approx(10.0)


def test_elevation_45_deg():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    t = Target("t", east=100.0, north=0.0, up=100.0)
    assert sc.elevation_deg(s, t) == pytest.approx(45.0)


def test_elevation_negative_below():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, height_m=50.0)
    t = Target("t", east=100.0, north=0.0, up=0.0)
    assert sc.elevation_deg(s, t) < 0.0


def test_bearing_to_east():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    t = Target("t", east=100.0, north=0.0)
    assert sc.bearing_to(s, t) == pytest.approx(90.0)


def test_in_coverage_inside():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    assert sc.in_coverage(s, Target(east=500.0, north=0.0))


def test_in_coverage_beyond_range():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    assert not sc.in_coverage(s, Target(east=1500.0, north=0.0))


def test_in_coverage_inside_deadzone():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, min_range_m=200.0)
    assert not sc.in_coverage(s, Target(east=100.0, north=0.0))


def test_in_coverage_on_annulus_edges():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, min_range_m=200.0)
    assert sc.in_coverage(s, Target(east=200.0, north=0.0))
    assert sc.in_coverage(s, Target(east=1000.0, north=0.0))


# -- slew time ----------------------------------------------------------------

def test_slew_time_basic():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, slew_rate_dps=45.0)
    assert sc.slew_time_s(s, 0.0, 90.0) == pytest.approx(2.0)


def test_slew_time_shortest_way():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, slew_rate_dps=10.0)
    # 350 -> 10 is 20 deg the short way, not 340.
    assert sc.slew_time_s(s, 350.0, 10.0) == pytest.approx(2.0)


def test_slew_time_includes_settle():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, slew_rate_dps=90.0, settle_s=0.5)
    assert sc.slew_time_s(s, 0.0, 90.0) == pytest.approx(1.5)


def test_slew_time_never_negative():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    assert sc.slew_time_s(s, 123.0, 123.0) == pytest.approx(0.0)


# -- cueing -------------------------------------------------------------------

def test_cue_static_target_points_at_it():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, slew_rate_dps=45.0)
    cmd = cue_sensor(s, Target("t", east=0.0, north=500.0), lead=True)
    assert cmd.bearing_deg == pytest.approx(0.0)
    assert cmd.in_coverage
    assert cmd.feasible
    assert cmd.lead_time_s == 0.0  # stationary => no lead


def test_cue_returns_command_type():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    assert isinstance(cue_sensor(s, Target(east=100.0, north=0.0)), CueCommand)


def test_cue_out_of_range_infeasible_with_advisory():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    cmd = cue_sensor(s, Target("t", east=5000.0, north=0.0), lead=False)
    assert not cmd.in_coverage
    assert not cmd.feasible
    assert any("beyond" in a for a in cmd.advisories)


def test_cue_in_deadzone_advisory():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, min_range_m=300.0)
    cmd = cue_sensor(s, Target("t", east=50.0, north=0.0), lead=False)
    assert not cmd.feasible
    assert any("dead-zone" in a for a in cmd.advisories)


def test_cue_lead_points_ahead_of_current_position():
    # Fast crossing target: leading should aim where the target will be.
    s = Sensor("s", 0.0, 0.0, max_range_m=5000.0, slew_rate_dps=30.0)
    t = Target("t", east=0.0, north=1000.0, vel_east=200.0, vel_north=0.0)
    # Start pointed west so a real slew (and thus a real lead) is required.
    led = cue_sensor(s, t, from_bearing_deg=270.0, lead=True)
    now = cue_sensor(s, t, lead=False)
    assert led.lead_time_s > 0.0
    assert led.aim_east > now.aim_east  # aim point advanced east with the target


def test_cue_lead_fixed_point_consistent():
    # After leading, the commanded slew time should match the lead time used
    # (the fixed-point converged so pointing where the target will be).
    s = Sensor("s", 0.0, 0.0, max_range_m=8000.0, slew_rate_dps=20.0)
    t = Target("t", east=500.0, north=2000.0, vel_east=150.0, vel_north=-50.0)
    cmd = cue_sensor(s, t, from_bearing_deg=0.0, lead=True)
    # The fixed-point iteration converges: the slew to the commanded bearing
    # equals the lead time used to place that aim point.
    assert cmd.slew_time_s == pytest.approx(cmd.lead_time_s, abs=1e-3)


def test_cue_no_lead_flag():
    s = Sensor("s", 0.0, 0.0, max_range_m=5000.0)
    t = Target("t", east=0.0, north=1000.0, vel_east=100.0, vel_north=0.0)
    cmd = cue_sensor(s, t, lead=False)
    assert cmd.lead_time_s == 0.0
    assert cmd.aim_east == pytest.approx(0.0)


def test_cue_bad_iterations():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    with pytest.raises(ValueError):
        cue_sensor(s, Target(east=1.0, north=1.0, vel_east=1.0), iterations=0)


def test_cue_to_dict_json():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    cmd = cue_sensor(s, Target("t", east=100.0, north=100.0))
    json.dumps(cmd.to_dict())


def test_cue_deterministic():
    s = Sensor("s", 0.0, 0.0, max_range_m=5000.0, slew_rate_dps=25.0)
    t = Target("t", east=300.0, north=1500.0, vel_east=120.0, vel_north=30.0)
    assert cue_sensor(s, t).to_dict() == cue_sensor(s, t).to_dict()


# -- select_tracker -----------------------------------------------------------

def test_select_tracker_picks_least_slew():
    t = Target("t", east=0.0, north=100.0)  # due north of origin
    aligned = Sensor("aligned", 0.0, 0.0, max_range_m=1000.0)  # from_bearing 0 == target
    off = Sensor("off", 0.0, 0.0, max_range_m=1000.0)
    # aligned starts pointing north (0), off starts pointing east (90)
    best = sc.select_tracker([off, aligned], t, from_bearings={"off": 90.0, "aligned": 0.0})
    assert best.sensor_id == "aligned"


def test_select_tracker_none_in_coverage():
    t = Target("t", east=9000.0, north=0.0)
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    assert sc.select_tracker([s], t) is None


def test_select_tracker_skips_out_of_coverage():
    t = Target("t", east=0.0, north=500.0)
    near = Sensor("near", 0.0, 0.0, max_range_m=1000.0)
    far = Sensor("far", 0.0, 0.0, max_range_m=100.0)  # cannot see target at 500
    best = sc.select_tracker([far, near], t)
    assert best.sensor_id == "near"


# -- coverage windows ---------------------------------------------------------

def test_windows_stationary_inside_open_ended():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    wins = sc.coverage_windows(s, Target(east=100.0, north=0.0))
    assert wins == [(0.0, math.inf)]


def test_windows_stationary_outside_empty():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    assert sc.coverage_windows(s, Target(east=5000.0, north=0.0)) == []


def test_windows_crossing_has_finite_exit():
    # Target starts inside at origin sensor, flies east and out.
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    t = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    wins = sc.coverage_windows(s, t)
    assert len(wins) == 1
    enter, exit_ = wins[0]
    assert enter == pytest.approx(0.0)
    assert exit_ == pytest.approx(10.0)  # 1000 m / 100 m/s


def test_windows_approaching_from_outside():
    # Target 2000 m west flying east at 100 m/s: enters at 1000 m (t=10), exits at 3000 m (t=30).
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    t = Target("t", east=-2000.0, north=0.0, vel_east=100.0, vel_north=0.0)
    wins = sc.coverage_windows(s, t)
    assert len(wins) == 1
    enter, exit_ = wins[0]
    assert enter == pytest.approx(10.0)
    assert exit_ == pytest.approx(30.0)


def test_windows_never_enters():
    # Passes north of the sensor beyond max range.
    s = Sensor("s", 0.0, 0.0, max_range_m=500.0)
    t = Target("t", east=-2000.0, north=1000.0, vel_east=100.0, vel_north=0.0)
    assert sc.coverage_windows(s, t) == []


def test_windows_deadzone_splits_into_two():
    # Fly straight through the sensor across a dead-zone: two windows.
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, min_range_m=200.0)
    t = Target("t", east=-2000.0, north=0.0, vel_east=100.0, vel_north=0.0)
    wins = sc.coverage_windows(s, t)
    assert len(wins) == 2
    # First window: enter outer (1000 m => t=10) until enter deadzone (200 m => t=18).
    assert wins[0][0] == pytest.approx(10.0)
    assert wins[0][1] == pytest.approx(18.0)
    # Second window: leave deadzone (200 m => t=22) until leave outer (1000 m => t=30).
    assert wins[1][0] == pytest.approx(22.0)
    assert wins[1][1] == pytest.approx(30.0)


def test_windows_offset_deadzone_not_crossed_single_window():
    # Path clips the disk but stays outside the dead-zone => single window.
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, min_range_m=200.0)
    t = Target("t", east=-2000.0, north=900.0, vel_east=100.0, vel_north=0.0)
    wins = sc.coverage_windows(s, t)
    assert len(wins) == 1


def test_windows_ascending_order():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0, min_range_m=200.0)
    t = Target("t", east=-2000.0, north=0.0, vel_east=100.0, vel_north=0.0)
    wins = sc.coverage_windows(s, t)
    times = [x for w in wins for x in w]
    assert times == sorted(times)


@pytest.mark.parametrize("speed", [10.0, 50.0, 100.0, 250.0])
@pytest.mark.parametrize("radius", [300.0, 1000.0, 2500.0])
def test_windows_exit_matches_radius_over_speed(speed, radius):
    # Target starting at the sensor flying straight out exits at radius/speed.
    s = Sensor("s", 0.0, 0.0, max_range_m=radius)
    t = Target("t", east=0.0, north=0.0, vel_east=speed, vel_north=0.0)
    wins = sc.coverage_windows(s, t)
    assert len(wins) == 1
    assert wins[0][1] == pytest.approx(radius / speed)


def test_current_window_when_inside():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    t = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    win = sc.current_window(s, t)
    assert win is not None and win[0] == pytest.approx(0.0)


def test_current_window_when_outside_none():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    t = Target("t", east=-2000.0, north=0.0, vel_east=100.0, vel_north=0.0)
    assert sc.current_window(s, t) is None


def test_time_to_exit_inside():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    t = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    assert sc.time_to_exit(s, t) == pytest.approx(10.0)


def test_time_to_exit_outside_none():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    assert sc.time_to_exit(s, Target(east=5000.0, north=0.0)) is None


def test_time_to_exit_stationary_infinite():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    assert sc.time_to_exit(s, Target(east=100.0, north=0.0)) == math.inf


# -- pairwise handoff ---------------------------------------------------------

def _line_of_sensors():
    # Three overlapping sensors along the east axis; target flies east through all.
    s0 = Sensor("s0", 0.0, 0.0, max_range_m=1200.0, role="detector")
    s1 = Sensor("s1", 2000.0, 0.0, max_range_m=1200.0)
    s2 = Sensor("s2", 4000.0, 0.0, max_range_m=1200.0)
    tgt = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    return s0, s1, s2, tgt


def test_handoff_selects_next_sensor():
    s0, s1, s2, tgt = _line_of_sensors()
    plan = plan_handoff(s0, [s1, s2], tgt)
    assert isinstance(plan, HandoffPlan)
    assert plan.to_sensor == "s1"


def test_handoff_seamless_when_overlapping():
    s0, s1, s2, tgt = _line_of_sensors()
    plan = plan_handoff(s0, [s1, s2], tgt)
    # s0 exit at 1200 m => t=12; s1 enters at 800 m => t=8. Overlap.
    assert plan.current_exit_t == pytest.approx(12.0)
    assert plan.next_enter_t == pytest.approx(8.0)
    assert plan.overlap_s == pytest.approx(4.0)
    assert plan.gap_s == pytest.approx(0.0)
    assert plan.seamless


def test_handoff_recommended_time_in_overlap():
    s0, s1, s2, tgt = _line_of_sensors()
    plan = plan_handoff(s0, [s1, s2], tgt)
    assert plan.next_enter_t <= plan.recommended_handoff_t <= plan.current_exit_t


def test_handoff_gap_reported_when_disjoint():
    # Two sensors with a gap between their footprints.
    s0 = Sensor("s0", 0.0, 0.0, max_range_m=1000.0)
    s1 = Sensor("s1", 5000.0, 0.0, max_range_m=1000.0)
    tgt = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    plan = plan_handoff(s0, [s1], tgt)
    # s0 exit at 1000 m => t=10; s1 enter at 4000 m => t=40. Gap of 30 s.
    assert plan.to_sensor == "s1"
    assert plan.gap_s == pytest.approx(30.0)
    assert not plan.seamless
    assert any("gap" in a.lower() for a in plan.advisories)


def test_handoff_none_when_no_successor():
    s0 = Sensor("s0", 0.0, 0.0, max_range_m=2000.0)
    s1 = Sensor("s1", 0.0, 0.0, max_range_m=500.0)  # entirely inside s0's reach
    tgt = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    plan = plan_handoff(s0, [s1], tgt)
    assert plan.to_sensor is None
    assert plan.gap_s == math.inf
    assert any("No candidate" in a for a in plan.advisories)


def test_handoff_prefers_longest_custody_on_tie():
    # Two successors both seamless with equal enter time; pick the longer-reaching.
    s0 = Sensor("s0", 0.0, 0.0, max_range_m=1000.0)
    short = Sensor("short", 1500.0, 0.0, max_range_m=800.0)
    long_ = Sensor("long", 1500.0, 0.0, max_range_m=1500.0)
    tgt = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    plan = plan_handoff(s0, [short, long_], tgt)
    assert plan.to_sensor == "long"


def test_handoff_excludes_self():
    s0, s1, s2, tgt = _line_of_sensors()
    # from_sensor also present in candidates; must not pick itself.
    plan = plan_handoff(s0, [s0, s1, s2], tgt)
    assert plan.to_sensor == "s1"


def test_handoff_from_sensor_not_currently_covering():
    s0 = Sensor("s0", 0.0, 0.0, max_range_m=1000.0)
    s1 = Sensor("s1", 2000.0, 0.0, max_range_m=1500.0)
    # Target already east of s0 coverage.
    tgt = Target("t", east=1500.0, north=0.0, vel_east=100.0, vel_north=0.0)
    plan = plan_handoff(s0, [s1], tgt)
    assert any("not currently" in a for a in plan.advisories)


def test_handoff_successor_ready_true_without_bearings():
    s0, s1, s2, tgt = _line_of_sensors()
    plan = plan_handoff(s0, [s1, s2], tgt)
    assert plan.successor_ready


def test_handoff_successor_not_ready_when_slow_slew():
    # Successor with a very slow slew and no lead time cannot get pointed in time.
    s0 = Sensor("s0", 0.0, 0.0, max_range_m=1000.0)
    s1 = Sensor("s1", 1200.0, 0.0, max_range_m=1000.0, slew_rate_dps=0.1)
    tgt = Target("t", east=0.0, north=0.0, vel_east=500.0, vel_north=0.0)
    # Point s1 far away so it needs a big slew right as the target arrives.
    plan = plan_handoff(s0, [s1], tgt, from_bearings={"s1": 180.0})
    assert not plan.successor_ready
    assert any("slew" in a for a in plan.advisories)


def test_handoff_to_dict_json():
    s0, s1, s2, tgt = _line_of_sensors()
    d = plan_handoff(s0, [s1, s2], tgt).to_dict()
    json.dumps(d)
    assert d["to_sensor"] == "s1"


def test_handoff_to_dict_none_gap_serialises():
    s0 = Sensor("s0", 0.0, 0.0, max_range_m=2000.0)
    s1 = Sensor("s1", 0.0, 0.0, max_range_m=500.0)
    tgt = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    d = plan_handoff(s0, [s1], tgt).to_dict()
    # inf gap maps to a JSON-safe value.
    js = json.dumps(d)
    assert "Infinity" not in js
    assert d["next_enter_t"] is None


def test_handoff_deterministic():
    s0, s1, s2, tgt = _line_of_sensors()
    assert plan_handoff(s0, [s1, s2], tgt).to_dict() == plan_handoff(s0, [s1, s2], tgt).to_dict()


# -- custody schedule ---------------------------------------------------------

def test_schedule_seamless_chain():
    s0, s1, s2, tgt = _line_of_sensors()
    sch = schedule_custody([s0, s1, s2], tgt)
    assert isinstance(sch, CustodySchedule)
    assert sch.seamless
    assert sch.gap_s == pytest.approx(0.0)
    ids = [seg.sensor_id for seg in sch.segments]
    assert ids == ["s0", "s1", "s2"]


def test_schedule_num_handoffs():
    s0, s1, s2, tgt = _line_of_sensors()
    sch = schedule_custody([s0, s1, s2], tgt)
    assert sch.num_handoffs == 2


def test_schedule_covered_time():
    s0, s1, s2, tgt = _line_of_sensors()
    sch = schedule_custody([s0, s1, s2], tgt)
    # s0 covers 0..12, s2 exits at (4000+1200)/100 = 52. Continuous => 52 s covered.
    assert sch.first_acquire_t == pytest.approx(0.0)
    assert sch.final_loss_t == pytest.approx(52.0)
    assert sch.covered_s == pytest.approx(52.0)


def test_schedule_reports_gap():
    s0 = Sensor("s0", 0.0, 0.0, max_range_m=1000.0)
    s1 = Sensor("s1", 5000.0, 0.0, max_range_m=1000.0)
    tgt = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    sch = schedule_custody([s0, s1], tgt)
    assert not sch.seamless
    assert sch.gap_s == pytest.approx(30.0)  # 10..40
    assert len(sch.gaps) == 1
    assert sch.gaps[0] == pytest.approx((10.0, 40.0))


def test_schedule_no_coverage():
    s = Sensor("s", 0.0, 0.0, max_range_m=500.0)
    tgt = Target("t", east=0.0, north=5000.0, vel_east=100.0, vel_north=0.0)
    sch = schedule_custody([s], tgt)
    assert sch.segments == ()
    assert sch.covered_s == 0.0
    assert any("cannot be observed" in a for a in sch.advisories)


def test_schedule_minimises_handoffs_prefers_longer_reach():
    # A single wide sensor spans the whole path; two narrow ones also cover parts.
    wide = Sensor("wide", 2000.0, 0.0, max_range_m=2500.0)
    narrow_a = Sensor("na", 1000.0, 0.0, max_range_m=600.0)
    narrow_b = Sensor("nb", 3000.0, 0.0, max_range_m=600.0)
    tgt = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    sch = schedule_custody([narrow_a, wide, narrow_b], tgt)
    # Greedy should hold the wide sensor the whole way => single segment, 0 handoffs.
    assert sch.num_handoffs == 0
    assert sch.segments[0].sensor_id == "wide"
    assert sch.seamless


def test_schedule_requires_horizon_for_stationary():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    tgt = Target("t", east=0.0, north=0.0)  # stationary inside coverage
    with pytest.raises(ValueError):
        schedule_custody([s], tgt)


def test_schedule_stationary_with_horizon():
    s = Sensor("s", 0.0, 0.0, max_range_m=1000.0)
    tgt = Target("t", east=0.0, north=0.0)
    sch = schedule_custody([s], tgt, horizon_s=100.0)
    assert sch.covered_s == pytest.approx(100.0)
    assert sch.seamless
    assert sch.final_loss_t == pytest.approx(100.0)


def test_schedule_horizon_caps_moving_target():
    s0, s1, s2, tgt = _line_of_sensors()
    sch = schedule_custody([s0, s1, s2], tgt, horizon_s=20.0)
    assert sch.final_loss_t == pytest.approx(20.0)
    assert sch.covered_s == pytest.approx(20.0)


def test_schedule_to_dict_json():
    s0, s1, s2, tgt = _line_of_sensors()
    d = schedule_custody([s0, s1, s2], tgt).to_dict()
    js = json.dumps(d)
    assert "Infinity" not in js
    assert len(d["segments"]) == 3


def test_schedule_deterministic():
    s0, s1, s2, tgt = _line_of_sensors()
    a = schedule_custody([s0, s1, s2], tgt).to_dict()
    b = schedule_custody([s0, s1, s2], tgt).to_dict()
    assert a == b


def test_schedule_order_independent_result():
    # Feeding sensors in a different order yields the same covered time / gaps.
    s0, s1, s2, tgt = _line_of_sensors()
    a = schedule_custody([s0, s1, s2], tgt)
    b = schedule_custody([s2, s0, s1], tgt)
    assert a.covered_s == pytest.approx(b.covered_s)
    assert a.gap_s == pytest.approx(b.gap_s)
    assert a.first_acquire_t == pytest.approx(b.first_acquire_t)
    assert a.final_loss_t == pytest.approx(b.final_loss_t)


def test_segment_duration():
    seg = sc.CustodySegment("s", 5.0, 12.0)
    assert seg.duration_s == pytest.approx(7.0)
    json.dumps(seg.to_dict())


@pytest.mark.parametrize("spacing", [1500.0, 1800.0, 2000.0])
def test_schedule_overlapping_line_is_seamless(spacing):
    # Sensors spaced closer than 2*radius always overlap => seamless custody.
    radius = 1200.0
    sensors = [Sensor(f"s{i}", i * spacing, 0.0, max_range_m=radius) for i in range(4)]
    tgt = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    sch = schedule_custody(sensors, tgt)
    assert sch.seamless
    assert sch.gap_s == pytest.approx(0.0)


@pytest.mark.parametrize("gap_spacing", [4000.0, 5000.0, 6000.0])
def test_schedule_spaced_line_has_gaps(gap_spacing):
    # Sensors spaced wider than 2*radius leave gaps.
    radius = 1000.0
    sensors = [Sensor(f"s{i}", i * gap_spacing, 0.0, max_range_m=radius) for i in range(3)]
    tgt = Target("t", east=0.0, north=0.0, vel_east=100.0, vel_north=0.0)
    sch = schedule_custody(sensors, tgt)
    assert not sch.seamless
    assert sch.gap_s > 0.0
    assert len(sch.gaps) == 2


# -- cross-check: handoff agrees with schedule's first transition -------------

def test_handoff_matches_schedule_first_transition():
    s0, s1, s2, tgt = _line_of_sensors()
    plan = plan_handoff(s0, [s1, s2], tgt)
    sch = schedule_custody([s0, s1, s2], tgt)
    assert plan.to_sensor == sch.segments[1].sensor_id


def test_time_to_exit_matches_schedule_first_segment_end():
    s0, s1, s2, tgt = _line_of_sensors()
    exit_t = sc.time_to_exit(s0, tgt)
    sch = schedule_custody([s0, s1, s2], tgt)
    assert exit_t == pytest.approx(sch.segments[0].end_t)
