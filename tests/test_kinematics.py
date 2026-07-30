"""Tests for the track-kinematics and track-quality scorer."""

from __future__ import annotations

import pytest

from frontline_drones import kinematics
from frontline_drones.kinematics import Fix, track_features, velocity_enu


def _straight_track(n=8, speed=5.0, dt=1.0, alt=50.0):
    return [(i * dt, i * speed * dt, 0.0, alt) for i in range(n)]


def test_empty_track_zero_quality():
    tf = track_features([])
    assert tf.num_fixes == 0
    assert tf.track_quality == 0.0


def test_single_fix_zero_quality():
    tf = track_features([(0.0, 0.0, 0.0)])
    assert tf.num_fixes == 1
    assert tf.track_quality == 0.0


def test_straight_track_is_transit():
    tf = track_features(_straight_track(speed=10.0))
    assert tf.motion_pattern == "transit"
    assert tf.straightness == pytest.approx(1.0, abs=1e-6)


def test_straight_track_mean_speed():
    tf = track_features(_straight_track(n=6, speed=10.0, dt=1.0))
    assert tf.mean_speed_mps == pytest.approx(10.0)
    assert tf.max_speed_mps == pytest.approx(10.0)


def test_straight_track_heading_east():
    tf = track_features(_straight_track())
    assert tf.heading_deg == pytest.approx(90.0, abs=1e-3)


def test_hover_track_pattern():
    # Tiny jitter around a point => hover.
    fixes = [(i, 0.1 * (i % 2), 0.1 * ((i + 1) % 2), 30.0) for i in range(8)]
    tf = track_features(fixes)
    assert tf.motion_pattern == "hover"
    assert tf.mean_speed_mps < kinematics.HOVER_SPEED_MPS


def test_climb_rate_positive():
    fixes = [(i, 0.0, 0.0, 10.0 * i) for i in range(6)]
    tf = track_features(fixes)
    assert tf.mean_climb_mps == pytest.approx(10.0, rel=1e-6)


def test_descent_rate_negative():
    fixes = [(i, 0.0, 0.0, 100.0 - 5.0 * i) for i in range(6)]
    tf = track_features(fixes)
    assert tf.mean_climb_mps < 0


def test_displacement_less_than_path_for_curved():
    # Out and back: large path, small displacement.
    fixes = [(0, 0, 0), (1, 10, 0), (2, 0, 0)]
    tf = track_features(fixes)
    assert tf.displacement_m < tf.path_length_m
    assert tf.straightness < 0.5


def test_quality_improves_with_more_fixes():
    short = track_features(_straight_track(n=2))
    longer = track_features(_straight_track(n=8))
    assert longer.track_quality > short.track_quality


def test_quality_in_unit_interval():
    tf = track_features(_straight_track(n=8))
    assert 0.0 <= tf.track_quality <= 1.0


def test_implausible_speed_penalized():
    fast = [(i, i * 200.0, 0.0, 50.0) for i in range(8)]  # 200 m/s
    normal = [(i, i * 10.0, 0.0, 50.0) for i in range(8)]
    assert track_features(fast).track_quality < track_features(normal).track_quality


def test_fixes_sorted_by_time():
    unordered = [(2, 20, 0), (0, 0, 0), (1, 10, 0)]
    tf = track_features(unordered)
    assert tf.heading_deg == pytest.approx(90.0, abs=1e-3)
    assert tf.duration_s == pytest.approx(2.0)


def test_accepts_fix_objects():
    fixes = [Fix(t=i, east=i * 5.0, north=0.0, up=40.0) for i in range(5)]
    tf = track_features(fixes)
    assert tf.mean_speed_mps == pytest.approx(5.0)


def test_accepts_dict_samples():
    fixes = [{"t": i, "east": i * 5.0, "north": 0.0, "up": 40.0} for i in range(5)]
    tf = track_features(fixes)
    assert tf.mean_speed_mps == pytest.approx(5.0)


def test_accepts_xyz_dict_aliases():
    fixes = [{"t": i, "x": i * 3.0, "y": 0.0, "z": 10.0} for i in range(4)]
    tf = track_features(fixes)
    assert tf.mean_speed_mps == pytest.approx(3.0)


def test_zero_duration_gives_zero_quality():
    # All fixes at the same timestamp => no positive interval.
    tf = track_features([(0, 0, 0), (0, 10, 0), (0, 20, 0)])
    assert tf.track_quality == 0.0


def test_erratic_track_high_turn_rate():
    # Zig-zag with big heading reversals and low straightness.
    fixes = [
        (0, 0, 0), (1, 10, 10), (2, 20, 0), (3, 30, 10), (4, 40, 0), (5, 50, 10)
    ]
    tf = track_features(fixes)
    assert tf.mean_turn_rate_dps > 0


def test_to_dict_keys():
    d = track_features(_straight_track()).to_dict()
    for k in (
        "num_fixes", "duration_s", "path_length_m", "displacement_m", "straightness",
        "mean_speed_mps", "max_speed_mps", "mean_climb_mps", "mean_turn_rate_dps",
        "heading_deg", "motion_pattern", "track_quality",
    ):
        assert k in d


def test_velocity_enu_basic():
    v = velocity_enu([(0, 0, 0, 0), (1, 10, 5, 2)])
    assert v == (10.0, 5.0, 2.0)


def test_velocity_enu_uses_last_two():
    v = velocity_enu([(0, 0, 0), (1, 5, 0), (2, 25, 0)])
    assert v == (20.0, 0.0, 0.0)


def test_velocity_enu_too_few():
    assert velocity_enu([(0, 0, 0)]) == (0.0, 0.0, 0.0)


def test_velocity_enu_zero_dt():
    assert velocity_enu([(0, 0, 0), (0, 5, 0)]) == (0.0, 0.0, 0.0)


def test_deterministic():
    a = track_features(_straight_track()).to_dict()
    b = track_features(_straight_track()).to_dict()
    assert a == b


def test_quality_penalizes_sampling_gaps():
    regular = [(i, i * 5.0, 0.0, 50.0) for i in range(8)]
    gappy = [(0, 0, 50), (1, 5, 50), (2, 10, 50), (3, 15, 50),
             (4, 20, 50), (5, 25, 50), (6, 30, 50), (20, 100, 50)]
    assert track_features(gappy).track_quality < track_features(regular).track_quality
