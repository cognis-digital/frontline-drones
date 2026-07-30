"""Behavioral intent classification from track kinematics."""
import math

import pytest

from frontline_drones.intent import (
    IntentClass,
    IntentThresholds,
    classify_intent,
    coordinated_ingress,
    track_features,
)

ASSET = (0.0, 0.0)


def straight_approach(n=10, start=1000.0, step=120.0):
    # Moving straight in toward the asset along +x axis from far away.
    return [(start - i * step, 0.0) for i in range(n)]


def transit_past(n=10, y=800.0, step=150.0):
    # Straight line passing the asset at a constant offset (not closing much).
    return [(-700.0 + i * step, y) for i in range(n)]


def loiter(n=12, r=500.0):
    # Wandering small displacements near a point offset from the asset.
    pts = []
    cx, cy = 500.0, 500.0
    offsets = [(0, 0), (30, 10), (10, 30), (-20, 15), (5, -25), (25, 5),
               (-15, -10), (0, 20), (18, -12), (-8, 8), (12, 3), (-5, -18)]
    for i in range(n):
        ox, oy = offsets[i % len(offsets)]
        pts.append((cx + ox, cy + oy))
    return pts


def orbit(n=16, r=600.0):
    # Steady CCW circle at constant range r around the asset.
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


# ---- features ----

def test_features_need_two_points():
    with pytest.raises(ValueError):
        track_features([(0, 0)], ASSET)


def test_bad_dt():
    with pytest.raises(ValueError):
        track_features([(0, 0), (1, 1)], ASSET, dt_s=0)


def test_straight_track_high_straightness():
    f = track_features(straight_approach(), ASSET)
    assert f.straightness > 0.95


def test_approach_positive_closure():
    f = track_features(straight_approach(), ASSET)
    assert f.closure_rate > 0   # range shrinking


def test_orbit_constant_range_low_cv():
    f = track_features(orbit(), ASSET)
    assert f.range_cv < 0.05
    assert abs(f.turn_consistency) > 0.8


def test_features_to_dict():
    d = track_features(straight_approach(), ASSET).to_dict()
    for k in ("mean_speed", "straightness", "closure_rate", "turn_consistency",
              "range_cv", "mean_range"):
        assert k in d


# ---- classification ----

def test_classify_approach():
    a = classify_intent(straight_approach(), ASSET)
    assert a.intent is IntentClass.APPROACH
    assert a.confidence >= 0.4


def test_classify_transit():
    a = classify_intent(transit_past(), ASSET)
    assert a.intent is IntentClass.TRANSIT


def test_classify_orbit():
    a = classify_intent(orbit(), ASSET)
    assert a.intent is IntentClass.RECON_ORBIT


def test_classify_loiter():
    a = classify_intent(loiter(), ASSET)
    assert a.intent is IntentClass.LOITER


def test_assessment_to_dict():
    d = classify_intent(straight_approach(), ASSET).to_dict()
    for k in ("intent", "confidence", "reasoning", "features"):
        assert k in d


def test_reasoning_present():
    assert classify_intent(straight_approach(), ASSET).reasoning


# ---- coordinated ingress (swarm) ----

def test_coordinated_ingress_true():
    tracks = [straight_approach() for _ in range(4)]
    assert coordinated_ingress(tracks, ASSET, min_closing=3)


def test_coordinated_ingress_false_when_transit():
    tracks = [transit_past() for _ in range(4)]
    assert not coordinated_ingress(tracks, ASSET, min_closing=3)


def test_coordinated_ingress_threshold():
    tracks = [straight_approach(), straight_approach(), transit_past()]
    assert not coordinated_ingress(tracks, ASSET, min_closing=3)
    assert coordinated_ingress(tracks, ASSET, min_closing=2)


# ---- property sweeps ----

@pytest.mark.parametrize("bearing", [0, 45, 90, 135, 180, 225, 270, 315])
def test_approach_from_any_bearing(bearing):
    b = math.radians(bearing)
    n, start, step = 10, 1000.0, 120.0
    track = [((start - i * step) * math.cos(b), (start - i * step) * math.sin(b))
             for i in range(n)]
    a = classify_intent(track, ASSET)
    assert a.intent is IntentClass.APPROACH


@pytest.mark.parametrize("n", [8, 12, 16, 24])
def test_orbit_various_resolutions(n):
    a = classify_intent(orbit(n=n), ASSET)
    assert a.intent is IntentClass.RECON_ORBIT


@pytest.mark.parametrize("offset", [400, 600, 800, 1200])
def test_transit_at_various_offsets(offset):
    a = classify_intent(transit_past(y=offset), ASSET)
    assert a.intent in (IntentClass.TRANSIT, IntentClass.UNKNOWN)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_coordinated_ingress_counts(k):
    tracks = [straight_approach() for _ in range(k)] + [transit_past()]
    assert coordinated_ingress(tracks, ASSET, min_closing=k) is (k >= 1)


@pytest.mark.parametrize("step", [60, 90, 120, 200])
def test_faster_approach_higher_closure(step):
    # Non-crossing inbound leg along +x: range shrinks by exactly `step` per point.
    n = 8
    start = n * step + 500.0   # stays positive -> never overshoots the asset
    track = [(start - i * step, 0.0) for i in range(n)]
    f = track_features(track, ASSET)
    assert f.closure_rate == pytest.approx(step, rel=0.05)
