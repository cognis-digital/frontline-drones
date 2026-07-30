"""Tests for multi-track swarm / coordinated-cluster detection."""

from __future__ import annotations

import pytest

from frontline_drones.swarm import (
    SwarmReport,
    TrackPoint,
    analyze_swarm,
    cluster_tracks,
    heading_coherence,
    mean_heading_deg,
)

# -- heading_coherence --------------------------------------------------------

def test_coherence_identical_headings():
    assert heading_coherence([90.0, 90.0, 90.0]) == pytest.approx(1.0)


def test_coherence_opposite_cancels():
    assert heading_coherence([0.0, 180.0]) == pytest.approx(0.0, abs=1e-9)


def test_coherence_orthogonal_spread():
    # N, E, S, W cancel to zero resultant.
    assert heading_coherence([0.0, 90.0, 180.0, 270.0]) == pytest.approx(0.0, abs=1e-9)


def test_coherence_in_unit_interval():
    c = heading_coherence([10.0, 20.0, 35.0, 15.0])
    assert 0.0 <= c <= 1.0


def test_coherence_empty_is_zero():
    assert heading_coherence([]) == 0.0


def test_coherence_ignores_none():
    assert heading_coherence([90.0, None, 90.0]) == pytest.approx(1.0)


def test_coherence_wraps_around_zero():
    # 350 and 10 are 20 deg apart, should be highly coherent.
    assert heading_coherence([350.0, 10.0]) > 0.98


# -- mean_heading_deg ---------------------------------------------------------

def test_mean_heading_basic():
    assert mean_heading_deg([80.0, 100.0]) == pytest.approx(90.0, abs=1e-6)


def test_mean_heading_wraps():
    assert mean_heading_deg([350.0, 10.0]) == pytest.approx(0.0, abs=1e-6)


def test_mean_heading_none_when_empty():
    assert mean_heading_deg([]) is None


def test_mean_heading_none_when_cancel():
    assert mean_heading_deg([0.0, 180.0]) is None


# -- cluster_tracks -----------------------------------------------------------

def test_single_cluster():
    pts = [TrackPoint("a", 0, 0), TrackPoint("b", 10, 0), TrackPoint("c", 20, 0)]
    groups = cluster_tracks(pts, link_radius_m=15)
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_two_clusters():
    pts = [TrackPoint("a", 0, 0), TrackPoint("b", 5, 0),
           TrackPoint("c", 1000, 0), TrackPoint("d", 1005, 0)]
    groups = cluster_tracks(pts, link_radius_m=20)
    assert len(groups) == 2


def test_single_link_chaining():
    # Chain of points each within radius of the next => one cluster.
    pts = [TrackPoint(str(i), i * 10.0, 0.0) for i in range(5)]
    groups = cluster_tracks(pts, link_radius_m=11)
    assert len(groups) == 1


def test_isolated_points_are_singletons():
    pts = [TrackPoint("a", 0, 0), TrackPoint("b", 500, 0), TrackPoint("c", 1000, 0)]
    groups = cluster_tracks(pts, link_radius_m=10)
    assert len(groups) == 3


def test_clusters_sorted_largest_first():
    pts = [TrackPoint("a", 0, 0), TrackPoint("b", 5, 0), TrackPoint("c", 10, 0),
           TrackPoint("d", 1000, 0)]
    groups = cluster_tracks(pts, link_radius_m=15)
    assert len(groups[0]) >= len(groups[1])


# -- analyze_swarm ------------------------------------------------------------

def test_empty_report():
    r = analyze_swarm([])
    assert isinstance(r, SwarmReport)
    assert r.num_tracks == 0
    assert not r.swarm_detected
    assert r.largest_cluster_size == 0


def test_coordinated_swarm_detected():
    tracks = [(0, 0, 90, 10), (20, 0, 91, 10), (40, 0, 89, 10)]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3, coherence_threshold=0.7)
    assert r.swarm_detected
    assert r.num_clusters == 1
    assert r.clusters[0].size == 3
    assert r.clusters[0].coordinated


def test_scattered_headings_not_coordinated():
    tracks = [(0, 0, 0, 10), (20, 0, 120, 10), (40, 0, 240, 10)]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3, coherence_threshold=0.7)
    assert not r.swarm_detected
    assert r.clusters[0].coordinated is False


def test_too_few_members_not_coordinated():
    tracks = [(0, 0, 90, 10), (20, 0, 90, 10)]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3)
    assert not r.swarm_detected


def test_spread_out_not_clustered():
    tracks = [(0, 0, 90, 10), (5000, 0, 90, 10), (10000, 0, 90, 10)]
    r = analyze_swarm(tracks, link_radius_m=100, min_size=3)
    assert r.num_clusters == 3
    assert not r.swarm_detected


def test_two_groups_one_coordinated():
    tracks = [
        (0, 0, 90, 10), (20, 0, 90, 10), (40, 0, 90, 10),   # tight, coherent
        (5000, 0, 0, 5), (5020, 0, 180, 5),                 # far pair, incoherent
    ]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3)
    assert r.swarm_detected
    assert len(r.coordinated_clusters) == 1


def test_centroid_computed():
    tracks = [(0, 0, 90), (10, 0, 90), (20, 0, 90)]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3)
    c = r.clusters[0]
    assert c.centroid_east == pytest.approx(10.0)
    assert c.centroid_north == pytest.approx(0.0)


def test_radius_and_spread():
    tracks = [(0, 0, 90), (10, 0, 90), (20, 0, 90)]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3)
    c = r.clusters[0]
    assert c.radius_m == pytest.approx(10.0)
    assert c.spread_m == pytest.approx((10 + 0 + 10) / 3)


def test_mean_speed_computed():
    tracks = [(0, 0, 90, 8), (10, 0, 90, 12), (20, 0, 90, 10)]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3)
    assert r.clusters[0].mean_speed_mps == pytest.approx(10.0)


def test_coordinated_without_headings_uses_size():
    # No headings supplied: a tight group of >= min_size still flags coordinated.
    tracks = [(0, 0), (10, 0), (20, 0)]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3)
    assert r.clusters[0].coordinated
    assert r.clusters[0].mean_heading_deg is None


def test_largest_cluster_size_reported():
    tracks = [(0, 0, 90), (10, 0, 90), (20, 0, 90), (30, 0, 90), (5000, 0, 90)]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3)
    assert r.largest_cluster_size == 4


def test_accepts_trackpoint_objects():
    tracks = [TrackPoint("x", 0, 0, 90, 10), TrackPoint("y", 10, 0, 90, 10),
              TrackPoint("z", 20, 0, 90, 10)]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3)
    assert r.swarm_detected
    assert set(r.clusters[0].track_ids) == {"x", "y", "z"}


def test_accepts_dicts():
    tracks = [{"track_id": "a", "east": 0, "north": 0, "heading_deg": 90, "speed_mps": 10},
              {"track_id": "b", "east": 10, "north": 0, "heading_deg": 90, "speed_mps": 10},
              {"track_id": "c", "east": 20, "north": 0, "heading_deg": 90, "speed_mps": 10}]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3)
    assert r.clusters[0].size == 3


def test_bad_link_radius_raises():
    with pytest.raises(ValueError):
        analyze_swarm([(0, 0), (1, 1)], link_radius_m=0)


def test_report_to_dict_keys():
    r = analyze_swarm([(0, 0, 90), (10, 0, 90), (20, 0, 90)], link_radius_m=50, min_size=3)
    d = r.to_dict()
    for k in ("num_tracks", "num_clusters", "largest_cluster_size", "swarm_detected",
              "clusters", "coordinated_clusters"):
        assert k in d


def test_cluster_to_dict_keys():
    r = analyze_swarm([(0, 0, 90), (10, 0, 90), (20, 0, 90)], link_radius_m=50, min_size=3)
    d = r.clusters[0].to_dict()
    for k in ("track_ids", "size", "centroid_east", "centroid_north", "radius_m",
              "spread_m", "heading_coherence", "mean_heading_deg", "mean_speed_mps",
              "coordinated"):
        assert k in d


def test_deterministic():
    tracks = [(0, 0, 90, 10), (20, 0, 90, 10), (40, 0, 90, 10), (5000, 0, 0, 5)]
    a = analyze_swarm(tracks, link_radius_m=50, min_size=3).to_dict()
    b = analyze_swarm(tracks, link_radius_m=50, min_size=3).to_dict()
    assert a == b


def test_track_ids_sorted_in_cluster():
    tracks = [TrackPoint("c", 0, 0, 90), TrackPoint("a", 10, 0, 90), TrackPoint("b", 20, 0, 90)]
    r = analyze_swarm(tracks, link_radius_m=50, min_size=3)
    assert list(r.clusters[0].track_ids) == ["a", "b", "c"]
