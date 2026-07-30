"""Tests for swarm size / spatial-extent / formation-geometry estimation."""

from __future__ import annotations

import math

import pytest

from frontline_drones.formation import (
    FORMATION_TYPES,
    FormationReport,
    Member,
    bounding_box,
    centroid,
    convex_hull,
    estimate_formation,
    max_pairwise_distance,
    nearest_neighbor_spacings,
    polygon_area,
)


def _rotate(members, deg):
    """Rotate an iterable of (e, n) tuples about the origin by ``deg``."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return [(e * c - n * s, e * s + n * c) for e, n in members]


def _translate(members, de, dn):
    return [(e + de, n + dn) for e, n in members]


# -- centroid -----------------------------------------------------------------

def test_centroid_basic():
    cx, cy = centroid([Member("a", 0, 0), Member("b", 10, 0), Member("c", 20, 0)])
    assert cx == pytest.approx(10.0)
    assert cy == pytest.approx(0.0)


def test_centroid_empty():
    assert centroid([]) == (0.0, 0.0)


def test_centroid_square():
    m = [Member("a", 0, 0), Member("b", 10, 0), Member("c", 10, 10), Member("d", 0, 10)]
    assert centroid(m) == pytest.approx((5.0, 5.0))


# -- bounding_box -------------------------------------------------------------

def test_bounding_box():
    m = [Member("a", -5, 2), Member("b", 10, -3), Member("c", 4, 8)]
    assert bounding_box(m) == (-5, -3, 10, 8)


def test_bounding_box_single():
    assert bounding_box([Member("a", 3, 4)]) == (3, 4, 3, 4)


# -- max_pairwise_distance ----------------------------------------------------

def test_max_pairwise_distance():
    m = [Member("a", 0, 0), Member("b", 3, 4), Member("c", 6, 8)]
    assert max_pairwise_distance(m) == pytest.approx(10.0)


def test_max_pairwise_distance_single_is_zero():
    assert max_pairwise_distance([Member("a", 1, 1)]) == 0.0


def test_max_pairwise_distance_empty_is_zero():
    assert max_pairwise_distance([]) == 0.0


# -- nearest_neighbor_spacings ------------------------------------------------

def test_nn_spacings_line():
    m = [Member("a", 0, 0), Member("b", 10, 0), Member("c", 20, 0)]
    sp = nearest_neighbor_spacings(m)
    assert sp == pytest.approx([10.0, 10.0, 10.0])


def test_nn_spacings_needs_two():
    assert nearest_neighbor_spacings([Member("a", 0, 0)]) == []
    assert nearest_neighbor_spacings([]) == []


def test_nn_spacings_picks_closest():
    m = [Member("a", 0, 0), Member("b", 3, 0), Member("c", 100, 0)]
    sp = nearest_neighbor_spacings(m)
    assert sp[0] == pytest.approx(3.0)
    assert sp[1] == pytest.approx(3.0)
    assert sp[2] == pytest.approx(97.0)


# -- convex_hull / polygon_area ----------------------------------------------

def test_convex_hull_square():
    m = [Member("a", 0, 0), Member("b", 10, 0), Member("c", 10, 10),
         Member("d", 0, 10), Member("e", 5, 5)]  # interior point excluded
    hull = convex_hull(m)
    assert set(hull) == {(0, 0), (10, 0), (10, 10), (0, 10)}


def test_convex_hull_collinear():
    m = [Member("a", 0, 0), Member("b", 5, 0), Member("c", 10, 0)]
    hull = convex_hull(m)
    assert hull == [(0.0, 0.0), (10.0, 0.0)]


def test_convex_hull_dedups_coincident():
    m = [Member("a", 0, 0), Member("b", 0, 0), Member("c", 0, 0)]
    assert convex_hull(m) == [(0.0, 0.0)]


def test_polygon_area_square():
    assert polygon_area([(0, 0), (10, 0), (10, 10), (0, 10)]) == pytest.approx(100.0)


def test_polygon_area_triangle():
    assert polygon_area([(0, 0), (10, 0), (0, 10)]) == pytest.approx(50.0)


def test_polygon_area_degenerate():
    assert polygon_area([(0, 0), (10, 0)]) == 0.0
    assert polygon_area([(0, 0)]) == 0.0


# -- estimate_formation: count & centroid ------------------------------------

def test_empty_report():
    r = estimate_formation([])
    assert isinstance(r, FormationReport)
    assert r.count == 0
    assert r.formation_type == "empty"
    assert r.compactness == 0.0
    assert r.density_per_km2 == 0.0


def test_count_matches_input():
    r = estimate_formation([(0, 0), (10, 0), (20, 0), (30, 0)])
    assert r.count == 4


def test_centroid_reported():
    r = estimate_formation([(0, 0), (10, 0), (20, 0)])
    assert r.centroid_east == pytest.approx(10.0)
    assert r.centroid_north == pytest.approx(0.0)


# -- formation-type classification -------------------------------------------

def test_single_member():
    assert estimate_formation([(0, 0)]).formation_type == "single"


def test_pair_member():
    assert estimate_formation([(0, 0), (50, 0)]).formation_type == "pair"


def test_line_no_heading():
    r = estimate_formation([(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)])
    assert r.formation_type == "line"
    assert r.anisotropy > 0.9


def test_column_along_heading():
    # Members strung east-west, all travelling east (compass 90) => column.
    tracks = [(x, 0.0, 90.0) for x in range(0, 50, 10)]
    r = estimate_formation(tracks)
    assert r.formation_type == "column"


def test_line_abreast_perpendicular_heading():
    # Members strung east-west but travelling north (compass 0) => abreast line.
    tracks = [(x, 0.0, 0.0) for x in range(0, 50, 10)]
    r = estimate_formation(tracks)
    assert r.formation_type == "line"


def test_wedge_vee():
    # Symmetric V opening from an apex at the origin, longer than it is wide so
    # the fore-aft axis is the major axis.
    tracks = [(0, 0), (30, 8), (30, -8), (60, 16), (60, -16), (90, 24), (90, -24)]
    r = estimate_formation(tracks)
    assert r.formation_type == "wedge"


def test_grid_regular():
    tracks = [(i * 10.0, j * 10.0) for i in range(3) for j in range(3)]
    r = estimate_formation(tracks)
    assert r.formation_type == "grid"
    assert r.regularity >= 0.75


def test_cluster_blob():
    # Irregular round-ish blob: low anisotropy, uneven spacing, not a grid.
    tracks = [(0, 0), (12, 2), (5, 11), (14, 13), (3, 6), (9, 15)]
    r = estimate_formation(tracks)
    assert r.formation_type == "cluster"


def test_formation_type_in_catalog():
    for tracks in (
        [],
        [(0, 0)],
        [(0, 0), (10, 0)],
        [(0, 0), (10, 0), (20, 0)],
        [(i * 10.0, j * 10.0) for i in range(3) for j in range(3)],
    ):
        assert estimate_formation(tracks).formation_type in FORMATION_TYPES


# -- geometry values ----------------------------------------------------------

def test_diameter_reported():
    r = estimate_formation([(0, 0), (30, 40)])
    assert r.diameter_m == pytest.approx(50.0)


def test_bbox_extents():
    r = estimate_formation([(0, 0), (40, 0), (40, 25), (0, 25)])
    assert r.bbox_width_m == pytest.approx(40.0)
    assert r.bbox_height_m == pytest.approx(25.0)


def test_major_axis_span_line():
    r = estimate_formation([(0, 0), (10, 0), (20, 0), (30, 0)])
    assert r.major_axis_m == pytest.approx(30.0)
    assert r.minor_axis_m == pytest.approx(0.0, abs=1e-9)


def test_hull_area_square():
    r = estimate_formation([(0, 0), (20, 0), (20, 20), (0, 20)])
    assert r.hull_area_m2 == pytest.approx(400.0)


def test_hull_area_line_is_zero():
    r = estimate_formation([(0, 0), (10, 0), (20, 0)])
    assert r.hull_area_m2 == pytest.approx(0.0)
    assert r.density_per_km2 == 0.0


def test_density_per_km2():
    # 4 members over a 1000 m x 1000 m square = 1 km^2 => 4 / km^2.
    r = estimate_formation([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    assert r.density_per_km2 == pytest.approx(4.0)


def test_dispersion_positive():
    r = estimate_formation([(0, 0), (10, 0), (0, 10), (10, 10)])
    assert r.dispersion_m > 0.0


def test_nn_stats_reported():
    r = estimate_formation([(0, 0), (10, 0), (20, 0), (30, 0)])
    assert r.nn_spacing_mean == pytest.approx(10.0)
    assert r.nn_spacing_min == pytest.approx(10.0)
    assert r.nn_spacing_std == pytest.approx(0.0, abs=1e-9)


def test_regular_line_high_regularity():
    r = estimate_formation([(0, 0), (10, 0), (20, 0), (30, 0)])
    assert r.regularity == pytest.approx(1.0)


def test_clumpy_low_regularity():
    # A tight triple plus one far outlier: nearest-neighbour spacing is uneven.
    r = estimate_formation([(0, 0), (1, 0), (2, 0), (100, 0)])
    assert r.regularity < 0.5


# -- input coercion -----------------------------------------------------------

def test_accepts_member_objects():
    r = estimate_formation([Member("x", 0, 0), Member("y", 10, 0), Member("z", 20, 0)])
    assert r.count == 3


def test_accepts_dicts():
    tracks = [
        {"track_id": "a", "east": 0, "north": 0},
        {"track_id": "b", "east": 10, "north": 0},
        {"track_id": "c", "east": 20, "north": 0},
    ]
    assert estimate_formation(tracks).count == 3


def test_accepts_xy_dicts():
    tracks = [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 20, "y": 0}]
    assert estimate_formation(tracks).count == 3


def test_accepts_tuples_with_heading():
    r = estimate_formation([(0, 0, 90), (10, 0, 90), (20, 0, 90)])
    assert r.count == 3


def test_accepts_swarm_trackpoint():
    from frontline_drones.swarm import TrackPoint

    tracks = [TrackPoint("a", 0, 0, 90, 10), TrackPoint("b", 10, 0, 90, 10),
              TrackPoint("c", 20, 0, 90, 10)]
    r = estimate_formation(tracks)
    assert r.count == 3
    # East-west string all heading east => column.
    assert r.formation_type == "column"


# -- to_dict ------------------------------------------------------------------

def test_report_to_dict_keys():
    r = estimate_formation([(0, 0), (10, 0), (20, 0)])
    d = r.to_dict()
    for k in (
        "count", "centroid_east", "centroid_north", "bbox_width_m", "bbox_height_m",
        "diameter_m", "major_axis_m", "minor_axis_m", "orientation_deg", "anisotropy",
        "dispersion_m", "nn_spacing_mean", "nn_spacing_min", "nn_spacing_std",
        "regularity", "compactness", "hull_area_m2", "density_per_km2", "formation_type",
    ):
        assert k in d


def test_to_dict_is_json_safe():
    import json

    r = estimate_formation([(0, 0, 90), (10, 0, 90), (20, 0, 90)])
    json.dumps(r.to_dict())  # must not raise


# -- determinism & invariants -------------------------------------------------

def test_deterministic():
    tracks = [(0, 0), (10, 3), (20, 1), (30, 4), (40, 2)]
    a = estimate_formation(tracks).to_dict()
    b = estimate_formation(tracks).to_dict()
    assert a == b


def test_order_independent_geometry():
    base = [(0, 0), (10, 5), (20, 1), (5, 12)]
    a = estimate_formation(base).to_dict()
    b = estimate_formation(list(reversed(base))).to_dict()
    for k in ("count", "diameter_m", "anisotropy", "hull_area_m2", "formation_type",
              "compactness", "dispersion_m"):
        assert a[k] == pytest.approx(b[k]) if isinstance(a[k], float) else a[k] == b[k]


@pytest.mark.parametrize("de,dn", [(0, 0), (100, 0), (0, -50), (1234.5, -678.9)])
def test_translation_invariance(de, dn):
    base = [(0, 0), (10, 5), (20, 1), (5, 12), (15, 9)]
    ref = estimate_formation(base)
    moved = estimate_formation(_translate(base, de, dn))
    assert moved.diameter_m == pytest.approx(ref.diameter_m)
    assert moved.anisotropy == pytest.approx(ref.anisotropy)
    assert moved.hull_area_m2 == pytest.approx(ref.hull_area_m2)
    assert moved.compactness == pytest.approx(ref.compactness)
    assert moved.formation_type == ref.formation_type
    assert moved.centroid_east == pytest.approx(ref.centroid_east + de)
    assert moved.centroid_north == pytest.approx(ref.centroid_north + dn)


@pytest.mark.parametrize("deg", [0, 30, 45, 90, 137, 180, 270])
def test_rotation_invariance_of_shape(deg):
    base = [(0, 0), (10, 4), (20, 1), (5, 9), (18, 7)]
    ref = estimate_formation(base)
    rot = estimate_formation(_rotate(base, deg))
    assert rot.diameter_m == pytest.approx(ref.diameter_m)
    assert rot.anisotropy == pytest.approx(ref.anisotropy)
    assert rot.hull_area_m2 == pytest.approx(ref.hull_area_m2)
    assert rot.compactness == pytest.approx(ref.compactness)
    assert rot.nn_spacing_mean == pytest.approx(ref.nn_spacing_mean)


@pytest.mark.parametrize("n", [3, 4, 5, 8, 12, 20])
def test_regular_line_is_line_any_length(n):
    r = estimate_formation([(i * 12.0, 0.0) for i in range(n)])
    assert r.formation_type == "line"
    assert r.count == n


@pytest.mark.parametrize("spacing", [5.0, 10.0, 25.0, 100.0])
def test_grid_stays_grid_across_spacing(spacing):
    tracks = [(i * spacing, j * spacing) for i in range(4) for j in range(4)]
    r = estimate_formation(tracks)
    assert r.formation_type == "grid"


@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0, 10.0])
def test_compactness_scale_invariant(scale):
    base = [(0, 0), (10, 0), (0, 10), (10, 10), (5, 5)]
    ref = estimate_formation(base)
    scaled = estimate_formation([(e * scale, n * scale) for e, n in base])
    # compactness is a ratio of distances => scale-free.
    assert scaled.compactness == pytest.approx(ref.compactness)
    assert scaled.anisotropy == pytest.approx(ref.anisotropy)


@pytest.mark.parametrize("tracks", [
    [(0, 0), (10, 0), (20, 0)],
    [(0, 0), (10, 5), (20, 1), (5, 12)],
    [(i * 10.0, j * 10.0) for i in range(3) for j in range(3)],
    [(0, 0), (100, 0), (50, 90)],
    [(0, 0), (1, 0), (500, 0), (501, 0)],
])
def test_scalar_fields_are_bounded(tracks):
    r = estimate_formation(tracks)
    assert 0.0 <= r.anisotropy <= 1.0
    assert 0.0 <= r.regularity <= 1.0
    assert 0.0 <= r.compactness <= 1.0
    assert 0.0 <= r.orientation_deg < 180.0
    assert r.diameter_m >= 0.0
    assert r.hull_area_m2 >= 0.0
    assert r.dispersion_m >= 0.0
    assert r.nn_spacing_min <= r.nn_spacing_mean + 1e-9


@pytest.mark.parametrize("deg", [0, 25, 90, 200, 359])
def test_column_detection_rotation_invariant(deg):
    # A moving column: string of members all sharing the travel heading.
    # Build along east, all heading east, then rotate positions AND heading.
    positions = [(x, 0.0) for x in range(0, 60, 10)]
    rot = _rotate(positions, deg)
    # _rotate turns the east axis CCW (toward north); compass bearing decreases,
    # so east (90) rotated by +deg is compass (90 - deg).
    heading = (90.0 - deg) % 360.0
    tracks = [(e, n, heading) for e, n in rot]
    r = estimate_formation(tracks)
    assert r.formation_type == "column"


def test_compactness_tight_vs_loose():
    tight = estimate_formation([(0, 0), (1, 0), (0, 1), (1, 1)])
    loose = estimate_formation([(0, 0), (1000, 0), (0, 1000), (1000, 1000)])
    # Same shape, but "tight" and "loose" have identical compactness (scale-free);
    # instead compare a genuinely more-dispersed arrangement.
    dispersed = estimate_formation([(0, 0), (1, 0), (0, 1), (1000, 1000)])
    assert dispersed.compactness < tight.compactness
    assert tight.compactness == pytest.approx(loose.compactness)


def test_single_and_empty_have_zero_extent():
    for tracks in ([], [(5, 5)]):
        r = estimate_formation(tracks)
        assert r.diameter_m == 0.0
        assert r.major_axis_m == 0.0
        assert r.minor_axis_m == 0.0


def test_two_coincident_members():
    # Degenerate but must not raise.
    r = estimate_formation([(0, 0), (0, 0)])
    assert r.count == 2
    assert r.diameter_m == 0.0
    assert r.formation_type == "pair"


def test_all_coincident_members():
    r = estimate_formation([(3, 3), (3, 3), (3, 3), (3, 3)])
    assert r.count == 4
    assert r.diameter_m == 0.0
    assert r.dispersion_m == pytest.approx(0.0)
    assert r.hull_area_m2 == 0.0
    assert r.formation_type in FORMATION_TYPES


def test_orientation_of_east_west_line():
    r = estimate_formation([(0, 0), (10, 0), (20, 0)])
    # Major axis runs east-west => compass bearing 90 (folded into [0,180)).
    assert r.orientation_deg == pytest.approx(90.0, abs=1e-6)


def test_orientation_of_north_south_line():
    r = estimate_formation([(0, 0), (0, 10), (0, 20)])
    # Major axis runs north-south => bearing 0 (folded into [0,180)).
    assert r.orientation_deg == pytest.approx(0.0, abs=1e-6) or \
        r.orientation_deg == pytest.approx(0.0)
