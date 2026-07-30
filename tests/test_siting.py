"""Tests for defensive C-UAS sensor-placement optimisation (coverage only)."""

from __future__ import annotations

import random

import pytest

from frontline_drones import siting
from frontline_drones.siting import CandidateSite, DemandPoint


# ---------------------------------------------------------------------------
# Dataclasses & converters
# ---------------------------------------------------------------------------
def test_demand_point_to_dict():
    d = DemandPoint("p", 1.0, 2.0, 3.0).to_dict()
    assert d == {"name": "p", "x": 1.0, "y": 2.0, "weight": 3.0}


def test_demand_point_default_weight():
    assert DemandPoint("p", 0.0, 0.0).weight == 1.0


def test_candidate_site_to_dict():
    d = CandidateSite("s", 1.0, 2.0, 500.0, 2.0).to_dict()
    assert d == {"name": "s", "x": 1.0, "y": 2.0, "radius_m": 500.0, "cost": 2.0}


def test_candidate_site_default_cost():
    assert CandidateSite("s", 0.0, 0.0, 100.0).cost == 1.0


def test_demand_from_latlon_origin_is_zero():
    p = siting.demand_from_latlon("p", 37.0, -122.0, 37.0, -122.0)
    assert p.x == pytest.approx(0.0, abs=1e-6)
    assert p.y == pytest.approx(0.0, abs=1e-6)


def test_demand_from_latlon_east_positive():
    p = siting.demand_from_latlon("p", 37.0, -121.99, 37.0, -122.0)
    assert p.x > 0
    assert p.y == pytest.approx(0.0, abs=1e-6)


def test_site_from_latlon_carries_radius_cost():
    s = siting.site_from_latlon("s", 37.01, -122.0, 800.0, 37.0, -122.0, cost=3.0)
    assert s.radius_m == 800.0
    assert s.cost == 3.0
    assert s.y > 0


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------
def test_grid_single_point_when_zero_area():
    pts = siting.grid_demand_points(0.0, 0.0, 10.0)
    assert len(pts) == 1
    assert (pts[0].x, pts[0].y) == (0.0, 0.0)


def test_grid_count_inclusive_of_edges():
    # 100 m wide at 50 m pitch => columns at 0, 50, 100 => 3 cols.
    pts = siting.grid_demand_points(100.0, 100.0, 50.0)
    assert len(pts) == 9  # 3 x 3


def test_grid_row_major_order():
    pts = siting.grid_demand_points(50.0, 50.0, 50.0)
    # r0c0, r0c1, r1c0, r1c1
    assert [p.name for p in pts] == ["g0_0", "g0_1", "g1_0", "g1_1"]


def test_grid_origin_offset_applied():
    pts = siting.grid_demand_points(0.0, 0.0, 10.0, origin=(5.0, -7.0))
    assert (pts[0].x, pts[0].y) == (5.0, -7.0)


def test_grid_weight_propagates():
    pts = siting.grid_demand_points(10.0, 10.0, 10.0, weight=2.5)
    assert all(p.weight == 2.5 for p in pts)


def test_grid_rejects_nonpositive_spacing():
    with pytest.raises(ValueError):
        siting.grid_demand_points(10.0, 10.0, 0.0)


def test_grid_rejects_negative_dimensions():
    with pytest.raises(ValueError):
        siting.grid_demand_points(-1.0, 10.0, 5.0)


# ---------------------------------------------------------------------------
# covers / coverage_sets
# ---------------------------------------------------------------------------
def test_covers_inside():
    assert siting.covers(CandidateSite("s", 0.0, 0.0, 100.0), DemandPoint("p", 50.0, 0.0))


def test_covers_on_boundary_is_inclusive():
    assert siting.covers(CandidateSite("s", 0.0, 0.0, 100.0), DemandPoint("p", 100.0, 0.0))


def test_covers_outside():
    assert not siting.covers(CandidateSite("s", 0.0, 0.0, 100.0), DemandPoint("p", 101.0, 0.0))


def test_coverage_sets_alignment():
    sites = [CandidateSite("a", 0.0, 0.0, 10.0), CandidateSite("b", 100.0, 0.0, 10.0)]
    pts = [DemandPoint("p0", 0.0, 0.0), DemandPoint("p1", 100.0, 0.0), DemandPoint("p2", 500.0, 0.0)]
    sets = siting.coverage_sets(sites, pts)
    assert sets == [frozenset({0}), frozenset({1})]


def test_coverage_sets_empty_when_no_points():
    sites = [CandidateSite("a", 0.0, 0.0, 10.0)]
    assert siting.coverage_sets(sites, []) == [frozenset()]


def test_coverable_indices_excludes_unreachable():
    sites = [CandidateSite("a", 0.0, 0.0, 10.0)]
    pts = [DemandPoint("p0", 0.0, 0.0), DemandPoint("p1", 9999.0, 0.0)]
    assert siting.coverable_indices(sites, pts) == frozenset({0})


# ---------------------------------------------------------------------------
# greedy_max_coverage — behaviour
# ---------------------------------------------------------------------------
def _two_site_setup():
    # Two disjoint clusters; each site covers one cluster.
    sites = [
        CandidateSite("west", 0.0, 0.0, 50.0),
        CandidateSite("east", 1000.0, 0.0, 50.0),
    ]
    pts = [
        DemandPoint("w0", 0.0, 0.0),
        DemandPoint("w1", 10.0, 0.0),
        DemandPoint("w2", 20.0, 0.0),
        DemandPoint("e0", 1000.0, 0.0),
    ]
    return sites, pts


def test_max_coverage_k0_empty():
    sites, pts = _two_site_setup()
    plan = siting.greedy_max_coverage(sites, pts, 0)
    assert plan.selected == ()
    assert plan.covered_weight == 0.0
    assert plan.coverage_fraction == 0.0


def test_max_coverage_k1_picks_biggest_cluster():
    sites, pts = _two_site_setup()
    plan = siting.greedy_max_coverage(sites, pts, 1)
    assert plan.selected == ("west",)  # covers 3 points vs east's 1
    assert plan.covered_weight == pytest.approx(3.0)


def test_max_coverage_k2_covers_all():
    sites, pts = _two_site_setup()
    plan = siting.greedy_max_coverage(sites, pts, 2)
    assert set(plan.selected) == {"west", "east"}
    assert plan.coverable_fraction == pytest.approx(1.0)
    assert plan.uncovered_points == ()


def test_max_coverage_stops_when_no_gain():
    sites, pts = _two_site_setup()
    # Ask for more sites than add value; only 2 useful sites exist.
    plan = siting.greedy_max_coverage(sites, pts, 5)
    assert len(plan.selected) == 2


def test_max_coverage_negative_k_raises():
    sites, pts = _two_site_setup()
    with pytest.raises(ValueError):
        siting.greedy_max_coverage(sites, pts, -1)


def test_max_coverage_marginal_steps_recorded():
    sites, pts = _two_site_setup()
    plan = siting.greedy_max_coverage(sites, pts, 2)
    assert plan.steps[0].site == "west"
    assert plan.steps[0].marginal_points == 3
    assert plan.steps[1].site == "east"
    assert plan.steps[1].marginal_points == 1


def test_max_coverage_marginal_gains_non_increasing():
    # A classic property of greedy max-coverage: marginal gains never rise.
    sites = [
        CandidateSite("a", 0.0, 0.0, 30.0),
        CandidateSite("b", 100.0, 0.0, 30.0),
        CandidateSite("c", 200.0, 0.0, 30.0),
    ]
    pts = (
        [DemandPoint(f"a{i}", i, 0.0) for i in range(5)]
        + [DemandPoint(f"b{i}", 100 + i, 0.0) for i in range(3)]
        + [DemandPoint(f"c{i}", 200 + i, 0.0) for i in range(1)]
    )
    plan = siting.greedy_max_coverage(sites, pts, 3)
    gains = [s.marginal_weight for s in plan.steps]
    assert gains == sorted(gains, reverse=True)


def test_max_coverage_weighted_prefers_high_value():
    sites = [
        CandidateSite("many", 0.0, 0.0, 10.0),
        CandidateSite("valuable", 1000.0, 0.0, 10.0),
    ]
    pts = [
        DemandPoint("m0", 0.0, 0.0, 1.0),
        DemandPoint("m1", 5.0, 0.0, 1.0),
        DemandPoint("vip", 1000.0, 0.0, 10.0),
    ]
    plan = siting.greedy_max_coverage(sites, pts, 1)
    assert plan.selected == ("valuable",)


def test_max_coverage_deterministic_tie_break_first_site():
    # Two identical-coverage sites; the earlier one must win, every run.
    sites = [
        CandidateSite("first", 0.0, 0.0, 10.0),
        CandidateSite("second", 0.0, 0.0, 10.0),
    ]
    pts = [DemandPoint("p", 0.0, 0.0)]
    for _ in range(5):
        assert siting.greedy_max_coverage(sites, pts, 1).selected == ("first",)


# ---------------------------------------------------------------------------
# greedy_set_cover
# ---------------------------------------------------------------------------
def test_set_cover_covers_all_reachable():
    sites, pts = _two_site_setup()
    plan = siting.greedy_set_cover(sites, pts)
    assert plan.coverable_fraction == pytest.approx(1.0)
    assert set(plan.selected) == {"west", "east"}


def test_set_cover_minimal_when_one_site_suffices():
    sites = [
        CandidateSite("big", 0.0, 0.0, 1000.0),
        CandidateSite("small", 0.0, 0.0, 10.0),
    ]
    pts = siting.grid_demand_points(100.0, 100.0, 50.0)
    plan = siting.greedy_set_cover(sites, pts)
    assert plan.selected == ("big",)


def test_set_cover_reports_unreachable_gap():
    sites = [CandidateSite("a", 0.0, 0.0, 10.0)]
    pts = [DemandPoint("in", 0.0, 0.0), DemandPoint("out", 9999.0, 0.0)]
    plan = siting.greedy_set_cover(sites, pts)
    assert plan.uncovered_points == (1,)
    assert any("outside every candidate footprint" in a for a in plan.advisories)


def test_set_cover_full_coverage_advisory():
    sites = [CandidateSite("big", 0.0, 0.0, 1000.0)]
    pts = siting.grid_demand_points(50.0, 50.0, 50.0)
    plan = siting.greedy_set_cover(sites, pts)
    assert any("Full detection coverage" in a for a in plan.advisories)


# ---------------------------------------------------------------------------
# budget_max_coverage
# ---------------------------------------------------------------------------
def test_budget_zero_selects_nothing():
    sites, pts = _two_site_setup()
    plan = siting.budget_max_coverage(sites, pts, 0.0)
    assert plan.selected == ()


def test_budget_respected():
    sites = [
        CandidateSite("a", 0.0, 0.0, 10.0, cost=1.0),
        CandidateSite("b", 100.0, 0.0, 10.0, cost=1.0),
        CandidateSite("c", 200.0, 0.0, 10.0, cost=1.0),
    ]
    pts = [DemandPoint("pa", 0.0, 0.0), DemandPoint("pb", 100.0, 0.0), DemandPoint("pc", 200.0, 0.0)]
    plan = siting.budget_max_coverage(sites, pts, 2.0)
    assert plan.total_cost <= 2.0 + 1e-9
    assert len(plan.selected) == 2


def test_budget_prefers_cost_efficient_site():
    # 'cheap' covers 2 pts at cost 1 (ratio 2); 'pricey' covers 3 pts at cost 10 (ratio 0.3).
    sites = [
        CandidateSite("pricey", 0.0, 0.0, 10.0, cost=10.0),
        CandidateSite("cheap", 1000.0, 0.0, 10.0, cost=1.0),
    ]
    pts = [
        DemandPoint("p0", 0.0, 0.0),
        DemandPoint("p1", 5.0, 0.0),
        DemandPoint("p2", 9.0, 0.0),
        DemandPoint("c0", 1000.0, 0.0),
        DemandPoint("c1", 1005.0, 0.0),
    ]
    plan = siting.budget_max_coverage(sites, pts, 1.0)
    assert plan.selected == ("cheap",)


def test_budget_negative_raises():
    sites, pts = _two_site_setup()
    with pytest.raises(ValueError):
        siting.budget_max_coverage(sites, pts, -1.0)


def test_budget_total_cost_accumulates():
    sites = [
        CandidateSite("a", 0.0, 0.0, 10.0, cost=2.0),
        CandidateSite("b", 100.0, 0.0, 10.0, cost=3.0),
    ]
    pts = [DemandPoint("pa", 0.0, 0.0), DemandPoint("pb", 100.0, 0.0)]
    plan = siting.budget_max_coverage(sites, pts, 10.0)
    assert plan.total_cost == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_no_points_returns_full_fraction():
    sites = [CandidateSite("a", 0.0, 0.0, 10.0)]
    plan = siting.greedy_max_coverage(sites, [], 1)
    assert plan.coverage_fraction == 1.0
    assert any("No demand points" in a for a in plan.advisories)


def test_no_sites_zero_coverage_advisory():
    pts = siting.grid_demand_points(50.0, 50.0, 50.0)
    plan = siting.greedy_max_coverage([], pts, 3)
    assert plan.selected == ()
    assert plan.coverage_fraction == 0.0
    assert any("zero detection coverage" in a for a in plan.advisories)


def test_all_points_unreachable():
    sites = [CandidateSite("a", 0.0, 0.0, 1.0)]
    pts = [DemandPoint("far", 5000.0, 0.0)]
    plan = siting.greedy_set_cover(sites, pts)
    assert plan.selected == ()
    assert plan.coverable_weight == 0.0
    assert plan.coverable_fraction == 1.0  # nothing reachable => vacuously "full"


def test_plan_to_dict_keys():
    sites, pts = _two_site_setup()
    d = siting.greedy_max_coverage(sites, pts, 2).to_dict()
    assert set(d) == {
        "selected",
        "covered_weight",
        "total_weight",
        "coverable_weight",
        "coverage_fraction",
        "coverable_fraction",
        "covered_points",
        "uncovered_points",
        "steps",
        "total_cost",
        "advisories",
    }


def test_step_to_dict_keys():
    sites, pts = _two_site_setup()
    step = siting.greedy_max_coverage(sites, pts, 1).steps[0]
    assert set(step.to_dict()) == {
        "site",
        "marginal_weight",
        "marginal_points",
        "cost",
        "cumulative_cost",
    }


def test_covered_and_uncovered_partition_all_points():
    sites, pts = _two_site_setup()
    plan = siting.greedy_max_coverage(sites, pts, 1)
    assert set(plan.covered_points) | set(plan.uncovered_points) == set(range(len(pts)))
    assert set(plan.covered_points) & set(plan.uncovered_points) == set()


# ---------------------------------------------------------------------------
# Parametrized property sweeps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 8])
def test_coverage_monotonic_in_k(k):
    """More allowed sites never reduces covered weight (greedy is nested)."""
    sites = [CandidateSite(f"s{i}", i * 40.0, 0.0, 25.0) for i in range(8)]
    pts = [DemandPoint(f"p{i}", i * 20.0, 0.0) for i in range(20)]
    plan_k = siting.greedy_max_coverage(sites, pts, k)
    plan_k1 = siting.greedy_max_coverage(sites, pts, k + 1)
    assert plan_k1.covered_weight >= plan_k.covered_weight - 1e-9


@pytest.mark.parametrize("k", [1, 2, 3, 6, 10])
def test_selected_count_never_exceeds_k(k):
    sites = [CandidateSite(f"s{i}", i * 40.0, 0.0, 25.0) for i in range(6)]
    pts = [DemandPoint(f"p{i}", i * 20.0, 0.0) for i in range(15)]
    plan = siting.greedy_max_coverage(sites, pts, k)
    assert len(plan.selected) <= k


@pytest.mark.parametrize("radius", [1.0, 10.0, 50.0, 200.0, 1000.0])
def test_larger_radius_never_reduces_coverage(radius):
    """A single site covers weakly monotonically more as its radius grows."""
    pts = siting.grid_demand_points(400.0, 400.0, 50.0)
    small = [CandidateSite("s", 200.0, 200.0, radius)]
    big = [CandidateSite("s", 200.0, 200.0, radius * 2)]
    p_small = siting.greedy_max_coverage(small, pts, 1)
    p_big = siting.greedy_max_coverage(big, pts, 1)
    assert p_big.covered_weight >= p_small.covered_weight - 1e-9


@pytest.mark.parametrize("seed", range(12))
def test_fraction_bounds_and_determinism(seed):
    """Fractions stay in [0,1] and the planner is reproducible for any layout."""
    rng = random.Random(seed)
    sites = [
        CandidateSite(f"s{i}", rng.uniform(0, 1000), rng.uniform(0, 1000), rng.uniform(20, 200))
        for i in range(rng.randint(1, 8))
    ]
    pts = [
        DemandPoint(f"p{i}", rng.uniform(0, 1000), rng.uniform(0, 1000), rng.uniform(0.5, 3.0))
        for i in range(rng.randint(1, 40))
    ]
    k = rng.randint(0, len(sites))
    a = siting.greedy_max_coverage(sites, pts, k)
    b = siting.greedy_max_coverage(sites, pts, k)
    assert a.to_dict() == b.to_dict()  # determinism
    assert 0.0 <= a.coverage_fraction <= 1.0 + 1e-9
    assert 0.0 <= a.coverable_fraction <= 1.0 + 1e-9
    assert a.covered_weight <= a.coverable_weight + 1e-9


@pytest.mark.parametrize("seed", range(12))
def test_set_cover_covers_everything_reachable(seed):
    rng = random.Random(1000 + seed)
    sites = [
        CandidateSite(f"s{i}", rng.uniform(0, 500), rng.uniform(0, 500), rng.uniform(50, 300))
        for i in range(rng.randint(2, 6))
    ]
    pts = [
        DemandPoint(f"p{i}", rng.uniform(0, 500), rng.uniform(0, 500))
        for i in range(rng.randint(5, 30))
    ]
    plan = siting.greedy_set_cover(sites, pts)
    coverable = siting.coverable_indices(sites, pts)
    # Every reachable point must be watched by the set-cover result.
    assert coverable <= set(plan.covered_points)


@pytest.mark.parametrize("seed", range(10))
def test_budget_never_overspends(seed):
    rng = random.Random(2000 + seed)
    sites = [
        CandidateSite(
            f"s{i}", rng.uniform(0, 500), rng.uniform(0, 500),
            rng.uniform(50, 200), cost=rng.uniform(0.5, 4.0),
        )
        for i in range(rng.randint(2, 7))
    ]
    pts = [DemandPoint(f"p{i}", rng.uniform(0, 500), rng.uniform(0, 500)) for i in range(20)]
    budget = rng.uniform(0.0, 8.0)
    plan = siting.budget_max_coverage(sites, pts, budget)
    assert plan.total_cost <= budget + 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_selecting_all_sites_reaches_coverable_ceiling(seed):
    """With k = number of sites, greedy watches every reachable point."""
    rng = random.Random(3000 + seed)
    n = rng.randint(1, 6)
    sites = [
        CandidateSite(f"s{i}", rng.uniform(0, 400), rng.uniform(0, 400), rng.uniform(60, 250))
        for i in range(n)
    ]
    pts = [DemandPoint(f"p{i}", rng.uniform(0, 400), rng.uniform(0, 400)) for i in range(25)]
    plan = siting.greedy_max_coverage(sites, pts, n)
    assert plan.covered_weight == pytest.approx(plan.coverable_weight)


@pytest.mark.parametrize("spacing", [10.0, 25.0, 50.0, 100.0])
def test_grid_points_within_bounds(spacing):
    pts = siting.grid_demand_points(200.0, 150.0, spacing)
    assert all(0.0 <= p.x <= 200.0 + 1e-9 for p in pts)
    assert all(0.0 <= p.y <= 150.0 + 1e-9 for p in pts)


@pytest.mark.parametrize(
    "dx,dy,radius,expected",
    [
        (0.0, 0.0, 100.0, True),
        (100.0, 0.0, 100.0, True),
        (0.0, 100.0, 100.0, True),
        (70.7, 70.7, 100.0, True),
        (71.0, 71.0, 100.0, False),
        (100.1, 0.0, 100.0, False),
    ],
)
def test_covers_circle_geometry(dx, dy, radius, expected):
    site = CandidateSite("s", 0.0, 0.0, radius)
    assert siting.covers(site, DemandPoint("p", dx, dy)) is expected


@pytest.mark.parametrize("n_useful", [1, 2, 3, 4])
def test_step_count_matches_selected(n_useful):
    # n_useful disjoint clusters, each reachable by exactly one site.
    sites = [CandidateSite(f"s{i}", i * 1000.0, 0.0, 20.0) for i in range(n_useful)]
    pts = [DemandPoint(f"p{i}", i * 1000.0, 0.0) for i in range(n_useful)]
    plan = siting.greedy_set_cover(sites, pts)
    assert len(plan.steps) == len(plan.selected) == n_useful


def test_cumulative_cost_monotone_in_steps():
    sites = [CandidateSite(f"s{i}", i * 1000.0, 0.0, 20.0, cost=float(i + 1)) for i in range(4)]
    pts = [DemandPoint(f"p{i}", i * 1000.0, 0.0) for i in range(4)]
    plan = siting.greedy_set_cover(sites, pts)
    cum = [s.cumulative_cost for s in plan.steps]
    assert cum == sorted(cum)


def test_integration_latlon_workflow():
    """End-to-end: build reachable geographic demand + sites, cover them."""
    ref_lat, ref_lon = 38.9, -77.0
    pts = [
        siting.demand_from_latlon("asset_a", 38.9010, -77.0000, ref_lat, ref_lon),
        siting.demand_from_latlon("asset_b", 38.9000, -77.0010, ref_lat, ref_lon),
    ]
    sites = [
        siting.site_from_latlon("post_1", 38.9005, -77.0005, 300.0, ref_lat, ref_lon),
        siting.site_from_latlon("post_2", 38.8500, -77.0500, 100.0, ref_lat, ref_lon),
    ]
    plan = siting.greedy_max_coverage(sites, pts, 1)
    assert plan.selected == ("post_1",)
    assert plan.coverable_fraction == pytest.approx(1.0)
