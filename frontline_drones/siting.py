"""Defensive C-UAS sensor-placement optimisation (coverage/awareness only).

Given a set of *candidate* detection-sensor sites and an area of airspace to
keep under observation, this module decides **where to put passive detection
sensors** so that as much of the area as possible is watched. It answers the
purely defensive question "which of my candidate sensor locations, chosen
together, give me the most detection coverage of the sky I care about?".

This is the classic *maximum-coverage / set-cover* problem. A demand set (a
grid of points, or an explicit list of places whose airspace matters) is
covered by a candidate site when the point falls inside that site's modeled
detection footprint. A greedy selection repeatedly adds the site that watches
the most still-unwatched, importance-weighted demand — the standard
``(1 - 1/e)``-approximation to maximum coverage.

Scope: the output is a *sensor emplacement plan* for **detection/awareness**.
Sites here are sensors (radar / acoustic / EO-IR / RF-DF observation posts),
never effectors. There is no engagement, weapon, jamming, targeting,
fire-control or gap-to-exploit content of any kind: coverage gaps reported here
are gaps in *our own detection* to be filled with more sensors, not openings to
attack. Pure standard library, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geo import enu_offset_m


@dataclass(frozen=True)
class DemandPoint:
    """A place whose airspace we want kept under detection coverage.

    Coordinates are metres in a local planar (ENU-style) frame; use
    :func:`demand_from_latlon` to build one from WGS-84 degrees.

    Attributes:
        name: Token identifying the point (e.g. a grid cell or asset).
        x: East offset, metres.
        y: North offset, metres.
        weight: Relative importance (>= 0); higher means more valuable to watch.
    """

    name: str
    x: float
    y: float
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y, "weight": self.weight}


@dataclass(frozen=True)
class CandidateSite:
    """A candidate location for a passive detection sensor (no engagement role).

    Attributes:
        name: Token identifying the site.
        x: East offset, metres.
        y: North offset, metres.
        radius_m: Modeled detection footprint radius, metres (> 0).
        cost: Relative emplacement cost (>= 0); used by budget-limited planning.
    """

    name: str
    x: float
    y: float
    radius_m: float
    cost: float = 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "radius_m": self.radius_m,
            "cost": self.cost,
        }


def demand_from_latlon(
    name: str,
    lat: float,
    lon: float,
    ref_lat: float,
    ref_lon: float,
    weight: float = 1.0,
) -> DemandPoint:
    """Build a :class:`DemandPoint` from WGS-84 degrees via a local ENU frame."""
    east, north = enu_offset_m(lat, lon, ref_lat, ref_lon)
    return DemandPoint(name=name, x=east, y=north, weight=weight)


def site_from_latlon(
    name: str,
    lat: float,
    lon: float,
    radius_m: float,
    ref_lat: float,
    ref_lon: float,
    cost: float = 1.0,
) -> CandidateSite:
    """Build a :class:`CandidateSite` from WGS-84 degrees via a local ENU frame."""
    east, north = enu_offset_m(lat, lon, ref_lat, ref_lon)
    return CandidateSite(name=name, x=east, y=north, radius_m=radius_m, cost=cost)


def grid_demand_points(
    width_m: float,
    height_m: float,
    spacing_m: float,
    origin: tuple[float, float] = (0.0, 0.0),
    weight: float = 1.0,
) -> list[DemandPoint]:
    """Tile a rectangular area with an evenly-spaced grid of demand points.

    The area spans ``[origin_x, origin_x + width_m] x [origin_y, origin_y +
    height_m]``; points are laid on a lattice of pitch ``spacing_m`` (inclusive
    of both edges). Deterministic row-major (north-then-east) ordering, so cell
    names and iteration order are stable across runs.
    """
    if width_m < 0 or height_m < 0:
        raise ValueError("width_m and height_m must be non-negative")
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    ox, oy = origin
    n_cols = int(math.floor(width_m / spacing_m + 1e-9)) + 1
    n_rows = int(math.floor(height_m / spacing_m + 1e-9)) + 1
    points: list[DemandPoint] = []
    for r in range(n_rows):
        for c in range(n_cols):
            points.append(
                DemandPoint(
                    name=f"g{r}_{c}",
                    x=ox + c * spacing_m,
                    y=oy + r * spacing_m,
                    weight=weight,
                )
            )
    return points


def covers(site: CandidateSite, point: DemandPoint) -> bool:
    """True when ``point`` lies within ``site``'s detection footprint."""
    return math.hypot(point.x - site.x, point.y - site.y) <= site.radius_m


def coverage_sets(
    sites: list[CandidateSite],
    points: list[DemandPoint],
) -> list[frozenset[int]]:
    """For each site, the frozenset of demand-point indices it covers.

    Index positions correspond to the order of ``points``; the returned list is
    aligned with ``sites``. This is the reusable coverage relation the greedy
    planners consume.
    """
    sets: list[frozenset[int]] = []
    for site in sites:
        covered = frozenset(i for i, p in enumerate(points) if covers(site, p))
        sets.append(covered)
    return sets


def _total_weight(points: list[DemandPoint]) -> float:
    return math.fsum(p.weight for p in points)


def coverable_indices(
    sites: list[CandidateSite],
    points: list[DemandPoint],
) -> frozenset[int]:
    """Indices of demand points that *some* candidate site can cover.

    Points outside every candidate footprint can never be watched by any
    selection; the planners treat this set as the achievable ceiling.
    """
    out: set[int] = set()
    for cov in coverage_sets(sites, points):
        out |= cov
    return frozenset(out)


@dataclass(frozen=True)
class PlacementStep:
    """One greedy pick: the site chosen and the weight of demand it newly watched.

    Attributes:
        site: Name of the site added at this step.
        marginal_weight: Importance-weighted demand this site covered that no
            previously selected site covered (its marginal gain).
        marginal_points: Count of newly-covered demand points.
        cost: Emplacement cost of the site added.
        cumulative_cost: Running total emplacement cost through this step.
    """

    site: str
    marginal_weight: float
    marginal_points: int
    cost: float
    cumulative_cost: float

    def to_dict(self) -> dict:
        return {
            "site": self.site,
            "marginal_weight": round(self.marginal_weight, 6),
            "marginal_points": self.marginal_points,
            "cost": self.cost,
            "cumulative_cost": self.cumulative_cost,
        }


@dataclass(frozen=True)
class PlacementPlan:
    """A defensive sensor-emplacement plan (detection coverage only).

    Attributes:
        selected: Names of chosen sites, in the order the greedy added them.
        covered_weight: Importance-weighted demand the plan watches.
        total_weight: Importance-weighted demand of the whole area.
        coverable_weight: Weighted demand that *any* selection could watch (the
            achievable ceiling given the candidate sites).
        coverage_fraction: ``covered_weight / total_weight`` in ``[0, 1]``.
        coverable_fraction: ``covered_weight / coverable_weight`` in ``[0, 1]``;
            1.0 means every reachable point is watched.
        covered_points: Sorted indices of demand points now watched.
        uncovered_points: Sorted indices of demand points still unwatched.
        steps: Per-pick marginal-gain breakdown.
        total_cost: Sum of emplacement costs of the selected sites.
        advisories: Plain-language notes about residual detection gaps.
    """

    selected: tuple[str, ...]
    covered_weight: float
    total_weight: float
    coverable_weight: float
    coverage_fraction: float
    coverable_fraction: float
    covered_points: tuple[int, ...]
    uncovered_points: tuple[int, ...]
    steps: tuple[PlacementStep, ...] = field(default_factory=tuple)
    total_cost: float = 0.0
    advisories: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "selected": list(self.selected),
            "covered_weight": round(self.covered_weight, 6),
            "total_weight": round(self.total_weight, 6),
            "coverable_weight": round(self.coverable_weight, 6),
            "coverage_fraction": round(self.coverage_fraction, 6),
            "coverable_fraction": round(self.coverable_fraction, 6),
            "covered_points": list(self.covered_points),
            "uncovered_points": list(self.uncovered_points),
            "steps": [s.to_dict() for s in self.steps],
            "total_cost": round(self.total_cost, 6),
            "advisories": list(self.advisories),
        }


def _build_plan(
    sites: list[CandidateSite],
    points: list[DemandPoint],
    chosen: list[int],
    cov_sets: list[frozenset[int]],
    steps: list[PlacementStep],
) -> PlacementPlan:
    """Assemble a :class:`PlacementPlan` from a finished greedy selection."""
    covered: set[int] = set()
    for idx in chosen:
        covered |= cov_sets[idx]

    total_w = _total_weight(points)
    coverable = coverable_indices(sites, points)
    coverable_w = math.fsum(points[i].weight for i in coverable)
    covered_w = math.fsum(points[i].weight for i in covered)

    all_idx = set(range(len(points)))
    uncovered = sorted(all_idx - covered)

    cov_frac = covered_w / total_w if total_w > 0 else 1.0
    coverable_frac = covered_w / coverable_w if coverable_w > 0 else 1.0
    total_cost = math.fsum(sites[i].cost for i in chosen)

    advisories: list[str] = []
    unreachable = sorted(all_idx - coverable)
    if not points:
        advisories.append("No demand points supplied; nothing to cover.")
    elif not chosen:
        advisories.append("No sites selected; the area has zero detection coverage.")
    if unreachable:
        advisories.append(
            f"{len(unreachable)} demand point(s) lie outside every candidate footprint and "
            f"can never be watched by any selection; add candidate sensor sites nearer them "
            f"to close this detection gap."
        )
    residual = sorted(coverable - covered)
    if residual:
        advisories.append(
            f"{len(residual)} reachable demand point(s) remain unwatched under this plan; "
            f"raise the site budget or add sensors to cover them."
        )
    if points and chosen and not residual and not unreachable:
        advisories.append("Full detection coverage of all demand points achieved.")

    return PlacementPlan(
        selected=tuple(sites[i].name for i in chosen),
        covered_weight=covered_w,
        total_weight=total_w,
        coverable_weight=coverable_w,
        coverage_fraction=cov_frac,
        coverable_fraction=coverable_frac,
        covered_points=tuple(sorted(covered)),
        uncovered_points=tuple(uncovered),
        steps=tuple(steps),
        total_cost=total_cost,
        advisories=tuple(advisories),
    )


def _greedy_select(
    sites: list[CandidateSite],
    points: list[DemandPoint],
    *,
    max_sites: int | None,
    max_cost: float | None,
    cost_weighted: bool,
    stop_when_covered: bool,
) -> PlacementPlan:
    """Core greedy loop shared by the public planners.

    At each step it scores every not-yet-selected site by the importance-weighted
    demand it would newly cover (optionally divided by its cost for a
    cost-efficiency ordering), adds the best, and repeats until a stopping
    condition trips. Ties break deterministically toward the earliest site in
    input order, so the result is fully reproducible.
    """
    cov_sets = coverage_sets(sites, points)
    weights = [p.weight for p in points]

    chosen: list[int] = []
    steps: list[PlacementStep] = []
    covered: set[int] = set()
    cum_cost = 0.0
    coverable = coverable_indices(sites, points)

    while True:
        if max_sites is not None and len(chosen) >= max_sites:
            break
        if stop_when_covered and covered >= coverable:
            break

        best_idx = -1
        best_score = 0.0
        best_gain = 0.0
        for i, cov in enumerate(cov_sets):
            if i in chosen:
                continue
            if max_cost is not None and cum_cost + sites[i].cost > max_cost + 1e-9:
                continue
            new = cov - covered
            if not new:
                continue
            gain = math.fsum(weights[j] for j in new)
            if gain <= 0.0:
                continue
            score = gain / sites[i].cost if cost_weighted and sites[i].cost > 0 else gain
            # Strict '>' keeps the earliest site on ties => deterministic.
            if score > best_score:
                best_score = score
                best_gain = gain
                best_idx = i

        if best_idx < 0:
            break

        new_pts = cov_sets[best_idx] - covered
        covered |= cov_sets[best_idx]
        chosen.append(best_idx)
        cum_cost += sites[best_idx].cost
        steps.append(
            PlacementStep(
                site=sites[best_idx].name,
                marginal_weight=best_gain,
                marginal_points=len(new_pts),
                cost=sites[best_idx].cost,
                cumulative_cost=cum_cost,
            )
        )

    return _build_plan(sites, points, chosen, cov_sets, steps)


def greedy_max_coverage(
    sites: list[CandidateSite],
    points: list[DemandPoint],
    k: int,
) -> PlacementPlan:
    """Pick at most ``k`` sensor sites to maximise weighted detection coverage.

    Greedy maximum-coverage: each step adds the site that newly watches the most
    importance-weighted demand. Stops after ``k`` picks or once no remaining site
    adds any coverage. This is the standard ``(1 - 1/e)``-approximation.

    ``k == 0`` yields an empty plan. Coverage/awareness only.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    return _greedy_select(
        sites,
        points,
        max_sites=k,
        max_cost=None,
        cost_weighted=False,
        stop_when_covered=True,
    )


def greedy_set_cover(
    sites: list[CandidateSite],
    points: list[DemandPoint],
) -> PlacementPlan:
    """Pick as few sensor sites as possible to watch every reachable demand point.

    Greedy set-cover: repeatedly add the site covering the most still-unwatched
    weighted demand until every *coverable* point is watched (points outside all
    footprints are reported as an unavoidable gap, never as a target). Awareness
    only.
    """
    return _greedy_select(
        sites,
        points,
        max_sites=None,
        max_cost=None,
        cost_weighted=False,
        stop_when_covered=True,
    )


def budget_max_coverage(
    sites: list[CandidateSite],
    points: list[DemandPoint],
    budget: float,
) -> PlacementPlan:
    """Maximise weighted detection coverage under an emplacement-cost ``budget``.

    Cost-efficient greedy: each step adds the affordable site with the best
    weighted-coverage-per-cost ratio, until the budget cannot fit another useful
    site. ``budget`` must be non-negative. Awareness only.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    return _greedy_select(
        sites,
        points,
        max_sites=None,
        max_cost=budget,
        cost_weighted=True,
        stop_when_covered=True,
    )
