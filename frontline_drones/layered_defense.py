"""Layered-defense coverage-gap & single-sensor-seam assessment (awareness only).

Given a set of *already-deployed* detection sensors — each with a position, a
maximum range and (optionally) a directional arc — this module audits how well
they blanket an area of airspace and, crucially, **how resilient that coverage
is**. It answers three defensive questions about an existing sensor layout:

1. Where are the **uncovered gaps** — points no sensor watches at all?
2. Where are the **single-sensor-dependency seams** — points watched by exactly
   one sensor, so a single outage, mask or maintenance window blinds that
   airspace?
3. Which sensors are **critical** (their loss opens a gap) versus which add
   **no protection** (every point they touch is already redundantly covered)?

This operationalizes the recurring 2024-2026 counter-UAS lesson that
*unintegrated C-UAS layers "added no protection"*: stacking more sensors only
helps if they close real gaps and seams, not if they pile redundant coverage on
airspace that was already double-covered while other sectors stay blind. The
"redundancy depth" of each point — the number of independent sensors that see
it — is the metric that exposes this.

Scope: the output is a *defensive coverage-assurance report*. Sensors here are
passive detection assets (radar / acoustic / EO-IR / RF-DF), never effectors.
"Gaps" and "seams" are holes in **our own detection** to be filled with more or
better-aimed sensors — never openings to exploit for attack. There is no
engagement, weapon, jamming, targeting, fire-control or terminal-guidance
content of any kind. Pure standard library, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geo import enu_offset_m

# A sensor whose field-of-view width is at or above this many degrees is treated
# as omnidirectional (no azimuth arc gate applied).
_OMNI_FOV_DEG = 360.0
# Angular slack (degrees) so a point exactly on an arc edge counts as covered.
_ANGLE_EPS_DEG = 1e-9
# Range slack (metres) so a point exactly on the range ring counts as covered.
_RANGE_EPS_M = 1e-9


@dataclass(frozen=True)
class Sensor:
    """A deployed passive detection sensor with a modeled coverage footprint.

    Coordinates are metres in a local planar (ENU-style) frame; use
    :func:`sensor_from_latlon` to build one from WGS-84 degrees. The footprint
    is an annular sector: azimuths within ``fov_deg`` of ``bearing_deg`` and
    ground ranges in ``[min_range_m, range_m]``. A ``fov_deg`` of 360 (the
    default) models an omnidirectional sensor with no azimuth gate.

    Attributes:
        name: Token identifying the sensor.
        x: East offset, metres.
        y: North offset, metres.
        range_m: Maximum detection range, metres (> 0).
        bearing_deg: Boresight/centre azimuth of the arc, compass degrees
            (north = 0, clockwise), normalised to ``[0, 360)``. Ignored when
            omnidirectional.
        fov_deg: Full angular width of the detection arc, degrees in
            ``(0, 360]``; 360 means omnidirectional.
        min_range_m: Inner blind radius, metres (>= 0, < ``range_m``); models a
            near-field dead zone. Defaults to 0.
        layer: Optional tag for the defensive layer / modality this sensor
            belongs to (e.g. ``"radar"``, ``"acoustic"``). Advisory grouping
            only.
    """

    name: str
    x: float
    y: float
    range_m: float
    bearing_deg: float = 0.0
    fov_deg: float = 360.0
    min_range_m: float = 0.0
    layer: str = ""

    def __post_init__(self) -> None:
        if self.range_m <= 0.0:
            raise ValueError("range_m must be positive")
        if not (0.0 < self.fov_deg <= 360.0):
            raise ValueError("fov_deg must be in (0, 360]")
        if self.min_range_m < 0.0:
            raise ValueError("min_range_m must be non-negative")
        if self.min_range_m >= self.range_m:
            raise ValueError("min_range_m must be less than range_m")
        # Normalise bearing into [0, 360) without touching the frozen instance
        # elsewhere (object.__setattr__ is the sanctioned frozen-dataclass path).
        object.__setattr__(self, "bearing_deg", self.bearing_deg % 360.0)

    @property
    def omnidirectional(self) -> bool:
        """True when the sensor applies no azimuth gate (``fov_deg`` >= 360)."""
        return self.fov_deg >= _OMNI_FOV_DEG

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "range_m": self.range_m,
            "bearing_deg": self.bearing_deg,
            "fov_deg": self.fov_deg,
            "min_range_m": self.min_range_m,
            "layer": self.layer,
        }


@dataclass(frozen=True)
class CoveragePoint:
    """A place whose airspace we want kept under resilient detection coverage.

    Coordinates are metres in a local planar (ENU-style) frame; use
    :func:`point_from_latlon` to build one from WGS-84 degrees.

    Attributes:
        name: Token identifying the point (e.g. a grid cell or defended asset).
        x: East offset, metres.
        y: North offset, metres.
        weight: Relative importance (>= 0); higher means more valuable to keep
            redundantly watched.
    """

    name: str
    x: float
    y: float
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y, "weight": self.weight}


def sensor_from_latlon(
    name: str,
    lat: float,
    lon: float,
    range_m: float,
    ref_lat: float,
    ref_lon: float,
    bearing_deg: float = 0.0,
    fov_deg: float = 360.0,
    min_range_m: float = 0.0,
    layer: str = "",
) -> Sensor:
    """Build a :class:`Sensor` from WGS-84 degrees via a local ENU frame."""
    east, north = enu_offset_m(lat, lon, ref_lat, ref_lon)
    return Sensor(
        name=name,
        x=east,
        y=north,
        range_m=range_m,
        bearing_deg=bearing_deg,
        fov_deg=fov_deg,
        min_range_m=min_range_m,
        layer=layer,
    )


def point_from_latlon(
    name: str,
    lat: float,
    lon: float,
    ref_lat: float,
    ref_lon: float,
    weight: float = 1.0,
) -> CoveragePoint:
    """Build a :class:`CoveragePoint` from WGS-84 degrees via a local ENU frame."""
    east, north = enu_offset_m(lat, lon, ref_lat, ref_lon)
    return CoveragePoint(name=name, x=east, y=north, weight=weight)


def grid_coverage_points(
    width_m: float,
    height_m: float,
    spacing_m: float,
    origin: tuple[float, float] = (0.0, 0.0),
    weight: float = 1.0,
) -> list[CoveragePoint]:
    """Tile a rectangular area with an evenly-spaced grid of coverage points.

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
    points: list[CoveragePoint] = []
    for r in range(n_rows):
        for c in range(n_cols):
            points.append(
                CoveragePoint(
                    name=f"g{r}_{c}",
                    x=ox + c * spacing_m,
                    y=oy + r * spacing_m,
                    weight=weight,
                )
            )
    return points


def _angular_separation_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute separation between two compass bearings, in [0, 180]."""
    d = abs((a_deg - b_deg) % 360.0)
    return min(d, 360.0 - d)


def bearing_to_point_deg(sensor: Sensor, point: CoveragePoint) -> float:
    """Compass bearing (north = 0, clockwise) from ``sensor`` to ``point``.

    Uses the local ENU planar frame (x = east, y = north). Returns 0.0 for a
    point coincident with the sensor, where bearing is undefined.
    """
    dx = point.x - sensor.x
    dy = point.y - sensor.y
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def sees(sensor: Sensor, point: CoveragePoint) -> bool:
    """True when ``point`` lies inside ``sensor``'s annular-sector footprint.

    The point must sit within the range band ``[min_range_m, range_m]`` and,
    unless the sensor is omnidirectional, within ``fov_deg / 2`` of the sensor's
    boresight bearing. Boundaries are inclusive within a tiny tolerance.
    """
    dx = point.x - sensor.x
    dy = point.y - sensor.y
    dist = math.hypot(dx, dy)
    if dist > sensor.range_m + _RANGE_EPS_M:
        return False
    if dist < sensor.min_range_m - _RANGE_EPS_M:
        return False
    if sensor.omnidirectional:
        return True
    if dist == 0.0:
        # At the sensor: inside range band and no meaningful azimuth => covered.
        return True
    brg = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    return _angular_separation_deg(brg, sensor.bearing_deg) <= sensor.fov_deg / 2.0 + _ANGLE_EPS_DEG


def covering_sensors(
    sensors: list[Sensor],
    points: list[CoveragePoint],
) -> list[frozenset[int]]:
    """For each point, the frozenset of sensor indices that see it.

    Index positions in each frozenset correspond to the order of ``sensors``;
    the returned list is aligned with ``points``. This is the reusable coverage
    relation the assessment consumes.
    """
    out: list[frozenset[int]] = []
    for p in points:
        out.append(frozenset(i for i, s in enumerate(sensors) if sees(s, p)))
    return out


def coverage_depth(
    sensors: list[Sensor],
    points: list[CoveragePoint],
) -> list[int]:
    """Number of independent sensors that see each point (its redundancy depth).

    A depth of 0 is an uncovered gap; 1 is a single-sensor-dependency seam; >= 2
    is redundantly / fusion-covered. Aligned with ``points``.
    """
    return [len(c) for c in covering_sensors(sensors, points)]


@dataclass(frozen=True)
class SensorFinding:
    """Per-sensor role in the layered-defense plan.

    Attributes:
        name: Sensor token.
        layer: The sensor's layer/modality tag (may be empty).
        covered_points: Count of coverage points this sensor sees.
        covered_weight: Importance-weighted demand this sensor sees.
        sole_points: Sorted indices this sensor *alone* covers; its loss turns
            each into a gap.
        is_critical: True when ``sole_points`` is non-empty (a single point of
            failure exists behind this sensor).
        adds_no_protection: True when every point it sees is already covered to
            at least the redundancy baseline by *other* sensors, so removing it
            leaves all points at or above ``min_depth`` — the "unintegrated
            layer that added no protection" case (also true if it sees nothing).
    """

    name: str
    layer: str
    covered_points: int
    covered_weight: float
    sole_points: tuple[int, ...]
    is_critical: bool
    adds_no_protection: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "layer": self.layer,
            "covered_points": self.covered_points,
            "covered_weight": round(self.covered_weight, 6),
            "sole_points": list(self.sole_points),
            "is_critical": self.is_critical,
            "adds_no_protection": self.adds_no_protection,
        }


@dataclass(frozen=True)
class CoverageAssessment:
    """Assessment of an existing sensor layout's coverage and resilience.

    All ``*_points`` fields are sorted tuples of indices into the input
    ``points`` list. All ``*_weight`` / ``*_fraction`` fields are
    importance-weighted.

    Attributes:
        n_sensors: Number of sensors assessed.
        n_points: Number of coverage points assessed.
        min_depth: Redundancy baseline; points below it are flagged.
        depths: Per-point redundancy depth, aligned with ``points``.
        gaps: Points with depth 0 (no sensor sees them).
        seams: Points with depth exactly 1 (single-sensor dependency).
        below_min_depth: Covered points with ``0 < depth < min_depth``.
        resilient_points: Points with ``depth >= min_depth``.
        max_depth: Largest per-point depth (0 if no points).
        min_covered_depth: Smallest depth among covered points (0 if none
            covered).
        total_weight: Weighted demand of the whole area.
        covered_weight: Weighted demand with depth >= 1.
        resilient_weight: Weighted demand with depth >= ``min_depth``.
        gap_weight: Weighted demand with depth 0.
        seam_weight: Weighted demand with depth exactly 1.
        coverage_fraction: ``covered_weight / total_weight`` in ``[0, 1]``.
        resilient_fraction: ``resilient_weight / total_weight`` in ``[0, 1]``.
        sensor_findings: Per-sensor role breakdown, aligned with ``sensors``.
        critical_sensors: Names of sensors with sole coverage (single points of
            failure).
        redundant_sensors: Names of sensors that add no protection at the
            baseline.
        advisories: Plain-language notes about gaps, seams and layer integration.
    """

    n_sensors: int
    n_points: int
    min_depth: int
    depths: tuple[int, ...]
    gaps: tuple[int, ...]
    seams: tuple[int, ...]
    below_min_depth: tuple[int, ...]
    resilient_points: tuple[int, ...]
    max_depth: int
    min_covered_depth: int
    total_weight: float
    covered_weight: float
    resilient_weight: float
    gap_weight: float
    seam_weight: float
    coverage_fraction: float
    resilient_fraction: float
    sensor_findings: tuple[SensorFinding, ...] = field(default_factory=tuple)
    critical_sensors: tuple[str, ...] = field(default_factory=tuple)
    redundant_sensors: tuple[str, ...] = field(default_factory=tuple)
    advisories: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_resilient(self) -> bool:
        """True when every point meets the ``min_depth`` redundancy baseline."""
        return not self.gaps and not self.below_min_depth

    def to_dict(self) -> dict:
        return {
            "n_sensors": self.n_sensors,
            "n_points": self.n_points,
            "min_depth": self.min_depth,
            "depths": list(self.depths),
            "gaps": list(self.gaps),
            "seams": list(self.seams),
            "below_min_depth": list(self.below_min_depth),
            "resilient_points": list(self.resilient_points),
            "max_depth": self.max_depth,
            "min_covered_depth": self.min_covered_depth,
            "total_weight": round(self.total_weight, 6),
            "covered_weight": round(self.covered_weight, 6),
            "resilient_weight": round(self.resilient_weight, 6),
            "gap_weight": round(self.gap_weight, 6),
            "seam_weight": round(self.seam_weight, 6),
            "coverage_fraction": round(self.coverage_fraction, 6),
            "resilient_fraction": round(self.resilient_fraction, 6),
            "is_resilient": self.is_resilient,
            "sensor_findings": [f.to_dict() for f in self.sensor_findings],
            "critical_sensors": list(self.critical_sensors),
            "redundant_sensors": list(self.redundant_sensors),
            "advisories": list(self.advisories),
        }


def assess_layered_defense(
    sensors: list[Sensor],
    points: list[CoveragePoint],
    min_depth: int = 2,
) -> CoverageAssessment:
    """Audit a deployed sensor layout for coverage gaps and single-sensor seams.

    For every coverage point this computes its *redundancy depth* (how many
    sensors see it) and classifies it: depth 0 is an uncovered **gap**, depth 1
    is a single-sensor-dependency **seam**, and ``depth >= min_depth`` is
    resilient. Each sensor is then labelled **critical** (it alone covers some
    point, so its loss opens a gap) and/or as adding **no protection** (every
    point it touches already meets the baseline without it — the classic
    unintegrated-layer failure).

    ``min_depth`` is the redundancy baseline (default 2, matching the >=2-source
    fusion baseline used elsewhere in the package); it is clamped to be >= 1.
    Advisory only — reports holes in *our own* detection, never openings to
    exploit.
    """
    min_depth = max(1, int(min_depth))

    cover = covering_sensors(sensors, points)
    depths = [len(c) for c in cover]
    weights = [p.weight for p in points]

    gaps: list[int] = []
    seams: list[int] = []
    below: list[int] = []
    resilient: list[int] = []
    for i, d in enumerate(depths):
        if d == 0:
            gaps.append(i)
        elif d == 1:
            seams.append(i)
        if 0 < d < min_depth:
            below.append(i)
        if d >= min_depth:
            resilient.append(i)

    total_w = math.fsum(weights)
    covered_w = math.fsum(weights[i] for i, d in enumerate(depths) if d >= 1)
    resilient_w = math.fsum(weights[i] for i in resilient)
    gap_w = math.fsum(weights[i] for i in gaps)
    seam_w = math.fsum(weights[i] for i in seams)

    covered_depths = [d for d in depths if d > 0]
    max_depth = max(depths) if depths else 0
    min_covered_depth = min(covered_depths) if covered_depths else 0

    # Per-sensor role: sole coverage (=> critical) and marginal protection.
    sole_by_sensor: list[list[int]] = [[] for _ in sensors]
    covered_by_sensor: list[list[int]] = [[] for _ in sensors]
    for pi, cset in enumerate(cover):
        for si in cset:
            covered_by_sensor[si].append(pi)
        if len(cset) == 1:
            (only,) = tuple(cset)
            sole_by_sensor[only].append(pi)

    findings: list[SensorFinding] = []
    critical: list[str] = []
    redundant: list[str] = []
    for si, s in enumerate(sensors):
        pts = covered_by_sensor[si]
        sole = sorted(sole_by_sensor[si])
        # Adds no protection at the baseline iff removing it keeps every point it
        # touches at >= min_depth, i.e. all its points already have depth >
        # min_depth via other sensors. A sensor covering nothing also qualifies.
        adds_no_protection = all(depths[pi] > min_depth for pi in pts)
        is_critical = bool(sole)
        findings.append(
            SensorFinding(
                name=s.name,
                layer=s.layer,
                covered_points=len(pts),
                covered_weight=math.fsum(weights[pi] for pi in pts),
                sole_points=tuple(sole),
                is_critical=is_critical,
                adds_no_protection=adds_no_protection,
            )
        )
        if is_critical:
            critical.append(s.name)
        if adds_no_protection:
            redundant.append(s.name)

    cov_frac = covered_w / total_w if total_w > 0 else 1.0
    res_frac = resilient_w / total_w if total_w > 0 else 1.0

    advisories: list[str] = []
    if not points:
        advisories.append("No coverage points supplied; nothing to assess.")
    elif not sensors:
        advisories.append("No sensors supplied; the area has zero detection coverage.")
    if gaps:
        advisories.append(
            f"{len(gaps)} coverage point(s) fall outside every sensor footprint — uncovered "
            f"gap(s) in the detection layer; reposition or add sensors to close them."
        )
    if seams:
        advisories.append(
            f"{len(seams)} coverage point(s) are watched by only a single sensor — "
            f"single-sensor-dependency seam(s); if that sensor is down, masked or in "
            f"maintenance, that airspace goes blind. Overlap a second sensor onto each seam."
        )
    only_below = [i for i in below if depths[i] > 1]
    if only_below:
        advisories.append(
            f"{len(only_below)} covered point(s) sit below the min_depth={min_depth} "
            f"redundancy baseline; add overlapping coverage to reach it."
        )
    if critical:
        shown = ", ".join(sorted(set(critical)))
        advisories.append(
            f"Critical single-point-of-failure sensor(s): {shown} — each is the sole cover for "
            f"some airspace, so its loss opens a gap. Prioritise backup coverage of their sectors."
        )
    if redundant and sensors:
        shown = ", ".join(sorted(set(redundant)))
        advisories.append(
            f"Sensor(s) adding no protection at the baseline: {shown} — every point they watch "
            f"is already covered to min_depth by others, so this is the classic 'added layers, "
            f"no added protection' pattern. Reaim or reposition them to close seams/gaps instead."
        )
    if points and sensors and not gaps and not below:
        advisories.append(
            f"All coverage points meet the min_depth={min_depth} redundancy baseline; no "
            f"uncovered gaps and no single-sensor seams remain."
        )

    return CoverageAssessment(
        n_sensors=len(sensors),
        n_points=len(points),
        min_depth=min_depth,
        depths=tuple(depths),
        gaps=tuple(gaps),
        seams=tuple(seams),
        below_min_depth=tuple(below),
        resilient_points=tuple(resilient),
        max_depth=max_depth,
        min_covered_depth=min_covered_depth,
        total_weight=total_w,
        covered_weight=covered_w,
        resilient_weight=resilient_w,
        gap_weight=gap_w,
        seam_weight=seam_w,
        coverage_fraction=cov_frac,
        resilient_fraction=res_frac,
        sensor_findings=tuple(findings),
        critical_sensors=tuple(dict.fromkeys(critical)),
        redundant_sensors=tuple(dict.fromkeys(redundant)),
        advisories=tuple(advisories),
    )
