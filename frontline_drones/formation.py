"""Swarm size, spatial-extent and formation-geometry estimation (awareness only).

Once :mod:`frontline_drones.swarm` has grouped a snapshot of tracks into a
coordinated cluster, an analyst still wants the *shape* of that cluster: how many
craft, how wide, how tightly packed, and whether they are flying abreast (a
line), nose-to-tail (a column), in a wedge, in a grid, or as a loose cloud.
Formation geometry changes the detection picture - a tight column threading a
corridor and a wide grid saturating an area are very different awareness
problems - and it is derived purely from the positions (and optionally headings)
the sensor network already holds.

This module estimates, for a set of member tracks:

* **count** - the number of resolved craft (the swarm size estimate);
* **centroid** - the mean member position in the local ENU plane;
* **spatial extent** - bounding box, principal-axis spans, maximum pairwise
  diameter and convex-hull footprint area;
* **compactness / spacing** - nearest-neighbour spacing statistics, dispersion
  about the centroid, packing density and a scale-free compactness index;
* **formation type** - a coarse geometric label (``line`` / ``column`` /
  ``echelon`` / ``wedge`` / ``grid`` / ``cluster`` / ``pair`` / ``single``)
  from a principal-component (covariance) analysis of the member positions plus
  a wedge/vee curvature test.

Scope: this is descriptive geometry for situational awareness only. It reports
*what the formation looks like*, never anything about acting on it - there is no
prediction-for-intercept, guidance, targeting or engagement logic. Pure stdlib,
deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A formation is treated as "regular" (grid-like) when the coefficient of
# variation of its nearest-neighbour spacing stays under this: evenly spaced
# members rather than clumps.
_GRID_REGULARITY = 0.75

# Std-ratio anisotropy above which a formation is effectively a straight line.
_LINE_ANISOTROPY = 0.93
# Anisotropy above which a formation is at least clearly elongated (echelon).
_ELONGATED_ANISOTROPY = 0.60
# |correlation| between along-axis position and lateral offset above which the
# formation opens out from an apex - a wedge / vee.
_WEDGE_SCORE = 0.60

_EPS = 1e-12

# Coarse formation-geometry labels.
FORMATION_TYPES: tuple[str, ...] = (
    "empty",
    "single",
    "pair",
    "line",
    "column",
    "echelon",
    "wedge",
    "grid",
    "cluster",
)


@dataclass(frozen=True)
class Member:
    """One member track's summary state for formation analysis.

    Attributes:
        track_id: Stable identifier for the track.
        east / north: Position in the local ENU plane, metres.
        heading_deg: Course over ground, compass degrees (optional). Used only to
            tell a line (abreast) from a column (nose-to-tail).
    """

    track_id: str
    east: float
    north: float
    heading_deg: float | None = None


def _as_member(item, idx: int) -> Member:
    """Coerce a caller-supplied member into a :class:`Member`."""
    if isinstance(item, Member):
        return item
    # Duck-type any object exposing east/north (e.g. a swarm ``TrackPoint``).
    if hasattr(item, "east") and hasattr(item, "north"):
        heading = getattr(item, "heading_deg", None)
        tid = getattr(item, "track_id", None)
        return Member(
            track_id=str(tid if tid is not None else idx),
            east=float(item.east),
            north=float(item.north),
            heading_deg=None if heading is None else float(heading),
        )
    if isinstance(item, dict):
        tid = str(item.get("track_id", item.get("id", idx)))
        heading = item.get("heading_deg")
        return Member(
            track_id=tid,
            east=float(item.get("east", item.get("x", 0.0))),
            north=float(item.get("north", item.get("y", 0.0))),
            heading_deg=None if heading is None else float(heading),
        )
    seq = list(item)
    east, north = float(seq[0]), float(seq[1])
    heading = float(seq[2]) if len(seq) > 2 else None
    return Member(track_id=str(idx), east=east, north=north, heading_deg=heading)


def centroid(members: list[Member]) -> tuple[float, float]:
    """Return the mean ``(east, north)`` position of the members, ENU metres."""
    n = len(members)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(m.east for m in members) / n
    cy = sum(m.north for m in members) / n
    return (cx, cy)


def bounding_box(members: list[Member]) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box as ``(min_e, min_n, max_e, max_n)``."""
    es = [m.east for m in members]
    ns = [m.north for m in members]
    return (min(es), min(ns), max(es), max(ns))


def max_pairwise_distance(members: list[Member]) -> float:
    """Return the maximum distance between any two members (formation diameter)."""
    n = len(members)
    best = 0.0
    for i in range(n):
        ei, ni = members[i].east, members[i].north
        for j in range(i + 1, n):
            d = math.hypot(members[j].east - ei, members[j].north - ni)
            if d > best:
                best = d
    return best


def nearest_neighbor_spacings(members: list[Member]) -> list[float]:
    """Distance from each member to its closest other member, metres.

    Returns an empty list when fewer than two members are supplied.
    """
    n = len(members)
    if n < 2:
        return []
    out: list[float] = []
    for i in range(n):
        ei, ni = members[i].east, members[i].north
        best = math.inf
        for j in range(n):
            if j == i:
                continue
            d = math.hypot(members[j].east - ei, members[j].north - ni)
            if d < best:
                best = d
        out.append(best)
    return out


def convex_hull(members: list[Member]) -> list[tuple[float, float]]:
    """Convex hull of the member positions as CCW ``(east, north)`` vertices.

    Andrew's monotone-chain algorithm. Deterministic; returns the unique points
    for degenerate (collinear or coincident) inputs.
    """
    pts = sorted({(m.east, m.north) for m in members})
    if len(pts) <= 2:
        return pts

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_area(vertices: list[tuple[float, float]]) -> float:
    """Absolute area of a polygon via the shoelace formula, m^2 (0 if <3 pts)."""
    n = len(vertices)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _covariance_axes(
    members: list[Member], cx: float, cy: float
) -> tuple[float, float, float, float, float]:
    """Principal-axis analysis of the member cloud about its centroid.

    Returns ``(sigma_major, sigma_minor, orientation_deg, ux, uy)`` where the
    sigmas are the standard deviations along the major/minor eigenvectors,
    ``orientation_deg`` is the compass bearing of the major axis in ``[0, 180)``
    and ``(ux, uy)`` is the unit major-axis vector in ENU (east, north).
    """
    n = len(members)
    if n == 0:
        return (0.0, 0.0, 0.0, 1.0, 0.0)
    sxx = sum((m.east - cx) ** 2 for m in members) / n
    syy = sum((m.north - cy) ** 2 for m in members) / n
    sxy = sum((m.east - cx) * (m.north - cy) for m in members) / n
    half = (sxx + syy) / 2.0
    diff = math.sqrt(max(0.0, ((sxx - syy) / 2.0) ** 2 + sxy * sxy))
    lam_major = max(0.0, half + diff)
    lam_minor = max(0.0, half - diff)
    # Major eigenvector orientation.
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)  # radians, math convention
    ux, uy = math.cos(theta), math.sin(theta)  # (east, north) components
    # Compass bearing of the axis, folded into [0, 180).
    bearing = (math.degrees(math.atan2(ux, uy)) + 180.0) % 180.0
    return (math.sqrt(lam_major), math.sqrt(lam_minor), bearing, ux, uy)


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation of two equal-length series; 0 on degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < _EPS or syy < _EPS:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def _angle_between(a_deg: float, b_deg: float) -> float:
    """Smallest absolute angle between two bearings, degrees in ``[0, 180]``."""
    d = abs(a_deg - b_deg) % 360.0
    return d if d <= 180.0 else 360.0 - d


@dataclass(frozen=True)
class FormationReport:
    """Estimated size, extent and geometry of a member set (awareness only).

    Attributes:
        count: Number of resolved member tracks (the swarm-size estimate).
        centroid_east / centroid_north: Mean member position, ENU metres.
        bbox_width_m / bbox_height_m: Axis-aligned bounding-box extents (E, N).
        diameter_m: Maximum distance between any two members.
        major_axis_m / minor_axis_m: Full member span along the principal axes.
        orientation_deg: Compass bearing of the major axis, ``[0, 180)``.
        anisotropy: Elongation of the cloud, ``1 - sigma_minor/sigma_major`` in
            ``[0, 1]`` (0 = round blob, ->1 = a straight line).
        dispersion_m: RMS distance of members from the centroid.
        nn_spacing_mean / nn_spacing_min / nn_spacing_std: Nearest-neighbour
            spacing statistics, metres.
        regularity: ``1 - CV`` of the nearest-neighbour spacing in ``[0, 1]``
            (high = evenly spaced / grid-like, low = clumpy).
        compactness: Scale-free tightness index ``mean_nn / (mean_nn +
            dispersion)`` in ``[0, 1]`` (->1 = members bunched relative to the
            formation size, ->0 = widely dispersed).
        hull_area_m2: Convex-hull footprint area.
        density_per_km2: Members per square kilometre of footprint (0 when the
            footprint is degenerate, e.g. a perfect line).
        formation_type: One of :data:`FORMATION_TYPES`.
    """

    count: int
    centroid_east: float
    centroid_north: float
    bbox_width_m: float
    bbox_height_m: float
    diameter_m: float
    major_axis_m: float
    minor_axis_m: float
    orientation_deg: float
    anisotropy: float
    dispersion_m: float
    nn_spacing_mean: float
    nn_spacing_min: float
    nn_spacing_std: float
    regularity: float
    compactness: float
    hull_area_m2: float
    density_per_km2: float
    formation_type: str

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "centroid_east": round(self.centroid_east, 3),
            "centroid_north": round(self.centroid_north, 3),
            "bbox_width_m": round(self.bbox_width_m, 3),
            "bbox_height_m": round(self.bbox_height_m, 3),
            "diameter_m": round(self.diameter_m, 3),
            "major_axis_m": round(self.major_axis_m, 3),
            "minor_axis_m": round(self.minor_axis_m, 3),
            "orientation_deg": round(self.orientation_deg, 3),
            "anisotropy": round(self.anisotropy, 4),
            "dispersion_m": round(self.dispersion_m, 3),
            "nn_spacing_mean": round(self.nn_spacing_mean, 3),
            "nn_spacing_min": round(self.nn_spacing_min, 3),
            "nn_spacing_std": round(self.nn_spacing_std, 3),
            "regularity": round(self.regularity, 4),
            "compactness": round(self.compactness, 4),
            "hull_area_m2": round(self.hull_area_m2, 3),
            "density_per_km2": round(self.density_per_km2, 4),
            "formation_type": self.formation_type,
        }


def _classify(
    count: int,
    anisotropy: float,
    wedge_score: float,
    regularity: float,
    members: list[Member],
    orientation_deg: float,
) -> str:
    """Assign a coarse formation-geometry label. Deterministic."""
    if count == 0:
        return "empty"
    if count == 1:
        return "single"
    if count == 2:
        return "pair"
    if anisotropy >= _LINE_ANISOTROPY:
        return _line_or_column(members, orientation_deg)
    if wedge_score >= _WEDGE_SCORE and anisotropy >= 0.40:
        return "wedge"
    if anisotropy >= _ELONGATED_ANISOTROPY:
        return "echelon"
    if regularity >= _GRID_REGULARITY and count >= 6:
        return "grid"
    return "cluster"


def _line_or_column(members: list[Member], orientation_deg: float) -> str:
    """Distinguish an abreast line from a nose-to-tail column via heading.

    A *column* travels along its long axis (heading parallel to the major axis);
    a *line* is abreast (heading perpendicular to the major axis). With no
    headings the neutral label ``line`` is used.
    """
    headings = [m.heading_deg for m in members if m.heading_deg is not None]
    if not headings:
        return "line"
    sx = sum(math.sin(math.radians(h)) for h in headings)
    sy = sum(math.cos(math.radians(h)) for h in headings)
    if abs(sx) < _EPS and abs(sy) < _EPS:
        return "line"
    mean_heading = (math.degrees(math.atan2(sx, sy)) + 360.0) % 360.0
    # Axis is undirected, so compare against the [0,180) axis bearing.
    offset = _angle_between(mean_heading % 180.0, orientation_deg)
    offset = min(offset, 180.0 - offset)  # fold to [0, 90]
    return "column" if offset <= 45.0 else "line"


def estimate_formation(members) -> FormationReport:
    """Estimate the size, spatial extent and geometry of a set of member tracks.

    ``members`` is any iterable of :class:`Member`, ``(east, north[, heading])``
    tuples, ``{"east","north","heading_deg"}`` dicts, or any object exposing
    ``east``/``north`` attributes (such as a swarm ``TrackPoint``). The result is
    a purely descriptive :class:`FormationReport` for situational awareness.
    Deterministic; order-independent for the derived geometry.
    """
    pts = [_as_member(m, i) for i, m in enumerate(members)]
    n = len(pts)

    if n == 0:
        return FormationReport(
            count=0,
            centroid_east=0.0,
            centroid_north=0.0,
            bbox_width_m=0.0,
            bbox_height_m=0.0,
            diameter_m=0.0,
            major_axis_m=0.0,
            minor_axis_m=0.0,
            orientation_deg=0.0,
            anisotropy=0.0,
            dispersion_m=0.0,
            nn_spacing_mean=0.0,
            nn_spacing_min=0.0,
            nn_spacing_std=0.0,
            regularity=0.0,
            compactness=0.0,
            hull_area_m2=0.0,
            density_per_km2=0.0,
            formation_type="empty",
        )

    cx, cy = centroid(pts)
    min_e, min_n, max_e, max_n = bounding_box(pts)
    diameter = max_pairwise_distance(pts)

    sig_major, sig_minor, orientation, ux, uy = _covariance_axes(pts, cx, cy)
    anisotropy = 0.0 if sig_major < _EPS else max(0.0, 1.0 - sig_minor / sig_major)

    # Along-axis / lateral projections about the centroid.
    axial = [(m.east - cx) * ux + (m.north - cy) * uy for m in pts]
    lateral = [-(m.east - cx) * uy + (m.north - cy) * ux for m in pts]
    major_axis_m = (max(axial) - min(axial)) if axial else 0.0
    minor_axis_m = (max(lateral) - min(lateral)) if lateral else 0.0
    wedge_score = abs(_pearson(axial, [abs(v) for v in lateral]))

    # Dispersion about the centroid (RMS radius).
    sq = [(m.east - cx) ** 2 + (m.north - cy) ** 2 for m in pts]
    dispersion = math.sqrt(sum(sq) / n)

    # Nearest-neighbour spacing statistics.
    spacings = nearest_neighbor_spacings(pts)
    if spacings:
        nn_mean = sum(spacings) / len(spacings)
        nn_min = min(spacings)
        var = sum((s - nn_mean) ** 2 for s in spacings) / len(spacings)
        nn_std = math.sqrt(var)
    else:
        nn_mean = nn_min = nn_std = 0.0

    regularity = 0.0 if nn_mean < _EPS else max(0.0, 1.0 - nn_std / nn_mean)
    compactness = (
        0.0 if (nn_mean + dispersion) < _EPS else nn_mean / (nn_mean + dispersion)
    )

    hull = convex_hull(pts)
    hull_area = polygon_area(hull)
    density = 0.0 if hull_area < _EPS else n / (hull_area / 1_000_000.0)

    formation_type = _classify(n, anisotropy, wedge_score, regularity, pts, orientation)

    return FormationReport(
        count=n,
        centroid_east=cx,
        centroid_north=cy,
        bbox_width_m=max_e - min_e,
        bbox_height_m=max_n - min_n,
        diameter_m=diameter,
        major_axis_m=major_axis_m,
        minor_axis_m=minor_axis_m,
        orientation_deg=orientation,
        anisotropy=anisotropy,
        dispersion_m=dispersion,
        nn_spacing_mean=nn_mean,
        nn_spacing_min=nn_min,
        nn_spacing_std=nn_std,
        regularity=regularity,
        compactness=compactness,
        hull_area_m2=hull_area,
        density_per_km2=density,
        formation_type=formation_type,
    )
