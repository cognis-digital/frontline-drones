"""Passive multi-sensor bearing/AoA emitter geolocation (awareness only).

A network of passive RF direction-finders (or acoustic bearing sensors) each
measures only a *bearing* to an emitter, not a range. Crossing two or more
bearing lines fixes the emitter's ground position - the classic "cross-fix" /
angle-of-arrival (AoA) triangulation an analyst does on paper, here as a small,
deterministic, stdlib-only solver.

Everything is a passive receive-side geometry calculation. This module localises
a *detected* emitter for situational awareness; it contains no tuning,
transmission, jamming, cueing-to-effector or engagement logic. Coordinates are a
local East-North (ENU) tangent plane in metres; bearings are compass degrees
(0 = North, increasing clockwise), matching :mod:`frontline_drones.geo`.

The N-sensor solve is the standard least-squares intersection of lines: each
bearing defines a line ``p_i + t * d_i``; the point minimising the sum of squared
perpendicular distances to all lines solves ``(sum A_i) x = sum A_i p_i`` where
``A_i = I - d_i d_i^T`` projects onto the direction orthogonal to bearing ``i``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geo import enu_offset_m, initial_bearing_deg

GEOLOCATION_CAVEAT = (
    "Passive bearing geolocation is an awareness aid: multipath, co-channel "
    "emitters and poor sensor geometry (near-parallel bearings) inflate the fix "
    "error. Treat the fix as an estimate with the reported quality, not truth."
)


def bearing_unit_vector(bearing_deg: float) -> tuple[float, float]:
    """Return the ENU (east, north) unit vector pointing along a compass bearing.

    Compass convention: 0 deg = North (0, 1), 90 deg = East (1, 0).
    """
    rad = math.radians(bearing_deg % 360.0)
    return (math.sin(rad), math.cos(rad))


@dataclass(frozen=True)
class BearingObservation:
    """One passive bearing measurement from a sensor at a known ENU position.

    Attributes:
        east / north: Sensor position in the local ENU plane, metres.
        bearing_deg: Measured bearing to the emitter, compass degrees.
        sensor_id: Optional label for the reporting sensor.
        weight: Relative trust in this bearing (higher = tighter). Must be > 0.
    """

    east: float
    north: float
    bearing_deg: float
    sensor_id: str = ""
    weight: float = 1.0

    def unit(self) -> tuple[float, float]:
        """The ENU unit direction of this bearing line."""
        return bearing_unit_vector(self.bearing_deg)


def _perp_distance(px: float, py: float, sx: float, sy: float, dx: float, dy: float) -> float:
    """Perpendicular distance from point (px,py) to the line through (sx,sy) dir (dx,dy)."""
    # Cross product of (point - sensor) with the unit direction.
    return abs((px - sx) * dy - (py - sy) * dx)


def _is_in_front(px: float, py: float, sx: float, sy: float, dx: float, dy: float) -> bool:
    """True when the fix lies in the half-plane the bearing points toward."""
    return (px - sx) * dx + (py - sy) * dy >= 0.0


def intersect_bearings(
    sensor_a: tuple[float, float],
    bearing_a_deg: float,
    sensor_b: tuple[float, float],
    bearing_b_deg: float,
) -> tuple[float, float] | None:
    """Closed-form ENU intersection of two bearing lines.

    Returns the (east, north) crossing point, or ``None`` when the two bearings
    are parallel (or anti-parallel) and therefore never cross. This is the exact
    two-sensor cross-fix; :func:`geolocate` generalises it to N sensors.
    """
    ax, ay = sensor_a
    bx, by = sensor_b
    ue, un = bearing_unit_vector(bearing_a_deg)
    ve, vn = bearing_unit_vector(bearing_b_deg)
    det = ve * un - ue * vn
    if abs(det) < 1e-12:
        return None
    # Solve a + s*u = b + t*v for s via Cramer's rule.
    rex, rey = bx - ax, by - ay
    s = (rex * (-vn) - (-ve) * rey) / det
    return (ax + s * ue, ay + s * un)


@dataclass(frozen=True)
class GeolocationFix:
    """Least-squares emitter fix from >= 2 passive bearings (awareness only).

    Attributes:
        east / north: Estimated emitter position, local ENU metres.
        num_bearings: Number of bearing observations used.
        residual_rms_m: RMS perpendicular distance of the fix to the bearing
            lines - a lower value means the bearings agree well.
        max_residual_m: Largest single-bearing perpendicular residual.
        gdop: Geometric dilution of precision (unitless, >= ~1). Large values flag
            poor geometry (near-parallel bearings) where the fix is ill-conditioned.
        min_subtended_deg: Smallest angle between any pair of bearing lines; small
            angles are the root cause of high GDOP.
        all_in_front: True when the fix lies ahead of every sensor's bearing (a
            fix behind a sensor indicates an ambiguous / spurious solution).
        well_conditioned: Convenience flag: good geometry and consistent bearings.
        caveat: Mandatory awareness caveat.
    """

    east: float
    north: float
    num_bearings: int
    residual_rms_m: float
    max_residual_m: float
    gdop: float
    min_subtended_deg: float
    all_in_front: bool
    well_conditioned: bool
    caveat: str = GEOLOCATION_CAVEAT

    def to_dict(self) -> dict:
        return {
            "east": round(self.east, 3),
            "north": round(self.north, 3),
            "num_bearings": self.num_bearings,
            "residual_rms_m": round(self.residual_rms_m, 3),
            "max_residual_m": round(self.max_residual_m, 3),
            "gdop": round(self.gdop, 4),
            "min_subtended_deg": round(self.min_subtended_deg, 3),
            "all_in_front": self.all_in_front,
            "well_conditioned": self.well_conditioned,
            "caveat": self.caveat,
        }


def _as_observation(item) -> BearingObservation:
    """Coerce a caller-supplied observation into a :class:`BearingObservation`."""
    if isinstance(item, BearingObservation):
        return item
    if isinstance(item, dict):
        return BearingObservation(
            east=float(item.get("east", item.get("x", 0.0))),
            north=float(item.get("north", item.get("y", 0.0))),
            bearing_deg=float(item["bearing_deg"]),
            sensor_id=str(item.get("sensor_id", "")),
            weight=float(item.get("weight", 1.0)),
        )
    seq = list(item)
    east, north, bearing = float(seq[0]), float(seq[1]), float(seq[2])
    weight = float(seq[3]) if len(seq) > 3 else 1.0
    return BearingObservation(east=east, north=north, bearing_deg=bearing, weight=weight)


def _min_subtended_deg(units: list[tuple[float, float]]) -> float:
    """Smallest angle (deg, in [0, 90]) between any pair of (undirected) bearing lines."""
    if len(units) < 2:
        return 0.0
    smallest = 180.0
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            ue, un = units[i]
            ve, vn = units[j]
            dot = max(-1.0, min(1.0, ue * ve + un * vn))
            ang = math.degrees(math.acos(abs(dot)))  # undirected: fold to [0, 90]
            smallest = min(smallest, ang)
    return smallest


def geolocate(observations, *, min_subtended_deg: float = 10.0) -> GeolocationFix:
    """Least-squares emitter geolocation from >= 2 passive bearing observations.

    ``observations`` is any iterable of :class:`BearingObservation`,
    ``(east, north, bearing_deg[, weight])`` tuples, or dicts. Bearings are
    weighted by their ``weight``. ``min_subtended_deg`` is the geometry threshold
    below which the fix is flagged as not ``well_conditioned``.

    Raises:
        ValueError: when fewer than two observations are supplied, or the bearing
            geometry is degenerate (all bearings parallel -> singular system).
    """
    obs = [_as_observation(o) for o in observations]
    if len(obs) < 2:
        raise ValueError("geolocation needs at least two bearing observations")

    # Build the weighted normal system  M x = b,  M = sum w_i A_i.
    m00 = m01 = m11 = 0.0
    b0 = b1 = 0.0
    units: list[tuple[float, float]] = []
    for o in obs:
        w = o.weight
        if w <= 0.0:
            raise ValueError("bearing weight must be positive")
        de, dn = o.unit()
        units.append((de, dn))
        # A_i = I - d d^T ; with unit d this is [[dn^2, -de*dn], [-de*dn, de^2]].
        a00 = dn * dn
        a01 = -de * dn
        a11 = de * de
        m00 += w * a00
        m01 += w * a01
        m11 += w * a11
        b0 += w * (a00 * o.east + a01 * o.north)
        b1 += w * (a01 * o.east + a11 * o.north)

    det = m00 * m11 - m01 * m01
    if abs(det) < 1e-12:
        raise ValueError("degenerate bearing geometry (all bearings parallel)")

    east = (m11 * b0 - m01 * b1) / det
    north = (-m01 * b0 + m00 * b1) / det

    # Residuals: perpendicular distance of the fix to each bearing line.
    residuals = [
        _perp_distance(east, north, o.east, o.north, u[0], u[1])
        for o, u in zip(obs, units, strict=True)
    ]
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    max_res = max(residuals)

    all_in_front = all(
        _is_in_front(east, north, o.east, o.north, u[0], u[1])
        for o, u in zip(obs, units, strict=True)
    )

    subtended = _min_subtended_deg(units)
    # GDOP proxy from the normal matrix: trace of the (unweighted-shape) inverse,
    # normalised so ideal orthogonal geometry approaches 1. As bearings become
    # parallel, det -> 0 and this grows, exactly as dilution of precision should.
    trace_inv = (m00 + m11) / det
    gdop = math.sqrt(abs(trace_inv)) * math.sqrt(len(obs)) if trace_inv > 0 else float("inf")

    well_conditioned = subtended >= min_subtended_deg and all_in_front and rms < 1e6

    return GeolocationFix(
        east=east,
        north=north,
        num_bearings=len(obs),
        residual_rms_m=rms,
        max_residual_m=max_res,
        gdop=gdop,
        min_subtended_deg=subtended,
        all_in_front=all_in_front,
        well_conditioned=well_conditioned,
    )


def bearing_from_positions(
    sensor_lat: float,
    sensor_lon: float,
    emitter_lat: float,
    emitter_lon: float,
) -> float:
    """Compass bearing (deg) from a sensor to an emitter, both WGS-84 degrees.

    Thin wrapper over :func:`frontline_drones.geo.initial_bearing_deg` so callers
    can synthesise bearing observations from known geometry (e.g. for testing a
    detection network's fix accuracy).
    """
    return initial_bearing_deg(sensor_lat, sensor_lon, emitter_lat, emitter_lon)


def observation_from_latlon(
    sensor_lat: float,
    sensor_lon: float,
    bearing_deg: float,
    *,
    ref_lat: float,
    ref_lon: float,
    sensor_id: str = "",
    weight: float = 1.0,
) -> BearingObservation:
    """Build a :class:`BearingObservation` for a sensor given in WGS-84 degrees.

    The sensor's position is projected into the local ENU plane about
    ``(ref_lat, ref_lon)`` using :func:`frontline_drones.geo.enu_offset_m`, so a
    whole detection network sharing one reference origin can be geolocated
    together by :func:`geolocate`.
    """
    east, north = enu_offset_m(sensor_lat, sensor_lon, ref_lat, ref_lon)
    return BearingObservation(
        east=east,
        north=north,
        bearing_deg=bearing_deg,
        sensor_id=sensor_id,
        weight=weight,
    )
