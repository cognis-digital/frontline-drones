"""Geospatial awareness primitives (situational-awareness only).

Small, dependency-free geodesy for counter-UAS **awareness**: how far away is a
track, what bearing, is it inside a protected zone, and when would a straight-line
track make its closest approach to a defended asset. Everything here answers
"where is it / is it a concern", never "how to engage it". There is no tasking,
cueing-to-effector, guidance or engagement content.

All angles are degrees, all distances metres, all coordinates WGS-84 decimal
degrees. Distances use the haversine great-circle formula on a spherical Earth,
which is accurate to well under 0.5% at the ranges a C-UAS site cares about.
Local displacement/velocity work uses a flat-Earth East-North-Up (ENU) tangent
plane, valid over the few-kilometre footprint of a fixed detection site.

Pure standard library, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Mean Earth radius (metres), WGS-84 authalic sphere.
EARTH_RADIUS_M = 6_371_008.8


def _to_rad(deg: float) -> float:
    return math.radians(deg)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS-84 points, in metres."""
    p1, p2 = _to_rad(lat1), _to_rad(lat2)
    dphi = _to_rad(lat2 - lat1)
    dlmb = _to_rad(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees in [0, 360)."""
    p1, p2 = _to_rad(lat1), _to_rad(lat2)
    dlmb = _to_rad(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def destination_point(
    lat: float, lon: float, bearing_deg: float, distance_m: float
) -> tuple[float, float]:
    """Return the WGS-84 point ``distance_m`` from (lat, lon) along ``bearing_deg``."""
    ang = distance_m / EARTH_RADIUS_M
    brg = _to_rad(bearing_deg)
    p1 = _to_rad(lat)
    l1 = _to_rad(lon)
    p2 = math.asin(
        math.sin(p1) * math.cos(ang) + math.cos(p1) * math.sin(ang) * math.cos(brg)
    )
    l2 = l1 + math.atan2(
        math.sin(brg) * math.sin(ang) * math.cos(p1),
        math.cos(ang) - math.sin(p1) * math.sin(p2),
    )
    lon_out = (math.degrees(l2) + 540.0) % 360.0 - 180.0  # normalise to [-180, 180)
    return round(math.degrees(p2), 9), round(lon_out, 9)


def enu_offset_m(
    lat: float, lon: float, ref_lat: float, ref_lon: float
) -> tuple[float, float]:
    """Local East/North offset (metres) of (lat, lon) from a reference origin.

    Flat-Earth equirectangular approximation on the tangent plane at the
    reference latitude - valid over the few-kilometre footprint of one site.
    """
    east = _to_rad(lon - ref_lon) * math.cos(_to_rad(ref_lat)) * EARTH_RADIUS_M
    north = _to_rad(lat - ref_lat) * EARTH_RADIUS_M
    return east, north


def slant_range_m(ground_range_m: float, altitude_m: float) -> float:
    """3-D slant range from a ground sensor to a target ``altitude_m`` overhead."""
    return math.hypot(ground_range_m, altitude_m)


def elevation_angle_deg(ground_range_m: float, altitude_m: float) -> float:
    """Elevation angle (degrees) to a target at ``altitude_m`` and ``ground_range_m``."""
    if ground_range_m == 0.0:
        return 90.0 if altitude_m > 0 else 0.0
    return math.degrees(math.atan2(altitude_m, ground_range_m))


@dataclass(frozen=True)
class ProtectedZone:
    """A circular protected area to raise awareness around (no engagement).

    Attributes:
        name: Canonical token for the zone.
        lat / lon: Zone centre, WGS-84 decimal degrees.
        radius_m: Protected radius, metres.
        label: Human-readable description.
    """

    name: str
    lat: float
    lon: float
    radius_m: float
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "radius_m": self.radius_m,
            "label": self.label,
        }


@dataclass(frozen=True)
class ZoneStatus:
    """Awareness report of a track's position relative to a protected zone.

    Attributes:
        zone: The zone token evaluated.
        range_m: Great-circle distance from the track to the zone centre.
        inside: True when the track is within ``radius_m`` of the centre.
        range_to_edge_m: Signed distance to the zone boundary (negative inside).
        bearing_from_center_deg: Bearing of the track as seen from the centre.
    """

    zone: str
    range_m: float
    inside: bool
    range_to_edge_m: float
    bearing_from_center_deg: float

    def to_dict(self) -> dict:
        return {
            "zone": self.zone,
            "range_m": round(self.range_m, 3),
            "inside": self.inside,
            "range_to_edge_m": round(self.range_to_edge_m, 3),
            "bearing_from_center_deg": round(self.bearing_from_center_deg, 3),
        }


def zone_status(zone: ProtectedZone, lat: float, lon: float) -> ZoneStatus:
    """Report a single track fix's position relative to ``zone`` (awareness only)."""
    rng = haversine_m(zone.lat, zone.lon, lat, lon)
    return ZoneStatus(
        zone=zone.name,
        range_m=rng,
        inside=rng <= zone.radius_m,
        range_to_edge_m=rng - zone.radius_m,
        bearing_from_center_deg=initial_bearing_deg(zone.lat, zone.lon, lat, lon),
    )


@dataclass(frozen=True)
class ClosestApproach:
    """Closest point of approach (CPA) of a constant-velocity track to an asset.

    Attributes:
        cpa_distance_m: Minimum distance the track reaches from the asset.
        time_to_cpa_s: Seconds from the sample epoch to the CPA (>=0; 0 if the
            track is already at or past its closest point / stationary).
        approaching: True when the track is currently closing on the asset.
        current_distance_m: Distance at the sample epoch.
    """

    cpa_distance_m: float
    time_to_cpa_s: float
    approaching: bool
    current_distance_m: float

    def to_dict(self) -> dict:
        return {
            "cpa_distance_m": round(self.cpa_distance_m, 3),
            "time_to_cpa_s": round(self.time_to_cpa_s, 3),
            "approaching": self.approaching,
            "current_distance_m": round(self.current_distance_m, 3),
        }


def closest_point_of_approach(
    position_enu: tuple[float, float, float],
    velocity_enu: tuple[float, float, float],
    asset_enu: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ClosestApproach:
    """CPA of a constant-velocity track to a static ``asset_enu`` point.

    Solves ``t* = -(r0 . v) / (v . v)`` for the time of minimum separation of a
    point moving at constant ``velocity_enu`` from ``position_enu``. A negative
    ``t*`` (CPA already in the past) or a stationary track clamps
    ``time_to_cpa_s`` to 0 and reports the current separation.

    All vectors are metres / metres-per-second in a local ENU frame. Awareness
    only: reports geometry, never a cue to act on it.
    """
    r0 = tuple(position_enu[i] - asset_enu[i] for i in range(3))
    v = velocity_enu
    current = math.sqrt(sum(c * c for c in r0))
    vv = sum(c * c for c in v)
    if vv == 0.0:
        # Stationary: closest approach is the current position, now.
        return ClosestApproach(current, 0.0, False, current)
    t_star = -sum(r0[i] * v[i] for i in range(3)) / vv
    approaching = t_star > 0.0
    t_eval = max(0.0, t_star)
    at = tuple(r0[i] + v[i] * t_eval for i in range(3))
    cpa = math.sqrt(sum(c * c for c in at))
    return ClosestApproach(cpa, t_eval, approaching, current)


@dataclass(frozen=True)
class ApproachAdvisory:
    """Awareness advisory combining zone status with a CPA projection."""

    zone: str
    inside: bool
    current_range_m: float
    cpa_distance_m: float
    time_to_cpa_s: float
    breaches_zone: bool
    advisories: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "zone": self.zone,
            "inside": self.inside,
            "current_range_m": round(self.current_range_m, 3),
            "cpa_distance_m": round(self.cpa_distance_m, 3),
            "time_to_cpa_s": round(self.time_to_cpa_s, 3),
            "breaches_zone": self.breaches_zone,
            "advisories": list(self.advisories),
        }


def approach_advisory(
    zone: ProtectedZone,
    position_enu: tuple[float, float, float],
    velocity_enu: tuple[float, float, float],
) -> ApproachAdvisory:
    """Combine zone containment and CPA into one awareness advisory.

    ``position_enu`` / ``velocity_enu`` are relative to the zone centre (use
    :func:`enu_offset_m` to build the position). Reports whether the track is
    inside the zone now, and whether its projected CPA breaches the radius.
    Pure warning output - never a mitigation instruction.
    """
    current = math.hypot(math.hypot(position_enu[0], position_enu[1]), position_enu[2])
    cpa = closest_point_of_approach(position_enu, velocity_enu, (0.0, 0.0, 0.0))
    inside = current <= zone.radius_m
    breaches = cpa.cpa_distance_m <= zone.radius_m

    advisories: list[str] = []
    if inside:
        advisories.append(f"Track is INSIDE protected zone '{zone.name}' now.")
    elif breaches and cpa.approaching:
        advisories.append(
            f"Track projected to breach zone '{zone.name}' "
            f"(CPA {cpa.cpa_distance_m:.0f} m < radius {zone.radius_m:.0f} m) "
            f"in ~{cpa.time_to_cpa_s:.0f} s if it holds course."
        )
    elif cpa.approaching:
        advisories.append(
            f"Track closing but projected to stay clear of zone '{zone.name}' "
            f"(CPA {cpa.cpa_distance_m:.0f} m)."
        )
    else:
        advisories.append(f"Track opening from zone '{zone.name}'.")

    return ApproachAdvisory(
        zone=zone.name,
        inside=inside,
        current_range_m=current,
        cpa_distance_m=cpa.cpa_distance_m,
        time_to_cpa_s=cpa.time_to_cpa_s,
        breaches_zone=inside or breaches,
        advisories=tuple(advisories),
    )
