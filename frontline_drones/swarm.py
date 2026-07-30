"""Multi-track swarm / coordinated-cluster detection (awareness only).

A single drone and a coordinated *swarm* are very different detection pictures,
and a swarm is what saturates a defence. Given a snapshot of the tracks a sensor
network is already holding (each with a position and, optionally, a velocity),
this module clusters them spatially and reports whether any cluster looks like a
*coordinated* group - many members, tightly grouped, moving with a coherent
heading and speed.

Scope: descriptive grouping for situational awareness only. It answers "how many
distinct groups are up, how big, how coherent", never anything about acting on
them. Spatial clustering is single-link (union-find) within a link radius; heading
coherence is the length of the mean unit-heading vector (1 = perfectly aligned,
0 = uniformly scattered). Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackPoint:
    """One track's summary state for swarm analysis.

    Attributes:
        track_id: Stable identifier for the track.
        east / north: Position in the local ENU plane, metres.
        heading_deg: Course over ground, compass degrees (optional).
        speed_mps: Ground speed, m/s (optional).
    """

    track_id: str
    east: float
    north: float
    heading_deg: float | None = None
    speed_mps: float | None = None


def _as_point(item, idx: int) -> TrackPoint:
    """Coerce a caller-supplied track into a :class:`TrackPoint`."""
    if isinstance(item, TrackPoint):
        return item
    if isinstance(item, dict):
        tid = str(item.get("track_id", item.get("id", idx)))
        heading = item.get("heading_deg")
        speed = item.get("speed_mps")
        return TrackPoint(
            track_id=tid,
            east=float(item.get("east", item.get("x", 0.0))),
            north=float(item.get("north", item.get("y", 0.0))),
            heading_deg=None if heading is None else float(heading),
            speed_mps=None if speed is None else float(speed),
        )
    seq = list(item)
    east, north = float(seq[0]), float(seq[1])
    heading = float(seq[2]) if len(seq) > 2 else None
    speed = float(seq[3]) if len(seq) > 3 else None
    return TrackPoint(track_id=str(idx), east=east, north=north,
                      heading_deg=heading, speed_mps=speed)


def heading_coherence(headings: list[float]) -> float:
    """Return the mean-resultant length of a set of compass headings, in [0, 1].

    1.0 means every heading is identical; 0.0 means they cancel out (uniformly
    spread). This is the standard circular-statistics measure of directional
    concentration and is the core of the "are they flying together" test.
    """
    vals = [h for h in headings if h is not None]
    if not vals:
        return 0.0
    sx = sum(math.sin(math.radians(h)) for h in vals)
    sy = sum(math.cos(math.radians(h)) for h in vals)
    return math.hypot(sx, sy) / len(vals)


def mean_heading_deg(headings: list[float]) -> float | None:
    """Circular mean of compass headings (deg), or ``None`` when none are given."""
    vals = [h for h in headings if h is not None]
    if not vals:
        return None
    sx = sum(math.sin(math.radians(h)) for h in vals)
    sy = sum(math.cos(math.radians(h)) for h in vals)
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:
        return None
    return (math.degrees(math.atan2(sx, sy)) + 360.0) % 360.0


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


@dataclass(frozen=True)
class Cluster:
    """One spatial cluster of tracks and its coordination summary.

    Attributes:
        track_ids: Member track ids, sorted.
        size: Number of members.
        centroid_east / centroid_north: Mean member position, ENU metres.
        radius_m: Distance from the centroid to the farthest member.
        spread_m: Mean member distance from the centroid.
        heading_coherence: Directional concentration of member headings, [0, 1].
        mean_heading_deg: Circular-mean heading, or None if unknown.
        mean_speed_mps: Mean member speed, or None if unknown.
        coordinated: True when the cluster meets the size + coherence thresholds.
    """

    track_ids: tuple[str, ...]
    size: int
    centroid_east: float
    centroid_north: float
    radius_m: float
    spread_m: float
    heading_coherence: float
    mean_heading_deg: float | None
    mean_speed_mps: float | None
    coordinated: bool

    def to_dict(self) -> dict:
        return {
            "track_ids": list(self.track_ids),
            "size": self.size,
            "centroid_east": round(self.centroid_east, 3),
            "centroid_north": round(self.centroid_north, 3),
            "radius_m": round(self.radius_m, 3),
            "spread_m": round(self.spread_m, 3),
            "heading_coherence": round(self.heading_coherence, 4),
            "mean_heading_deg": (
                None if self.mean_heading_deg is None else round(self.mean_heading_deg, 3)
            ),
            "mean_speed_mps": (
                None if self.mean_speed_mps is None else round(self.mean_speed_mps, 3)
            ),
            "coordinated": self.coordinated,
        }


@dataclass(frozen=True)
class SwarmReport:
    """Full swarm-awareness summary over a snapshot of tracks.

    Attributes:
        num_tracks: Total tracks analysed.
        clusters: All clusters, largest first.
        num_clusters: Count of clusters.
        largest_cluster_size: Size of the biggest cluster (0 when no tracks).
        swarm_detected: True when at least one cluster is ``coordinated``.
        coordinated_clusters: Just the coordinated clusters.
    """

    num_tracks: int
    clusters: tuple[Cluster, ...]
    num_clusters: int
    largest_cluster_size: int
    swarm_detected: bool
    coordinated_clusters: tuple[Cluster, ...]

    def to_dict(self) -> dict:
        return {
            "num_tracks": self.num_tracks,
            "num_clusters": self.num_clusters,
            "largest_cluster_size": self.largest_cluster_size,
            "swarm_detected": self.swarm_detected,
            "clusters": [c.to_dict() for c in self.clusters],
            "coordinated_clusters": [c.to_dict() for c in self.coordinated_clusters],
        }


def cluster_tracks(points: list[TrackPoint], link_radius_m: float) -> list[list[int]]:
    """Single-link spatial clustering: return member index lists per cluster.

    Two tracks are linked when they are within ``link_radius_m`` of each other;
    clusters are the connected components of that link graph. Result groups are
    sorted largest-first, ties broken by smallest first-member index.
    """
    n = len(points)
    uf = _UnionFind(n)
    r2 = link_radius_m * link_radius_m
    for i in range(n):
        for j in range(i + 1, n):
            de = points[i].east - points[j].east
            dn = points[i].north - points[j].north
            if de * de + dn * dn <= r2:
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    out = list(groups.values())
    out.sort(key=lambda g: (-len(g), g[0]))
    return out


def _summarize(members: list[TrackPoint], min_size: int, coherence_threshold: float) -> Cluster:
    size = len(members)
    cx = sum(m.east for m in members) / size
    cy = sum(m.north for m in members) / size
    dists = [math.hypot(m.east - cx, m.north - cy) for m in members]
    radius = max(dists)
    spread = sum(dists) / size
    headings = [m.heading_deg for m in members]
    coherence = heading_coherence(headings)
    mean_hdg = mean_heading_deg(headings)
    speeds = [m.speed_mps for m in members if m.speed_mps is not None]
    mean_spd = (sum(speeds) / len(speeds)) if speeds else None
    have_headings = any(h is not None for h in headings)
    coordinated = size >= min_size and (
        coherence >= coherence_threshold if have_headings else True
    )
    return Cluster(
        track_ids=tuple(sorted(m.track_id for m in members)),
        size=size,
        centroid_east=cx,
        centroid_north=cy,
        radius_m=radius,
        spread_m=spread,
        heading_coherence=coherence,
        mean_heading_deg=mean_hdg,
        mean_speed_mps=mean_spd,
        coordinated=coordinated,
    )


def analyze_swarm(
    tracks,
    *,
    link_radius_m: float = 150.0,
    min_size: int = 3,
    coherence_threshold: float = 0.7,
) -> SwarmReport:
    """Cluster a snapshot of tracks and flag coordinated swarms (awareness only).

    ``tracks`` is any iterable of :class:`TrackPoint`, ``(east, north[, heading[,
    speed]])`` tuples, or dicts. A cluster is flagged ``coordinated`` when it has
    at least ``min_size`` members and (if headings are known) a heading coherence
    of at least ``coherence_threshold``. Deterministic.
    """
    points = [_as_point(t, i) for i, t in enumerate(tracks)]
    if not points:
        return SwarmReport(0, (), 0, 0, False, ())
    if link_radius_m <= 0:
        raise ValueError("link_radius_m must be positive")

    groups = cluster_tracks(points, link_radius_m)
    clusters = tuple(
        _summarize([points[i] for i in g], min_size, coherence_threshold) for g in groups
    )
    coordinated = tuple(c for c in clusters if c.coordinated)
    largest = max((c.size for c in clusters), default=0)
    return SwarmReport(
        num_tracks=len(points),
        clusters=clusters,
        num_clusters=len(clusters),
        largest_cluster_size=largest,
        swarm_detected=bool(coordinated),
        coordinated_clusters=coordinated,
    )
